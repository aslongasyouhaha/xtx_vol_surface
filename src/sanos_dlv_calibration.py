"""Direct nonnegative DLV calibration with banded forward/adjoint solves.

This is our nonlinear calibration implementation, not the paper's LP algorithm.
The reported objective is the paper's original inverse-spread outside loss plus
epsilon midpoint loss. Smoothed-hinge stages initialize a final exact-loss stage.
The final DLV problem has no probability floor, clipping, density interpolation
or mass renormalization; optional interior constraints are initialization only.
"""
import time
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import minimize,linprog,brentq
from scipy import sparse
from sanos_core import call_kernel,paper_spread_weights,build_convex_order_problem
from sanos_dlv import DLVModel,_grid,recover_local_volatilities
from sanos_surface import audit_surface


def calibration_grid(inputs,*,central_points=121,wing_points=31):
    """Common grid independent of fitted LP atoms; resolution is configurable."""
    if central_points<3 or wing_points<2:raise ValueError('Insufficient calibration grid resolution')
    lo=min(float(np.min(v['anchor_k'])) for v in inputs.values())
    hi=max(float(np.max(v['anchor_k'])) for v in inputs.values())
    if not 0<lo<1<hi:raise ValueError('Model support must bracket one')
    left=max(lo,.85);right=min(hi,1.15)
    return np.unique(np.r_[np.linspace(lo,left,wing_points),np.linspace(left,right,central_points),
                           np.linspace(right,hi,wing_points),1.])


def _hinge(value,delta):
    if delta==0:return np.maximum(value,0),(value>0).astype(float)
    positive=np.maximum(value,0)
    return np.where(positive<delta,positive**2/(2*delta),positive-delta/2),np.minimum(positive/delta,1)


class DLVCalibrator:
    def __init__(self,inputs,anchors=None,*,mid_penalty=1e-8):
        if not inputs:raise ValueError('No quotes supplied')
        if not np.isfinite(mid_penalty) or mid_penalty<0:raise ValueError('Invalid midpoint penalty')
        self.items=dict(sorted(inputs.items()));self.exdates=list(self.items)
        self.anchors=_grid(calibration_grid(inputs) if anchors is None else anchors)
        at=np.flatnonzero(self.anchors==1.)
        if len(at)!=1:raise ValueError('Common grid must contain state one exactly')
        self.initial=np.zeros(len(self.anchors));self.initial[at[0]]=1.
        self.times=np.array([float(v.get('T',v['dte']/365.)) for v in self.items.values()])
        self.dt=np.diff(np.r_[0.,self.times]);self.variances=np.array([v['smooth_variance'] for v in self.items.values()])
        if np.any(~np.isfinite(self.times)) or np.any(self.dt<=0):raise ValueError('Times must be positive and increasing')
        if np.any(~np.isfinite(self.variances)) or np.any(self.variances<0) or np.any(np.diff(self.variances)<0):
            raise ValueError('Smoothing variances must be nonnegative and nondecreasing')
        self.kernels=[];self.bids=[];self.asks=[];self.mids=[];self.spreads=[]
        for v in self.items.values():
            k=np.asarray(v['market_k'],float);b=np.asarray(v['pure_bid'],float);a=np.asarray(v['pure_ask'],float)
            mid=np.asarray(v['pure_mid'],float)
            if k.ndim!=1 or not len(k) or b.shape!=k.shape or a.shape!=k.shape or mid.shape!=k.shape:
                raise ValueError('Invalid quote shapes')
            if np.any(~np.isfinite(k)) or np.any(k<=0) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(mid)):
                raise ValueError('Nonfinite quotes/strikes')
            paper_spread_weights(b,a)
            self.kernels.append(call_kernel(self.anchors,k,v['smooth_variance']))
            self.bids.append(b);self.asks.append(a);self.mids.append(mid);self.spreads.append(a-b)
        self.nquotes=sum(len(b) for b in self.bids);self.mid_penalty=float(mid_penalty)
        a=self.anchors;self.om_minus=a[1:-1]**2/((a[2:]-a[:-2])*(a[1:-1]-a[:-2]))
        self.om_plus=a[1:-1]**2/((a[2:]-a[:-2])*(a[2:]-a[1:-1]))
        # Initial variance-rate scaling changes coordinates, not constraints or weights.
        # A positive initial guess avoids an absorbing all-zero initial diffusion.
        atm=np.array([float(v.get('V_j',np.asarray(v['quotes'].atm_total_variance)[0]
                       if 'quotes' in v else max(v['smooth_variance']*4,.001))) for v in self.items.values()])
        forward=np.diff(np.r_[0.,atm-self.variances])/self.dt
        self.rate_scale=np.maximum(forward,.0025)
        self.shape=(len(self.items),len(self.anchors)-2)
        self.parameter_scale=np.broadcast_to(self.rate_scale[:,None],self.shape).copy()

    def volatilities(self,parameters):
        return np.sqrt(np.asarray(parameters).reshape(self.shape)*self.parameter_scale)

    def rescale_parameters(self,parameters):
        """Change coordinates around a warm start without changing the model."""
        rate=np.asarray(parameters).reshape(self.shape)*self.parameter_scale
        if np.any(~np.isfinite(rate)) or np.any(rate<0):raise ValueError('Invalid parameter rescaling')
        self.parameter_scale=np.maximum(rate,self.rate_scale[:,None]*.01)
        return (rate/self.parameter_scale).ravel()

    def interior_lp_initialization(self,*,loss_allowance=.1,time_limit=15.):
        """Choose a strictly interior warm start within an explicit LP-loss budget.

        The positive reference is a unit-mean probability distribution and is
        used only in initialization. The final nonlinear problem has no density
        floor or reference penalty. Failure never silently clips LP weights.
        """
        if not np.isfinite(loss_allowance) or loss_allowance<=0:raise ValueError('Positive finite seed allowance required')
        if not np.isfinite(time_limit) or time_limit<=0:raise ValueError('Positive finite seed time budget required')
        started=time.perf_counter();deadline=started+time_limit
        items={ex:dict(v,anchor_k=self.anchors.copy()) for ex,v in self.items.items()}
        problem=build_convex_order_problem(items)
        # The supplied endpoints already lie outside all observations; no extra states.
        if any(not np.array_equal(v['anchor_k'],self.anchors) for v in problem['items'].values()):
            raise ValueError('Seed support differs from calibration grid')
        A=problem['A'];E=problem['E'];b=problem['b'];cost=problem['c']/self.nquotes
        options=dict(presolve=True,primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10,time_limit=time_limit)
        control=linprog(cost,A_ub=A,b_ub=b,A_eq=E,b_eq=np.ones(E.shape[0]),bounds=(0,None),method='highs',options=options)
        if not control.success:raise ValueError('Initial LP control failed: '+control.message)
        a=self.anchors
        def tilted(lam):
            logw=lam*(a-1);w=np.exp(logw-logw.max());return w/w.sum()
        lam=brentq(lambda x:a@tilted(x)-1,-100.,100.)
        reference=tilted(lam)
        rows=[];cols=[];values=[];rhs=[];r=0;reference_vector=np.zeros(problem['cursor'])
        # Reference marginals increase in convex order from the unit point mass.
        for j,ex in enumerate(problem['exdates']):
            blend=(j+1)/(len(self.exdates)+1)
            target=blend*reference+(1-blend)*self.initial
            q0,q1=problem['q_offsets'][ex]
            reference_vector[q0:q1]=target
            for i in range(len(a)):
                rows.extend([r,r]);cols.extend([q0+i,problem['cursor']]);values.extend([-1.,target[i]])
                rhs.append(0.);r+=1
        positive=sparse.coo_matrix((values,(rows,cols)),shape=(r,problem['cursor']+1)).tocsr()
        # Keep the residual mixture component in convex order too, providing
        # positive calendar margins wherever the reference has time value.
        margin=np.zeros(A.shape[0]);ne=E.shape[0]
        cal0,cal1=problem['calendar_rows'];cal0-=ne;cal1-=ne
        margin[cal0:cal1]=np.maximum(-(A@reference_vector)[cal0:cal1],0.)
        augmented=sparse.vstack([sparse.hstack([A,sparse.csr_matrix(margin[:,None])]),positive,
            sparse.csr_matrix(np.r_[cost,0.][None,:])],format='csr')
        upper=np.r_[b,rhs,control.fun+loss_allowance/self.nquotes]
        eq=sparse.hstack([E,sparse.csr_matrix((E.shape[0],1))],format='csr')
        target=np.zeros(problem['cursor']+1);target[-1]=-1
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise ValueError('LP initialization time budget exhausted')
        options['time_limit']=remaining
        try:
            import highspy
        except ImportError:
            highspy=None
        if highspy is None:
            seed=linprog(target,A_ub=augmented,b_ub=upper,A_eq=eq,b_eq=np.ones(E.shape[0]),
                         bounds=[(0,None)]*problem['cursor']+[(0,1)],method='highs-ipm',options=options)
        else:
            from types import SimpleNamespace
            matrix=sparse.vstack([eq,augmented],format='csc');ne=eq.shape[0]
            h=highspy.Highs()
            for name,value in dict(output_flag=False,threads=1,solver='ipm',presolve='on',simplex_scale_strategy=0,
                primal_feasibility_tolerance=1e-10,dual_feasibility_tolerance=1e-10,
                ipm_optimality_tolerance=1e-10,time_limit=remaining).items():
                if h.setOptionValue(name,value)==highspy.HighsStatus.kError:raise ValueError('HiGHS option error: '+name)
            lp=highspy.HighsLp();lp.num_col_=len(target);lp.num_row_=matrix.shape[0]
            lp.col_cost_=target;lp.col_lower_=np.zeros(len(target));lp.col_upper_=np.r_[np.full(len(target)-1,np.inf),1.]
            lp.row_lower_=np.r_[np.ones(ne),np.full(len(upper),-np.inf)];lp.row_upper_=np.r_[np.ones(ne),upper]
            lp.a_matrix_.format_=highspy.MatrixFormat.kColwise;lp.a_matrix_.start_=matrix.indptr.astype(np.int32)
            lp.a_matrix_.index_=matrix.indices.astype(np.int32);lp.a_matrix_.value_=matrix.data
            h.passModel(lp);h.run();raw=h.getSolution();status=h.getModelStatus()
            seed=SimpleNamespace(success=status==highspy.HighsModelStatus.kOptimal,x=np.asarray(raw.col_value),message=h.modelStatusToString(status))
        if not seed.success or seed.x[-1]<=0:raise ValueError('No strictly positive LP initialization: '+seed.message)
        previous=self.initial;sigmas=[]
        for j,ex in enumerate(problem['exdates']):
            q=seed.x[slice(*problem['q_offsets'][ex])]
            sigma=recover_local_volatilities(a,previous,q,self.dt[j],tolerance=2e-7)
            sigmas.append(sigma[1:-1]);previous=q
        parameters=(np.asarray(sigmas)**2/self.parameter_scale).ravel()
        evaluation=self.evaluate(parameters,gradient=False)
        return parameters,dict(method='interior_lp',seconds=time.perf_counter()-started,
            reference_mass=float(seed.x[-1]),loss_allowance=loss_allowance,
            calendar_reference_margin=True,
            lp_control_objective=float(control.fun*self.nquotes),seed_objective=evaluation['objective'],
            min_seed_probability=float(min(seed.x[slice(*o)].min() for o in problem['q_offsets'].values())))

    def evaluate(self,parameters,*,smoothing=0.,gradient=True):
        """Exact adjoint gradient; smoothed loss is globally scaled by quote count."""
        u=np.asarray(parameters,float).reshape(self.shape)
        if np.any(~np.isfinite(u)) or np.any(u<0):raise ValueError('Nonnegative finite variance-rate parameters required')
        if not np.isfinite(smoothing) or smoothing<0:raise ValueError('Invalid hinge smoothing')
        previous=self.initial;weights=[];bands=[];fits=[];gq=[];loss=0.;actual=0.
        for j in range(len(self.times)):
            rate=u[j]*self.parameter_scale[j];minus=self.om_minus*rate*self.dt[j];plus=self.om_plus*rate*self.dt[j]
            ab=np.zeros((3,len(self.anchors)));ab[1]=1.
            ab[0,1:-1]=-minus;ab[1,1:-1]+=minus+plus;ab[2,1:-1]=-plus
            q=solve_banded((1,1),ab,previous,check_finite=True)
            if (np.any(~np.isfinite(q)) or q.min() < -2e-7 or abs(q.sum()-1)>2e-7
                or abs(self.anchors@q-1)>2e-7):
                raise FloatingPointError('Banded propagation failed probability/moment audit')
            fitted=self.kernels[j]@q;bid=self.bids[j];ask=self.asks[j];spread=self.spreads[j]
            upper=(fitted-ask)/spread;lower=(bid-fitted)/spread;mid=(fitted-self.mids[j])/spread
            hu,du=_hinge(upper,smoothing);hl,dl=_hinge(lower,smoothing)
            if smoothing:
                smooth_abs=np.hypot(mid,smoothing);hm=smooth_abs-smoothing;dm=mid/smooth_abs
            else:hm=abs(mid);dm=np.sign(mid)
            loss+=float(np.sum(hu+hl+self.mid_penalty*hm))
            actual+=float(np.sum(np.maximum(upper,0)+np.maximum(lower,0)+self.mid_penalty*abs(mid)))
            if gradient:gq.append(self.kernels[j].T@((du-dl+self.mid_penalty*dm)/spread)/self.nquotes)
            weights.append(q);bands.append(ab);fits.append(fitted);previous=q
        result=dict(objective=actual,optimization_loss=loss/self.nquotes,weights=np.asarray(weights),fits=fits)
        if gradient:
            future=np.zeros(len(self.anchors));grad=np.zeros(self.shape)
            for j in range(len(self.times)-1,-1,-1):
                ab=bands[j];transposed=np.zeros_like(ab)
                transposed[0,1:]=ab[2,:-1];transposed[1]=ab[1];transposed[2,:-1]=ab[0,1:]
                adjoint=solve_banded((1,1),transposed,gq[j]+future,check_finite=True)
                omega=(self.om_minus*(adjoint[1:-1]-adjoint[:-2])
                       +self.om_plus*(adjoint[1:-1]-adjoint[2:]))
                grad[j]=-self.dt[j]*weights[j][1:-1]*omega*self.parameter_scale[j]
                future=adjoint
            result['gradient']=grad.ravel()
        return result

    def calibrate(self,*,time_limit=30.,max_iterations=600,initial_parameters=None):
        if not np.isfinite(time_limit) or time_limit<=0:raise ValueError('Positive finite time budget required')
        if max_iterations<1:raise ValueError('max_iterations must be positive')
        started=time.perf_counter();deadline=started+time_limit
        x=np.ones(np.prod(self.shape)) if initial_parameters is None else np.asarray(initial_parameters,float).ravel().copy()
        initial=self.evaluate(x,gradient=False);best=dict(x=x.copy(),evaluation=initial)
        attempts=[];evaluations=0;invalid_evaluations=0;stop='stages_complete';last=None
        class BudgetExpired(Exception):pass
        for delta in [.05,.005,0.]:
            stage_start=time.perf_counter()
            def fun(value):
                nonlocal evaluations,invalid_evaluations
                if time.perf_counter()>=deadline:raise BudgetExpired()
                try:result=self.evaluate(value,smoothing=delta)
                except (FloatingPointError,np.linalg.LinAlgError):
                    invalid_evaluations+=1
                    return 1e50,np.zeros_like(value)
                evaluations+=1
                if result['objective']<best['evaluation']['objective']:
                    best.update(x=value.copy(),evaluation=result)
                return result['optimization_loss'],result['gradient']
            try:
                last=minimize(fun,x,method='L-BFGS-B',jac=True,bounds=[(0,None)]*len(x),
                              options=dict(maxiter=max_iterations,maxls=30,ftol=1e-13,gtol=1e-9,maxcor=20))
                x=last.x
                attempts.append(dict(smoothing=delta,seconds=time.perf_counter()-stage_start,success=bool(last.success),
                    message=str(last.message),iterations=last.nit,evaluations=last.nfev))
            except BudgetExpired:
                stop='time_limit';attempts.append(dict(smoothing=delta,seconds=time.perf_counter()-stage_start,
                                                     success=False,message='Shared time budget exhausted'));break
        selected=self.evaluate(best['x'],gradient=False)
        sigma=self.volatilities(best['x'])
        model=DLVModel(self.anchors,self.times,sigma,self.variances,tolerance=2e-7)
        audit=audit_surface(model.surface());raw_weights=model.weights
        audit['max_mass_error']=float(np.max(abs(raw_weights.sum(axis=1)-1)))
        audit['max_mean_error']=float(np.max(abs(raw_weights@self.anchors-1)))
        audit['min_probability']=float(raw_weights.min())
        transitions=[t.audit(tolerance=2e-7) for t in model.transitions]
        audit['transition_audits_passed']=all(t['passed'] for t in transitions)
        audit['max_transition_residual']=max(max(t['errors'].values()) for t in transitions)
        fits=[G@q for G,q in zip(self.kernels,raw_weights)]
        audit['max_forward_price_difference']=max(float(np.max(abs(a-b))) for a,b in zip(fits,selected['fits']))
        actual=sum(float(np.sum((np.maximum(b-fit,0)+np.maximum(fit-a,0)+self.mid_penalty*abs(fit-mid))/spread))
                   for fit,b,a,mid,spread in zip(fits,self.bids,self.asks,self.mids,self.spreads))
        inside=sum(int((np.maximum(b-fit,0)+np.maximum(fit-a,0)<=1e-9).sum())
                   for fit,b,a in zip(fits,self.bids,self.asks))
        last_is_selected=last is not None and np.array_equal(last.x,best['x'])
        converged=bool(stop=='stages_complete' and last is not None and last.success and last_is_selected)
        fit_target=inside==self.nquotes
        validated=bool(audit['passed'] and audit['transition_audits_passed'] and audit['max_mass_error']<=2e-7 and audit['max_mean_error']<=2e-7
                       and audit['min_probability']>=-2e-7)
        return dict(success=validated and (fit_target or converged),validated=validated,optimizer_converged=converged,
                    optimizer_reported_success=bool(last is not None and last.success),
                    selected_point_is_last_optimizer_result=last_is_selected,
                    fit_target_met=fit_target,stop_reason=stop,model=model,parameters=best['x'],
                    fits=fits,objective=actual,initial_objective=initial['objective'],
                    coverage=inside/self.nquotes,attempts=attempts,evaluations=evaluations,
                    invalid_evaluations=invalid_evaluations,
                    request_seconds=time.perf_counter()-started,audit=audit,
                    local_optimization=True,global_optimality_certified=False)
