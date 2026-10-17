"""Discrete local volatility: martingale transitions, recovery and nonlinear calibration."""

import numpy as np
import time as _time
from scipy.linalg import solve_banded
from scipy import sparse
from sanos_surface import SanosSurface, audit_surface
from scipy.optimize import minimize, linprog, brentq
from sanos_core import call_kernel, paper_spread_weights, build_convex_order_problem

def _grid(anchors):
    a=np.array(anchors,float,copy=True)
    if a.ndim!=1 or len(a)<3 or np.any(~np.isfinite(a)) or np.any(a<=0) or np.any(np.diff(a)<=0):
        raise ValueError('At least three finite positive increasing states required')
    return a


def latent_calls(anchors,weights):
    """Unsmoothed calls at the same states in O(N), via suffix moments."""
    a=_grid(anchors);q=np.asarray(weights,float)
    if q.shape!=a.shape or np.any(~np.isfinite(q)):raise ValueError('Invalid marginal shape or values')
    mass=np.r_[np.cumsum(q[:0:-1])[::-1],0.]
    moment=np.r_[np.cumsum((a*q)[:0:-1])[::-1],0.]
    return moment-a*mass


class DLVTransition:
    def __init__(self,anchors,local_volatilities,dt):
        self.anchors=_grid(anchors);n=len(self.anchors)
        sigma=np.asarray(local_volatilities,float)
        if sigma.shape==(n-2,):sigma=np.r_[0.,sigma,0.]
        if sigma.shape!=(n,) or np.any(~np.isfinite(sigma)) or np.any(sigma<0):
            raise ValueError('Nonnegative finite interior DLV values required')
        if sigma[0]!=0 or sigma[-1]!=0:raise ValueError('Boundary DLV values must be zero')
        self.sigma=sigma.copy();self.dt=float(dt)
        if not np.isfinite(self.dt) or self.dt<0:raise ValueError('dt must be finite and nonnegative')
        a=self.anchors;minus=np.zeros(n);plus=np.zeros(n)
        with np.errstate(over='ignore',invalid='ignore'):
            factor=(sigma[1:-1]*a[1:-1])**2*self.dt/(a[2:]-a[:-2])
            minus[1:-1]=factor/(a[1:-1]-a[:-2]);plus[1:-1]=factor/(a[2:]-a[1:-1])
        if np.any(~np.isfinite(minus)) or np.any(~np.isfinite(plus)):
            raise ValueError('DLV coefficients overflowed')
        self._bands=np.zeros((3,n))
        self._bands[0,1:]=-minus[1:]
        self._bands[1]=1+minus+plus
        self._bands[2,:-1]=-plus[:-1]
        self.matrix=sparse.diags([-plus[:-1],1+minus+plus,-minus[1:]],[-1,0,1],format='csc')
        for value in [self.anchors,self.sigma,self._bands]:value.flags.writeable=False

    def apply(self,weights):
        """Apply Q by solving Q^{-1} x=weights; do not clip or renormalize x."""
        q=np.asarray(weights,float)
        if q.ndim not in {1,2} or q.shape[0]!=len(self.anchors) or np.any(~np.isfinite(q)):
            raise ValueError('RHS must have shape (N,) or (N, number_of_rhs)')
        return solve_banded((1,1),self._bands,q,check_finite=True)

    def transition_matrix(self):
        """Explicit Q for small diagnostics; production propagation uses apply."""
        return self.apply(np.eye(len(self.anchors)))

    def audit(self,tolerance=1e-10):
        """Structural M-matrix/martingale checks plus sampled inverse columns."""
        a=self.anchors;M=self.matrix
        columns=np.unique([0,len(a)//2,len(a)-1]);rhs=np.zeros((len(a),len(columns)))
        rhs[columns,np.arange(len(columns))]=1.
        Q=self.apply(rhs)
        errors=dict(inverse_column_mass=float(np.max(abs(np.asarray(M.sum(axis=0)).ravel()-1))),
                    inverse_column_mean=float(np.max(abs(np.asarray(a@M).ravel()-a))),
                    solved_column_mass=float(np.max(abs(Q.sum(axis=0)-1))),
                    solved_column_mean=float(np.max(abs(a@Q-a[columns]))),
                    negative_probability=max(0.,float(-Q.min())),
                    solve_residual=float(np.max(abs(M@Q-rhs))))
        return dict(passed=max(errors.values())<=tolerance,tolerance=tolerance,errors=errors)


def recovery_diagnostics(anchors,previous,current,dt,*,tolerance=1e-10):
    """Report whether finite equation-(16) parameters can encode a marginal pair."""
    a=_grid(anchors);p=np.asarray(previous,float);q=np.asarray(current,float)
    if p.shape!=a.shape or q.shape!=a.shape or np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        raise ValueError('Invalid marginal arrays')
    if not np.isfinite(dt) or dt<=0:raise ValueError('Recovery requires positive finite dt')
    before=latent_calls(a,p);after=latent_calls(a,q);delta=after-before
    singular=(q[1:-1]<=0)&(delta[1:-1]>tolerance)
    result=dict(min_previous_weight=float(p.min()),min_current_weight=float(q.min()),
                previous_mass_error=float(abs(p.sum()-1)),current_mass_error=float(abs(q.sum()-1)),
                previous_mean_error=float(abs(a@p-1)),current_mean_error=float(abs(a@q-1)),
                calendar_violation=max(0.,float(-delta.min())),
                boundary_call_change=float(np.max(abs(delta[[0,-1]]))),
                singular_interior_indices=(np.flatnonzero(singular)+1).tolist())
    result['finite_recovery_possible']=bool(p.min()>=0 and q.min()>=0 and not singular.any()
        and max(result[k] for k in ['previous_mass_error','current_mass_error','previous_mean_error',
            'current_mean_error','calendar_violation','boundary_call_change'])<=tolerance)
    return result


def recover_local_volatilities(anchors,previous,current,dt,*,tolerance=1e-10):
    """Equation (16): use unsmoothed marginal calls, not market/smoothed calls.

    0/0 gives zero. Positive time value with zero final probability requires
    an infinite DLV and is rejected rather than replaced with an arbitrary cap.
    """
    a=_grid(anchors);p=np.asarray(previous,float);q=np.asarray(current,float)
    audit=recovery_diagnostics(a,p,q,dt,tolerance=tolerance)
    if not audit['finite_recovery_possible']:raise ValueError(f'No finite audited DLV recovery: {audit}')
    delta=latent_calls(a,q)-latent_calls(a,p)
    sigma=np.zeros(len(a));positive=q[1:-1]>0
    variance=np.zeros(len(a)-2)
    variance[positive]=(np.maximum(delta[1:-1][positive],0)*(a[2:]-a[:-2])[positive]
        /(dt*a[1:-1][positive]**2*q[1:-1][positive]))
    sigma[1:-1]=np.sqrt(variance)
    if np.any(~np.isfinite(sigma)):raise ValueError('Recovered DLV is not finite')
    reconstructed=DLVTransition(a,sigma,dt).apply(p)
    error=float(np.max(abs(reconstructed-q)))
    if error>tolerance:raise ValueError(f'DLV reconstruction residual {error} exceeds tolerance')
    return sigma


def marginal_from_latent_calls(anchors,calls,*,tolerance=1e-10):
    """Recover discrete probabilities from slope jumps, including both tails."""
    a=_grid(anchors);c=np.asarray(calls,float)
    if c.shape!=a.shape or np.any(~np.isfinite(c)):raise ValueError('Invalid latent call vector')
    if abs(c[0]-(1-a[0]))>tolerance or abs(c[-1])>tolerance:
        raise ValueError('Latent calls must have intrinsic left and zero right boundary values')
    q=np.diff(np.r_[-1.,np.diff(c)/np.diff(a),0.])
    if q.min() < -tolerance or abs(a@q-1)>tolerance:raise ValueError('Latent calls fail convexity/moment audit')
    return q


class DLVModel:
    def __init__(self,anchors,times,local_volatilities,smoothing_variances,*,initial_weights=None,tolerance=1e-10):
        a=_grid(anchors);t=np.asarray(times,float);v=np.asarray(smoothing_variances,float)
        if t.ndim!=1 or not len(t) or np.any(~np.isfinite(t)) or np.any(t<=0) or np.any(np.diff(t)<=0):
            raise ValueError('Positive increasing times required')
        sigma=np.asarray(local_volatilities,float)
        if sigma.shape not in {(len(t),len(a)),(len(t),len(a)-2)}:
            raise ValueError('DLV surface must have one row per time and one column per interior state')
        if v.shape!=t.shape or np.any(~np.isfinite(v)) or np.any(v<0) or np.any(np.diff(v)<0):
            raise ValueError('Nonnegative nondecreasing smoothing variances required')
        if initial_weights is None:
            at=np.flatnonzero(a==1.)
            if len(at)!=1:raise ValueError('Include state 1 for a deterministic initial martingale')
            initial=np.zeros(len(a));initial[at[0]]=1.
        else:initial=np.array(initial_weights,float,copy=True)
        if (initial.shape!=a.shape or np.any(~np.isfinite(initial)) or initial.min()<0
            or abs(initial.sum()-1)>tolerance or abs(a@initial-1)>tolerance):
            raise ValueError('Initial weights must be nonnegative with unit mass and mean')
        self.transitions=[];weights=[];previous=initial
        for j,time in enumerate(t):
            transition=DLVTransition(a,sigma[j],time-(t[j-1] if j else 0.))
            current=transition.apply(previous)
            if (current.min() < -tolerance or abs(current.sum()-1)>tolerance or abs(a@current-1)>tolerance):
                raise ValueError('DLV propagation failed probability/moment audit')
            self.transitions.append(transition);weights.append(current);previous=current
        self.anchors=a;self.times=t.copy();self.smoothing_variances=v.copy()
        self.initial_weights=initial;self.weights=np.asarray(weights)
        for value in [self.anchors,self.times,self.smoothing_variances,self.initial_weights,self.weights]:value.flags.writeable=False

    def surface(self,*,time_interpolation='atm_pchip'):
        return SanosSurface(self.times,[self.anchors]*len(self.times),self.weights,self.smoothing_variances,
                            time_interpolation=time_interpolation)

    @classmethod
    def from_npz(cls,path,*,tolerance=2e-7):
        """Reload parameters and repropagate; inspect the calibration summary for fit/convergence."""
        with np.load(path,allow_pickle=False) as data:
            model=cls(data['anchors'],data['times'],data['local_volatilities'],data['smoothing_variances'],
                      initial_weights=data['initial_weights'] if 'initial_weights' in data else None,tolerance=tolerance)
            if 'weights' in data:
                saved=data['weights']
                if saved.shape!=model.weights.shape or np.max(abs(saved-model.weights))>tolerance:
                    raise ValueError('Saved probabilities do not match propagated DLV parameters')
        return model


def embed_common_grid(anchors,weights):
    """Exactly embed heterogeneous marginals; do not interpolate their densities."""
    if len(anchors)!=len(weights) or not len(anchors):raise ValueError('Marginal arrays required')
    grids=[_grid(a) for a in anchors];common=np.unique(np.r_[1.,np.concatenate(grids)])
    embedded=np.zeros((len(grids),len(common)))
    for j,(a,q) in enumerate(zip(grids,weights)):
        q=np.asarray(q,float)
        if q.shape!=a.shape or np.any(~np.isfinite(q)):raise ValueError('Invalid marginal weights')
        embedded[j,np.searchsorted(common,a)]=q
    return common,embedded


# Nonlinear calibration shares the transition and recovery implementation above.
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
        started=_time.perf_counter();deadline=started+time_limit
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
        remaining=deadline-_time.perf_counter()
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
        return parameters,dict(method='interior_lp',seconds=_time.perf_counter()-started,
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
        started=_time.perf_counter();deadline=started+time_limit
        x=np.ones(np.prod(self.shape)) if initial_parameters is None else np.asarray(initial_parameters,float).ravel().copy()
        initial=self.evaluate(x,gradient=False);best=dict(x=x.copy(),evaluation=initial)
        attempts=[];evaluations=0;invalid_evaluations=0;stop='stages_complete';last=None
        class BudgetExpired(Exception):pass
        for delta in [.05,.005,0.]:
            stage_start=_time.perf_counter()
            def fun(value):
                nonlocal evaluations,invalid_evaluations
                if _time.perf_counter()>=deadline:raise BudgetExpired()
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
                attempts.append(dict(smoothing=delta,seconds=_time.perf_counter()-stage_start,success=bool(last.success),
                    message=str(last.message),iterations=last.nit,evaluations=last.nfev))
            except BudgetExpired:
                stop='time_limit';attempts.append(dict(smoothing=delta,seconds=_time.perf_counter()-stage_start,
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
                    request_seconds=_time.perf_counter()-started,audit=audit,
                    local_optimization=True,global_optimality_certified=False)
