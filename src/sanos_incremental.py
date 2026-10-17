"""Persistent HiGHS LP for quote-only updates on a frozen SANOS model.

Install requirements.txt. ETA, forwards/normalized strikes, anchors,
expiry membership and smoothing variances are fixed for this object's lifetime.
Changes to any of those require constructing a new session.
Default updates reuse the full global basis directly. The working-set
experiment is opt-in: its cold local LP and certification can cost more.
"""
from __future__ import annotations
import copy
import time
import numpy as np
from scipy import sparse
from scipy.special import ndtr
from sanos_core import build_convex_order_problem, spread_weights, paper_spread_weights

NUMERICS_PROFILES={
    'unscaled':{'simplex_scale_strategy':0},
    'bounded_scaling':{'simplex_scale_strategy':2,'allowed_matrix_scale_factor':5},
    'direct_unscaled':{'simplex_scale_strategy':2,'simplex_unscaled_solution_strategy':2},
    'legacy':{'simplex_scale_strategy':2,'allowed_matrix_scale_factor':20},
    'ipm_seed':{'solver':'ipm','presolve':'on','simplex_scale_strategy':0},
}


def native_model(problem):
    matrix = sparse.vstack([problem['E'],problem['A']],format='csc')
    ne = problem['E'].shape[0]
    return matrix, np.r_[np.ones(ne),np.full(problem['A'].shape[0],-np.inf)], np.r_[np.ones(ne),problem['b']]


def compile_native_model(problem,*,scaling='none'):
    """Equivalent OTM rows and scaled slack units, not a relaxation of the LP.

    'otm' is an experimental paper-objective row transform. Real-data controls
    did not consistently match full-objective accuracy or improve warm latency;
    it is intentionally NOT the default compilation.
    """
    if scaling=='none':
        matrix,lower,_=native_model(problem)
        return dict(matrix=matrix,lower=lower,column_scale=np.ones(problem['cursor']),
                    objective_scale=1.,quote_scales={},scaling='none')
    if scaling=='otm':
        # Exact put-call row transformation using unit mass/mean equalities.
        # Evaluate OTM kernels directly, avoiding subtraction of ITM prices.
        blocks=[];midblocks=[];cursor=problem['cursor'];nmid=sum(b-a for a,b in problem['mid_offsets'].values())
        for ex,v in problem['items'].items():
            k=v['market_k'][:,None];a=v['anchor_k'][None,:];variance=v['smooth_variance']
            if variance>0:
                s=np.sqrt(variance);d1=(np.log(a/k)+.5*variance)/s;d2=d1-s
                kernel=np.where(k<1,k*ndtr(-d2)-a*ndtr(-d1),a*ndtr(d1)-k*ndtr(d2))
            else:kernel=np.where(k<1,np.maximum(k-a,0),np.maximum(a-k,0))
            G=sparse.csr_matrix(kernel);n=len(v['market_k']);q0,q1=problem['q_offsets'][ex]
            blocks.append(sparse.bmat([[G,-sparse.eye(n)],[-G,-sparse.eye(n)]],format='csr'))
            if nmid:
                m0,m1=problem['mid_offsets'][ex]
                part=sparse.hstack([sparse.csr_matrix((n,q0)),G,sparse.csr_matrix((n,cursor-q1))],format='csr')
                error=sparse.coo_matrix((-np.ones(n),(np.arange(n),np.arange(m0,m1))),shape=(n,cursor)).tocsr()
                midblocks.append((part+error,-part+error))
        market=sparse.block_diag(blocks,format='csr')
        market=sparse.hstack([market,sparse.csr_matrix((market.shape[0],nmid))])
        mr=market.shape[0];cr=problem['calendar_constraint_count']
        parts=[problem['E'],market,problem['A'][mr:mr+cr]]
        if nmid:
            parts.extend([sparse.vstack([x[0] for x in midblocks]),sparse.vstack([x[1] for x in midblocks])])
        matrix=sparse.vstack(parts,format='csc')
        lower=np.r_[np.ones(problem['E'].shape[0]),np.full(problem['A'].shape[0],-np.inf)]
        return dict(matrix=matrix,lower=lower,column_scale=np.ones(cursor),objective_scale=1.,quote_scales={},scaling='otm')
    if scaling!='otm_spread':raise ValueError('Unknown scaling')
    if problem.get('mid_penalty',0):raise ValueError('OTM-spread transform is a legacy-only experiment')
    blocks=[];scales={};column_scale=np.ones(problem['cursor'])
    for ex,v in problem['items'].items():
        k=v['market_k'][:,None];a=v['anchor_k'][None,:];variance=v['smooth_variance']
        if variance>0:
            s=np.sqrt(variance);d1=(np.log(a/k)+.5*variance)/s;d2=d1-s
            kernel=np.where(k<1,k*ndtr(-d2)-a*ndtr(-d1),a*ndtr(d1)-k*ndtr(d2))
        else: kernel=np.where(k<1,np.maximum(k-a,0),np.maximum(a-k,0))
        spread=np.asarray(v['pure_ask'])-np.asarray(v['pure_bid'])
        scale=np.maximum(spread,np.quantile(spread,.10));scales[ex]=scale
        column_scale[slice(*problem['z_offsets'][ex])]=scale
        G=sparse.csr_matrix(kernel/scale[:,None]);n=len(scale)
        blocks.append(sparse.bmat([[G,-sparse.eye(n)],[-G,-sparse.eye(n)]],format='csr'))
    market_rows=sum(2*len(v['market_k']) for v in problem['items'].values())
    matrix=sparse.vstack([problem['E'],sparse.block_diag(blocks),problem['A'][market_rows:]],format='csc')
    factor=float(np.median((problem['c']*column_scale)[problem['c']>0]))
    lower=np.r_[np.ones(problem['E'].shape[0]),np.full(problem['A'].shape[0],-np.inf)]
    return dict(matrix=matrix,lower=lower,column_scale=column_scale,objective_scale=factor,quote_scales=scales,scaling=scaling)


def native_quote_vectors(problem,compiled,quotes):
    if compiled['scaling']=='none':return quote_vectors(problem,quotes)
    if compiled['scaling']=='otm':
        upper,costs=quote_vectors(problem,quotes);cursor=problem['E'].shape[0]
        for ex,v in problem['items'].items():
            intrinsic=np.maximum(1-v['market_k'],0);n=len(intrinsic)
            upper[cursor:cursor+n]-=intrinsic;upper[cursor+n:cursor+2*n]+=intrinsic;cursor+=2*n
            if ex in problem['mid_rows']:
                plus,minus,_=problem['mid_rows'][ex]
                upper[plus:plus+n]-=intrinsic;upper[minus:minus+n]+=intrinsic
        return upper,costs
    _,costs=quote_vectors(problem,quotes)
    bounds=[]
    for ex,v in problem['items'].items():
        bid,ask=quotes[ex];intrinsic=np.maximum(1-v['market_k'],0);scale=compiled['quote_scales'][ex]
        bounds.extend(np.r_[(ask-intrinsic)/scale,-(bid-intrinsic)/scale])
    upper=np.r_[np.ones(problem['E'].shape[0]),bounds,np.zeros(problem['calendar_constraint_count'])]
    return upper,costs*compiled['column_scale']/compiled['objective_scale']


def quote_vectors(problem, quotes):
    """Validate the whole update before changing the solver; same row order as core."""
    if set(quotes) != set(problem['items']):
        raise ValueError('Expiry set changed; rebuild the model.')
    bounds=[]; costs=problem['c'].copy()
    for ex,v in problem['items'].items():
        bid,ask=(np.asarray(a,dtype=float) for a in quotes[ex])
        n=len(v['market_k'])
        if bid.shape != (n,) or ask.shape != (n,) or np.any(~np.isfinite(bid)) or np.any(bid<0):
            raise ValueError('Quote count changed or invalid bid; rebuild for structural changes.')
        weights=paper_spread_weights(bid,ask) if problem.get('weight_mode')=='paper' else spread_weights(bid,ask)
        bounds.extend(np.r_[ask,-bid])
        lo,hi=problem['z_offsets'][ex];costs[lo:hi]=weights
        if ex in problem.get('mid_offsets',{}):
            lo,hi=problem['mid_offsets'][ex];costs[lo:hi]=problem['mid_penalty']*weights
    upper=np.r_[np.ones(problem['E'].shape[0]),bounds,np.zeros(problem['calendar_constraint_count'])]
    if problem.get('mid_penalty',0):
        targets=np.concatenate([quote_midpoint(v,*quotes[ex]) for ex,v in problem['items'].items()])
        upper=np.r_[upper,targets,-targets]
    return upper,costs


def quote_midpoint(item,bid,ask):
    # Preserve an explicitly supplied initial midpoint; later two-sided updates
    # define their own arithmetic midpoint without needing yesterday's target.
    if np.array_equal(bid,item['pure_bid']) and np.array_equal(ask,item['pure_ask']):
        return np.asarray(item['pure_mid'])
    return .5*(np.asarray(bid)+np.asarray(ask))


def expiry_columns(problem,ex):
    columns=np.arange(problem['q_offsets'][ex][0],problem['z_offsets'][ex][1])
    if ex in problem.get('mid_offsets',{}):columns=np.r_[columns,np.arange(*problem['mid_offsets'][ex])]
    return columns


def recompute_loss_variables(problem,x,quotes):
    for ex,v in problem['items'].items():
        fit=v['G']@x[slice(*problem['q_offsets'][ex])];bid,ask=quotes[ex]
        x[slice(*problem['z_offsets'][ex])]=np.maximum(bid-fit,0)+np.maximum(fit-ask,0)
        if ex in problem.get('mid_offsets',{}):
            x[slice(*problem['mid_offsets'][ex])]=abs(fit-quote_midpoint(v,bid,ask))


class PersistentSanosLP:
    def __init__(self,inputs,*,time_limit=60.,problem=None,compiled=None,numerics_profile='adaptive',
                 working_set=False,working_set_seconds=.75,working_set_expiries=5,objective_mode=None):
        import highspy
        self.highspy=highspy
        if objective_mode not in {None,'paper','legacy'}:raise ValueError('Unknown objective mode')
        if problem is None:
            mode=objective_mode or ('legacy' if numerics_profile=='legacy' else 'paper')
            self.problem=build_convex_order_problem(copy.deepcopy(inputs),weight_mode=mode,
                                                  mid_penalty=1e-8 if mode=='paper' else 0.)
        else:
            self.problem=copy.deepcopy(problem)
            if objective_mode is not None and (self.problem['weight_mode']!=objective_mode or
                self.problem.get('mid_penalty',0)!=(1e-8 if objective_mode=='paper' else 0.)):
                raise ValueError('Supplied problem disagrees with requested objective mode')
        self.objective_mode=('paper' if self.problem['weight_mode']=='paper' and self.problem.get('mid_penalty')==1e-8
                             else 'legacy_outside_only' if self.problem['weight_mode']=='legacy' and not self.problem.get('mid_penalty')
                             else 'custom')
        self.compiled=compile_native_model(self.problem) if compiled is None else copy.deepcopy(compiled)
        self.matrix,self.lower=self.compiled['matrix'],self.compiled['lower']
        self.quotes={ex:(np.asarray(v['pure_bid']).copy(),np.asarray(v['pure_ask']).copy())
                     for ex,v in self.problem['items'].items()}
        self.upper,self.costs=native_quote_vectors(self.problem,self.compiled,self.quotes)
        self.audit_cache=prepare_validation(self.problem)
        self.time_limit=float(time_limit)
        if not np.isfinite(self.time_limit) or self.time_limit<=0:raise ValueError('time_limit must be positive and finite.')
        if numerics_profile not in {'adaptive','legacy'}:raise ValueError('Unknown numerics_profile')
        self.numerics_profile=numerics_profile
        self.working_set=bool(working_set) and numerics_profile=='adaptive'
        self.working_set_seconds=float(working_set_seconds)
        self.working_set_expiries=int(working_set_expiries)
        if not np.isfinite(self.working_set_seconds) or self.working_set_seconds<=0 or self.working_set_expiries<1:
            raise ValueError('Working-set budget and expiry count must be positive')
        self._accepted_x=None
        self._certified_result=None
        self._profile_name='unscaled' if numerics_profile=='adaptive' else 'legacy'
        self._create_solver(self._profile_name)
        self.last_update_seconds=0.

    def _create_solver(self,profile):
        highspy=self.highspy
        self.h=highspy.Highs()
        for name,value in [('output_flag',False),('threads',1),('random_seed',0),
                           ('simplex_strategy',1),('solver','simplex'),('presolve','off'),
                           ('primal_feasibility_tolerance',1e-9 if self.objective_mode=='paper' else 1e-7),
                           ('dual_feasibility_tolerance',1e-9 if self.objective_mode=='paper' else 1e-7)]:
            self._check(self.h.setOptionValue(name,value))
        for name,value in NUMERICS_PROFILES[profile].items():self._check(self.h.setOptionValue(name,value))
        lp=highspy.HighsLp()
        lp.num_col_=self.matrix.shape[1];lp.num_row_=self.matrix.shape[0]
        lp.col_cost_=self.costs;lp.col_lower_=np.zeros(lp.num_col_);lp.col_upper_=np.full(lp.num_col_,np.inf)
        lp.row_lower_=self.lower;lp.row_upper_=self.upper
        lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_=self.matrix.indptr.astype(np.int32)
        lp.a_matrix_.index_=self.matrix.indices.astype(np.int32)
        lp.a_matrix_.value_=self.matrix.data
        self._check(self.h.passModel(lp))
        self._profile_name=profile
        self._last_success=True

    def _check(self,status):
        if status == self.highspy.HighsStatus.kError:
            raise RuntimeError('HiGHS API error')

    def update_quotes(self,quotes):
        started=time.perf_counter()
        upper,costs=native_quote_vectors(self.problem,self.compiled,quotes)
        rows=np.flatnonzero(upper != self.upper).astype(np.int32)
        cols=np.flatnonzero(costs != self.costs).astype(np.int32)
        if len(rows): self._check(self.h.changeRowsBounds(len(rows),rows,self.lower[rows],upper[rows]))
        if len(cols): self._check(self.h.changeColsCost(len(cols),cols,costs[cols]))
        self.upper,self.costs=upper,costs
        self.quotes={ex:(np.asarray(bid).copy(),np.asarray(ask).copy()) for ex,(bid,ask) in quotes.items()}
        self.last_update_seconds=time.perf_counter()-started
        return dict(updated_rows=len(rows),updated_costs=len(cols))

    def _remember_full(self,out):
        if out['success']:
            self._accepted_x=out['x'].copy()
            self._accepted_dual=np.asarray(self.h.getSolution().row_dual).copy()
            self._accepted_upper,self._accepted_costs=quote_vectors(self.problem,self.quotes)
            self._certified_result=None

    def surface(self,*,time_interpolation='atm_pchip'):
        """Snapshot the last accepted fit only if it matches current quotes."""
        from sanos_surface import SanosSurface
        if self._accepted_x is None:raise ValueError('Solve and validate before querying a surface')
        upper,costs=quote_vectors(self.problem,self.quotes)
        if not np.array_equal(upper,self._accepted_upper) or not np.array_equal(costs,self._accepted_costs):
            raise ValueError('Quotes changed; solve and validate the update first')
        p=self.problem;exdates=p['exdates']
        return SanosSurface([float(p['items'][ex].get('T',p['items'][ex]['dte']/365.)) for ex in exdates],
            [p['items'][ex]['anchor_k'] for ex in exdates],
            [self._accepted_x[slice(*p['q_offsets'][ex])] for ex in exdates],
            [p['items'][ex]['smooth_variance'] for ex in exdates],time_interpolation=time_interpolation)

    def solve(self,*,discard_basis=False,validate=True):
        started=time.perf_counter();local=None
        if self.working_set and self._accepted_x is not None and not discard_basis:
            upper,costs=quote_vectors(self.problem,self.quotes)
            changed=not (np.array_equal(upper,self._accepted_upper) and np.array_equal(costs,self._accepted_costs))
            if not changed and self._certified_result is not None:
                out=copy.deepcopy(self._certified_result)
                out.update(optimization_path='cached_certified',solution_reused=True,iterations=0,
                           solve_seconds=0.,validation_seconds=0.,request_seconds=time.perf_counter()-started)
                return out
            if changed:
                from sanos_solver_research import restricted_update
                local_budget=min(self.working_set_seconds,.2*self.time_limit)
                local_deadline=min(started+self.time_limit,time.perf_counter()+local_budget)
                try:
                    local=restricted_update(self,self._accepted_x,self._accepted_dual,
                        self._accepted_upper,self._accepted_costs,deadline=local_deadline,
                        max_expiries=self.working_set_expiries)
                except (ValueError,RuntimeError) as error:
                    local=dict(success=False,reason=str(error),attempts=[])
                if local['success']:
                    attempts=local['attempts'];out=dict(success=True,x=local['x'],**local['audit'])
                    out.update(objective=local['audit']['actual_objective'],model_status=None,
                        objective_mode=self.objective_mode,
                        weight_mode=self.problem['weight_mode'],mid_penalty=self.problem.get('mid_penalty',0),
                        message='Original constraints and global objective-gap tolerance passed',
                        optimization_path='certified_working_set',solution_reused=False,
                        global_lower_bound=local['lower_bound'],global_objective_gap=local['objective_gap'],
                        objective_tolerance=1e-6,working_set_attempts=attempts,
                        iterations=sum(max(a['iterations'],0) for a in attempts),
                        solve_seconds=sum(a['solve_seconds']+sum(b['solve_seconds'] for b in a.get('dual_polishing',[])) for a in attempts),
                        numerics_profile=self._profile_name,request_seconds=time.perf_counter()-started,
                        update_seconds=self.last_update_seconds)
                    self._accepted_x=out['x'].copy();self._accepted_dual=local['dual'].copy()
                    self._accepted_upper,self._accepted_costs=upper.copy(),costs.copy()
                    self._certified_result=copy.deepcopy(out)
                    return out
        elapsed=time.perf_counter()-started
        out=self._solve_full(discard_basis=discard_basis,validate=validate,
                             budget=max(1e-6,self.time_limit-elapsed))
        out.update(optimization_path='full',solution_reused=False,objective_mode=self.objective_mode,
                   weight_mode=self.problem['weight_mode'],mid_penalty=self.problem.get('mid_penalty',0))
        if local is not None:
            out.update(working_set_attempts=local['attempts'],working_set_fallback_reason=local.get('reason'),
                       working_set_seconds=elapsed)
        self._remember_full(out)
        out['request_seconds']=time.perf_counter()-started
        return out

    def _solve_full(self,*,discard_basis=False,validate=True,budget=None):
        if self.numerics_profile=='legacy':
            return self._solve_once(discard_basis=discard_basis,validate=validate,remaining_time=budget)
        started=time.perf_counter();attempts=[];setup_seconds=0.;out=None
        profiles=[self._profile_name]+[p for p in NUMERICS_PROFILES if p!=self._profile_name]
        deadline=started+(self.time_limit if budget is None else budget)
        for index,profile in enumerate(profiles):
            if time.perf_counter()>=deadline:break
            if index:
                setup=time.perf_counter();self._create_solver(profile);setup_seconds+=time.perf_counter()-setup
            remaining=max(1e-6,deadline-time.perf_counter())
            # A fallback never bypasses the original-unit audit, even when a
            # caller opts out of the redundant benchmark audit in legacy mode.
            out=self._solve_once(discard_basis=discard_basis if not index else False,
                                 validate=True,remaining_time=remaining)
            attempts.append(dict(profile=profile,success=out['success'],message=out['message'],
                solve_seconds=out['solve_seconds'],iterations=out['iterations'],
                validation_seconds=out.get('validation_seconds',0.),
                max_constraint_violation=out.get('max_constraint_violation'),
                convex_order_min=out.get('convex_order_min')))
            if out['success']:
                if profile=='ipm_seed':self._check(self.h.setOptionValue('solver','simplex'))
                break
        if out is None:raise RuntimeError('No time available for a solver attempt')
        out.update(numerics_profile=self._profile_name,numerics_attempts=attempts,
                   solve_seconds=sum(a['solve_seconds'] for a in attempts),
                   total_simplex_iterations=sum(max(a['iterations'],0) for a in attempts),
                   validation_seconds=sum(a['validation_seconds'] for a in attempts),
                   retry_setup_seconds=setup_seconds,request_seconds=time.perf_counter()-started)
        return out

    def _solve_once(self,*,discard_basis=False,validate=True,remaining_time=None):
        if discard_basis or getattr(self,'_last_success',True) is False:self._check(self.h.clearSolver())
        basis_valid=bool(self.h.getBasis().valid)
        # HiGHS' run clock accumulates across runs on the same solver object.
        budget=self.time_limit if remaining_time is None else remaining_time
        self._check(self.h.setOptionValue('time_limit',self.h.getRunTime()+budget))
        self._last_success=False
        started=time.perf_counter();api_status=self.h.run();solve_seconds=time.perf_counter()-started
        started=time.perf_counter();status=self.h.getModelStatus()
        raw=self.h.getSolution() if status==self.highspy.HighsModelStatus.kOptimal else None
        x=np.asarray(raw.col_value)*self.compiled['column_scale'] if raw else np.zeros(self.problem['cursor'])
        extraction_seconds=time.perf_counter()-started
        info=self.h.getInfo()
        out=dict(success=status==self.highspy.HighsModelStatus.kOptimal,
                    model_status=int(status),message=self.h.modelStatusToString(status),x=x,
                    objective=self.h.getObjectiveValue()*self.compiled['objective_scale'],iterations=info.simplex_iteration_count,
                    basis_before_run=basis_valid,solve_seconds=solve_seconds,
                    run_api_status=int(api_status),
                    extraction_seconds=extraction_seconds,update_seconds=self.last_update_seconds)
        if validate and out['success']:
            upper,costs=quote_vectors(self.problem,self.quotes)
            out.update(validate_frame(self.problem,upper,costs,x,self.quotes,audit_cache=self.audit_cache))
            out['success']=out['success'] and out['validated']
            if not out['success']:out['message']='Optimal status rejected by independent constraints'
        self._last_success=out['success']
        return out


def prepare_validation(problem):
    matrix,lower,_=native_model(problem);pairs=[];items=problem['items'];exdates=list(items)
    for prev,nxt in zip(exdates[:-1],exdates[1:]):
        a,b=items[prev]['anchor_k'],items[nxt]['anchor_k'];knots=np.union1d(a,b)
        pairs.append((prev,nxt,np.maximum(a[None,:]-knots[:,None],0),np.maximum(b[None,:]-knots[:,None],0)))
    return dict(matrix=matrix,lower=lower,pairs=pairs)


def validate_frame(problem,upper,costs,x,quotes,*,tolerance=2e-7,audit_cache=None):
    """Audit original constraints and actual fit, separately from solver timings."""
    started=time.perf_counter()
    cache=prepare_validation(problem) if audit_cache is None else audit_cache
    matrix,lower=cache['matrix'],cache['lower']
    activity=matrix@x
    violation=max(float(np.max(lower-activity)),float(np.max(activity-upper)),float(-x.min()))
    # Independent exact convex-order certificate on all union knots.
    convex_min=0.;items=problem['items']
    for prev,nxt,pa,pb in cache['pairs']:
        qa=x[slice(*problem['q_offsets'][prev])];qb=x[slice(*problem['q_offsets'][nxt])]
        diff=pb@qb-pa@qa
        convex_min=min(convex_min,float(diff.min()))
    objective=0.;mid_objective=0.;total=inside=0
    for ex,v in items.items():
        q=x[slice(*problem['q_offsets'][ex])];fit=v['G']@q;bid,ask=quotes[ex]
        outside=np.maximum(bid-fit,0)+np.maximum(fit-ask,0)
        weights=paper_spread_weights(bid,ask) if problem.get('weight_mode')=='paper' else spread_weights(bid,ask)
        objective+=float(weights@outside)
        mid_objective+=float(problem.get('mid_penalty',0)*(weights@abs(fit-quote_midpoint(v,bid,ask))))
        total+=len(fit);inside+=int((outside<=1e-9).sum())
    return dict(validated=bool(np.all(np.isfinite(x)) and violation<=tolerance and convex_min>=-tolerance),
                max_constraint_violation=violation,convex_order_min=convex_min,
                actual_outside_objective=objective,actual_mid_penalty=mid_objective,
                actual_objective=objective+mid_objective,linear_objective=float(costs@x),coverage=inside/total,
                validation_seconds=time.perf_counter()-started)
