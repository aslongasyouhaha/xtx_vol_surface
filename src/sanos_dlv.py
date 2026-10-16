"""SANOS section 3.2, equations (14)-(17), on a common positive state grid.

Matrices act on column probability vectors: Q[i,j] conditions on state j.
Compute q_next by a tridiagonal solve, never by a dense inverse. Finite DLV
recovery can fail for sparse target marginals; no mass floors or renormalizing
are used to conceal that failure.
"""
import numpy as np
from scipy.linalg import solve_banded
from scipy import sparse
from sanos_surface import SanosSurface


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
