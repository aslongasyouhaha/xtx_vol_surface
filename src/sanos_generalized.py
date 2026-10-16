"""Exact finite-path reference for paper (18),(19),(22); no path pruning.

Intended for validation, not a scalable production fitter. Variances supplied
here are already scaled (eta is not applied a second time).
"""
import numpy as np
from scipy.optimize import linprog
from sanos_core import call_kernel


def prices(states, variances, weights, strikes):
    out = np.zeros(len(strikes))
    for a, v, q in zip(states, variances, weights):
        out += q * call_kernel([a], strikes, v)[:, 0]
    return out


class ExactProduct:
    def __init__(self, max_paths=100000):
        self.states = np.ones(1)
        self.variances = np.zeros(1)
        self.weights = np.ones(1)
        self.max_paths = max_paths

    def kernel(self, anchors, increments, strikes):
        a, v = np.asarray(anchors), np.asarray(increments)
        if a.ndim != 1 or v.shape != a.shape or np.any(a <= 0) or np.any(v < 0) or not np.all(np.isfinite(np.r_[a,v])):
            raise ValueError('Invalid positive multipliers or nonnegative variances')
        if len(a)*len(self.states) > self.max_paths:
            raise ValueError('Exact path budget exceeded; no truncation performed')
        return np.column_stack([prices(self.states*x, self.variances+y,
                                       self.weights, strikes) for x,y in zip(a,v)])

    def advance(self, anchors, increments, q):
        a, v, q = map(lambda x: np.asarray(x, float), (anchors, increments, q))
        self.kernel(a, v, np.array([1.]))
        if q.shape != a.shape or np.any(~np.isfinite(q)) or np.any(q < 0) or abs(q.sum()-1)>2e-7 or abs(a@q-1)>2e-7:
            raise ValueError('Weights must be nonnegative with unit mass and mean')
        self.states = (self.states[:,None]*a).ravel()
        self.variances = (self.variances[:,None]+v).ravel()
        self.weights = (self.weights[:,None]*q).ravel()

    def calls(self, strikes):
        return prices(self.states, self.variances, self.weights, strikes)

    def fit_step(self, anchors, increments, strikes, bid, ask, mid=None, *,
                 priority_mask=None, core_loss_tolerance=1e-5):
        """Sequential LP with paper spread loss and 1e-8 midpoint penalty.

        Optional priority_mask selects a two-stage objective: minimize core
        outside loss first, then wing outside loss with the core optimum locked.
        The tolerance is TOTAL spread-normalized loss, not a per-quote budget.
        Product of independent unit-mean positive increments is a martingale;
        hence latent calendar inequalities in GNR hold by conditional Jensen.
        """
        A = self.kernel(anchors, increments, strikes)
        bid, ask = np.asarray(bid), np.asarray(ask)
        if np.any(~np.isfinite(np.r_[bid,ask])) or np.any(ask <= bid):
            raise ValueError('Invalid quote spreads')
        mid = (bid+ask)/2 if mid is None else np.asarray(mid)
        n,m = A.shape
        I,Z = np.eye(n),np.zeros((n,n))
        if priority_mask is not None:
            mask=np.asarray(priority_mask,bool)
            if mask.shape!=(n,):raise ValueError('priority_mask must match quotes')
            if not np.isfinite(core_loss_tolerance) or core_loss_tolerance<0:
                raise ValueError('Invalid core loss tolerance')
            if mask.any():
                return self._fit_core_first(A,anchors,increments,bid,ask,mid,mask,core_loss_tolerance)
        res = linprog(np.r_[np.zeros(m),1/(ask-bid),1e-8/(ask-bid)],
            A_ub=np.block([[A,-I,Z],[-A,-I,Z],[A,Z,-I],[-A,Z,-I]]),
            b_ub=np.r_[ask,-bid,mid,-mid],
            A_eq=np.c_[np.array([np.ones(m),anchors]),np.zeros((2,2*n))],
            b_eq=[1,1], bounds=(0,None),method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not res.success:
            raise RuntimeError(res.message)
        q=res.x[:m]
        self.advance(anchors,increments,q)
        return {'q':q,'fit':A@q,'objective':float(res.fun)}

    def _fit_core_first(self,A,anchors,increments,bid,ask,mid,mask,tolerance):
        n,m=A.shape;I=np.eye(n);Z=np.zeros((n,n));w=1/(ask-bid)
        inequalities=np.block([[A,-I,Z],[-A,-I,Z],[A,Z,-I],[-A,Z,-I]])
        rhs=np.r_[ask,-bid,mid,-mid]
        E=np.c_[np.array([np.ones(m),anchors]),np.zeros((2,2*n))]
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
        corecost=np.r_[np.zeros(m),w*mask,np.zeros(n)]
        first=linprog(corecost,A_ub=inequalities,b_ub=rhs,A_eq=E,b_eq=[1,1],
            bounds=(0,None),method='highs',options=options)
        if not first.success:raise RuntimeError(f'Core stage: {first.message}')
        optimum=max(0.,float(first.fun));limit=optimum+tolerance
        outside=lambda f:np.maximum(bid-f,0)+np.maximum(f-ask,0)
        first_error=outside(A@first.x[:m])
        hard=bool(optimum<=1e-8 and np.all(first_error[mask]<=1e-9))
        bounds=[(0,None)]*(m+2*n)
        if hard:
            for j in np.flatnonzero(mask):bounds[m+j]=(0,0)
        wingcost=np.r_[np.zeros(m),w*(~mask),1e-8*w]
        second=linprog(wingcost,A_ub=np.vstack([inequalities,corecost]),b_ub=np.r_[rhs,limit],
            A_eq=E,b_eq=[1,1],bounds=bounds,method='highs',options=options)
        def valid(x):
            if x is None or np.any(~np.isfinite(x)) or np.any(x[:m]<0):return False
            residual=max(float(np.max(abs(E@x-1))),float(np.max(inequalities@x-rhs)),float(-x.min()))
            error=outside(A@x[:m]);coreloss=float(np.sum(w[mask]*error[mask]))
            return residual<=2e-7 and coreloss<=limit and (not hard or np.all(error[mask]<=1e-9))
        # A failed/numerically inaccurate second solve cannot sacrifice core fit.
        if second.success and valid(second.x):selected=second;stage=2
        elif valid(first.x):selected=first;stage=1
        else:raise RuntimeError('Core-priority candidate failed original-unit/core-loss audit')
        q=selected.x[:m];fit=A@q;error=outside(fit)
        self.advance(anchors,increments,q)
        return dict(q=q,fit=fit,objective=float(np.sum(w*(error+1e-8*abs(fit-mid)))),
            fit_priority='core_first',core_optimal_outside_loss=optimum,
            core_actual_outside_loss=float(np.sum(w[mask]*error[mask])),core_loss_limit=limit,
            core_hard_bid_ask=hard,selected_stage=stage,
            second_stage_status=str(second.message),second_stage_accepted=stage==2)
