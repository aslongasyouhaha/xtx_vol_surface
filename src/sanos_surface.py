"""Pure-call surface interpolation, SANOS equations (4)-(5), Remark 2.13.

T is in ACT/365 years; k is strike divided by that maturity's forward.
No time extrapolation or implicit interpolation of forwards/discounts.
Raw calibrated weights are retained, including solver-scale residuals.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr, ndtri
from sanos_core import call_kernel


class SanosSurface:
    def __init__(self,times,anchors,weights,variances,*,time_interpolation='atm_pchip',tolerance=2e-7):
        self.times=np.array(times,float,copy=True)
        self.anchors=tuple(np.array(a,float,copy=True) for a in anchors)
        self.weights=tuple(np.array(q,float,copy=True) for q in weights)
        self.variances=np.array(variances,float,copy=True)
        self.time_interpolation=time_interpolation
        if time_interpolation not in {'linear','atm_pchip'}:raise ValueError('Unknown time interpolation')
        if not np.isfinite(tolerance) or tolerance<0:raise ValueError('Invalid tolerance')
        n=len(self.times)
        if (self.times.ndim!=1 or not n or np.any(~np.isfinite(self.times)) or
            np.any(self.times<=0) or np.any(np.diff(self.times)<=0)):
            raise ValueError('Times must be finite, positive, strictly increasing')
        if len(self.anchors)!=n or len(self.weights)!=n or self.variances.shape!=(n,):
            raise ValueError('Mismatched slice arrays')
        if (np.any(~np.isfinite(self.variances)) or np.any(self.variances<0) or
            np.any(np.diff(self.variances)<0)):
            raise ValueError('Smoothing variances must be nonnegative and nondecreasing')
        for a,q in zip(self.anchors,self.weights):
            if (a.ndim!=1 or q.shape!=a.shape or not len(a) or np.any(~np.isfinite(a)) or
                np.any(~np.isfinite(q)) or np.any(a<=0) or np.any(np.diff(a)<=0)):
                raise ValueError('Invalid anchor or weight arrays')
            if q.min() < -tolerance or abs(q.sum()-1)>tolerance or abs(a@q-1)>tolerance:
                raise ValueError('Slice fails probability/moment audit')
        for j in range(n-1):
            knots=np.union1d(self.anchors[j],self.anchors[j+1])
            prev=np.maximum(self.anchors[j][None,:]-knots[:,None],0)@self.weights[j]
            nxt=np.maximum(self.anchors[j+1][None,:]-knots[:,None],0)@self.weights[j+1]
            if np.min(nxt-prev)<-tolerance:raise ValueError('Slices fail latent convex-order audit')
        self.atm_prices=np.array([self._slice(j,np.array([1.]))[0] for j in range(n)])
        if np.any(self.atm_prices<0) or np.any(self.atm_prices>=1):raise ValueError('Invalid fitted ATM prices')
        self.atm_variances=4*ndtri((1+self.atm_prices)/2)**2
        self._atm_curve=PchipInterpolator(self.times,self.atm_variances,extrapolate=False) if n>1 else None
        for a in (*self.anchors,*self.weights,self.times,self.variances,self.atm_prices,self.atm_variances):
            a.flags.writeable=False

    def _slice(self,j,k):
        # k=0 is an exact moment evaluation, avoiding log(0).
        out=np.empty(k.shape,float);zero=k==0
        out[zero]=self.anchors[j]@self.weights[j]
        out[~zero]=call_kernel(self.anchors[j],k[~zero],self.variances[j])@self.weights[j]
        return out

    def pure_call(self,T,k):
        """Broadcast T and k elementwise; use T[:,None], k[None,:] for a grid.

        PCHIP interpolates *fitted* ATM total variance, then converts to a
        strike-independent convex-combination weight. It is not IV interpolation
        at each strike. Prices are continuous in T; C1 at all strikes is not
        claimed. Flat ATM intervals use the paper's linear-time alternative.
        """
        T,k=np.broadcast_arrays(np.asarray(T,float),np.asarray(k,float));shape=T.shape
        t=T.ravel();strikes=k.ravel()
        if np.any(~np.isfinite(t)) or np.any(~np.isfinite(strikes)) or np.any(strikes<0):
            raise ValueError('Finite times and finite nonnegative normalized strikes required')
        if np.any(t<self.times[0]) or np.any(t>self.times[-1]):
            raise ValueError('Time outside calibrated range; extrapolation is disabled')
        if len(self.times)==1:result=self._slice(0,strikes)
        else:
            index=np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-2)
            result=np.empty_like(strikes)
            for j in np.unique(index):
                mask=index==j;tm=t[mask];km=strikes[mask]
                u=(tm-self.times[j])/(self.times[j+1]-self.times[j])
                alpha=u
                delta=self.atm_prices[j+1]-self.atm_prices[j]
                if self.time_interpolation=='atm_pchip' and abs(delta)>1e-12:
                    w=self._atm_curve(tm)
                    atm=2*ndtr(np.sqrt(np.maximum(w,0))/2)-1
                    alpha=(atm-self.atm_prices[j])/delta
                    if np.any(alpha < -1e-10) or np.any(alpha > 1+1e-10):
                        raise ValueError('ATM interpolation failed convex-weight check')
                    alpha=np.clip(alpha,0,1)  # alpha roundoff only; never alter q/prices.
                alpha=np.where(tm==self.times[j],0,np.where(tm==self.times[j+1],1,alpha))
                result[mask]=(1-alpha)*self._slice(j,km)+alpha*self._slice(j+1,km)
        result=result.reshape(shape)
        return float(result) if result.ndim==0 else result

    def pure_call_days(self,dte,k):
        return self.pure_call(np.asarray(dte,float)/365.,k)

    def call_price(self,T,strike,*,forward,discount_factor):
        """Cash call with caller-supplied F(T), DF(T); no hidden curve policy."""
        T,K,F,D=np.broadcast_arrays(np.asarray(T,float),np.asarray(strike,float),
                                    np.asarray(forward,float),np.asarray(discount_factor,float))
        if np.any(~np.isfinite(F)) or np.any(F<=0) or np.any(~np.isfinite(D)) or np.any(D<=0):
            raise ValueError('Positive finite forward and discount factor required')
        value=D*F*self.pure_call(T,K/F)
        return float(value) if value.ndim==0 else value

    @classmethod
    def from_solution(cls,solution,**kwargs):
        if not solution.get('success'):raise ValueError('Cannot build surface from rejected solution')
        exdates=solution['exdates'];items=solution['inputs']
        return cls([float(items[ex].get('T',items[ex]['dte']/365.)) for ex in exdates],
                   [items[ex]['anchor_k'] for ex in exdates],
                   [solution['q_by_expiry'][ex] for ex in exdates],
                   [items[ex]['smooth_variance'] for ex in exdates],**kwargs)

    @classmethod
    def from_directory(cls,directory,**kwargs):
        directory=Path(directory);summary=json.loads((directory/'summary.json').read_text(encoding='utf8'))
        if not summary.get('success'):raise ValueError('Saved calibration was rejected')
        with np.load(directory/'model.npz',allow_pickle=False) as data:
            prefixes=sorted(k[:-3] for k in data.files if k.endswith('__q'))
            times=[float(data[ex+'__T']) if ex+'__T' in data else float(data[ex+'__dte'])/365 for ex in prefixes]
            return cls(times,[data[ex+'__anchors'] for ex in prefixes],[data[ex+'__q'] for ex in prefixes],
                       [float(data[ex+'__smooth_variance']) for ex in prefixes],**kwargs)


def audit_surface(surface,*,tolerance=2e-7):
    """Supplement constructor's structural checks with interior-time grid checks."""
    times=surface.times
    if len(times)>1:
        times=np.unique(np.concatenate([np.linspace(a,b,5) for a,b in zip(times[:-1],times[1:])]))
    k=np.linspace(0,max(3.,max(a[-1] for a in surface.anchors)),601)
    prices=surface.pure_call(times[:,None],k[None,:])
    slopes=np.diff(prices,axis=1)/np.diff(k)
    residuals=dict(lower_price_violation=float(np.max(np.maximum(1-k,0)-prices)),
                   upper_price_violation=float(np.max(prices-1)),
                   increasing_strike_violation=float(np.max(slopes)),
                   slope_below_minus_one=float(np.max(-1-slopes)),
                   convexity_violation=float(np.max(-np.diff(slopes,axis=1))),
                   calendar_violation=float(np.max(-np.diff(prices,axis=0))) if len(times)>1 else 0.)
    maximum=max(0.,*residuals.values())
    return dict(passed=bool(np.all(np.isfinite(prices)) and maximum<=tolerance),
                tolerance=tolerance,max_violation=maximum,n_times=len(times),n_strikes=len(k),
                min_dte=float(surface.times[0]*365),max_dte=float(surface.times[-1]*365),
                time_interpolation=surface.time_interpolation,residuals=residuals)
