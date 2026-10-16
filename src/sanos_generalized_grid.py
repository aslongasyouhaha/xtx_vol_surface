"""Conservative grid approximation of independent multiplicative increments.

Analytically integrated stochastic rounding in PRICE (not log-price)
preserves each conditional mean up to audited Gaussian tail truncation.
No probability renormalization is performed.
Fitted kernels and saved distributions use the identical transition operator.
This is an engineering approximation, not the paper's full path tensor.
"""
import numpy as np
from scipy.special import ndtr
from scipy.signal import fftconvolve
from sanos_generalized import ExactProduct


class GridProduct(ExactProduct):
    def __init__(self, nx=16385, quadrature=128, tolerance=2e-7):
        self.h=7/(nx-1)
        self.left=int(np.ceil(4/self.h))
        self.x=(np.arange(nx)-self.left)*self.h
        self.states=np.exp(self.x)
        self.weights=np.zeros(nx);self.weights[self.left]=1.
        # quadrature is retained for compatibility; integration is now analytic.
        self.tolerance=tolerance
        self._cache=None

    def calls(self,strikes):
        k=np.asarray(strikes)
        sm=np.r_[np.cumsum(self.weights[::-1])[::-1],0.]
        sx=np.r_[np.cumsum((self.weights*self.states)[::-1])[::-1],0.]
        i=np.searchsorted(self.states,k,side='right')
        return sx[i]-k*sm[i]

    def _propagate(self,a,v):
        if not np.isfinite(a+v) or a<=0 or v<0:raise ValueError('Invalid increment')
        if v==0:
            start=int(np.floor(np.log(a)/self.h))
            frac=(a-np.exp(start*self.h))/(np.exp((start+1)*self.h)-np.exp(start*self.h))
            kernel=np.array([1-frac,frac])
        else:
            sd=np.sqrt(v);mu=np.log(a)-.5*v
            width=(10+sd)*sd
            start=int(np.floor((mu-width)/self.h))
            end=int(np.ceil((mu+width)/self.h))
            edges=np.arange(start,end+1)*self.h
            def differences(z):
                # Survival subtraction avoids cancellation in the upper tail.
                return np.where(z[:-1]>0,ndtr(-z[:-1])-ndtr(-z[1:]),np.diff(ndtr(z)))
            prob=differences((edges-mu)/sd)
            moment=a*differences((edges-mu-v)/sd)
            lower,upper=np.exp(edges[:-1]),np.exp(edges[1:])
            down=(upper*prob-moment)/(upper-lower)
            up=(moment-lower*prob)/(upper-lower)
            if min(down.min(),up.min()) < -1e-12:raise ValueError('Invalid integrated projection')
            kernel=np.zeros(end-start+1)
            kernel[:-1]+=np.maximum(down,0);kernel[1:]+=np.maximum(up,0)
            offsets=np.arange(start,end+1)
            if abs(kernel.sum()-1)>1e-11 or abs(kernel@np.exp(offsets*self.h)-a)>1e-10:
                raise ValueError('Integrated increment loses probability or mean')
        full=fftconvolve(self.weights,kernel)
        # FFT roundoff is audited, not silently accepted at arbitrary scale.
        if full.min() < -1e-14:raise ValueError('Negative transition probabilities')
        full=np.maximum(full,0.)
        out=np.zeros_like(self.weights)
        l=max(0,-start);r=min(len(full),len(out)-start)
        out[l+start:r+start]=full[l:r]
        expected_mass=self.weights.sum();expected_mean=(self.weights@self.states)*a
        if abs(out.sum()-expected_mass)>1e-11 or abs(out@self.states-expected_mean)>1e-10:
            raise ValueError('Grid boundary loss or transition moment error; expand domain')
        return out

    def kernel(self,anchors,increments,strikes):
        a,v=np.asarray(anchors),np.asarray(increments)
        if a.ndim!=1 or a.shape!=v.shape:raise ValueError('Invalid increments')
        columns=np.column_stack([self._propagate(x,y) for x,y in zip(a,v)])
        k=np.asarray(strikes)
        idx=np.searchsorted(self.states,k,side='right')
        sm=np.vstack([np.cumsum(columns[::-1],axis=0)[::-1],np.zeros(len(a))])
        sx=np.vstack([np.cumsum((columns*self.states[:,None])[::-1],axis=0)[::-1],np.zeros(len(a))])
        self._cache=(a.copy(),v.copy(),columns)
        return sx[idx]-k[:,None]*sm[idx]

    def advance(self,anchors,increments,q):
        a,v,q=map(lambda x:np.asarray(x,float),(anchors,increments,q))
        if q.shape!=a.shape or np.any(~np.isfinite(q)) or np.any(q<0) or abs(q.sum()-1)>self.tolerance or abs(a@q-1)>self.tolerance:
            raise ValueError(f'Invalid unit-mean weights: min={q.min()}, mass={q.sum()-1}, mean={a@q-1}')
        if self._cache is None or not np.array_equal(a,self._cache[0]) or not np.array_equal(v,self._cache[1]):
            self.kernel(a,v,[1.])
        candidate=self._cache[2]@q
        if abs(candidate.sum()-1)>self.tolerance or abs(candidate@self.states-1)>self.tolerance:
            raise ValueError('Distribution fails mass/mean audit')
        self.weights=candidate;self._cache=None
