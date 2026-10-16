"""Audit IV wing reversals on saved LPs without refitting or tail extrapolation."""
import _bootstrap  # project paths for direct script execution
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/outputs/wing_turns'

def otm(k,a,w,q):
    k=np.asarray(k)[:,None];a=np.asarray(a)[None,:]
    s=np.sqrt(w);d1=np.log(a/k)/s+s/2;d2=d1-s
    return np.where(k<1,k*ndtr(-d2)-a*ndtr(-d1),a*ndtr(d1)-k*ndtr(d2))@q

def iv(k,price,T):
    lo=np.zeros(len(k));hi=np.full(len(k),4.)
    valid=(price>1e-12)&np.isfinite(price)
    for _ in range(65):
        mid=(lo+hi)/2;s=mid*np.sqrt(T);d1=-np.log(k)/np.maximum(s,1e-30)+s/2;d2=d1-s
        val=np.where(k<1,k*ndtr(-d2)-ndtr(-d1),ndtr(d1)-k*ndtr(d2))
        below=val<price;lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    result=(lo+hi)/2;result[~valid]=np.nan
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted((ROOT/'data/outputs/daily_validation').glob('*/*/lp/model.npz')):
        date=path.parents[1].name;quotes=pd.read_csv(path.parent/'quotes.csv')
        with np.load(path) as model:
            expiries=sorted(quotes.exdate.unique());fig,axs=plt.subplots(int(np.ceil(len(expiries)/3)),3,figsize=(15,3.2*np.ceil(len(expiries)/3)),squeeze=False)
            for ax,e in zip(axs.flat,expiries):
                f=quotes[quotes.exdate==e].sort_values('k');a=model[e+'__anchors'];q=model[e+'__q'];w=float(model[e+'__smooth_variance']);T=float(model[e+'__T'])
                k=np.linspace(f.k.min(),f.k.max(),501);v=iv(k,otm(k,a,w,q),T);slope=np.gradient(v,k)
                ax.plot(k,v,color='red',lw=1,label='LP');ax.scatter(f.k,f.source_otm_iv,s=4,color='black',label='filtered market')
                ax.scatter(f.loc[f.core,'k'],f.loc[f.core,'source_otm_iv'],s=6,color='blue',label='core')
                ax.set_title(f'{date} {int(model[e+"__dte"])}D');ax.grid(alpha=.2)
                for side,mask,sign in [('left',k<min(1,k[100]),1),('right',k>max(1,k[-101]),-1)]:
                    reversal=mask&np.isfinite(v)&(sign*slope>.01)
                    if side=='right':
                        # A negative equity skew alone is not a right-wing downturn.
                        risen=np.maximum.accumulate(np.where(k>1,slope,-np.inf))>.01
                        reversal &= risen
                    else:
                        falling_later=np.maximum.accumulate(np.where(k<1,-slope,-np.inf)[::-1])[::-1]>.01
                        reversal &= falling_later
                    # Report slope-direction reversals, not a universal smile requirement.
                    idx=np.flatnonzero(reversal);n=len(idx)
                    corelo=f.loc[f.core,'k'].min();corehi=f.loc[f.core,'k'].max()
                    rows.append(dict(date=date,expiry=e,dte=int(model[e+'__dte']),side=side,turn=n>=3,
                        reversed_grid_points=n,first_k=k[idx[0]] if n else None,last_k=k[idx[-1]] if n else None,
                        overlaps_core=bool(n and np.any((k[idx]>=corelo)&(k[idx]<=corehi))),
                        anchor_min=a.min(),anchor_max=a.max(),smooth_iv=np.sqrt(w/T),
                        q_min=q.min(),edge_iv=v[0 if side=='left' else -1]))
            for ax in list(axs.flat)[len(expiries):]:ax.set_visible(False)
            axs.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/f'{date}.png',dpi=120);plt.close(fig)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'turns.csv',index=False)
    summary=frame.groupby('side').agg(slices=('turn','size'),turns=('turn','sum'),core_overlap=('overlaps_core','sum'))
    print(summary.to_string());print(frame[frame.turn].groupby(['date','side']).size().to_string())
    (OUT/'summary.json').write_text(json.dumps(summary.to_dict(),indent=2),encoding='utf8')

if __name__=='__main__':main()
