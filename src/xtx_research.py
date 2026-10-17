"""Diagnostics on the same input selection and models as the main pipeline."""
import json
from pathlib import Path
from xtx_plotting import otm,iv
import numpy as np
import pandas as pd
from sanos_core import solve_convex_order_lp
from sanos_surface import SanosSurface,audit_surface
from sanos_generalized import ExactProduct
from sanos_generalized_grid import GridProduct
from xtx_inputs import core_mask


def run_testcases(cases,output,data_dir=None):
    """Calibrate and audit shipped synthetic fixtures using the maintained LP."""
    from xtx_inputs import load_testcase
    output=Path(output);output.mkdir(parents=True,exist_ok=True);reports=[]
    for case in cases:
        inputs,config=load_testcase(case,data_dir)
        solution=solve_convex_order_lp(inputs)
        if not solution['success']:raise ValueError(solution['message'])
        surface=SanosSurface.from_solution(solution);audit=audit_surface(surface)
        destination=output/case;destination.mkdir(parents=True,exist_ok=True)
        arrays={};frames=[]
        for expiry,item in solution['inputs'].items():
            key=str(expiry.date());q=solution['q_by_expiry'][expiry]
            frame=item['quotes'].copy();fit=solution['fit_by_expiry'][expiry]
            frame['fit']=fit;frame['outside']=np.maximum(frame.pure_bid-fit,0)+np.maximum(fit-frame.pure_ask,0)
            frame['inside']=frame.outside<=1e-9;frames.append(frame)
            arrays[key+'__anchors']=item['anchor_k'];arrays[key+'__q']=q
            arrays[key+'__smooth_variance']=np.asarray(item['smooth_variance'])
            arrays[key+'__T']=np.asarray(item['T']);arrays[key+'__dte']=np.asarray(item['dte'])
        frame=pd.concat(frames,ignore_index=True)
        loss=float(np.sum(frame.outside/(frame.pure_ask-frame.pure_bid)))
        full=bool(frame.inside.all());expected=config['expected']
        passed=bool(audit['passed'] and full==expected['all_quotes_inside'] and loss+1e-6>=expected['minimum_outside_loss'])
        report=dict(case=case,data_kind='synthetic',success=bool(audit['passed']),testcase_passed=passed,
            quotes=len(frame),expiries=len(inputs),coverage=float(frame.inside.mean()),
            outside_loss=loss,fit_target_met=full,surface_audit=audit,expected=expected)
        frame.to_csv(destination/'quotes.csv',index=False)
        np.savez_compressed(destination/'model.npz',**arrays)
        (destination/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        reports.append(report);print(json.dumps(report),flush=True)
    (output/'results.json').write_text(json.dumps(reports,indent=2),encoding='utf8')
    if not all(r['testcase_passed'] for r in reports):raise ValueError('Synthetic testcase expectation failed')
    return reports


def market_calendar(quotes,*,points=301,tolerance=1e-9):
    """Adjacent-expiry interpolated quote compatibility, not an execution test.

    ask_next < bid_prev rules out ordered prices inside the interpolated bands.
    bid_next < ask_prev alone does not. Interpolation can introduce artifacts.
    """
    groups=[]
    for expiry,g in quotes.groupby('exdate',sort=True):
        g=g.groupby('k',as_index=False)[['pure_bid','pure_mid','pure_ask']].median().sort_values('k')
        if len(g)>=2:groups.append((expiry,g))
    rows=[];curves=[]
    for (prev,a),(nxt,b) in zip(groups[:-1],groups[1:]):
        lo=max(a.k.min(),b.k.min());hi=min(a.k.max(),b.k.max())
        if lo>=hi:
            rows.append(dict(previous=str(prev),next=str(nxt),status='no_common_strike_range'));continue
        k=np.linspace(lo,hi,points)
        interp=lambda f,c:np.interp(k,f.k,f[c])
        mid=interp(b,'pure_mid')-interp(a,'pure_mid')
        margin=interp(b,'pure_ask')-interp(a,'pure_bid')
        rows.append(dict(previous=str(prev),next=str(nxt),status='checked',min_k=float(lo),max_k=float(hi),
            min_mid_difference=float(mid.min()),min_compatible_order_margin=float(margin.min()),
            mid_inversion_points=int(np.sum(mid < -tolerance)),
            interpolated_band_conflict_points=int(np.sum(margin < -tolerance))))
        curves.append(dict(previous=str(prev),next=str(nxt),k=k,mid_difference=mid,compatible_order_margin=margin))
    return rows,curves


def eta_control(inputs,eta,*,time_limit=45):
    """Full joint LP on unchanged observations/states; no automatic eta choice."""
    if not np.isfinite(eta) or not 0<=eta<1:raise ValueError('eta must be in [0,1)')
    items={e:dict(v,smooth_variance=float(eta)*float(v['quotes'].atm_total_variance.iloc[0])) for e,v in inputs.items()}
    sol=solve_convex_order_lp(items,time_limit=time_limit)
    row=dict(eta=float(eta),success=bool(sol['success']),message=sol['message'],solve_seconds=sol['solve_seconds'])
    if not sol['success']:return row
    n=inside=nc=ic=0;outside_total=0.
    for e,v in sol['inputs'].items():
        fit=sol['fit_by_expiry'][e];outside=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
        core=core_mask(v['quotes']);n+=len(fit);inside+=int(np.sum(outside<=1e-9))
        nc+=int(core.sum());ic+=int(np.sum(outside[core]<=1e-9))
        outside_total+=float(np.sum(outside/(v['pure_ask']-v['pure_bid'])))
    audit=audit_surface(SanosSurface.from_solution(sol))
    row.update(success=bool(audit['passed']),quotes=n,expiries=len(items),coverage=inside/n,
        core_coverage=ic/nc if nc else None,outside_loss=outside_total,
        actual_objective=sol['actual_objective'],surface_audit=audit)
    return row


def generalized_accuracy(grids=(4097,16385),*,steps=3):
    """Identical multipliers/variances/weights: analytic reference vs grid model."""
    k=np.linspace(.6,1.4,201);rows=[]
    for nx in grids:
        model=GridProduct(nx);exact=ExactProduct();previous=model.calls(k)
        for step in range(steps):
            a=[.9,1,1.1];v=[.001,.002,.003];q=np.array([.2,.6,.2])
            fitted=model.kernel(a,v,k)@q;model.advance(a,v,q);exact.advance(a,v,q)
            prices=model.calls(k)
            rows.append(dict(nx=int(nx),step=step+1,max_price_error=float(np.max(abs(prices-exact.calls(k)))),
                reprice_error=float(np.max(abs(prices-fitted))),mass_error=float(abs(model.weights.sum()-1)),
                mean_error=float(abs(model.weights@model.states-1)),calendar_min=float(np.min(prices-previous))))
            previous=prices
    return rows


def shape(p,x,e):
    v=p['items'][e];k=np.linspace(v['market_k'].min(),v['market_k'].max(),1501)
    q=x[slice(*p['q_offsets'][e])];T=v.get('T',v['dte']/365)
    sigma=iv(k,otm(k,v['anchor_k'],v['smooth_variance'],q),T)
    idx=np.flatnonzero(k<min(1,k[300]));left=sigma[idx]
    return k,sigma,dict(left_edge_iv=float(left[0]),left_peak_iv=float(np.nanmax(left)),
        left_turn_amplitude=float(np.nanmax(left)-left[0]),
        left_peak_k=float(k[idx[np.nanargmax(left)]]),anchor_min=float(v['anchor_k'].min()))


def run_wing_scan(output,baseline_dir):
    import matplotlib.pyplot as plt
    from xtx_plotting import otm,iv
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True);rows=[]
    for path in sorted(Path(baseline_dir).glob('*/*/lp/model.npz')):
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
            axs.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/f'{date}.png',dpi=120);plt.close(fig)
    frame=pd.DataFrame(rows);frame.to_csv(output/'turns.csv',index=False)
    summary=frame.groupby('side').agg(slices=('turn','size'),turns=('turn','sum'),core_overlap=('overlaps_core','sum'))
    print(summary.to_string());print(frame[frame.turn].groupby(['date','side']).size().to_string())
    (output/'summary.json').write_text(json.dumps(summary.to_dict(),indent=2),encoding='utf8')


def run_wing_boundary(output):
    import matplotlib.pyplot as plt
    import json
    from pathlib import Path
    from xtx_results import save_fit,load_saved_lp
    from sanos_incremental import PersistentSanosLP
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True);rows=[]
    for date,block,eta,dtes in [('2025-01-02','january',.25,[88]),('2025-04-03','april',.1,[43,88])]:
        p,x,_=load_saved_lp(date,block,eta)
        variants=[('saved_baseline',p,x)]
        for name,left in [('add_state_020',.2)]:
            changed={e:dict(v,anchor_k=np.unique(np.r_[left,v['anchor_k']])) for e,v in p['items'].items()}
            session=PersistentSanosLP(changed,time_limit=45);r=session.solve()
            if not r['success']:
                rows.append(dict(date=date,variant=name,success=False,message=r['message']))
                print(json.dumps(rows[-1]),flush=True);continue
            report=save_fit(output/date/name,date,session.problem,r['x'],dict(eta=eta,variant=name))
            variants.append((name,session.problem,r['x']))
            print(json.dumps(dict(date=date,variant=name,coverage=report['coverage'],core_coverage=report['core_coverage'])),flush=True)
        for dte in dtes:
            e=next(e for e,v in p['items'].items() if v['dte']==dte);g=p['items'][e]['quotes']
            fig,axs=plt.subplots(1,2,figsize=(12,4))
            axs[0].scatter(g.k,g.source_otm_iv,s=8,c='black',label='eligible market mid')
            T=p['items'][e].get('T',dte/365);kq=g.k.to_numpy();intrinsic=np.maximum(1-kq,0)
            bidiv=iv(kq,g.pure_bid.to_numpy()-intrinsic,T);askiv=iv(kq,g.pure_ask.to_numpy()-intrinsic,T)
            for ax in axs:ax.fill_between(kq,bidiv,askiv,color='gray',alpha=.25,label='eligible bid/ask')
            curves={}
            for name,pp,xx in variants:
                k,s,stats=shape(pp,xx,e);curves[name]=(k,s)
                fit=pp['items'][e]['G']@xx[slice(*pp['q_offsets'][e])]
                outside=np.maximum(g.pure_bid.to_numpy()-fit,0)+np.maximum(fit-g.pure_ask.to_numpy(),0)
                row=dict(date=date,dte=dte,variant=name,success=True,coverage=float(np.mean(outside<=1e-9)),
                    core_coverage=float(np.mean(outside[core_mask(g)]<=1e-9)),**stats)
                rows.append(row)
                for ax in axs:ax.plot(k,s,label=name)
            lo=g.k.min();hi=min(1,lo+.2*(g.k.max()-lo));axs[1].set_xlim(lo,hi)
            for ax in axs:ax.set_xlabel('k');ax.set_ylabel('OTM IV');ax.grid(alpha=.2);ax.legend(fontsize=7)
            fig.suptitle(f'{date} {dte}D: identical quotes, eta={eta}; only add common left state 0.2')
            fig.tight_layout();fig.savefig(output/f'{date}_{dte}D.png',dpi=150);plt.close(fig)
        (output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
