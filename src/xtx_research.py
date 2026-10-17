"""Diagnostics on the same input selection and models as the main pipeline."""

import json
import numpy as np
import pandas as pd
import time
import copy
from pathlib import Path
from xtx_plotting import otm, iv
from sanos_core import solve_convex_order_lp
from sanos_surface import SanosSurface, audit_surface
from sanos_generalized import ExactProduct, GridProduct
from xtx_io import core_mask, load_inputs
from sanos_dlv import DLVModel, DLVTransition, latent_calls, recover_local_volatilities, recovery_diagnostics, embed_common_grid, DLVCalibrator, calibration_grid
from threadpoolctl import threadpool_limits

def run_testcases(cases,output,data_dir=None):
    """Calibrate and audit shipped synthetic fixtures using the maintained LP."""
    from xtx_io import load_testcase
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
    from xtx_io import save_fit,load_saved_lp
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


def run_dlv_recovery(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    a=np.array([.5,.7,.85,1.,1.15,1.4,1.8]);times=np.array([7,30,60,120])/365.
    vols=np.array([(.18+.04*j)*(1+.4*(a[1:-1]-1)**2) for j in range(len(times))])
    started=time.perf_counter();model=DLVModel(a,times,vols,[.0001,.0004,.0008,.0016])
    elapsed=time.perf_counter()-started;roundtrips=[];previous=model.initial_weights
    for j,transition in enumerate(model.transitions):
        current=model.weights[j]
        sigma=recover_local_volatilities(a,previous,current,transition.dt)
        rebuilt=DLVTransition(a,sigma,transition.dt).apply(previous)
        roundtrips.append(dict(transition=j,operator_audit=transition.audit(),
            maximum_sigma_error=float(np.max(abs(sigma-transition.sigma))),
            maximum_probability_error=float(np.max(abs(rebuilt-current))),
            maximum_latent_price_error=float(np.max(abs(latent_calls(a,rebuilt)-latent_calls(a,current))))))
        previous=current
    report=dict(generated_seconds=elapsed,roundtrips=roundtrips,
                max_mass_error=float(np.max(abs(model.weights.sum(axis=1)-1))),
                max_mean_error=float(np.max(abs(model.weights@a-1))),surface_audit=audit_surface(model.surface()))
    assert report['surface_audit']['passed'] and all(r['operator_audit']['passed'] for r in roundtrips)
    assert max(r['maximum_probability_error'] for r in roundtrips)<1e-10
    np.savez(out/'synthetic_model.npz',anchors=a,times=times,local_volatilities=vols,weights=model.weights)
    (out/'synthetic_results.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('SYNTHETIC '+json.dumps(report),flush=True)
    real_rows=[]
    for date in ([] if args.synthetic_only else args.dates):
        surface=SanosSurface.from_directory(args.baseline_dir/date)
        common,weights=embed_common_grid(surface.anchors,surface.weights)
        initial=np.zeros(len(common));initial[np.flatnonzero(common==1.)[0]]=1.;previous=initial
        summaries=[]
        for j,current in enumerate(weights):
            dt=surface.times[j]-(surface.times[j-1] if j else 0.)
            diagnostic=recovery_diagnostics(common,previous,current,dt,tolerance=2e-7)
            row=dict(date=date,dte=float(surface.times[j]*365),common_grid_size=len(common),
                     negative_probability_count=int((current<0).sum()),
                     singular_nodes=len(diagnostic['singular_interior_indices']),**diagnostic)
            # Finite eligibility is necessary; numerical round-trip is still checked.
            row['recovered']=False
            if diagnostic['finite_recovery_possible']:
                try:
                    sigma=recover_local_volatilities(common,previous,current,dt,tolerance=2e-7)
                    row.update(recovered=True,max_sigma=float(sigma.max()))
                except ValueError as error:row['recovery_error']=str(error)
            real_rows.append(row);summaries.append(dict(dte=row['dte'],negative_weights=row['negative_probability_count'],
                                                       singular_nodes=row['singular_nodes'],recovered=row['recovered']))
            previous=current
        np.savez(out/(date+'_common_grid.npz'),anchors=common,weights=weights,initial=initial)
        print(json.dumps(dict(date=date,common_grid_size=len(common),transitions=summaries)),flush=True)
    (out/'lp_encoding_diagnostics.json').write_text(json.dumps(real_rows,indent=2),encoding='utf8')
    pd.DataFrame([{k:v for k,v in row.items() if k!='singular_interior_indices'} for row in real_rows]).to_csv(out/'lp_encoding_summary.csv',index=False)


def run_dlv_calibration(args):
    args.output.mkdir(parents=True,exist_ok=True);rows=[]
    with threadpool_limits(limits=1):
        for date in args.dates:
            eta=.1 if date=='2025-04-01' else .25
            inputs=load_inputs(date,eta)
            grid=calibration_grid(inputs,central_points=args.central_points,wing_points=args.wing_points)
            c=DLVCalibrator(inputs,grid)
            print(json.dumps(dict(date=date,event='start',eta=eta,states=len(grid),parameters=int(np.prod(c.shape)))),flush=True)
            started=time.perf_counter();initial=None;initialization=dict(method='atm')
            if args.initialization=='lp_interior':
                try:
                    initial,initialization=c.interior_lp_initialization(loss_allowance=args.seed_loss_allowance,
                        time_limit=min(20.,args.time_limit*.6))
                    initial=c.rescale_parameters(initial)
                except ValueError as error:
                    initialization=dict(method='atm_fallback',lp_seed_error=str(error),seconds=time.perf_counter()-started)
            remaining=max(.001,args.time_limit-(time.perf_counter()-started))
            result=c.calibrate(time_limit=remaining,max_iterations=args.max_iterations,initial_parameters=initial)
            calibration_seconds=time.perf_counter()-started
            out=args.output/date;out.mkdir(parents=True,exist_ok=True)
            frames=[]
            for (ex,v),fit in zip(c.items.items(),result['fits']):
                g=v['quotes'][['exdate','dte','T','k','source_cp_flag','volume','pure_bid','pure_mid','pure_ask','source_otm_iv']].copy()
                g['fit']=fit;g['outside']=np.maximum(g.pure_bid-fit,0)+np.maximum(fit-g.pure_ask,0)
                g['outside_spreads']=g.outside/(g.pure_ask-g.pure_bid);g['inside']=g.outside.le(1e-9)
                g['core']=core_mask(v['quotes']);frames.append(g)
            quotes=pd.concat(frames,ignore_index=True);quotes.to_csv(out/'quotes.csv',index=False)
            per=quotes.groupby(['exdate','dte']).agg(n_quotes=('inside','size'),coverage=('inside','mean'),
                                                   max_outside_spreads=('outside_spreads','max')).reset_index()
            per.to_csv(out/'per_expiry.csv',index=False)
            model=result['model']
            np.savez(out/'model.npz',anchors=model.anchors,times=model.times,smoothing_variances=model.smoothing_variances,
                     local_volatilities=c.volatilities(result['parameters']),
                     weights=model.weights,initial_weights=model.initial_weights)
            control=copy.deepcopy(inputs)
            for v in control.values():v['anchor_k']=grid.copy()
            lp=solve_convex_order_lp(control,time_limit=30.)
            row={k:v for k,v in result.items() if k not in {'model','parameters','fits'}}
            row.update(initialization=initialization,calibration_seconds=calibration_seconds)
            row.update(date=date,eta=eta,states=len(grid),parameters=int(np.prod(c.shape)),quotes=len(quotes),
                       core_quotes=int(quotes.core.sum()),core_coverage=float(quotes.loc[quotes.core,'inside'].mean()),
                       core_outside_gt_01_spread=int((quotes.core & quotes.outside_spreads.gt(.1)).sum()),
                       max_local_volatility=float(c.volatilities(result['parameters']).max()),
                       same_grid_lp_success=lp['success'],same_grid_lp_seconds=lp['solve_seconds'])
            if lp['success']:
                inside=total=0
                for ex,v in lp['inputs'].items():
                    mask=core_mask(v['quotes']);fit=lp['fit_by_expiry'][ex]
                    outside=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
                    total+=int(mask.sum());inside+=int((outside[mask]<=1e-9).sum())
                row.update(same_grid_lp_core_coverage=inside/total,same_grid_lp_objective=lp['actual_objective'])
            (out/'summary.json').write_text(json.dumps(row,indent=2),encoding='utf8')
            rows.append(row);(args.output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
            pd.DataFrame([{k:v for k,v in r.items() if k not in {'attempts','audit','initialization'}} for r in rows]).to_csv(args.output/'summary.csv',index=False)
            print(json.dumps(row),flush=True)
