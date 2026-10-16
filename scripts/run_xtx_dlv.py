"""Validate DLV generation/recovery and diagnose exact encoding of saved LPs."""
import _bootstrap  # project paths for direct script execution
import json
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sanos_dlv import (DLVModel,DLVTransition,latent_calls,recover_local_volatilities,
                        recovery_diagnostics,embed_common_grid)
from sanos_surface import SanosSurface,audit_surface

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-01-02','2025-04-01','2025-06-02'])
    parser.add_argument('--synthetic-only',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data/outputs'/'dlv_validation')
    args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
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
        surface=SanosSurface.from_directory(ROOT/'data/outputs'/'baseline_paper_gap001'/date)
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


if __name__=='__main__':main()
