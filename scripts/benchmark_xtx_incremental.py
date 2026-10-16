"""Cold/warm Python/C++ HiGHS benchmark on labeled synthetic quote updates.

No intraday data are available. Each sequence freezes ETA, normalized strikes,
forwards, expiry membership and anchors. This is not a tick-to-quote benchmark.
"""
import _bootstrap  # project paths for direct script execution
from pathlib import Path
import argparse
import json
import struct
import subprocess
import sys
import time
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
DEPS=ROOT/'data/outputs'/'native_deps'
if (DEPS/'highspy').exists(): sys.path.insert(0,str(DEPS))
from sanos_core import build_convex_order_problem
from sanos_incremental import (PersistentSanosLP,quote_vectors,native_model,validate_frame,prepare_validation,
                               compile_native_model,native_quote_vectors)
from xtx_inputs import load_inputs,core_mask


def frames(problem,scenario='broad'):
    """Base, local/broad small moves, then widened spreads; fixed model geometry."""
    updates=[];rng=np.random.default_rng(20261015)
    base={ex:(np.asarray(v['pure_bid']).copy(),np.asarray(v['pure_ask']).copy()) for ex,v in problem['items'].items()}
    candidates=[(ex,i) for ex,v in problem['items'].items() for i in np.flatnonzero(core_mask(v['quotes']))]
    order=rng.permutation(len(candidates)) if scenario=='sparse' else None;counts=[0,1,5,10,25]
    for frame in range(5):
        selected=set(candidates[i] for i in order[:counts[frame]]) if scenario=='sparse' else None
        quotes={}
        for ex,v in problem['items'].items():
            bid,ask=base[ex];spread=ask-bid;core=core_mask(v['quotes'])
            selection=core.copy()
            if scenario=='sparse':selection=np.array([(ex,i) in selected for i in range(len(bid))])
            elif frame==1: selection &= rng.random(len(bid))<.1
            movement=np.zeros(len(bid))
            if frame: movement[selection]=(.05 if scenario=='sparse' or frame%2 else -.05)*spread[selection]
            widen=np.where(core,.025*spread,0) if frame==4 and scenario=='broad' else np.zeros(len(bid))
            # Selected core OTM bids are positive; nevertheless reject invalid
            # synthetic prices instead of clipping and changing spread weights.
            new_bid=bid+movement-widen;new_ask=ask+movement+widen
            if np.any(new_bid<0): raise ValueError('Synthetic bid below zero')
            quotes[ex]=(new_bid,new_ask)
        updates.append((quotes,*quote_vectors(problem,quotes)))
    return updates


def write_binary(path,problem,compiled,updates):
    matrix,lower=compiled['matrix'],compiled['lower']
    with path.open('wb') as f:
        f.write(b'XTXLP001');f.write(struct.pack('<4Q',matrix.shape[1],matrix.shape[0],matrix.nnz,len(updates)))
        for array,dtype in [(matrix.indptr,'<i4'),(matrix.indices,'<i4'),(matrix.data,'<f8'),(lower,'<f8')]:
            f.write(np.asarray(array,dtype=dtype).tobytes())
        for quotes,_,_ in updates:
            upper,costs=native_quote_vectors(problem,compiled,quotes)
            f.write(np.asarray(costs,dtype='<f8').tobytes());f.write(np.asarray(upper,dtype='<f8').tobytes())


def benchmark_summary(records):
    import pandas as pd
    df=pd.DataFrame(records).query('frame > 0')
    counts=df.groupby(['date','method']).agg(all_passed=('success','all'),total_updates=('success','size'))
    timings=df[df.success].groupby(['date','method']).agg(
        median_seconds=('total_seconds','median'),min_seconds=('total_seconds','min'),
        max_seconds=('total_seconds','max'),median_iterations=('iterations','median'),
        successful_updates=('success','size'))
    return counts.join(timings,how='left')


def paired_language_comparison(records):
    import pandas as pd
    df=pd.DataFrame(records)
    cpp=df[(df.method=='cpp_warm') & (df.frame>0) & df.success]
    py=df[(df.method=='python_warm') & (df.frame>0) & df.success]
    paired=cpp.merge(py,on=['date','frame'],suffixes=('_cpp','_py'))
    return paired.groupby('date').agg(paired_frames=('frame','size'),
        cpp_median=('total_seconds_cpp','median'),python_median=('total_seconds_py','median'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',nargs='+',default=['2025-04-01','2025-06-02'])
    parser.add_argument('--time-limit',type=float,default=60.)
    parser.add_argument('--skip-scipy',action='store_true')
    parser.add_argument('--scenario',choices=['broad','sparse'],default='broad')
    parser.add_argument('--cpp-exe',type=Path,default=DEPS/'incremental_highs.exe')
    parser.add_argument('--dll',type=Path,default=DEPS/'highs-1.15.1'/'bin'/'highs.dll')
    args=parser.parse_args()
    suffix='' if args.scenario=='broad' else '_sparse'
    out=ROOT/'data/outputs'/('incremental_benchmark'+suffix);out.mkdir(parents=True,exist_ok=True)
    records=[];profiles=[]

    def save(row):
        records.append(row)
        (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
        import pandas as pd
        pd.DataFrame(records).to_csv(out/'results.csv',index=False)
        print(json.dumps(row),flush=True)

    for date in args.dates:
        eta=.1 if date=='2025-04-01' else .25
        started=time.perf_counter();inputs=load_inputs(date,eta,baseline_variant='baseline');data_seconds=time.perf_counter()-started
        started=time.perf_counter();problem=build_convex_order_problem(inputs,weight_mode='legacy',mid_penalty=0.);build_seconds=time.perf_counter()-started
        updates=frames(problem,args.scenario)
        audit_cache=prepare_validation(problem)
        profile=dict(date=date,eta=eta,data_seconds=data_seconds,build_seconds=build_seconds,
                     kernel_seconds=problem['kernel_seconds'],assembly_seconds=problem['assembly_seconds'],
                     n_vars=problem['cursor'],n_quotes=sum(len(v['market_k']) for v in problem['items'].values()))
        profiles.append(profile);print('PROFILE '+json.dumps(profile),flush=True)
        if not args.skip_scipy:
            t=time.perf_counter()
            res=linprog(problem['c'],A_ub=problem['A'],b_ub=problem['b'],A_eq=problem['E'],
                b_eq=np.ones(problem['E'].shape[0]),bounds=(0,None),method='highs-ds',
                options=dict(presolve=False,time_limit=args.time_limit,
                             primal_feasibility_tolerance=1e-7,dual_feasibility_tolerance=1e-7))
            row=dict(date=date,eta=eta,method='scipy_cold',frame=0,solve_seconds=time.perf_counter()-t,
                     success=bool(res.success),message=res.message,iterations=int(res.nit),
                     total_seconds=time.perf_counter()-t)
            if res.success:
                row.update(validate_frame(problem,updates[0][1],updates[0][2],res.x,updates[0][0],audit_cache=audit_cache));row['objective']=float(res.fun)
                row['success']=row['success'] and row['validated']
            save(row)
        compiled=compile_native_model(problem)
        path=out/f'{date}.bin';write_binary(path,problem,compiled,updates)
        for mode in ['cold','warm']:
            solpath=out/f'{date}_cpp_{mode}.solutions'
            completed=subprocess.run([str(args.cpp_exe),str(args.dll),str(path),mode,str(solpath),str(args.time_limit)],
                                     capture_output=True,text=True)
            if completed.returncode:
                raise RuntimeError(f'C++ benchmark failed: {completed.stderr}')
            (out/f'{date}_cpp_{mode}.jsonl').write_text(completed.stdout,encoding='utf8')
            cpp_rows=[json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
            if len(cpp_rows)!=len(updates):raise RuntimeError('Incomplete C++ result sequence')
            with solpath.open('rb') as f:
                for frame,row in enumerate(cpp_rows):
                    status,objective=struct.unpack('<id',f.read(12))
                    x=np.frombuffer(f.read(problem['cursor']*8),dtype='<f8').copy()*compiled['column_scale']
                    row.update(method=row.pop('mode'),date=date,eta=eta,
                               objective=objective*compiled['objective_scale'],success=status==7)
                    if row['success']:
                        row.update(validate_frame(problem,updates[frame][1],updates[frame][2],x,updates[frame][0],audit_cache=audit_cache))
                        row['success']=row['success'] and row['validated']
                    save(row)
        for method in ['python_cold','python_reuse_no_basis','python_warm']:
            session=None
            for frame,(quotes,upper,costs) in enumerate(updates):
                started=time.perf_counter()
                if session is None or method=='python_cold':
                    session=PersistentSanosLP(inputs,problem=problem,compiled=compiled,time_limit=args.time_limit,numerics_profile='legacy')
                session.update_quotes(quotes)
                setup_seconds=time.perf_counter()-started
                row=session.solve(discard_basis=method=='python_reuse_no_basis',validate=False)
                row.update(total_seconds=time.perf_counter()-started,setup_seconds=setup_seconds,
                           method=method,date=date,eta=eta,frame=frame,version=session.h.version())
                x=row.pop('x')
                if row['success']:
                    row.update(validate_frame(problem,upper,costs,x,quotes,audit_cache=audit_cache));row['success']=row['success'] and row['validated']
                save(row)
        (out/'profiles.json').write_text(json.dumps(profiles,indent=2),encoding='utf8')
    # Same-version cold solutions anchor every objective comparison. Differences
    # beyond explicit tolerance are failures, not speed improvements.
    for row in records:
        controls=[r for r in records if r['date']==row['date'] and r['frame']==row['frame'] and r['method']=='cpp_cold' and r['success']]
        if row.get('success') and controls:
            delta=abs(row['actual_outside_objective']-controls[0]['actual_outside_objective'])
            row['outside_objective_difference_to_cpp_cold']=delta
            row['objective_agrees']=bool(delta<=1e-6)
            row['success']=row['success'] and row['objective_agrees']
    (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    import pandas as pd
    pd.DataFrame(records).to_csv(out/'results.csv',index=False)
    summary=benchmark_summary(records)
    summary.to_csv(out/'summary.csv');print(summary.to_string(),flush=True)
    paired_language_comparison(records).to_csv(out/'paired_language_comparison.csv')
    metadata=dict(synthetic_updates=True,frames_per_date=5,seed=20261015,
                  solver='dual simplex',presolve='off',threads=1,primal_dual_tolerance=1e-7,
                  independent_feasibility_tolerance=2e-7,objective_comparison_tolerance=1e-6,
                  fixed_geometry=True,exclude_data_io_and_validation_from_total=True)
    metadata.update(native_otm_and_spread_scaling=False,scipy_uses_original_unscaled_form=True)
    metadata.update(scenario=args.scenario,sparse_quote_counts=[0,1,5,10,25] if args.scenario=='sparse' else None)
    metadata.update(sparse_incremental_dirty_quotes=[0,1,4,5,15] if args.scenario=='sparse' else None)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    if not all(row['success'] for row in records):raise SystemExit('One or more benchmarks failed validation or optimality.')


if __name__=='__main__':main()
