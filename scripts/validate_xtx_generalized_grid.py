"""Independent fixed-parameter analytic-projection grid accuracy controls."""
import _bootstrap  # project paths for direct script execution
import json
from pathlib import Path
import numpy as np
from sanos_generalized import ExactProduct
from sanos_generalized_grid import GridProduct
from xtx_inputs import load_inputs


def main():
    out=Path('data/outputs/generalized_grid');out.mkdir(parents=True,exist_ok=True)
    results=[];k=np.linspace(.6,1.4,201)
    for nx in [4097,16385,65537]:
        grid=GridProduct(nx);exact=ExactProduct()
        for _ in range(3):
            a=[.9,1,1.1];v=[.001,.002,.003];q=[.2,.6,.2]
            grid.advance(a,v,q);exact.advance(a,v,q)
        results.append(dict(case='synthetic_exact',nx=nx,integration='analytic',
            max_price_error=float(np.max(abs(grid.calls(k)-exact.calls(k)))) ))
    for date in ['2025-01-02','2025-04-01','2025-06-02']:
        saved=np.load(out/f'{date}_16385.npz');items=load_inputs(date,.1)
        model=GridProduct(65537);rows=[]
        for j,it in enumerate(items.values()):
            model.advance(it['anchor_k'],saved[f'increment_{j}'],saved[f'q_{j}'])
            old=GridProduct(16385,128);old.weights=saved[f'density_{j}']
            difference=abs(model.calls(it['market_k'])-old.calls(it['market_k']))
            rows.append(dict(dte=int(it['dte']),max_price_difference=float(difference.max()),
                max_difference_spreads=float(np.max(difference/(it['pure_ask']-it['pure_bid'])))))
        results.append(dict(case=date,comparison='fixed q: 16385 versus 65537 grid points',rows=rows))
    (out/'accuracy_controls.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
