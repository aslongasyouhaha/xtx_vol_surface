"""IV comparison using only the observations in a saved calibration."""
import _bootstrap  # project paths for direct script execution
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from diagnose_xtx_wing_turns import otm,iv

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=Path('data/outputs/daily_validation/june/2025-06-02/lp'))
    parser.add_argument('--output',type=Path,default=Path('data/outputs/calibration_quotes/2025-06-02.html'))
    args=parser.parse_args();quotes=pd.read_csv(args.model_dir/'quotes.csv');fig=go.Figure()
    palette=['#636efa','#ef553b','#00cc96','#ab63fa','#ffa15a','#19d3f3','#ff6692','#b6e880','#ff97ff']
    count=0
    with np.load(args.model_dir/'model.npz') as model:
        for j,(e,g) in enumerate(quotes.groupby('exdate',sort=True)):
            g=g.sort_values('k');prefix=str(e);T=float(model[prefix+'__T']);dte=int(model[prefix+'__dte'])
            k=np.linspace(g.k.min(),g.k.max(),501)
            fit=iv(k,otm(k,model[prefix+'__anchors'],float(model[prefix+'__smooth_variance']),model[prefix+'__q']),T)
            color=palette[j%len(palette)];name=f'{e} ({dte}D)'
            fig.add_trace(go.Scatter(x=g.k,y=g.source_otm_iv,mode='markers',name='calibration quotes '+name,
                marker=dict(size=4,color=color),legendgroup=name))
            fig.add_trace(go.Scatter(x=k,y=fit,mode='lines',name='LP '+name,
                line=dict(color=color),legendgroup=name,connectgaps=False))
            count+=len(g)
    fig.update_layout(title='OTM IV: calibration observations and LP (each expiry inside its quote range)',
        xaxis_title='Normalized strike k',yaxis_title='OTM implied volatility',width=1200,height=720,
        meta=dict(model_dir=str(args.model_dir),calibration_quotes=count,raw_filtered_out_quotes_shown=0))
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.write_html(args.output,include_plotlyjs=True)
    print(f'Saved {args.output}; {count} calibration observations; no excluded quotes.')

if __name__=='__main__':main()
