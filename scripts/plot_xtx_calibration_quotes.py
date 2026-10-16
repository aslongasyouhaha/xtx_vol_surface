"""IV comparison using only the observations in a saved calibration."""
import _bootstrap  # project paths for direct script execution
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from diagnose_xtx_wing_turns import otm,iv

def calibration_iv_figure(quotes,model):
    """Build an IV overlay from the supplied calibration observations only."""
    fig=go.Figure()
    palette=['#636efa','#ef553b','#00cc96','#ab63fa','#ffa15a','#19d3f3','#ff6692','#b6e880','#ff97ff']
    for j,(e,g) in enumerate(quotes.groupby('exdate',sort=True)):
        g=g.sort_values('k');prefix=str(e);T=float(model[prefix+'__T']);dte=int(model[prefix+'__dte'])
        k=np.linspace(g.k.min(),g.k.max(),501)
        fit=iv(k,otm(k,model[prefix+'__anchors'],float(model[prefix+'__smooth_variance']),model[prefix+'__q']),T)
        color=palette[j%len(palette)];name=f'{e} ({dte}D)'
        fig.add_trace(go.Scatter(x=g.k,y=g.source_otm_iv,mode='markers',name='market OTM IV '+name,
            marker=dict(size=5,color=color),legendgroup=name,
            hovertemplate='k=%{x:.4f}<br>Market IV=%{y:.4%}<extra>%{fullData.name}</extra>'))
        fig.add_trace(go.Scatter(x=k,y=fit,mode='lines',name='joint LP OTM IV '+name,
            line=dict(color=color,width=2),legendgroup=name,connectgaps=False,
            hovertemplate='k=%{x:.4f}<br>Fitted IV=%{y:.4%}<extra>%{fullData.name}</extra>'))
    date=str(quotes['date'].iloc[0])
    fig.update_layout(title=f'OTM IV slices: market OTM IV vs fitted OTM IV (joint LP), SPXW {date}',
        xaxis_title='Normalized strike k',yaxis_title='OTM implied volatility',height=650,
        template='plotly',legend=dict(groupclick='togglegroup'),
        margin=dict(l=65,r=30,t=85,b=60),
        meta=dict(calibration_quotes=len(quotes),raw_filtered_out_quotes_shown=0))
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=Path('data/outputs/daily_validation/june/2025-06-02/lp'))
    parser.add_argument('--output',type=Path,default=Path('data/outputs/calibration_quotes/2025-06-02.html'))
    args=parser.parse_args();quotes=pd.read_csv(args.model_dir/'quotes.csv')
    with np.load(args.model_dir/'model.npz') as model:
        fig=calibration_iv_figure(quotes,model)
    fig.update_layout(meta=dict(model_dir=str(args.model_dir),calibration_quotes=len(quotes),raw_filtered_out_quotes_shown=0))
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.write_html(args.output,include_plotlyjs=True)
    print(f'Saved {args.output}; {len(quotes)} calibration observations; no excluded quotes.')

if __name__=='__main__':main()
