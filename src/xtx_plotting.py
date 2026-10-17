"""Shared OTM prices, IV inversion and calibration figures."""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.special import ndtr

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


def run_plot(args):
    quotes=pd.read_csv(args.model_dir/'quotes.csv')
    with np.load(args.model_dir/'model.npz') as model:
        fig=calibration_iv_figure(quotes,model)
    fig.update_layout(meta=dict(model_dir=str(args.model_dir),calibration_quotes=len(quotes),raw_filtered_out_quotes_shown=0))
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.write_html(args.output,include_plotlyjs=True)
    print(f'Saved {args.output}; {len(quotes)} calibration observations; no excluded quotes.')
