"""Market quote compatibility for every adjacent eligible expiry."""
import _bootstrap
import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xtx_inputs import load_inputs
from xtx_research import market_calendar

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date',default='2025-06-02')
    p.add_argument('--output',type=Path,default=Path('data/outputs/market_calendar'))
    args=p.parse_args();items=load_inputs(args.date,.25)
    quotes=pd.concat([v['quotes'] for v in items.values()],ignore_index=True)
    rows,curves=market_calendar(quotes);out=args.output/args.date;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'pairs.csv',index=False)
    metadata=dict(date=args.date,expiries=len(items),pairs=len(curves),interpretation='Interpolated quote-band compatibility; not executable arbitrage.')
    (out/'summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    if curves:
        fig,axs=plt.subplots(len(curves),1,figsize=(10,2.5*len(curves)),squeeze=False)
        for ax,c in zip(axs.flat,curves):
            ax.plot(c['k'],c['mid_difference'],label='mid_next - mid_previous')
            ax.plot(c['k'],c['compatible_order_margin'],label='ask_next - bid_previous')
            ax.axhline(0,color='gray',ls='--');ax.set_title(c['previous']+' -> '+c['next']);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/'calendar.png',dpi=120);plt.close(fig)
    print(json.dumps(metadata))
if __name__=='__main__':main()
