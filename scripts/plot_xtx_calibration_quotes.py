"""IV comparison using only the observations in a saved calibration."""
import _bootstrap  # project paths for direct script execution
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from xtx_plotting import calibration_iv_figure



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
