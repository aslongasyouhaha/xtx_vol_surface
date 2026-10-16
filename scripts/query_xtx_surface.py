"""Query normalized calls on a calibrated SANOS surface, with no time extrapolation."""
import _bootstrap  # project paths for direct script execution
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sanos_surface import SanosSurface


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--dte',nargs='+',type=float,required=True)
    parser.add_argument('--k',nargs='+',type=float,required=True)
    parser.add_argument('--interpolation',choices=['atm_pchip','linear'],default='atm_pchip')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    surface=SanosSurface.from_directory(args.model_dir,time_interpolation=args.interpolation)
    dte,k=np.meshgrid(args.dte,args.k,indexing='ij')
    values=surface.pure_call_days(dte,k)
    frame=pd.DataFrame(dict(dte=dte.ravel(),k=k.ravel(),pure_call=values.ravel()))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(args.output,index=False)
    print(frame.to_csv(index=False),end='')


if __name__=='__main__':main()
