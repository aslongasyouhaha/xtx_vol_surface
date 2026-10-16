import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression
from sanos_core import paper_spread_weights,solve_convex_order_lp,call_kernel,build_model_anchor_grid
from test_sanos_core import example_inputs


class PaperSettingsTests(unittest.TestCase):
    def test_anchor_gap_fill_retains_source_strikes_without_count_cap(self):
        source=np.r_[np.linspace(.8,1.,200),1.25]
        grid=build_model_anchor_grid(source,max_gap=.01)
        self.assertGreater(len(grid),200)
        self.assertLessEqual(np.diff(grid).max(),.01+1e-14)
        self.assertTrue(np.isin(source,grid).all())
        np.testing.assert_array_equal(grid,build_model_anchor_grid(source[::-1],max_gap=.01))
        for gap in [0,-1,np.inf]:
            with self.assertRaises(ValueError):build_model_anchor_grid(source,max_gap=gap)
    def test_weights_are_unclipped_unscaled_inverse_spreads(self):
        np.testing.assert_allclose(paper_spread_weights([0,0,0],[1e-4,.01,1]),[10000,100,1])
        with self.assertRaises(ValueError):paper_spread_weights([1],[1])

    def test_mid_absolute_error_is_in_the_lp_and_reconstructed_objective(self):
        inputs=example_inputs()
        first=next(iter(inputs));inputs[first]['pure_mid']=inputs[first]['pure_mid'].copy()
        inputs[first]['pure_mid'][0]+=3e-5
        result=solve_convex_order_lp(inputs)
        self.assertTrue(result['success'],result['message'])
        actual=0.;mid_error=0.
        for ex,v in result['inputs'].items():
            fit=result['fit_by_expiry'][ex];w=1/(v['pure_ask']-v['pure_bid'])
            outside=np.maximum(v['pure_bid']-fit,0)+np.maximum(fit-v['pure_ask'],0)
            error=abs(fit-v['pure_mid']);mid_error+=float(w@error)
            actual+=float(w@(outside+1e-8*error))
            m=result['x'][slice(*result['mid_offsets'][ex])]
            self.assertGreaterEqual(float((m-error).min()),-2e-7)
        self.assertGreater(mid_error,0)
        self.assertAlmostEqual(result['actual_objective'],actual,places=12)
        self.assertLess(result['objective_reconstruction_error'],1e-6)
        self.assertTrue(result['mid_tiebreak']);self.assertEqual(result['weight_mode'],'paper')
        with self.assertRaises(ValueError):solve_convex_order_lp(inputs,mid_penalty=-1)

    def test_atm_backbone_uses_bid_not_mid_iv(self):
        k=np.array([.99,1.,1.01]);T=.1
        bid=call_kernel([1.],k,.2**2*T)[:,0]
        mid=call_kernel([1.],k,.3**2*T)[:,0]
        chain=pd.DataFrame(dict(k=k,pure_bid=bid,pure_mid=mid,T=T,dte=36,
                               exdate=pd.Timestamp('2025-02-07'),strike_price=k*100))
        env=dict(np=np,pd=pd,norm=norm,brentq=brentq,IsotonicRegression=IsotonicRegression,call_surface=chain)
        from xtx_data import build_variance_backbone
        surface,rows,_=build_variance_backbone(chain)
        atm=rows.iloc[0]
        self.assertAlmostEqual(atm.atm_iv,.2,places=8)
        self.assertAlmostEqual(atm.atm_total_variance_raw,.004,places=10)
        self.assertAlmostEqual(surface.bs_iv.iloc[1],.3,places=8)


if __name__=='__main__':unittest.main()
