import unittest
import numpy as np
from test_sanos_daily import daily_items
from sanos_core import build_convex_order_problem,solve_convex_order_lp
from explore_xtx_fit_preserving_qp import fit_preserving_qp

class FitPreservingQPTests(unittest.TestCase):
    def test_temporal_improvement_cannot_worsen_any_quote(self):
        inputs=daily_items();p=build_convex_order_problem(inputs);lp=solve_convex_order_lp(inputs)
        quotes={e:(v['pure_bid'],v['pure_ask']) for e,v in p['items'].items()}
        refs={e:(v['pure_mid']+.01,np.ones(len(v['market_k']),bool)) for e,v in p['items'].items()}
        for coordinates in ['absolute','delta','moments']:
            with self.subTest(coordinates=coordinates):
                r=fit_preserving_qp(p,lp['x'],quotes,refs,time_limit=5,coordinates=coordinates)
                self.assertTrue(r['success'],r)
                self.assertTrue(r['audit']['validated'])
                self.assertLess(r['qp_reference_rms'],r['lp_reference_rms'])
                self.assertLessEqual(r['qp_outside_loss'],r['lp_outside_loss']+1e-8)
                for e,v in p['items'].items():
                    bid,ask=quotes[e]
                    errors=[]
                    for x in [lp['x'],r['x']]:
                        fit=v['G']@x[slice(*p['q_offsets'][e])]
                        errors.append(np.maximum(bid-fit,0)+np.maximum(fit-ask,0))
                    self.assertLessEqual(float(np.max(errors[1]-errors[0])),1e-9)

if __name__=='__main__':unittest.main()
