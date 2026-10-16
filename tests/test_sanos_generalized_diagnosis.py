import unittest
import numpy as np
from sanos_core import call_kernel
from sanos_generalized_grid import GridProduct
from diagnose_xtx_generalized_april import relaxed_marginal


class DiagnosisTests(unittest.TestCase):
    def test_multiplicative_grid_kernels_commute(self):
        first=GridProduct(1025);second=GridProduct(1025)
        for m,order in [(first,[0,1]),(second,[1,0])]:
            steps=[([.9,1.1],[.001,.003],[.5,.5]),([.8,1,1.2],[.002,.001,.004],[.2,.6,.2])]
            for j in order:m.advance(*steps[j])
        np.testing.assert_allclose(first.calls(np.linspace(.5,1.5,51)),
                                   second.calls(np.linspace(.5,1.5,51)),atol=1e-12)

    def test_free_next_marginal_fits_convex_ordered_quotes(self):
        m=GridProduct(513);k=np.linspace(.8,1.2,21)
        c=call_kernel([1],k,.01)[:,0]
        result=relaxed_marginal(m,dict(market_k=k,pure_bid=c-.001,
            pure_ask=c+.001,pure_mid=c))
        self.assertTrue(result['success'])
        self.assertEqual(result['coverage'],1.)
        self.assertLess(result['actual_outside_objective'],1e-7)

    def test_previous_price_is_unavoidable_lower_bound(self):
        m=GridProduct(513);m.advance([.8,1.2],[0,0],[.5,.5])
        k=np.array([.95,1.,1.05]);c=np.maximum(1-k,0.)
        bid=c+.001;ask=c+.002
        result=relaxed_marginal(m,dict(market_k=k,pure_bid=bid,pure_ask=ask,pure_mid=(bid+ask)/2))
        lower=np.maximum(m.calls(k)-ask,0).sum()/.001
        self.assertTrue(result['success'])
        self.assertGreaterEqual(result['actual_outside_objective'],lower-1e-5)


if __name__=='__main__':unittest.main()
