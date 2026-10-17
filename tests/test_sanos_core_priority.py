import unittest
import numpy as np
from sanos_generalized import ExactProduct
from sanos_generalized import GridProduct

class CorePriorityTests(unittest.TestCase):
    def data(self):
        a=np.array([.5,1.,1.5]);v=np.zeros(3)
        k=np.r_[1.,np.linspace(1.2,1.3,10)]
        mid=np.r_[.05,.4*(1.5-k[1:])];mask=np.arange(len(k))==0
        return a,v,k,mid-.001,mid+.001,mid,mask

    def test_wings_cannot_buy_a_worse_feasible_core(self):
        a,v,k,bid,ask,mid,mask=self.data()
        original=ExactProduct().fit_step(a,v,k,bid,ask,mid)
        self.assertGreater(bid[0]-original['fit'][0] if original['fit'][0]<bid[0] else original['fit'][0]-ask[0],.01)
        for cls in [ExactProduct,lambda:GridProduct(4097)]:
            model=cls();r=model.fit_step(a,v,k,bid,ask,mid,priority_mask=mask)
            self.assertTrue(r['core_hard_bid_ask'])
            self.assertLessEqual(max(bid[0]-r['fit'][0],r['fit'][0]-ask[0]),1e-9)
            self.assertLessEqual(r['core_actual_outside_loss'],r['core_loss_limit'])
            self.assertAlmostEqual(r['q'].sum(),1,places=10)
            self.assertAlmostEqual(a@r['q'],1,places=10)
            np.testing.assert_allclose(r['fit'],model.calls(k),atol=1e-12)

    def test_unavoidable_core_error_is_optimized_instead_of_hard_infeasibility(self):
        a,v,k,bid,ask,mid,mask=self.data();mask[:2]=True
        r=ExactProduct().fit_step(a,v,k,bid,ask,mid,priority_mask=mask)
        self.assertFalse(r['core_hard_bid_ask'])
        self.assertGreater(r['core_optimal_outside_loss'],1)
        self.assertLessEqual(r['core_actual_outside_loss'],r['core_loss_limit'])

    def test_invalid_priority_inputs(self):
        a,v,k,bid,ask,mid,mask=self.data()
        with self.assertRaises(ValueError):ExactProduct().fit_step(a,v,k,bid,ask,mid,priority_mask=[True])
        with self.assertRaises(ValueError):ExactProduct().fit_step(a,v,k,bid,ask,mid,priority_mask=mask,core_loss_tolerance=-1)

if __name__=='__main__':unittest.main()
