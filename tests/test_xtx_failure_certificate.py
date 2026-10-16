import unittest
import numpy as np
from diagnose_xtx_fit_failure import minimum_reduced_cost,independent_certificate

class Prefix:
    states=np.array([.8,1.2])
    weights=np.array([.5,.5])
    def calls(self,k):
        k=np.asarray(k)
        return np.sum(np.maximum(self.states-k[...,None],0)*self.weights,axis=-1)

class FailureCertificateTests(unittest.TestCase):
    def test_event_sweep_matches_all_exact_kinks(self):
        prefix=Prefix();k=np.array([.7,1.,1.3]);t=np.array([2.,-4.,3.])
        r=minimum_reduced_cost(prefix.states,prefix.weights,k,t,.3,2.)
        nodes=np.unique(np.r_[0,(k[:,None]/prefix.states).ravel()])
        prices=np.array([np.zeros(len(k)) if a==0 else a*prefix.calls(k/a) for a in nodes])
        expected=np.min(prices@t-.3-r['lam1']*nodes)
        self.assertAlmostEqual(r['minimum'],expected,places=12)
        self.assertGreaterEqual(t.sum()*(prefix.weights@prefix.states)-r['lam1'],-1e-14)

    def test_certificate_lower_bound_respects_known_jensen_optimum(self):
        v=dict(market_k=np.array([1.]),pure_bid=np.array([.04]),pure_ask=np.array([.05]),anchor_k=np.array([.5,1,1.5]))
        r=independent_certificate(Prefix(),v,np.array([True]))
        self.assertTrue(r['certificate_valid'],r)
        # Any independent mean-one multiplier can only increase this call.
        # The identity multiplier attains price .1, hence minimum loss is 5.
        self.assertGreater(r['all_multiplier_outside_loss_lower_bound'],4.9)
        self.assertLessEqual(r['all_multiplier_outside_loss_lower_bound'],5+1e-8)

if __name__=='__main__':unittest.main()
