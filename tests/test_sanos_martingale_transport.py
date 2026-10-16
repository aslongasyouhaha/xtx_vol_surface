import unittest
import numpy as np
from sanos_martingale_transport import project_source,calls,fit_state_dependent_transition,couple_marginals


class TransportTests(unittest.TestCase):
    def test_numerical_marginal_relaxation_is_explicit(self):
        p=np.array([1e-12,1-2e-12,1e-12])
        r=couple_marginals([.5,1,1.5],p,[1.],[1.],marginal_tolerance=1e-10)
        self.assertTrue(r['validated'])
        self.assertEqual(r['marginal_tolerance'],1e-10)
        self.assertLess(r['marginal_l1_error'],1e-9)
        self.assertGreater(r['marginal_l1_error'],0)
        np.testing.assert_allclose(r['Q']@r['target_states'],r['source_states'],atol=1e-9)

    def test_fixed_marginals_are_reproduced(self):
        r=couple_marginals([.8,1.2],[.5,.5],[.5,1,1.5],[.25,.5,.25])
        self.assertTrue(r['validated'])
        np.testing.assert_allclose(r['source_weights']@r['Q'],[.25,.5,.25],atol=1e-10)
        np.testing.assert_allclose(r['Q']@r['target_states'],r['source_states'],atol=1e-10)

    def test_fixed_coupling_rejects_reverse_convex_order(self):
        with self.assertRaisesRegex(ValueError,'convex order'):
            couple_marginals([.8,1.2],[.5,.5],[1.],[1.])

    def test_fixed_coupling_removes_only_exact_zeros(self):
        p=np.array([1e-12,1-2e-12,1e-12,0.])
        r=couple_marginals([.8,1,1.2,2],p,[.8,1,1.2,2],p)
        self.assertTrue(r['validated'])
        np.testing.assert_array_equal(r['source_indices'],[0,1,2])
        np.testing.assert_array_equal(r['target_indices'],[0,1,2])

    def test_resource_cap_precedes_quadratic_allocation(self):
        a=np.array([.5,1.,1.5]);p=np.array([0,1.,0])
        k=np.linspace(.6,1.4,400)
        with self.assertRaisesRegex(ValueError,'resource cap'):
            fit_state_dependent_transition(a,p,dict(market_k=k))

    def test_projection_preserves_conditional_means_and_quote_calls(self):
        a=np.array([.5,.8,1.,1.2,1.5]);p=np.array([0,.25,.5,.25,0])
        grid=np.array([.5,.9,1.1,1.5]);q,P=project_source(a,p,grid)
        np.testing.assert_allclose(P@grid,a,atol=1e-15)
        np.testing.assert_allclose(np.asarray(P.sum(axis=1)).ravel(),1)
        np.testing.assert_allclose(calls(grid,q,grid),calls(a,p,grid),atol=1e-15)

    def test_constructive_state_dependent_fit(self):
        a=np.array([.5,.8,1.,1.2,1.5]);p=np.array([0,.25,.5,.25,0])
        target=np.array([.1,.15,.5,.15,.1]);k=np.array([.7,.9,1.,1.1,1.3])
        c=calls(a,target,k)
        r=fit_state_dependent_transition(a,p,dict(market_k=k,pure_bid=c-.001,
            pure_ask=c+.001,pure_mid=c))
        self.assertTrue(r['validated'])
        self.assertEqual(r['coverage'],1.)
        np.testing.assert_allclose(r['Q']@r['grid'],r['grid'],atol=1e-9)
        np.testing.assert_allclose(np.asarray(p@r['projection']).ravel()@r['Q'],r['target'],atol=1e-14)

    def test_calendar_conflict_cannot_be_fitted_away(self):
        a=np.array([.5,1.,1.5]);p=np.array([.5,0,.5]);k=np.array([1.])
        r=fit_state_dependent_transition(a,p,dict(market_k=k,pure_bid=np.array([.01]),
            pure_ask=np.array([.02]),pure_mid=np.array([.015])))
        self.assertTrue(r['validated'])
        self.assertEqual(r['coverage'],0.)
        self.assertGreaterEqual(r['outside_objective'],23-1e-7)


if __name__=='__main__':unittest.main()
