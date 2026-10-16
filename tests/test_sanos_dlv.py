import unittest
import numpy as np
from sanos_dlv import (DLVTransition,DLVModel,latent_calls,recover_local_volatilities,
                        marginal_from_latent_calls,recovery_diagnostics,embed_common_grid)
from sanos_surface import audit_surface


class DLVTests(unittest.TestCase):
    def test_three_state_operator_has_paper_column_orientation(self):
        t=DLVTransition([.5,1,1.5],[.2],.25)
        expected=np.array([[1,-.02,0],[0,1.04,0],[0,-.02,1.]])
        np.testing.assert_allclose(t.matrix.toarray(),expected,atol=1e-15)
        q=t.apply([0,1,0]);np.testing.assert_allclose(q,[.02/1.04,1/1.04,.02/1.04])
        Q=t.transition_matrix();np.testing.assert_allclose(Q.sum(axis=0),1,atol=1e-14)
        np.testing.assert_allclose(np.array([.5,1,1.5])@Q,[.5,1,1.5],atol=1e-14)
        self.assertGreater(np.max(abs(Q.sum(axis=1)-1)),.01)
        self.assertTrue(t.audit()['passed'])

    def test_nonuniform_grid_transition_preserves_each_conditional_mean(self):
        a=np.array([.4,.75,1.,1.4,2.])
        t=DLVTransition(a,[.15,.3,.2],.4);Q=t.transition_matrix()
        self.assertGreaterEqual(Q.min(),0)
        np.testing.assert_allclose(Q.sum(axis=0),1,atol=1e-13)
        np.testing.assert_allclose(a@Q,a,atol=1e-13)
        self.assertTrue(t.audit()['passed'])

    def test_zero_volatility_or_zero_time_is_identity(self):
        for sigma,dt in [([0,0,0],.4),([.2,.3,.4],0)]:
            np.testing.assert_array_equal(DLVTransition([.5,.8,1,1.5,2],sigma,dt).transition_matrix(),np.eye(5))

    def test_finite_dlv_recovery_and_call_probability_roundtrip(self):
        a=np.array([.5,.8,1.,1.35,1.9]);before=np.array([0,0,1.,0,0])
        t=DLVTransition(a,[.2,.25,.3],.2);after=t.apply(before)
        calls=latent_calls(a,after)
        direct=np.maximum(a[None,:]-a[:,None],0)@after
        np.testing.assert_allclose(calls,direct,atol=1e-15)
        np.testing.assert_allclose(marginal_from_latent_calls(a,calls),after,atol=2e-15)
        sigma=recover_local_volatilities(a,before,after,.2)
        np.testing.assert_allclose(sigma,t.sigma,atol=1e-11)
        np.testing.assert_allclose(DLVTransition(a,sigma,.2).apply(before),after,atol=1e-13)

    def test_zero_over_zero_is_zero_but_positive_over_zero_is_rejected(self):
        a=[.5,1,1.5];before=[0,1,0]
        np.testing.assert_array_equal(recover_local_volatilities(a,before,before,.1),[0,0,0])
        # This target is in convex order but needs infinite interior DLV.
        after=[.5,0,.5];report=recovery_diagnostics(a,before,after,.1)
        self.assertEqual(report['singular_interior_indices'],[1])
        self.assertFalse(report['finite_recovery_possible'])
        with self.assertRaisesRegex(ValueError,'finite'):recover_local_volatilities(a,before,after,.1)

    def test_common_grid_embedding_preserves_original_calls_and_mean(self):
        grids=[[.5,1,1.5],[.25,1,1.75]];qs=[[.1,.8,.1],[.2,.6,.2]]
        common,q=embed_common_grid(grids,qs);k=np.linspace(0,2,51)
        for i in range(2):
            np.testing.assert_allclose(np.maximum(common[None,:]-k[:,None],0)@q[i],
                np.maximum(np.asarray(grids[i])[None,:]-k[:,None],0)@qs[i],atol=1e-15)
            self.assertAlmostEqual(common@q[i],1.)

    def test_dlv_generated_model_connects_to_validated_continuous_surface(self):
        a=[.5,.75,1,1.3,1.8];times=np.array([30,60,120])/365
        model=DLVModel(a,times,[[.2,.25,.3]]*3,[.001,.002,.004])
        np.testing.assert_allclose(model.weights.sum(axis=1),1,atol=1e-13)
        np.testing.assert_allclose(model.weights@model.anchors,1,atol=1e-13)
        self.assertTrue(audit_surface(model.surface())['passed'])
        for previous,current in zip(model.weights[:-1],model.weights[1:]):
            self.assertGreaterEqual((latent_calls(a,current)-latent_calls(a,previous)).min(),-1e-14)

    def test_invalid_inputs_are_rejected_without_parameter_clipping(self):
        for anchors,sigma,dt in [([.5,1,1],[.2],.1),([0,1,2],[.2],.1),
                                  ([.5,1,2],[-.2],.1),([.5,1,2],[1,.2,0],.1),([.5,1,2],[.2],-1)]:
            with self.assertRaises(ValueError):DLVTransition(anchors,sigma,dt)
        with self.assertRaises(ValueError):DLVModel([.5,.9,1.5],[.1],[[.2]],[.01])
        with self.assertRaises(ValueError):recover_local_volatilities([.5,1,1.5],[0,1,0],[.1,.8,.1],0)


if __name__=='__main__':unittest.main()
