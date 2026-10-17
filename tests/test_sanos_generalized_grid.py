import unittest
import numpy as np
from sanos_generalized import ExactProduct
from sanos_generalized import GridProduct


class GridTests(unittest.TestCase):
    def test_initial_state_and_multistep_moments(self):
        model=GridProduct(4097)
        self.assertEqual(model.states[model.left],1.)
        k=np.geomspace(.1,3,401);previous=model.calls(k)
        for _ in range(5):
            a=np.array([.9,1,1.1]);v=np.array([.001,.003,.002]);q=[.2,.6,.2]
            A=model.kernel(a,v,k);model.advance(a,v,q)
            current=model.calls(k)
            np.testing.assert_allclose(current,A@q,atol=1e-14)
            self.assertLess(abs(model.weights.sum()-1),1e-11)
            self.assertLess(abs(model.weights@model.states-1),1e-11)
            self.assertGreaterEqual(np.min(current-previous),-1e-12)
            previous=current

    def test_exact_reference_and_refinement(self):
        errors=[];k=np.linspace(.7,1.3,41)
        for nx,nq in [(4097,64),(16385,256)]:
            grid=GridProduct(nx,nq);exact=ExactProduct()
            for _ in range(2):
                a=[.95,1.05];v=[.002,.004];q=[.5,.5]
                grid.advance(a,v,q);exact.advance(a,v,q)
            errors.append(np.max(abs(grid.calls(k)-exact.calls(k))))
        self.assertLess(errors[-1],errors[0])
        self.assertLess(errors[-1],2e-5)

    def test_tail_loss_rejected(self):
        model=GridProduct(1025)
        with self.assertRaises(ValueError):model.kernel([100],[.01],[1])

    def test_fit_and_saved_density_agree(self):
        m=GridProduct(4097);k=np.linspace(.8,1.2,31)
        a=[.9,1,1.1];v=[.002,.001,.003]
        c=m.kernel(a,v,k)@np.array([.2,.6,.2])
        r=m.fit_step(a,v,k,c-1e-5,c+1e-5)
        np.testing.assert_allclose(r['fit'],m.calls(k),atol=1e-14)


if __name__=='__main__':unittest.main()
