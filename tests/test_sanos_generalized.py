import unittest
import numpy as np
from sanos_generalized import ExactProduct
from sanos_core import call_kernel


class GeneralizedTests(unittest.TestCase):
    def test_explicit_two_step_and_calendar(self):
        a=np.array([.8,1.,1.2]);q=np.array([.25,.5,.25]);v=np.array([.01,.02,.03])
        k=np.linspace(.1,3,101);m=ExactProduct()
        m.advance(a,v,q);first=m.calls(k)
        kernel=m.kernel(a,v,k)
        expected=sum(q[i]*q[j]*call_kernel([a[i]*a[j]],k,v[i]+v[j])[:,0]
                     for i in range(3) for j in range(3))
        np.testing.assert_allclose(kernel@q,expected,atol=1e-15)
        m.advance(a,v,q)
        np.testing.assert_allclose(m.calls(k),expected,atol=1e-15)
        self.assertGreaterEqual(np.min(expected-first),-1e-15)
        self.assertAlmostEqual(m.weights@m.states,1)

    def test_sequential_fit_reprices_synthetic_targets(self):
        truth=ExactProduct();fit=ExactProduct()
        a=np.array([.9,1.,1.1]);v=np.array([.001,.002,.003]);q=np.array([.2,.6,.2])
        k=np.linspace(.7,1.3,31)
        for _ in range(3):
            truth.advance(a,v,q);c=truth.calls(k)
            result=fit.fit_step(a,v,k,c-1e-6,c+1e-6)
            self.assertLessEqual(np.max(abs(result['fit']-c)),1.001e-6)
            np.testing.assert_allclose(result['fit'],fit.calls(k),atol=1e-15)

    def test_zero_variance_and_budget(self):
        m=ExactProduct(max_paths=3)
        m.advance([.8,1.2],[0,0],[.5,.5])
        np.testing.assert_allclose(m.calls([1]),[.1])
        with self.assertRaises(ValueError):m.advance([.8,1.2],[0,0],[.5,.5])
        with self.assertRaises(ValueError):ExactProduct().advance([1],[.1],[-1])


if __name__=='__main__':unittest.main()
