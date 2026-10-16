import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sanos_core import call_kernel
from sanos_surface import SanosSurface,audit_surface


def example(**kwargs):
    return SanosSurface([.1,.3,.7],[[.7,1.,1.3]]*3,
                        [[.1,.8,.1],[.2,.6,.2],[.3,.4,.3]],[.01,.02,.04],**kwargs)


class SurfaceTests(unittest.TestCase):
    def test_exact_slices_and_linear_midpoint_price_combination(self):
        s=example(time_interpolation='linear');k=np.array([0.,.8,1.,1.2,2.])
        for j,T in enumerate(s.times):
            expected=call_kernel(s.anchors[j],k[1:],s.variances[j])@s.weights[j]
            np.testing.assert_allclose(s.pure_call(T,k[1:]),expected,atol=1e-14)
        np.testing.assert_allclose(s.pure_call(.2,k),.5*(s.pure_call(.1,k)+s.pure_call(.3,k)),atol=1e-14)
        self.assertAlmostEqual(s.pure_call(.2,0.),1.)

    def test_atm_pchip_uses_fitted_variance_and_one_weight_for_every_strike(self):
        s=example();T=.2;k=np.array([.8,1.,1.2])
        lo=s.pure_call(.1,k);hi=s.pure_call(.3,k);mid=s.pure_call(T,k)
        alpha=(mid-lo)/(hi-lo)
        np.testing.assert_allclose(alpha,np.full(3,alpha[0]),atol=1e-12)
        self.assertGreater(alpha[0],0);self.assertLess(alpha[0],1)
        from scipy.special import ndtr
        expected=2*ndtr(np.sqrt(s._atm_curve(T))/2)-1
        self.assertAlmostEqual(s.pure_call(T,1.),expected,places=14)
        self.assertFalse(np.allclose(s.atm_variances,s.variances))

    def test_intermediate_times_preserve_bounds_strike_convexity_and_calendar_order(self):
        for mode in ['linear','atm_pchip']:
            result=audit_surface(example(time_interpolation=mode))
            self.assertTrue(result['passed'],result)

    def test_cash_conversion_broadcasting_and_explicit_domain_errors(self):
        s=example();values=s.pure_call(np.array([.15,.4])[:,None],np.array([.8,1.,1.2])[None,:])
        self.assertEqual(values.shape,(2,3))
        self.assertAlmostEqual(s.call_price(.2,110,forward=100,discount_factor=.98),98*s.pure_call(.2,1.1))
        self.assertAlmostEqual(s.pure_call_days(73,1.),s.pure_call(.2,1.))
        for T,k in [(.09,1.),(.71,1.),(.2,-1),(.2,np.nan)]:
            with self.assertRaises(ValueError):s.pure_call(T,k)
        with self.assertRaises(ValueError):s.call_price(.2,100,forward=0,discount_factor=1)

    def test_flat_atm_and_single_expiry(self):
        s=SanosSurface([.1,.2],[[1.],[1.]],[[1.],[1.]],[.01,.01])
        self.assertAlmostEqual(s.pure_call(.15,1.),s.pure_call(.1,1.))
        single=SanosSurface([.1],[[1.]],[[1.]],[.01])
        self.assertTrue(audit_surface(single)['passed'])
        with self.assertRaises(ValueError):single.pure_call(.2,1.)

    def test_invalid_moments_and_reversed_convex_order_are_rejected(self):
        with self.assertRaises(ValueError):SanosSurface([.1],[[1.]],[[.9]],[.01])
        with self.assertRaisesRegex(ValueError,'convex-order'):
            SanosSurface([.1,.2],[[.7,1.,1.3]]*2,[[.4,.2,.4],[.1,.8,.1]],[.01,.02])

    def test_saved_model_reload_and_failed_calibration_rejection(self):
        s=example()
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'summary.json').write_text(json.dumps({'success':True}))
            data={}
            for j in range(3):
                ex=f'2025-0{j+1}-01'
                for key,value in [('T',s.times[j]),('q',s.weights[j]),('anchors',s.anchors[j]),('smooth_variance',s.variances[j])]:
                    data[ex+'__'+key]=value
            np.savez(p/'model.npz',**data)
            loaded=SanosSurface.from_directory(p)
            np.testing.assert_allclose(loaded.pure_call(.2,[.8,1,1.2]),s.pure_call(.2,[.8,1,1.2]))
            (p/'summary.json').write_text(json.dumps({'success':False}))
            with self.assertRaises(ValueError):SanosSurface.from_directory(p)

    def test_construct_from_actual_lp_solution(self):
        from sanos_core import solve_convex_order_lp
        from test_sanos_core import example_inputs
        sol=solve_convex_order_lp(example_inputs())
        s=SanosSurface.from_solution(sol)
        self.assertTrue(audit_surface(s)['passed'])
        for ex,v in sol['inputs'].items():
            np.testing.assert_allclose(s.pure_call_days(v['dte'],v['market_k']),sol['fit_by_expiry'][ex],atol=1e-14)
        with self.assertRaises(ValueError):SanosSurface.from_solution({'success':False})

    def test_preserves_raw_solver_weights_and_copies_inputs(self):
        q=np.array([-1e-9,1+2e-9,-1e-9]);a=np.array([.7,1.,1.3])
        s=SanosSurface([.1],[a],[q],[.01]);q[:]=999;a[:]=999
        self.assertEqual(s.weights[0][0],-1e-9)
        self.assertFalse(s.weights[0].flags.writeable)
        self.assertAlmostEqual(s.pure_call(.1,0.),1.)


if __name__=='__main__':unittest.main()
