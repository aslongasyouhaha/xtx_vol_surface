import unittest
import tempfile
from pathlib import Path
import numpy as np
from sanos_dlv import DLVModel
from sanos_dlv_calibration import DLVCalibrator,calibration_grid


def synthetic_inputs():
    anchors=np.array([.5,.8,1.,1.3,1.8]);times=np.array([30,60])/365
    variances=np.array([.001,.002])
    model=DLVModel(anchors,times,[[.15,.22,.3],[.2,.3,.25]],variances)
    inputs={}
    strikes=np.linspace(.85,1.15,9)
    for j,T in enumerate(times):
        mid=model.surface().pure_call(T,strikes)
        inputs[j]=dict(dte=int(round(T*365)),T=T,anchor_k=anchors,market_k=strikes,
            smooth_variance=variances[j],V_j=.04*T,pure_bid=np.maximum(mid-1e-4,0),
            pure_ask=mid+1e-4,pure_mid=mid)
    return inputs,model


class CalibrationTests(unittest.TestCase):
    def test_banded_forward_matches_independent_dlv_model(self):
        inputs,model=synthetic_inputs();c=DLVCalibrator(inputs,model.anchors)
        sigma=np.array([[.15,.22,.3],[.2,.3,.25]])
        x=(sigma**2/c.rate_scale[:,None]).ravel()
        result=c.evaluate(x)
        np.testing.assert_allclose(result['weights'],model.weights,atol=1e-14)
        self.assertLess(result['objective'],1e-6)

    def test_adjoint_gradient_includes_future_expiry_effects(self):
        inputs,model=synthetic_inputs();c=DLVCalibrator(inputs,model.anchors)
        x=np.array([.8,1.2,.9,1.3,.7,1.1])
        for smoothing in [.05,0.]:
            exact=c.evaluate(x,smoothing=smoothing)['gradient'];finite=[]
            for j in range(len(x)):
                step=np.zeros_like(x);step[j]=1e-5
                finite.append((c.evaluate(x+step,smoothing=smoothing,gradient=False)['optimization_loss']-
                               c.evaluate(x-step,smoothing=smoothing,gradient=False)['optimization_loss'])/(2e-5))
            np.testing.assert_allclose(exact,finite,rtol=2e-5,atol=2e-7)

    def test_direct_calibration_reduces_actual_paper_loss_and_preserves_moments(self):
        inputs,model=synthetic_inputs();c=DLVCalibrator(inputs,model.anchors)
        result=c.calibrate(time_limit=3,max_iterations=250)
        self.assertTrue(result['validated'],result)
        self.assertLess(result['objective'],result['initial_objective']*.01)
        self.assertGreaterEqual(result['coverage'],.95)
        np.testing.assert_allclose(result['model'].weights.sum(axis=1),1,atol=1e-12)
        np.testing.assert_allclose(result['model'].weights@model.anchors,1,atol=1e-12)

    def test_time_limit_is_reported_without_claiming_optimality(self):
        inputs,model=synthetic_inputs();c=DLVCalibrator(inputs,model.anchors)
        result=c.calibrate(time_limit=1e-9)
        self.assertEqual(result['stop_reason'],'time_limit')
        self.assertFalse(result['optimizer_converged']);self.assertFalse(result['global_optimality_certified'])
        self.assertTrue(result['validated'])
        with self.assertRaises(ValueError):c.evaluate(np.full(np.prod(c.shape),-1.))

    def test_parameter_rescaling_preserves_model_and_adjoint(self):
        inputs,model=synthetic_inputs();c=DLVCalibrator(inputs,model.anchors)
        x=np.array([.8,1.2,.9,1.3,.7,1.1]);before=c.evaluate(x,gradient=False)
        x=c.rescale_parameters(x);after=c.evaluate(x)
        np.testing.assert_allclose(after['weights'],before['weights'],atol=1e-14)
        step=np.zeros_like(x);step[2]=1e-5
        finite=(c.evaluate(x+step,gradient=False)['optimization_loss']-c.evaluate(x-step,gradient=False)['optimization_loss'])/2e-5
        self.assertAlmostEqual(after['gradient'][2],finite,places=5)

    def test_interior_seed_is_a_finite_initialization_not_a_final_density_constraint(self):
        inputs,model=synthetic_inputs();c=DLVCalibrator(inputs,model.anchors)
        x,report=c.interior_lp_initialization(loss_allowance=.1,time_limit=3)
        self.assertGreater(report['reference_mass'],0)
        self.assertGreater(report['min_seed_probability'],0)
        x=c.rescale_parameters(x)
        result=c.calibrate(initial_parameters=x,time_limit=2,max_iterations=200)
        self.assertTrue(result['validated'])
        self.assertLessEqual(result['objective'],report['seed_objective']+1e-6)
        # Nonnegativity, not the seed's positive lower bounds, defines final parameters.
        zero=np.zeros_like(x)
        self.assertGreaterEqual(c.evaluate(zero)['weights'].min(),0)

    def test_saved_calibration_parameters_repropagate_and_detect_inconsistent_weights(self):
        inputs,model=synthetic_inputs()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.npz'
            data=dict(anchors=model.anchors,times=model.times,local_volatilities=[[.15,.22,.3],[.2,.3,.25]],
                      smoothing_variances=model.smoothing_variances,initial_weights=model.initial_weights,weights=model.weights)
            np.savez(path,**data);loaded=DLVModel.from_npz(path)
            np.testing.assert_allclose(loaded.surface().pure_call_days(45,[.9,1.,1.1]),
                                       model.surface().pure_call_days(45,[.9,1.,1.1]),atol=1e-14)
            data['weights']=np.ones_like(model.weights);np.savez(path,**data)
            with self.assertRaises(ValueError):DLVModel.from_npz(path)


if __name__=='__main__':unittest.main()
