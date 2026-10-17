import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

deps=Path(__file__).resolve().parents[1]/'data/outputs'/'native_deps'
if (deps/'highspy').exists():sys.path.insert(0,str(deps))
try:
    import highspy
except ImportError:
    highspy=None

from test_sanos_core import example_inputs
from sanos_core import build_convex_order_problem,solve_convex_order_lp
from sanos_incremental import (PersistentSanosLP,compile_native_model,native_quote_vectors,
                               quote_vectors)


@unittest.skipIf(highspy is None,'Install requirements.txt for native solver tests')
class IncrementalTests(unittest.TestCase):
    def test_default_quote_update_uses_retained_global_basis_without_local_attempt(self):
        s=PersistentSanosLP(example_inputs())
        self.assertTrue(s.solve()['success'])
        quotes={e:(b.copy()+1e-5,a.copy()+1e-5) for e,(b,a) in s.quotes.items()}
        s.update_quotes(quotes)
        with patch('sanos_solver_research.restricted_update',side_effect=AssertionError('Unexpected local rebuild')):
            r=s.solve()
        self.assertTrue(r['success'])
        self.assertTrue(r['basis_before_run'])
        self.assertEqual(r['optimization_path'],'full')
        self.assertNotIn('working_set_attempts',r)

    def test_unchanged_model_has_zero_simplex_iterations(self):
        session=PersistentSanosLP(example_inputs())
        initial=session.solve();self.assertTrue(initial['success'])
        repeated=session.solve();self.assertTrue(repeated['success'])
        self.assertEqual(repeated['iterations'],0)
        np.testing.assert_allclose(initial['x'],repeated['x'],atol=1e-8)

    def test_fallback_rejects_solver_optimal_but_invalid_candidate(self):
        session=PersistentSanosLP(example_inputs())
        real_solve=session._solve_once
        calls=[]
        def first_invalid_then_real(**kwargs):
            calls.append(session._profile_name)
            result=real_solve(**kwargs)
            if len(calls)==1:
                # Simulate an independent-audit rejection of an Optimal status.
                result.update(success=False,validated=False,max_constraint_violation=1e-3)
            return result
        with patch.object(session,'_solve_once',side_effect=first_invalid_then_real):
            result=session.solve()
        self.assertTrue(result['success'],result['message'])
        self.assertEqual(calls,['unscaled','bounded_scaling'])
        self.assertEqual(len(result['numerics_attempts']),2)
        self.assertFalse(result['numerics_attempts'][0]['success'])
        self.assertEqual(result['numerics_profile'],'bounded_scaling')

    def test_otm_scaling_preserves_feasibility_and_objective(self):
        inputs=example_inputs();p=build_convex_order_problem(inputs,weight_mode='legacy',mid_penalty=0.);compiled=compile_native_model(p,scaling='otm_spread')
        sol=solve_convex_order_lp(inputs,mid_penalty=0.,weight_mode='legacy');self.assertTrue(sol['success'])
        quotes={ex:(v['pure_bid'],v['pure_ask']) for ex,v in p['items'].items()}
        upper,costs=native_quote_vectors(p,compiled,quotes)
        scaled=sol['x']/compiled['column_scale']
        self.assertLess(np.max(compiled['matrix']@scaled-upper),1e-7)
        self.assertAlmostEqual(float(costs@scaled)*compiled['objective_scale'],sol['objective'],places=10)

    def test_quote_update_retains_basis_and_matches_cold_optimum(self):
        inputs=example_inputs();warm=PersistentSanosLP(inputs,working_set=False)
        first=warm.solve();self.assertTrue(first['success'],first['message'])
        quotes={}
        for index,(ex,v) in enumerate(inputs.items()):
            q=np.array([.1,.8,.1]) if index==0 else np.array([.35,.3,.35])
            mid=v['G']@q;quotes[ex]=(mid-1e-4,mid+1e-4)
        changes=warm.update_quotes(quotes);self.assertGreater(changes['updated_rows'],0)
        result=warm.solve();self.assertTrue(result['success'],result['message'])
        self.assertTrue(result['basis_before_run'])
        cold=PersistentSanosLP(inputs);cold.update_quotes(quotes);other=cold.solve()
        self.assertTrue(other['success'],other['message'])
        self.assertAlmostEqual(result['actual_outside_objective'],other['actual_outside_objective'],places=6)

    def test_invalid_update_leaves_previous_quotes_usable(self):
        session=PersistentSanosLP(example_inputs());self.assertTrue(session.solve()['success'])
        quotes={ex:(bid.copy(),ask.copy()) for ex,(bid,ask) in session.quotes.items()}
        ex=next(iter(quotes));bid,ask=quotes[ex];ask[0]=bid[0]
        with self.assertRaises(ValueError):session.update_quotes(quotes)
        self.assertTrue(session.solve()['success'])
        with self.assertRaisesRegex(ValueError,'Expiry set'):
            session.update_quotes({})


if __name__=='__main__':unittest.main()
