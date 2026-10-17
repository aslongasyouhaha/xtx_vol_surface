import time
import unittest
from unittest.mock import patch
import numpy as np
from test_sanos_incremental import highspy
from test_sanos_core import example_inputs
from sanos_incremental import PersistentSanosLP, quote_vectors
from sanos_solver_research import mean_one_minimum, global_lower_bound, restricted_update


class BoundTests(unittest.TestCase):
    def test_moment_minimum(self):
        self.assertEqual(mean_one_minimum([.5,1,1.5],[0,3,0]),0)
        self.assertEqual(mean_one_minimum([.5,2],[2,5]),3)
        with self.assertRaises(ValueError):mean_one_minimum([1.1,2],[0,0])


@unittest.skipIf(highspy is None,'Requires native HiGHS')
class WorkingSetTests(unittest.TestCase):
    def seed_update(self):
        s=PersistentSanosLP(example_inputs(),working_set=True)
        self.assertTrue(s.solve()['success'])
        quotes={ex:(b.copy(),a.copy()) for ex,(b,a) in s.quotes.items()}
        ex=next(iter(quotes));mid=s.problem['items'][ex]['G']@np.array([.1,.8,.1])
        quotes[ex]=(mid-1e-4,mid+1e-4);s.update_quotes(quotes)
        return s

    def test_random_repaired_duals_bound_known_zero_optimum(self):
        s=PersistentSanosLP(example_inputs());upper,costs=quote_vectors(s.problem,s.quotes)
        rng=np.random.default_rng(12)
        for _ in range(20):
            dual=rng.normal(size=len(upper))
            result=global_lower_bound(s.problem,s.audit_cache['matrix'],upper,costs,dual)
            self.assertLessEqual(result['lower_bound'],1e-9)
            for ex in s.problem['items']:
                self.assertGreaterEqual(result['reduced_cost'][slice(*s.problem['z_offsets'][ex])].min(),0)

    def test_public_local_cache_and_continued_update(self):
        s=self.seed_update();result=s.solve()
        self.assertTrue(result['success']);self.assertEqual(result['optimization_path'],'certified_working_set')
        self.assertTrue(result['validated']);self.assertLessEqual(abs(result['global_objective_gap']),1e-6)
        result['x'][:]=999
        cached=s.solve();self.assertTrue(cached['solution_reused']);self.assertLess(cached['x'].max(),999)
        self.assertEqual(cached['iterations'],0)
        ex=next(iter(s.quotes));quotes={k:(b.copy(),a.copy()) for k,(b,a) in s.quotes.items()}
        mid=s.problem['items'][ex]['G']@np.array([.15,.7,.15]);quotes[ex]=(mid-1e-4,mid+1e-4)
        s.update_quotes(quotes);again=s.solve();self.assertTrue(again['success']);self.assertTrue(again['validated'])
        cold=PersistentSanosLP(example_inputs(),working_set=False);cold.update_quotes(quotes);reference=cold.solve()
        self.assertTrue(reference['success'])
        self.assertLessEqual(abs(again['actual_outside_objective']-reference['actual_outside_objective']),1e-6)

    def test_feasible_candidate_without_global_certificate_is_rejected(self):
        s=self.seed_update()
        original=global_lower_bound
        def weak(*args,**kwargs):
            result=original(*args,**kwargs);result['lower_bound']=-1000.;return result
        with patch('sanos_solver_research.global_lower_bound',side_effect=weak):
            local=restricted_update(s,s._accepted_x,s._accepted_dual,s._accepted_upper,s._accepted_costs,
                                    deadline=time.perf_counter()+2,max_expiries=2)
        self.assertFalse(local['success'])

    def test_failed_certificate_falls_back_to_full_solver(self):
        s=self.seed_update()
        with patch('sanos_solver_research.restricted_update',return_value=dict(success=False,reason='unproven',attempts=[])):
            result=s.solve()
        self.assertTrue(result['success']);self.assertTrue(result['validated'])
        self.assertEqual(result['optimization_path'],'full')
        self.assertEqual(result['working_set_fallback_reason'],'unproven')


if __name__=='__main__':unittest.main()
