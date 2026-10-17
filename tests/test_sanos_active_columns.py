import time
import unittest
from unittest.mock import patch
import numpy as np
from test_sanos_incremental import highspy
from test_sanos_core import example_inputs
from sanos_incremental import PersistentSanosLP
from sanos_solver_research import active_state_update,prepare_active_model
from sanos_solver_research import global_lower_bound


@unittest.skipIf(highspy is None,'Requires native HiGHS')
class ActiveColumnTests(unittest.TestCase):
    def session(self):
        s=PersistentSanosLP(example_inputs())
        self.assertTrue(s.solve()['success'])
        return s

    def test_full_objective_certification_and_original_constraints(self):
        s=self.session();previous=s._accepted_x.copy()
        quotes={e:(b.copy()+1e-5,a.copy()+1e-5) for e,(b,a) in s.quotes.items()}
        s.update_quotes(quotes);r=active_state_update(s,previous,budget=2)
        self.assertTrue(r['success'],r)
        self.assertTrue(r['audit']['validated'])
        self.assertLessEqual(abs(r['global_gap']),1e-6)
        # Experiment does not corrupt accepted state of the full fallback model.
        np.testing.assert_array_equal(s._accepted_x,previous)
        cold=s.solve();self.assertTrue(cold['success'])
        self.assertLessEqual(abs(cold['actual_objective']-r['audit']['actual_objective']),1e-6)

    def test_prepared_model_can_handle_two_successive_quote_updates(self):
        s=self.session();model=prepare_active_model(s,s._accepted_x);previous=s._accepted_x.copy()
        for _ in range(2):
            s.update_quotes({e:(b+1e-5,a+1e-5) for e,(b,a) in s.quotes.items()})
            r=active_state_update(s,previous,prepared=model,budget=2)
            self.assertTrue(r['success'],r)
            self.assertTrue(r['audit']['validated']);previous=r['x']

    def test_feasibility_without_global_certificate_is_never_accepted(self):
        s=self.session()
        def weak(*args,**kwargs):
            c=global_lower_bound(*args,**kwargs);c['lower_bound']-=1000;return c
        with patch('sanos_solver_research.global_lower_bound',side_effect=weak):
            r=active_state_update(s,s._accepted_x,budget=2)
        self.assertFalse(r['success'])
        self.assertEqual(r['reason'],'no_global_certificate')


if __name__=='__main__':unittest.main()
