import unittest
from unittest.mock import patch
import numpy as np
from test_sanos_incremental import highspy
from test_sanos_core import example_inputs
from sanos_incremental import PersistentSanosLP
from sanos_solver_research import solve_generated


@unittest.skipIf(highspy is None,'Requires native HiGHS')
class GenerationTests(unittest.TestCase):
    def incompatible_quotes(self):
        inputs=example_inputs()
        for v,q in zip(inputs.values(),[[.45,.1,.45],[.05,.9,.05]]):
            mid=v['G']@np.array(q);v['pure_bid']=mid-1e-4;v['pure_ask']=mid+1e-4
        return inputs

    def test_violated_calendar_rows_are_added_and_full_objective_matches(self):
        s=PersistentSanosLP(self.incompatible_quotes(),working_set=False)
        generated=solve_generated(s);full=s.solve()
        self.assertTrue(full['success']);self.assertTrue(generated['success'],generated)
        self.assertGreater(generated['attempts'][0]['violated_omitted_rows'],0)
        self.assertGreater(len(generated['attempts']),1)
        self.assertTrue(generated['audit']['validated'])
        self.assertLessEqual(abs(generated['audit']['actual_outside_objective']-full['actual_outside_objective']),1e-6)

    def test_iteration_limit_never_accepts_relaxed_infeasible_surface(self):
        s=PersistentSanosLP(self.incompatible_quotes(),working_set=False)
        result=solve_generated(s,max_rounds=1)
        self.assertFalse(result['success'])
        self.assertGreater(result['attempts'][0]['max_omitted_violation'],2e-7)

    def test_failed_global_bound_never_accepts_feasible_surface(self):
        s=PersistentSanosLP(example_inputs(),working_set=False)
        with patch('sanos_solver_research.global_lower_bound',return_value={'lower_bound':-1000.}):
            result=solve_generated(s)
        self.assertFalse(result['success'])


if __name__=='__main__':unittest.main()
