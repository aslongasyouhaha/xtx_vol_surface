import unittest
import numpy as np
from scipy.optimize import linprog
from sanos_core import build_convex_order_problem
from sanos_incremental import quote_vectors,validate_frame
from test_sanos_core import example_inputs


class CalendarReductionTests(unittest.TestCase):
    def test_next_knots_match_union_objective_and_full_constraint_audit(self):
        inputs=example_inputs()
        for i,v in enumerate(inputs.values()):
            v['anchor_k']=np.array([.8,.87,1.,1.17,1.25]) if i%2==0 else np.array([.8,.93,1.,1.1,1.25])
        old=build_convex_order_problem(inputs,calendar_grid='union')
        new=build_convex_order_problem(inputs,calendar_grid='next')
        self.assertLess(new['A'].shape[0],old['A'].shape[0])
        objectives=[]
        for p in [old,new]:
            r=linprog(p['c'],A_ub=p['A'],b_ub=p['b'],A_eq=p['E'],b_eq=np.ones(p['E'].shape[0]),bounds=(0,None),method='highs')
            self.assertTrue(r.success)
            quotes={e:(v['pure_bid'],v['pure_ask']) for e,v in p['items'].items()}
            upper,cost=quote_vectors(old,quotes)
            audit=validate_frame(old,upper,cost,r.x,quotes)
            self.assertTrue(audit['validated'])
            objectives.append(audit['actual_objective'])
        self.assertLess(abs(objectives[0]-objectives[1]),1e-6)

    def test_reduction_requires_supported_mode(self):
        with self.assertRaises(ValueError):build_convex_order_problem(example_inputs(),calendar_grid='previous')


if __name__=='__main__':unittest.main()
