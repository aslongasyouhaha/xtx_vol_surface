import copy
import unittest
import numpy as np
from test_sanos_incremental import highspy
from test_sanos_core import example_inputs
from sanos_core import build_convex_order_problem,solve_convex_order_lp
from sanos_incremental import PersistentSanosLP,quote_vectors,prepare_validation,compile_native_model,native_quote_vectors
from sanos_solver_research import global_lower_bound


@unittest.skipIf(highspy is None,'Requires native HiGHS')
class PaperIncrementalTests(unittest.TestCase):
    def test_paper_otm_rows_preserve_full_objective_and_midpoint_constraints(self):
        inputs=example_inputs();p=build_convex_order_problem(inputs)
        compiled=compile_native_model(p,scaling='otm');solution=solve_convex_order_lp(inputs)
        quotes={e:(v['pure_bid'],v['pure_ask']) for e,v in p['items'].items()}
        upper,costs=native_quote_vectors(p,compiled,quotes)
        activity=compiled['matrix']@solution['x']
        self.assertLessEqual(float(np.max(activity-upper)),2e-7)
        self.assertLessEqual(float(np.max(compiled['lower']-activity)),2e-7)
        np.testing.assert_array_equal(costs,p['c'])
        # Targets include all globally appended midpoint rows after quote changes.
        quotes={e:(b+1e-5,a+3e-5) for e,(b,a) in quotes.items()}
        upper,_=native_quote_vectors(p,compiled,quotes)
        for e,v in p['items'].items():
            plus,minus,n=p['mid_rows'][e];target=sum(quotes[e])*.5-np.maximum(1-v['market_k'],0)
            np.testing.assert_allclose(upper[plus:plus+n],target)
            np.testing.assert_allclose(upper[minus:minus+n],-target)

    def test_paper_otm_solver_and_update_match_original_positive_loss(self):
        inputs=example_inputs();p=build_convex_order_problem(inputs)
        s=PersistentSanosLP(inputs,problem=p,compiled=compile_native_model(p,scaling='otm'))
        self.assertTrue(s.solve()['success'])
        updated=copy.deepcopy(inputs);quotes={}
        for (e,v),q in zip(updated.items(),[[.4,.2,.4],[.1,.8,.1]]):
            mid=v['G']@np.array(q);v.update(pure_bid=mid-1e-4,pure_ask=mid+1e-4,pure_mid=mid)
            quotes[e]=(v['pure_bid'],v['pure_ask'])
        s.update_quotes(quotes);result=s.solve();reference=solve_convex_order_lp(updated)
        self.assertTrue(result['success']);self.assertTrue(reference['success'])
        self.assertGreater(result['actual_objective'],.01)
        self.assertLess(abs(result['actual_objective']-reference['actual_objective']),1e-6)

    def test_shared_matrix_costs_targets_and_midpoint_update(self):
        inputs=example_inputs();s=PersistentSanosLP(inputs)
        expected=build_convex_order_problem(inputs,weight_mode='paper',mid_penalty=1e-8)
        np.testing.assert_array_equal(s.problem['c'],expected['c'])
        self.assertEqual((s.problem['A']-expected['A']).nnz,0)
        self.assertEqual(s.objective_mode,'paper')
        upper,costs=quote_vectors(expected,s.quotes)
        np.testing.assert_array_equal(upper,np.r_[np.ones(expected['E'].shape[0]),expected['b']])
        np.testing.assert_array_equal(costs,expected['c'])
        ex=next(iter(inputs));quotes={key:(b.copy(),a.copy()) for key,(b,a) in s.quotes.items()}
        quotes[ex]=(quotes[ex][0]+1e-5,quotes[ex][1]+3e-5)
        s.update_quotes(quotes);upper,costs=quote_vectors(s.problem,s.quotes)
        plus,minus,n=s.problem['mid_rows'][ex]
        np.testing.assert_allclose(upper[plus:plus+n],sum(quotes[ex])*.5)
        np.testing.assert_allclose(upper[minus:minus+n],-sum(quotes[ex])*.5)
        w=1/(quotes[ex][1]-quotes[ex][0])
        np.testing.assert_allclose(costs[slice(*s.problem['z_offsets'][ex])],w)
        np.testing.assert_allclose(costs[slice(*s.problem['mid_offsets'][ex])],1e-8*w)

    def test_paper_warm_and_independent_cold_agree_on_positive_loss(self):
        inputs=example_inputs();warm=PersistentSanosLP(inputs);self.assertTrue(warm.solve()['success'])
        updated=copy.deepcopy(inputs);quotes={}
        for (ex,v),q in zip(updated.items(),[[.4,.2,.4],[.1,.8,.1]]):
            mid=v['G']@np.array(q);v['pure_bid']=mid-1e-4;v['pure_ask']=mid+1e-4;v['pure_mid']=mid
            quotes[ex]=(v['pure_bid'],v['pure_ask'])
        warm.update_quotes(quotes);result=warm.solve();cold=solve_convex_order_lp(updated)
        self.assertTrue(result['success'],result['message']);self.assertTrue(cold['success'],cold['message'])
        self.assertEqual(result['objective_mode'],'paper')
        self.assertGreater(result['actual_objective'],0.01)
        self.assertLessEqual(abs(result['actual_objective']-cold['actual_objective']),1e-6)
        self.assertAlmostEqual(result['actual_objective'],result['actual_outside_objective']+result['actual_mid_penalty'])

    def test_repaired_dual_bound_accounts_for_both_unbounded_error_variables(self):
        s=PersistentSanosLP(example_inputs());upper,costs=quote_vectors(s.problem,s.quotes)
        rng=np.random.default_rng(8)
        for _ in range(10):
            cert=global_lower_bound(s.problem,s.audit_cache['matrix'],upper,costs,rng.normal(size=len(upper)))
            self.assertLessEqual(cert['lower_bound'],1e-8)
            for kind in ['z_offsets','mid_offsets']:
                for offsets in s.problem[kind].values():
                    self.assertGreaterEqual(cert['reduced_cost'][slice(*offsets)].min(),0.)

    def test_surface_snapshot_requires_a_fit_of_current_quotes(self):
        s=PersistentSanosLP(example_inputs())
        with self.assertRaises(ValueError):s.surface()
        self.assertTrue(s.solve()['success']);snapshot=s.surface()
        original=snapshot.pure_call_days(45,1.)
        quotes={ex:(b.copy(),a.copy()) for ex,(b,a) in s.quotes.items()}
        ex=next(iter(quotes));quotes[ex]=(quotes[ex][0],quotes[ex][1]+1e-5)
        s.update_quotes(quotes)
        with self.assertRaises(ValueError):s.surface()
        self.assertTrue(s.solve()['success']);self.assertGreater(s.surface().pure_call_days(45,1.),0)
        self.assertEqual(snapshot.pure_call_days(45,1.),original)


if __name__=='__main__':unittest.main()
