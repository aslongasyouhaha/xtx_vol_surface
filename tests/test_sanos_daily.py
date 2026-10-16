import copy
import unittest
import numpy as np
import pandas as pd
from test_sanos_incremental import highspy
from test_sanos_core import example_inputs
from sanos_core import build_convex_order_problem,call_kernel,solve_convex_order_lp
from sanos_daily import unit_call,implied_total_variance,snapshot,daily_references,stability_qp,seed_previous_marginals,active_support_qp


def daily_items():
    items=example_inputs()
    for e,v in items.items():
        k=v['market_k'];v['quotes']=pd.DataFrame(dict(k=k,T=v['dte']/365,
            source_otm_iv=.35,source_cp_flag=np.where(k<1,'P','C')))
    return items


class DailyTests(unittest.TestCase):
    def test_reference_inversion_and_time_roll(self):
        k=np.array([.8,1,1.2]);w=np.array([.01,.04,.09])
        np.testing.assert_allclose(implied_total_variance(k,unit_call(k,w)),w,atol=1e-12)
        items=daily_items();p=build_convex_order_problem(items);x=np.zeros(p['cursor'])
        for e,v in p['items'].items():
            a,b=p['q_offsets'][e];x[a:b]=[0,1,0]
        old=snapshot('2025-01-01',p,x)
        newitems=copy.deepcopy(items)
        for v in newitems.values():v['dte']-=1
        current=build_convex_order_problem(newitems);refs=daily_references(old,current,'2025-01-02')
        for e,(price,mask) in refs.items():
            before=p['items'][e];now=current['items'][e]
            expected=unit_call(now['market_k'][mask],before['smooth_variance']*now['dte']/before['dte'])
            np.testing.assert_allclose(price[mask],expected,atol=1e-12)
        with self.assertRaises(ValueError):daily_references(old,current,'2024-12-31')

    def test_restricted_qp_retains_full_constraints_and_loss_budget(self):
        items=daily_items();p=build_convex_order_problem(items);lp=solve_convex_order_lp(items)
        quotes={e:(v['pure_bid'],v['pure_ask']) for e,v in p['items'].items()}
        refs={e:(v['pure_mid']+.02,np.ones(len(v['market_k']),bool)) for e,v in p['items'].items()}
        r=active_support_qp(p,lp['x'],quotes,refs,loss_allowance=.01,time_limit=4)
        self.assertTrue(r['success'],r)
        self.assertTrue(r['audit']['validated'])
        self.assertEqual(len(r['x']),p['cursor'])
        self.assertLessEqual(r['allowed_state_count'],r['full_state_count'])
        self.assertLessEqual(r['audit']['actual_outside_objective'],r['lp_outside_loss']+.010001)
        self.assertLessEqual(r['qp_movement_rms'],r['lp_movement_rms']+1e-8)
        self.assertGreaterEqual(r['feasible_step_alpha'],0)
        self.assertLessEqual(r['feasible_step_alpha'],1)

    def test_qp_preserves_daily_fit_when_prior_prices_are_higher(self):
        items=daily_items();p=build_convex_order_problem(items);lp=solve_convex_order_lp(items)
        quotes={e:(v['pure_bid'],v['pure_ask']) for e,v in p['items'].items()}
        refs={e:(v['pure_mid']+.01,np.ones(len(v['market_k']),bool)) for e,v in p['items'].items()}
        r=stability_qp(p,lp['x'],quotes,refs,time_limit=4)
        self.assertTrue(r['success'],r)
        self.assertTrue(r['audit']['validated'])
        self.assertLessEqual(r['qp_movement_rms'],r['lp_movement_rms']+1e-8)
        for e,v in p['items'].items():
            fit=v['G']@r['x'][slice(*p['q_offsets'][e])]
            self.assertTrue(np.all(fit<refs[e][0]))
        no=stability_qp(p,lp['x'],quotes,{},time_limit=1)
        self.assertFalse(no['success'])


@unittest.skipIf(highspy is None,'Requires native HiGHS')
class DailySeedTests(unittest.TestCase):
    def test_prior_primal_guess_does_not_reuse_incompatible_basis(self):
        from sanos_incremental import PersistentSanosLP
        items=daily_items();s=PersistentSanosLP(items);initial=s.solve()
        previous=snapshot('2025-01-01',s.problem,initial['x'])
        current=copy.deepcopy(items)
        for v in current.values():v['dte']-=1
        nextday=PersistentSanosLP(current);seed=seed_previous_marginals(nextday,previous,'2025-01-02')
        self.assertGreater(seed['expiries_seeded'],0);self.assertFalse(seed['old_basis_reused'])
        self.assertTrue(nextday.solve()['success'])
        with self.assertRaises(ValueError):seed_previous_marginals(nextday,previous,'2024-12-31')


if __name__=='__main__':unittest.main()
