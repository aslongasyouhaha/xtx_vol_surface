import unittest

import numpy as np
import pandas as pd

from sanos_core import add_common_boundary_anchors, call_kernel, solve_convex_order_lp, solve_convex_order_qp


def example_inputs():
    anchors = np.array([0.8, 1.0, 1.2])
    strikes = np.array([0.85, 0.95, 1.0, 1.05, 1.15])
    items = {}
    for dte, variance, q in [(30, .01, [0, 1, 0]), (60, .02, [.25, .5, .25])]:
        G = call_kernel(anchors, strikes, variance)
        mid = G @ np.array(q)
        items[pd.Timestamp('2025-01-01') + pd.Timedelta(days=dte)] = {
            'dte': dte, 'smooth_variance': variance, 'market_k': strikes,
            'anchor_k': anchors, 'G': G, 'pure_bid': mid-1e-4,
            'pure_mid': mid, 'pure_ask': mid+1e-4,
        }
    return items


class ConvexOrderTests(unittest.TestCase):
    def test_qp_reference_preserves_constraints_and_tracks_known_surface(self):
        items = example_inputs()
        refs = {ex: (v['pure_mid'], np.ones(len(v['market_k']), bool)) for ex,v in items.items()}
        result = solve_convex_order_qp(items, refs, stability_strength=.1)
        self.assertTrue(result['success'], result['message'])
        self.assertLess(result['normalized_outside_loss'], 1e-6)
        self.assertLess(result['reference_change_mse'], 1e-3)
        self.assertLess(result['residuals']['max_eq_residual'], 1e-7)
        self.assertGreaterEqual(result['convex_order_checks'].min_convex_order_diff.min(), -1e-7)

    def test_qp_negative_strength_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            solve_convex_order_qp(example_inputs(), {}, stability_strength=-1)

    def test_stronger_temporal_penalty_trades_quote_fit_for_stability(self):
        items = example_inputs()
        refs = {ex: (v['G'] @ np.array([.45,.1,.45]),
                     np.ones(len(v['market_k']),bool)) for ex,v in items.items()}
        weak = solve_convex_order_qp(items, refs, stability_strength=.0001)
        strong = solve_convex_order_qp(items, refs, stability_strength=1.)
        self.assertTrue(weak['success'],weak['message'])
        self.assertTrue(strong['success'],strong['message'])
        self.assertLess(strong['reference_change_mse'],weak['reference_change_mse'])
        self.assertGreater(strong['normalized_outside_loss'],weak['normalized_outside_loss'])

    def test_known_convex_order_surface_fits_and_validates(self):
        result = solve_convex_order_lp(example_inputs())
        self.assertTrue(result['success'], result['message'])
        self.assertLess(abs(result['objective']), 1e-6)
        self.assertEqual(result['n_vars'], 26)
        self.assertLess(result['residuals']['max_eq_residual'], 1e-7)
        self.assertGreaterEqual(result['calendar_checks'].min_calendar_diff.min(), -1e-7)
        self.assertGreaterEqual(result['convex_order_checks'].min_convex_order_diff.min(), -1e-7)

    def test_boundary_preparation_is_idempotent_and_does_not_mutate(self):
        original = example_inputs()
        original[max(original)]['anchor_k'] = np.array([.75, 1.0, 1.25])
        prepared = add_common_boundary_anchors(original)
        twice = add_common_boundary_anchors(prepared)
        for ex in original:
            self.assertEqual(len(original[ex]['anchor_k']), 3)
            np.testing.assert_array_equal(prepared[ex]['anchor_k'], twice[ex]['anchor_k'])

    def test_decreasing_smoothing_variance_rejected(self):
        items = example_inputs()
        items[max(items)]['smooth_variance'] = .005
        with self.assertRaisesRegex(ValueError, 'nondecreasing'):
            solve_convex_order_lp(items)

    def test_expiry_outside_business_scope_rejected(self):
        items = example_inputs()
        items[max(items)]['dte'] = 121
        with self.assertRaisesRegex(ValueError, 'outside'):
            solve_convex_order_lp(items)


if __name__ == '__main__':
    unittest.main()
