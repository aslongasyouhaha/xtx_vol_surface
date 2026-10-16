import unittest
import numpy as np
import pandas as pd
from test_sanos_daily import daily_items
from xtx_research import market_calendar,eta_control


class ResearchTests(unittest.TestCase):
    def bands(self,second_bid,second_ask):
        return pd.DataFrame(dict(exdate=['2025-01-01']*2+['2025-02-01']*2,
            k=[.9,1.1,.9,1.1],pure_bid=[.10,.05,*second_bid],
            pure_ask=[.12,.07,*second_ask],pure_mid=[.11,.06,*((np.array(second_bid)+second_ask)/2)]))

    def test_overlapping_bands_are_not_a_calendar_conflict(self):
        rows,_=market_calendar(self.bands([.09,.04],[.13,.08]))
        self.assertEqual(rows[0]['interpolated_band_conflict_points'],0)

    def test_later_ask_below_earlier_bid_reports_band_conflict(self):
        rows,_=market_calendar(self.bands([.07,.02],[.09,.04]))
        self.assertGreater(rows[0]['interpolated_band_conflict_points'],0)

    def test_single_expiry_and_nonoverlapping_ranges(self):
        frame=self.bands([.1,.05],[.13,.08])
        self.assertEqual(market_calendar(frame.iloc[:2]),([],[]))
        frame.loc[2:,'k']=[1.2,1.3]
        rows,curves=market_calendar(frame)
        self.assertEqual(rows[0]['status'],'no_common_strike_range')
        self.assertEqual(curves,[])

    def test_eta_control_uses_all_expiries_without_mutating_geometry(self):
        inputs=daily_items()
        for v in inputs.values():v['quotes']['atm_total_variance']=v['smooth_variance']/.25
        before={e:(v['anchor_k'].copy(),v['market_k'].copy(),v['smooth_variance']) for e,v in inputs.items()}
        row=eta_control(inputs,.25,time_limit=5)
        self.assertTrue(row['success'],row)
        self.assertEqual(row['expiries'],len(inputs))
        self.assertEqual(row['quotes'],sum(len(v['market_k']) for v in inputs.values()))
        for e,v in inputs.items():
            np.testing.assert_array_equal(v['anchor_k'],before[e][0])
            np.testing.assert_array_equal(v['market_k'],before[e][1])
            self.assertEqual(v['smooth_variance'],before[e][2])

if __name__=='__main__':unittest.main()
