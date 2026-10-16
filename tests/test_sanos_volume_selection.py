"""Exercise production quote selection without external market data."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sanos_core import call_kernel
from unittest.mock import patch
from xtx_data import select_calibration_quotes,build_joint_inputs


class VolumeSelectionTests(unittest.TestCase):
    def build(self,minimum):
        k=np.linspace(.8,1.2,25);expiry=pd.Timestamp('2025-02-01')
        chain=pd.DataFrame(dict(date=pd.Timestamp('2025-01-02'),exdate=expiry,dte=30,T=30/365,
            symbol=[f'Q{i}' for i in range(25)],cp_flag=np.where(k<1,'P','C'),strike=k*100,k=k,
            best_bid=.5,best_offer=.6,mid=.55,pure_bid=.005,pure_mid=.0055,pure_ask=.006,
            impl_volatility=.2,forward=100.,df=1.,volume=np.r_[np.full(20,100.),0.,99.,np.nan,1.,0.],
            open_interest=10,bid_ask_spread=.1))
        atm=pd.DataFrame(dict(exdate=[expiry],atm_k=[1.],atm_iv=[.2],atm_total_variance_raw=[.01],
            atm_total_variance=[.01],atm_total_variance_adjustment=[0.],variance_monotone_violation=[False]))
        env=dict(np=np,pd=pd,option_chain=chain,atm_rows=atm,FIT_MAX_DTE=120,FIT_MIN_DAILY_VOLUME=minimum,
            implied_vol_from_pure_option=lambda *args:.2,bs_vega_forward_unit=lambda k,iv,T:.1*np.sqrt(T),
            ETA=.25,MIN_QUOTES_PER_EXPIRY=8,
            build_kernel_matrix=lambda anchor_k,market_k,total_variance:call_kernel(anchor_k,market_k,total_variance))
        with patch('xtx_data.implied_vol_from_pure_option',env['implied_vol_from_pure_option']),patch('xtx_data.bs_vega_forward_unit',env['bs_vega_forward_unit']):
            quotes,anchors=select_calibration_quotes(chain,atm,minimum)
        env['sanos_quotes']=quotes
        env['joint_inputs']=build_joint_inputs(quotes,anchors)
        return env

    def test_volume_filters_observations_without_removing_latent_states(self):
        filtered=self.build(100);control=self.build(None)
        quotes=filtered['sanos_quotes']
        self.assertEqual(len(quotes),20)
        self.assertTrue(quotes.volume.ge(100).all())
        f=next(iter(filtered['joint_inputs'].values()));c=next(iter(control['joint_inputs'].values()))
        np.testing.assert_array_equal(f['anchor_k'],c['anchor_k'])
        self.assertEqual(len(f['market_k']),20);self.assertEqual(len(c['market_k']),25)
        self.assertGreater(len(f['anchor_k']),len(f['market_k']))


if __name__=='__main__':unittest.main()
