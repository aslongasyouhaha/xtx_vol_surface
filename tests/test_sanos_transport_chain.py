import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy import sparse
from sanos_core import call_kernel
from sanos_martingale_transport import calls
from xtx_transport_research import fit_chain


class ChainTests(unittest.TestCase):
    def test_saved_multistep_chain_replays_from_initial_state(self):
        items={};k=np.linspace(.8,1.2,21)
        for j in range(2):
            c=call_kernel([1],k,.01*(j+1))[:,0]
            items[j]=dict(dte=j+1,market_k=k,pure_bid=c-.001,pure_ask=c+.001,pure_mid=c)
        with TemporaryDirectory() as directory:
            out=Path(directory);result=fit_chain(items,out)
            self.assertTrue(result['validated'])
            self.assertEqual(result['coverage'],1.)
            previous=None
            for j in range(2):
                z=np.load(out/f'transition_{j}.npz');P=sparse.load_npz(out/f'projection_{j}.npz')
                if previous is not None:np.testing.assert_allclose(z['original_weights'],previous,atol=1e-14)
                actual=np.asarray(z['original_weights']@P).ravel()@z['Q']
                np.testing.assert_allclose(actual,z['target'],atol=1e-14)
                np.testing.assert_allclose(calls(z['grid'],actual,z['market_k']),z['fit'],atol=1e-14)
                previous=actual
                z.close()


if __name__=='__main__':unittest.main()
