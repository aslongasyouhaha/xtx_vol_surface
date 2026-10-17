"""Public fixtures exercise successful fits and a deliberately inconsistent market."""
import pytest
from sanos_surface import SanosSurface
from xtx_research import run_testcases
import xtx_io


@pytest.mark.parametrize('case',xtx_io.TESTCASES)
def test_public_fixture_needs_no_licensed_data(case,tmp_path,monkeypatch):
    def unavailable(*args,**kwargs):
        raise AssertionError('Public testcases must not read licensed market data')
    monkeypatch.setattr(xtx_io,'build_baseline_inputs',unavailable)
    report=run_testcases([case],tmp_path)[0]
    assert report['testcase_passed'] and report['surface_audit']['passed']
    surface=SanosSurface.from_directory(tmp_path/case)
    prices=surface.pure_call_days([30,45,60,75,90],1.)
    assert (prices[1:]>=prices[:-1]-1e-9).all()
    if case=='conflicting_quotes':
        # Identical contracts with disjoint equally wide bands force this loss.
        assert report['outside_loss']>=36.5-1e-6
        assert report['coverage']<1 and not report['fit_target_met']
    else:
        assert report['coverage']==1 and report['outside_loss']<1e-6
