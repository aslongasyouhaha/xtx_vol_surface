"""Public command-line workflows run without any licensed market files."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def research(*arguments):
    return subprocess.run(
        [sys.executable, 'scripts/research_xtx.py', *map(str, arguments)],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=30,
    )


def test_public_case_can_be_calibrated_then_plotted(tmp_path):
    output = tmp_path / 'models'
    research('testcases', '--cases', 'clean', '--output', output)
    report = json.loads((output / 'clean/summary.json').read_text(encoding='utf-8'))
    assert report['testcase_passed'] and report['fit_target_met']
    html = tmp_path / 'figures/iv.html'
    research('plot', '--model-dir', output / 'clean', '--output', html)
    content = html.read_text(encoding='utf-8')
    assert 'Plotly.newPlot' in content and 'market OTM IV' in content


def test_dlv_recovery_command_is_self_contained(tmp_path):
    research('dlv-recovery', '--synthetic-only', '--output', tmp_path)
    report = json.loads((tmp_path / 'synthetic_results.json').read_text(encoding='utf-8'))
    assert report['surface_audit']['passed']
    assert report['max_mass_error'] < 1e-10
    assert report['max_mean_error'] < 1e-10
    assert all(row['maximum_probability_error'] < 1e-10 for row in report['roundtrips'])
