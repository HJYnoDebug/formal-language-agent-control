"""The harness's own scripted checks (agent/harness/verify_*.py), run with the test suite so that they cannot go stale
unnoticed (2026-10-18: verify_llm_client still expected the old 429 window)."""
import os
import subprocess
import sys

import pytest

HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "harness")


@pytest.mark.parametrize("script", ["verify_llm_client.py", "verify_run_manifest.py"])
def test_harness_verify_script(script):
    out = subprocess.run([sys.executable, script], cwd=HARNESS, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and "All checks passed." in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]
