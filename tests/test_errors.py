"""Tests that invalid input raises proper exceptions, also under python -O.

The library used to validate input with assert statements, which python -O
strips, so invalid input could silently give wrong results.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from abeliantensors import TensorZ2


def run_optimized(code):
    """Run `code` in a fresh `python -O` and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-O", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def broken_tensor():
    """Return a TensorZ2 whose (0, 0) block has the wrong shape."""
    T = TensorZ2.random([[1, 1], [1, 1]], dirs=[1, -1])
    T.sects[(0, 0)] = np.zeros((5, 5))
    return T


def test_check_consistency_reports_what_is_wrong():
    with pytest.raises(AssertionError, match="shape"):
        broken_tensor().check_consistency()


def test_check_consistency_works_under_python_O():
    result = run_optimized(
        """
        import numpy as np
        from abeliantensors import TensorZ2
        T = TensorZ2.random([[1, 1], [1, 1]], dirs=[1, -1])
        T.sects[(0, 0)] = np.zeros((5, 5))
        T.check_consistency()
        """
    )
    assert result.returncode != 0
    assert "AssertionError" in result.stderr
