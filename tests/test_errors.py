"""Tests that invalid input raises proper exceptions, also under python -O.

The library used to validate input with assert statements, which python -O
strips, so invalid input could silently give wrong results.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from abeliantensors import AbelianTensor, Tensor, TensorZ2


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
        from abeliantensors import AbelianTensor, Tensor, TensorZ2
        T = TensorZ2.random([[1, 1], [1, 1]], dirs=[1, -1])
        T.sects[(0, 0)] = np.zeros((5, 5))
        T.check_consistency()
        """
    )
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_init_requires_qhape():
    with pytest.raises(TypeError, match="qhape"):
        AbelianTensor([[1, 1]], dirs=[1])


def test_init_rejects_mismatched_form():
    with pytest.raises(ValueError, match="dirs"):
        TensorZ2([[1, 1]], dirs=[1, -1])
    with pytest.raises(ValueError, match="qhape"):
        TensorZ2([[1, 1]], qhape=[[0]], dirs=[1])
    with pytest.raises(ValueError, match="defval"):
        TensorZ2([[1, 1]], dirs=[1], charge=1, defval=1.0)


def test_factories_reject_mismatched_form():
    with pytest.raises(ValueError, match="dirs"):
        TensorZ2.random([[1, 1]], dirs=[1, -1])
    with pytest.raises(ValueError, match="qhape"):
        TensorZ2.random([[1, 1]], qhape=[[0]], dirs=[1])
    with pytest.raises(ValueError, match="qim"):
        TensorZ2.eye([1, 1], qim=[0])


def test_init_validation_works_under_python_O():
    result = run_optimized(
        """
        from abeliantensors import TensorZ2
        TensorZ2([[1, 1]], dirs=[1, -1])
        """
    )
    assert "ValueError" in result.stderr


def z2(shape, dirs, **kwargs):
    return TensorZ2.random(shape, dirs=dirs, **kwargs)


def test_binary_ops_reject_mismatched_forms():
    A = z2([[2, 1], [1, 2]], [1, -1])
    B = z2([[1, 2], [1, 2]], [1, -1])
    with pytest.raises(ValueError, match="form"):
        _ = A + B
    with pytest.raises(ValueError, match="form"):
        _ = A.allclose(B)


def test_dot_rejects_incompatible_indices():
    A = z2([[2, 1], [1, 2]], [1, -1])
    B = z2([[1, 2], [1, 2]], [1, -1])
    with pytest.raises(ValueError, match="compatible"):
        _ = A.dot(B, (0, 0))
    with pytest.raises(ValueError, match="same number"):
        _ = A.dot(B, ([0, 1], [0]))
    with pytest.raises(ValueError, match="compatible"):
        _ = A.matrix_dot(A)


def test_trace_rejects_incompatible_indices():
    with pytest.raises(ValueError, match="compatible"):
        _ = z2([[2, 1], [1, 2]], [1, -1]).trace()
    with pytest.raises(ValueError, match="compatible"):
        _ = Tensor.random([2, 3]).trace()


def test_join_and_split_reject_mismatched_arguments():
    T = z2([[1, 1], [1, 1], [1, 1]], [1, 1, 1])
    with pytest.raises(ValueError, match="dirs"):
        _ = T.join_indices([0, 1], [2], dirs=[1])
    with pytest.raises(ValueError, match="same length"):
        _ = T.split_indices([0, 1], [[[1], [1]]], qims=[[[0], [0]]])
    with pytest.raises(ValueError, match="same length"):
        _ = Tensor.random([2, 3]).split_indices([0, 1], [[1, 2]])


def test_multiply_diag_rejects_bad_arguments():
    T = z2([[1, 1], [1, 1]], [1, -1])
    D = z2([[1, 1]], [-1], invar=False)
    with pytest.raises(ValueError, match="direction"):
        _ = T.multiply_diag(D, 1, direction="x")
    with pytest.raises(ValueError, match="vector"):
        _ = T.multiply_diag(T, 1)
    with pytest.raises(ValueError, match="charge"):
        _ = T.multiply_diag(z2([[1, 1]], [-1], charge=1), 1)


def test_diag_rejects_bad_input():
    with pytest.raises(ValueError, match="1 or 2 indices"):
        _ = z2([[1, 1]] * 3, [1, -1, 1]).diag()
    with pytest.raises(ValueError, match="compatible"):
        _ = z2([[2, 1], [1, 2]], [1, -1]).diag()


def test_trace_validation_works_under_python_O():
    result = run_optimized(
        """
        from abeliantensors import TensorZ2
        TensorZ2.random([[2, 1], [1, 2]], dirs=[1, -1]).trace()
        """
    )
    assert "ValueError" in result.stderr


def test_decompositions_require_invariant_tensors():
    N = z2([[1, 2], [2, 1]], [1, -1], invar=False)
    with pytest.raises(ValueError, match="invariant"):
        _ = N.svd([0], [1])
    with pytest.raises(ValueError, match="invariant"):
        _ = N.eig([0], [1])
    shifted = z2([[1, 2], [2, 1]], [1, -1]) + 1.0
    with pytest.raises(ValueError, match="defval"):
        _ = shifted.svd([0], [1])


def test_eig_rejects_bad_matrices():
    with pytest.raises(ValueError, match="charge"):
        _ = z2([[1, 2], [1, 2]], [1, -1], charge=1).eig([0], [1])
    # eig() orients the dirs itself, so only matrix_eig sees bad dirs.
    with pytest.raises(ValueError, match="dirs"):
        _ = z2([[1, 2], [1, 2]], [1, 1]).matrix_eig()
    with pytest.raises(ValueError, match="square"):
        _ = z2([[1, 2], [2, 2]], [1, -1]).eig([0], [1])


def test_matrix_dot_and_diag_require_invariant_matrices():
    M = z2([[1, 1], [1, 1]], [1, -1])
    N = z2([[1, 1], [1, 1]], [1, -1], invar=False)
    v = z2([[1, 1]], [1], invar=False)
    with pytest.raises(ValueError, match="invariant"):
        _ = M.matrix_dot(N)
    with pytest.raises(ValueError, match="invariant"):
        _ = N.matrix_dot(M)
    with pytest.raises(ValueError, match="invariant"):
        _ = v.matrix_dot(N)
    with pytest.raises(ValueError, match="invariant"):
        _ = N.matrix_dot(v.flip_dir(0))
    with pytest.raises(ValueError, match="invariant"):
        _ = N.diag()


def test_svd_validation_works_under_python_O():
    result = run_optimized(
        """
        from abeliantensors import TensorZ2
        N = TensorZ2.random([[1, 2], [2, 1]], dirs=[1, -1], invar=False)
        N.svd([0], [1])
        """
    )
    assert "ValueError" in result.stderr
