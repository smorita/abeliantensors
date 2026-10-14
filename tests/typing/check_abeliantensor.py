"""Type-checking assertions for AbelianTensor.

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import Any, assert_type

import numpy as np

from abeliantensors import AbelianGroup, AbelianTensor, CyclicGroup, TensorZ2
from abeliantensors._typing import Dim, Qim, Qnum

T = TensorZ2.random([[1, 1], [1, 1]], dirs=[1, -1])

# Attributes
assert_type(T.shape, list[Dim])
assert_type(T.qhape, list[Qim])
assert_type(T.dirs, list[int])
assert_type(T.charge, Qnum)
assert_type(T.sects, dict[tuple[Qnum, ...], np.ndarray])
assert_type(T.symmetry, AbelianGroup[Any])
assert_type(T.invar, bool)

# Operators return the class of the operand.
assert_type(T + T, TensorZ2)
assert_type(T + 1.0, TensorZ2)
assert_type(2.0 * T, TensorZ2)
assert_type(T * T, TensorZ2)
assert_type(T / 2, TensorZ2)
assert_type(T**2, TensorZ2)
assert_type(-T, TensorZ2)
assert_type(abs(T), TensorZ2)
assert_type(T == T, TensorZ2)
assert_type(T != 0, TensorZ2)
assert_type(T < T, TensorZ2)
assert_type(T >= 0, TensorZ2)
assert_type(T.sqrt(), TensorZ2)
assert_type(T.exp(), TensorZ2)
assert_type(T.abs(), TensorZ2)

# Creation
assert_type(
    AbelianTensor.random(
        [[1, 1]], qhape=[[0, 1]], dirs=[1], symmetry=CyclicGroup(2)
    ),
    AbelianTensor,
)
assert_type(AbelianTensor([[1]], qhape=[[0]], dirs=[1]), AbelianTensor)
assert_type(TensorZ2.eye([1, 1]), TensorZ2)
assert_type(
    TensorZ2.from_ndarray(np.eye(2), shape=[[1, 1], [1, 1]], dirs=[1, -1]),
    TensorZ2,
)
assert_type(T.to_ndarray(), np.ndarray)
assert_type(T.empty_like(), TensorZ2)
assert_type(T.view(), TensorZ2)
assert_type(T.copy(), TensorZ2)

# Elements and values
assert_type(T[(0, 0)], np.ndarray)
assert_type(T.value(), Any)
assert_type(T.sum(), Any)
assert_type(T.max(), Any)
assert_type(T.any(), bool | np.bool_)
assert_type(T.all(), bool | np.bool_)
assert_type(T.allclose(T), bool)
assert_type(T.isscalar(), bool)
assert_type(T.is_full(), bool)
assert_type(len(T), int)
assert_type(bool(T.trace()), bool)

# Element-wise and form operations
assert_type(T.conj(), TensorZ2)
assert_type(T.conjugate(), TensorZ2)
assert_type(T.real(), TensorZ2)
assert_type(T.imag(), TensorZ2)
assert_type(T.astype(np.complex128), TensorZ2)
assert_type(T.diag(), TensorZ2)
assert_type(T.flip_dir(0), TensorZ2)
assert_type(T.expand_dims(0), TensorZ2)
assert_type(T.is_valid_key((0, 0)), bool)
assert_type(T.compatible_indices(T, 0, 1), bool)
assert_type(T.check_consistency(), bool)
assert_type(TensorZ2.check_form_match(tensor1=T, tensor2=T), bool)
assert_type(repr(T), str)
