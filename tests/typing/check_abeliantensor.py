"""Type-checking assertions for AbelianTensor.

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import Any, assert_type

import numpy as np

from abeliantensors import AbelianGroup, TensorZ2
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
