"""Type-checking assertions for Tensor (a np.ndarray subclass).

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from abeliantensors import Tensor

D = Tensor.random([2, 2])

# Operators keep the Tensor type (declared under TYPE_CHECKING).
assert_type(D + D, Tensor)
assert_type(D + 1.0, Tensor)
assert_type(2.0 * D, Tensor)
assert_type(D - D, Tensor)
assert_type(1.0 - D, Tensor)
assert_type(D * D, Tensor)
assert_type(D / 2, Tensor)
assert_type(D // 2, Tensor)
assert_type(D % 2, Tensor)
assert_type(D**2, Tensor)
assert_type(-D, Tensor)
assert_type(+D, Tensor)
assert_type(abs(D), Tensor)
assert_type(D == D, Tensor)
assert_type(D != 0, Tensor)
assert_type(D < D, Tensor)
assert_type(D >= 0, Tensor)
assert_type((D + D).svd(0, 1), tuple[Tensor, Tensor, Tensor])

E = Tensor.random([2, 2])
E += 1
assert_type(E, Tensor)
E *= 2.0
assert_type(E.svd(0, 1), tuple[Tensor, Tensor, Tensor])

# Methods inherited from ndarray, declared to return Tensor.
assert_type(D.conj(), Tensor)
assert_type(D.transpose(), Tensor)
assert_type(D.swapaxes(0, 1), Tensor)
assert_type(D.copy(), Tensor)
assert_type(D.astype(np.complex128), Tensor)

# divmod returns a tuple of arrays at runtime, so it is not declared and
# keeps numpy's typing (a pair of ndarrays, not a single Tensor).
quotient, remainder = divmod(D, 2)
assert_type(quotient, npt.NDArray[Any])
