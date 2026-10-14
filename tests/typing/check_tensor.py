"""Type-checking assertions for Tensor (a np.ndarray subclass).

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

from abeliantensors import Tensor, TensorZ2

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

# Creation
assert_type(Tensor([2, 2]), Tensor)
assert_type(Tensor.zeros([2, 3]), Tensor)
assert_type(Tensor.eye(3), Tensor)
assert_type(Tensor.from_ndarray(np.eye(2)), Tensor)
assert_type(Tensor.from_ndarray([[1.0, 0.0], [0.0, 1.0]]), Tensor)
assert_type(Tensor.from_ndarray(2.5), Tensor)
assert_type(D.to_ndarray(), np.ndarray)

# Methods defined in Tensor
assert_type(D.diag(), Tensor)
assert_type(D.conjugate(), Tensor)
assert_type(D.sqrt(), Tensor)
assert_type(D.exp(), Tensor)
assert_type(D.abs(), Tensor)
assert_type(D.real(), Tensor)
assert_type(D.flip_dir(0), Tensor)
assert_type(D.expand_dims(0), Tensor)
assert_type(D.trace(), Tensor)
assert_type(D.value(), Any)
assert_type(D.sum(), Any)
assert_type(D.isscalar(), bool)
assert_type(D.allclose(D), bool)
assert_type(D.compatible_indices(D, 0, 1), bool)
assert_type(Tensor.check_form_match(tensor1=D, tensor2=D), bool)
assert_type(D.join_indices([0, 1]), Tensor)
assert_type(
    D.join_indices([0, 1], return_transposed_shape_data=True),
    tuple[Tensor, tuple[int, ...], None, None],
)
assert_type(D.join_indices([0, 1]).split_indices(0, [2, 2]), Tensor)
assert_type(D.multiply_diag(D.diag(), 1), Tensor)
assert_type(D.dot(D, (1, 0)), Tensor)
assert_type(D.matrix_dot(D), Tensor)
assert_type(D.matrix_svd(), tuple[Tensor, Tensor, Tensor, float])
assert_type(D.matrix_eig(), tuple[Tensor, Tensor, float])


def swap_classes(T: Tensor | TensorZ2) -> None:
    """The same code type-checks for Tensor and TensorZ2."""
    U, S, V = T.svd(0, 1)
    R = U.dot(S.diag(), (1, 0)).dot(V, (1, 0))
    assert_type(R, Tensor | TensorZ2)
    assert_type((R - T).norm(), float)


# Matrix multiplication keeps the Tensor type too.
assert_type(D @ D, Tensor)
assert_type((D @ D).svd(0, 1), tuple[Tensor, Tensor, Tensor])
F = Tensor.random([2, 2])
F @= D
assert_type(F, Tensor)
