"""Type-checking assertions for methods defined in TensorCommon.

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import Any, assert_type

from abeliantensors import Tensor, TensorU1, TensorZ2, TensorZ2Z3

T = TensorZ2.random([[1, 1], [1, 1]], dirs=[1, -1])
assert_type(T, TensorZ2)
assert_type(TensorZ2.zeros([[1, 1]], dirs=[1]), TensorZ2)
assert_type(TensorZ2.ones([[1, 1]], dirs=[1]), TensorZ2)
assert_type(TensorZ2.empty([[1, 1]], dirs=[1]), TensorZ2)
assert_type(TensorU1.random([[1, 1]], qhape=[[0, 1]], dirs=[1]), TensorU1)
assert_type(TensorZ2Z3.random([[1] * 6], dirs=[1]), TensorZ2Z3)
assert_type(T.dot(T, (1, 0)), TensorZ2)
assert_type(T.dot(T, ([1], [0])), TensorZ2)
assert_type(T.norm(), float)
assert_type(T.norm_sq(), float)
assert_type(T.form_str(), str)

D = Tensor.random([2, 2])
assert_type(D, Tensor)
assert_type(D.norm(), float)

# Return shapes that depend on flags (@overload in TensorCommon).
assert_type(T.to_matrix([0], [1]), TensorZ2)
assert_type(
    T.to_matrix([0], [1], return_transposed_shape_data=True),
    tuple[TensorZ2, Any, Any, Any],
)
assert_type(T.svd(0, 1), tuple[TensorZ2, TensorZ2, TensorZ2])
assert_type(
    T.svd(0, 1, return_rel_err=True),
    tuple[TensorZ2, TensorZ2, TensorZ2, float],
)
assert_type(T.eig(0, 1), tuple[TensorZ2, TensorZ2])
assert_type(T.eig(0, 1, return_rel_err=True), tuple[TensorZ2, TensorZ2, float])
assert_type(T.split(0, 1), tuple[TensorZ2, TensorZ2])
assert_type(
    T.split(0, 1, return_sings=True), tuple[TensorZ2, TensorZ2, TensorZ2]
)
assert_type(
    T.split(0, 1, return_rel_err=True), tuple[TensorZ2, TensorZ2, float]
)
assert_type(
    T.split(0, 1, return_rel_err=True, return_sings=True),
    tuple[TensorZ2, TensorZ2, TensorZ2, float],
)
assert_type(D.svd(0, 1), tuple[Tensor, Tensor, Tensor])
assert_type(D.split([0], [1]), tuple[Tensor, Tensor])


def with_flags(flag: bool) -> None:
    """Flags given as bool variables give the union of the return shapes."""
    assert_type(
        T.svd(0, 1, return_rel_err=flag),
        tuple[TensorZ2, TensorZ2, TensorZ2]
        | tuple[TensorZ2, TensorZ2, TensorZ2, float],
    )
    assert_type(
        T.split(0, 1, return_rel_err=flag, return_sings=flag),
        tuple[TensorZ2, TensorZ2]
        | tuple[TensorZ2, TensorZ2, TensorZ2]
        | tuple[TensorZ2, TensorZ2, float]
        | tuple[TensorZ2, TensorZ2, TensorZ2, float],
    )
