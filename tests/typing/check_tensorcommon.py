"""Type-checking assertions for methods defined in TensorCommon.

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import assert_type

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
