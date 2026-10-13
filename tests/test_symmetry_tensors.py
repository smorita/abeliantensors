"""Tests for how AbelianTensor and its subclasses use symmetry groups."""

import numpy as np
import pytest

from abeliantensors import (
    AbelianTensor,
    CyclicGroup,
    ProductGroup,
    TensorU1,
    TensorZ2,
    TensorZ2Z2,
    TensorZ2Z3,
    TensorZ3,
    TensorZN,
    U1Group,
)


def test_symmetry_attribute_of_existing_classes():
    assert TensorZ2.random([[1, 1]], dirs=[1]).symmetry == CyclicGroup(2)
    assert TensorZ3.random([[1, 1, 1]], dirs=[1]).symmetry == CyclicGroup(3)
    T = TensorU1.random([[1, 1]], qhape=[[0, 5]], dirs=[1])
    assert T.symmetry == U1Group()


def test_abelian_tensor_defaults_to_u1():
    T = AbelianTensor.random([[1, 1]], qhape=[[0, 5]], dirs=[1])
    assert T.symmetry == U1Group()
    assert T.charge == 0


def test_qodulus_argument_is_deprecated_and_converted():
    with pytest.warns(DeprecationWarning, match="qodulus"):
        T = AbelianTensor.random([[2, 1]], qhape=[[0, 1]], dirs=[1], qodulus=2)
    assert T.symmetry == CyclicGroup(2)
    with pytest.warns(DeprecationWarning, match="qodulus"):
        T = AbelianTensor([[2, 1]], qhape=[[0, 1]], dirs=[1], qodulus=None)
    assert T.symmetry == U1Group()


def test_qodulus_attribute_is_deprecated():
    T = TensorZ3.random([[1, 1, 1]], dirs=[1])
    with pytest.warns(DeprecationWarning, match="qodulus"):
        assert T.qodulus == 3
    U = TensorU1.random([[1, 1]], qhape=[[0, 5]], dirs=[1])
    with pytest.warns(DeprecationWarning, match="qodulus"):
        assert U.qodulus is None


def test_qodulus_and_symmetry_together_is_type_error():
    with pytest.raises(TypeError):
        AbelianTensor(
            [[1]], qhape=[[0]], dirs=[1], qodulus=2, symmetry=CyclicGroup(2)
        )


def test_fixed_symmetry_class_rejects_other_symmetry():
    with pytest.raises(ValueError, match="symmetry"):
        TensorZ2.random([[1, 1]], dirs=[1], symmetry=CyclicGroup(3))


def test_mixing_symmetries_is_rejected():
    A = AbelianTensor.random(
        [[1, 1]], qhape=[[0, 1]], dirs=[1], symmetry=CyclicGroup(2)
    )
    B = AbelianTensor.random(
        [[1, 1]], qhape=[[0, 1]], dirs=[1], symmetry=CyclicGroup(3)
    )
    with pytest.raises(AssertionError):
        _ = A + B
    with pytest.raises(AssertionError):
        _ = A.allclose(B)


def test_legacy_subclass_with_qodulus_class_attribute():
    with pytest.warns(DeprecationWarning, match="qodulus"):

        class TensorZ5(TensorZN):
            qodulus = 5

    T = TensorZ5.random([[1] * 5, [1] * 5], dirs=[1, -1])
    assert T.symmetry == CyclicGroup(5)
    assert T.qhape == [[0, 1, 2, 3, 4]] * 2


def test_from_ndarray_with_unnormalized_qhape_roundtrips():
    a = np.arange(36.0).reshape(6, 6)
    T = TensorZ3.from_ndarray(
        a,
        shape=[[1, 2, 3]] * 2,
        qhape=[[-1, 0, 1]] * 2,
        dirs=[1, 1],
        invar=False,
    )
    assert T.qhape == [[0, 1, 2]] * 2
    assert np.allclose(T.to_ndarray(), a)


def test_numpy_integer_qnums_become_python_ints():
    T = TensorU1.random(
        [[1, 1]],
        qhape=[[np.int64(0), np.int64(3)]],
        dirs=[1],
        charge=np.int64(3),
    )
    assert all(type(q) is int for q in T.qhape[0])
    assert type(T.charge) is int
    assert (3,) in T.sects


def test_default_qim_requires_one_or_all_sectors():
    with pytest.raises(ValueError, match="qhape"):
        TensorZ3.random([[1, 1]], dirs=[1])
    assert TensorZ3.random([[2]], dirs=[1]).qhape == [[0]]


def test_eye_respects_dtype():
    T = TensorZ3.eye([1, 2, 3], dtype=np.complex128)
    assert T.dtype == np.complex128
    assert np.allclose(T.to_ndarray(), np.eye(6))


def test_repr_and_str_show_symmetry():
    T = TensorZ2.random([[1, 1]], dirs=[1])
    assert "symmetry=CyclicGroup(2)" in repr(T)
    assert ", defval=" in repr(T)
    assert "symmetry = CyclicGroup(2)" in str(T)


def test_product_group_classes():
    Z2, Z3 = CyclicGroup(2), CyclicGroup(3)
    assert TensorZ2Z2.symmetry == ProductGroup(Z2, Z2)
    assert TensorZ2Z3.symmetry == ProductGroup(Z2, Z3)
    T = TensorZ2Z3.zeros([[1] * 6, [1] * 6], dirs=[1, -1])
    assert T.charge == (0, 0)
    assert T.qhape[0] == ProductGroup(Z2, Z3).elements()


def test_product_default_qhape_requires_one_or_all_sectors():
    with pytest.raises(ValueError, match="qhape"):
        TensorZ2Z3.random([[1, 1]], dirs=[1])
    assert TensorZ2Z3.random([[2]], dirs=[1]).qhape == [[(0, 0)]]


def test_charges_compose_in_product_group():
    A = TensorZ2Z3.random([[1] * 6, [1] * 6], dirs=[1, -1], charge=(1, 2))
    B = TensorZ2Z3.random([[1] * 6, [1] * 6], dirs=[1, -1], charge=(1, 2))
    C = A.dot(B, (1, 0))
    assert C.charge == (0, 1)
    assert np.allclose(C.to_ndarray(), A.to_ndarray() @ B.to_ndarray())
    assert A.conj().charge == (1, 1)


def test_product_qnums_given_as_lists_are_normalized():
    T = TensorZ2Z3.random(
        [[1, 2]], qhape=[[[0, 0], [1, 2]]], dirs=[1], charge=[1, 2]
    )
    assert T.qhape == [[(0, 0), (1, 2)]]
    assert T.charge == (1, 2)
    assert ((1, 2),) in T.sects


def test_binary_op_with_differing_tuple_charges_warns():
    A = TensorZ2Z3.random([[1] * 6], dirs=[1], charge=(1, 0))
    B = TensorZ2Z3.random([[1] * 6], dirs=[1], charge=(0, 1))
    with pytest.warns(UserWarning, match=r"\(1, 0\) and \(0, 1\)"):
        _ = A + B


def test_product_qodulus_attribute_is_attribute_error():
    T = TensorZ2Z3.random([[1] * 6], dirs=[1])
    with pytest.warns(DeprecationWarning), pytest.raises(AttributeError):
        _ = T.qodulus


def test_product_tensor_does_not_mix_with_cyclic():
    A = TensorZ2Z2.random([[1] * 4], dirs=[1])
    B = TensorZ2.random([[1, 1]], dirs=[1])
    with pytest.raises(AssertionError):
        _ = A + B


def test_scalar_with_identity_tuple_charge_stays_invariant():
    """The identity (0, 0) of a product group is a truthy tuple; it must not
    be mistaken for a non-zero charge.
    """
    A = TensorZ2Z3.random([[1] * 6], dirs=[1])
    B = TensorZ2Z3.random([[1] * 6], dirs=[-1])
    s = A.dot(B, (0, 0))
    assert s.isscalar()
    assert s.invar
    assert (s * 2.0).invar
    assert (s + s).invar
