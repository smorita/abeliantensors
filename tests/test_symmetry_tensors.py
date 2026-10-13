"""Tests for how AbelianTensor and its subclasses use symmetry groups."""

import warnings

import numpy as np
import pytest

from abeliantensors import (
    AbelianTensor,
    CyclicGroup,
    FiniteAbelianTensor,
    ProductGroup,
    Tensor,
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
    with pytest.raises(ValueError, match="different symmetries"):
        _ = A + B
    with pytest.raises(ValueError, match="different symmetries"):
        _ = A.allclose(B)
    with pytest.raises(ValueError, match="different symmetries"):
        _ = A.dot(B.flip_dir(0), (0, 0))
    with pytest.raises(ValueError, match="different symmetries"):
        _ = A.matrix_dot(B.flip_dir(0))
    D = AbelianTensor.random(
        [[1, 1]],
        qhape=[[0, 1]],
        dirs=[-1],
        invar=False,
        symmetry=CyclicGroup(3),
    )
    M = AbelianTensor.random(
        [[1, 1], [1, 1]],
        qhape=[[0, 1], [0, 1]],
        dirs=[1, -1],
        symmetry=CyclicGroup(2),
    )
    with pytest.raises(ValueError, match="different symmetries"):
        _ = M.multiply_diag(D, 1)


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
    with pytest.raises(ValueError, match="different symmetries"):
        _ = A + B
    with pytest.raises(ValueError, match="different symmetries"):
        _ = A.dot(B.flip_dir(0), (0, 0))


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


@pytest.mark.parametrize(
    "make",
    [
        lambda: AbelianTensor([[1]], qhape=[[0]], dirs=[1], qodulus=2),
        lambda: AbelianTensor.random(
            [[1, 1]], qhape=[[0, 1]], dirs=[1], qodulus=2
        ),
        lambda: AbelianTensor.zeros(
            [[1, 1]], qhape=[[0, 1]], dirs=[1], qodulus=2
        ),
        lambda: TensorZ2.eye([1, 1], qodulus=2),
        lambda: TensorZ2.random([[1, 1]], dirs=[1], qodulus=2),
    ],
    ids=["init", "random", "zeros", "eye", "subclass-random"],
)
def test_qodulus_warning_points_at_caller(make):
    """Python shows a DeprecationWarning by default only when it is attributed
    to the caller's code, not to a frame inside the library.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make()
    deprecations = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecations
    assert deprecations[0].filename == __file__


def test_malformed_product_key_is_key_error():
    T = TensorZ2Z2.random([[1] * 4, [1] * 4], dirs=[1, -1])
    with pytest.raises(KeyError, match="quantum numbers"):
        _ = T[(0, 0)]


def test_check_form_match_accepts_deprecated_qodulus():
    with pytest.warns(DeprecationWarning, match="qodulus"):
        assert AbelianTensor.check_form_match(
            qhape1=[[0, 1, 2]],
            shape1=[[1, 2, 3]],
            dirs1=[1],
            qhape2=[[0, 2, 1]],
            shape2=[[1, 2, 3]],
            dirs2=[-1],
            qodulus=3,
        )


def test_tensor_check_form_match_accepts_symmetry():
    assert Tensor.check_form_match(
        shape1=[2, 3], shape2=[2, 3], symmetry=CyclicGroup(2)
    )


def test_default_qhape_with_infinite_group_asks_for_qhape():
    class TensorZ2U1(FiniteAbelianTensor):
        symmetry = ProductGroup(CyclicGroup(2), U1Group())

    with pytest.raises(ValueError, match="qhape"):
        TensorZ2U1.random([[1, 1]], dirs=[1])
    T = TensorZ2U1.random([[1, 1]], qhape=[[(0, 0), (1, 5)]], dirs=[1])
    assert T.qhape == [[(0, 0), (1, 5)]]


def test_subclass_without_symmetry_is_not_fixed():
    class MyTensor(AbelianTensor):
        pass

    assert MyTensor([[1]], qhape=[[0]], dirs=[1]).symmetry == U1Group()
    T = MyTensor([[1]], qhape=[[0]], dirs=[1], symmetry=CyclicGroup(3))
    assert T.symmetry == CyclicGroup(3)
