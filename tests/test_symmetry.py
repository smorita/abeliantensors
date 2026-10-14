"""Tests for the abelian group objects in abeliantensors.symmetry."""

import copy
import pickle

import numpy as np
import pytest

from abeliantensors import CyclicGroup, ProductGroup, U1Group

Z2 = CyclicGroup(2)
Z3 = CyclicGroup(3)
FINITE_GROUPS = [
    Z2,
    Z3,
    ProductGroup(Z2, Z2),
    ProductGroup(Z2, Z3),
]


@pytest.mark.parametrize("g", FINITE_GROUPS, ids=repr)
def test_group_axioms(g):
    """Identity, inverses, closure, commutativity and associativity hold for
    all elements.
    """
    els = g.elements()
    assert len(els) == len(set(els))
    assert g.identity in els
    for a in els:
        assert g.normalize(a) == a
        assert g.add(a, g.identity) == a
        assert g.add(a, g.neg(a)) == g.identity
        for b in els:
            assert g.add(a, b) in els
            assert g.add(a, b) == g.add(b, a)
            for c in els:
                assert g.add(g.add(a, b), c) == g.add(a, g.add(b, c))


@pytest.mark.parametrize("g", FINITE_GROUPS, ids=repr)
def test_signed_and_sum(g):
    els = g.elements()
    for a in els:
        assert g.signed(a, 1) == a
        assert g.signed(a, -1) == g.neg(a)
    expected = g.identity
    for a in els:
        expected = g.add(expected, a)
    assert g.sum(els) == expected
    assert g.sum(iter(els)) == expected
    assert g.sum([]) == g.identity


@pytest.mark.parametrize("d", [0, 2, -2])
@pytest.mark.parametrize("g", [*FINITE_GROUPS, U1Group()], ids=repr)
def test_signed_rejects_non_unit_direction(g, d):
    with pytest.raises(ValueError, match="direction"):
        g.signed(g.identity, d)


def test_elements():
    assert Z3.elements() == [0, 1, 2]
    assert ProductGroup(Z2, Z3).elements() == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    assert U1Group().elements() is None
    assert ProductGroup(U1Group(), Z2).elements() is None


def test_normalize_valid_inputs():
    assert Z3.normalize(-1) == 2
    assert Z3.normalize(np.int64(4)) == 1
    assert type(Z3.normalize(np.int64(4))) is int
    assert U1Group().normalize(np.int64(-5)) == -5
    assert type(U1Group().normalize(np.int64(-5))) is int
    assert ProductGroup(Z2, Z3).normalize([3, -1]) == (1, 2)
    assert ProductGroup(Z2, Z3).normalize((np.int64(1), 2)) == (1, 2)
    assert ProductGroup(U1Group(), Z2).normalize((-7, 3)) == (-7, 1)


@pytest.mark.parametrize(
    "g, q",
    [
        (Z2, 0.5),
        (Z2, "1"),
        (Z2, (0,)),
        (U1Group(), 1.5),
        (ProductGroup(Z2, Z3), 1),
        (ProductGroup(Z2, Z3), (0,)),
        (ProductGroup(Z2, Z3), (0, 0, 0)),
        (ProductGroup(Z2, Z3), (0.5, 0)),
    ],
    ids=repr,
)
def test_normalize_rejects_invalid_inputs(g, q):
    with pytest.raises(ValueError):
        g.normalize(q)


def test_equality_and_hash():
    assert CyclicGroup(2) == CyclicGroup(2)
    assert hash(CyclicGroup(2)) == hash(CyclicGroup(2))
    assert CyclicGroup(2) != CyclicGroup(3)
    assert CyclicGroup(2) != U1Group()
    assert U1Group() == U1Group()
    assert ProductGroup(Z2, Z3) == ProductGroup(CyclicGroup(2), CyclicGroup(3))
    assert ProductGroup(Z2, Z3) != ProductGroup(Z3, Z2)
    assert len({CyclicGroup(2), CyclicGroup(2), U1Group()}) == 2


def test_constructor_validation():
    with pytest.raises(ValueError):
        CyclicGroup(0)
    with pytest.raises(ValueError):
        ProductGroup(Z2)
    with pytest.raises(TypeError):
        ProductGroup(Z2, "Z3")
    with pytest.raises(TypeError):
        ProductGroup(Z2, ProductGroup(Z2, Z2))


@pytest.mark.parametrize("g", [*FINITE_GROUPS, U1Group()], ids=repr)
def test_copies_are_the_same_object(g):
    assert copy.copy(g) is g
    assert copy.deepcopy(g) is g


def test_repr():
    assert repr(Z2) == "CyclicGroup(2)"
    assert repr(U1Group()) == "U1Group()"
    assert (
        repr(ProductGroup(Z2, Z3))
        == "ProductGroup(CyclicGroup(2), CyclicGroup(3))"
    )


@pytest.mark.parametrize("g", [*FINITE_GROUPS, U1Group()], ids=repr)
def test_pickle_roundtrip(g):
    h = pickle.loads(pickle.dumps(g))
    assert h == g
    assert hash(h) == hash(g)
    assert h.elements() == g.elements()
