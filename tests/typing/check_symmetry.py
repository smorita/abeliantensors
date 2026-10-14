"""Type-checking assertions for abeliantensors.symmetry.

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import assert_type

from abeliantensors import CyclicGroup, ProductGroup, U1Group

Z3 = CyclicGroup(3)
assert_type(Z3.identity, int)
assert_type(Z3.add(1, 2), int)
assert_type(Z3.neg(1), int)
assert_type(Z3.normalize(-1), int)
assert_type(Z3.signed(1, -1), int)
assert_type(Z3.sum([1, 2]), int)
assert_type(Z3.elements(), list[int] | None)

U1 = U1Group()
assert_type(U1.add(1, 2), int)

P = ProductGroup(CyclicGroup(2), CyclicGroup(3))
assert_type(P.identity, tuple[int, ...])
assert_type(P.add((1, 0), (1, 2)), tuple[int, ...])
assert_type(P.normalize([3, -1]), tuple[int, ...])
assert_type(P.elements(), list[tuple[int, ...]] | None)
assert_type(P.factors, tuple[CyclicGroup | U1Group, ...])
