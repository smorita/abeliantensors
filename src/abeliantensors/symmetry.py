"""Abelian groups that define the symmetry of an `AbelianTensor`.

A group object defines how quantum numbers (charges) combine. Tensors treat
quantum numbers as opaque hashable values and delegate all arithmetic on them
to their group. Groups are immutable values: two groups are equal if they
describe the same group, and copying a group returns the group itself.
"""

import itertools as itt
import numbers


def _as_int(q):
    """Return `q` as a Python int, or raise ValueError if it is not an
    integer.
    """
    if type(q) is int:
        return q
    if isinstance(q, numbers.Integral):
        return int(q)
    raise ValueError(f"Quantum number must be an integer, got {q!r}.")


class AbelianGroup:
    """Base class for abelian groups of quantum numbers.

    Subclasses implement `identity`, `add`, `neg`, `normalize`, `elements`
    and `_key`. The group operation is written additively.
    """

    identity = None

    def add(self, a, b):
        """Return the group product of `a` and `b`."""
        raise NotImplementedError

    def neg(self, a):
        """Return the inverse of `a`."""
        raise NotImplementedError

    def normalize(self, q):
        """Return `q` in canonical form. Raise ValueError if `q` is not a
        valid quantum number of this group.
        """
        raise NotImplementedError

    def elements(self):
        """Return a list of all the elements, or None if the group is
        infinite.
        """
        return None

    def signed(self, a, d):
        """Return `a` for direction ``d == 1`` and its inverse for
        ``d == -1``.
        """
        if d == 1:
            return a
        if d == -1:
            return self.neg(a)
        raise ValueError(f"direction must be +1 or -1, got {d!r}")

    def sum(self, qs):
        """Return the group product of all the quantum numbers in `qs`."""
        res = self.identity
        for q in qs:
            res = self.add(res, q)
        return res

    def _key(self):
        """Return a tuple that identifies the group within its class."""
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self):
        return hash((type(self), self._key()))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class CyclicGroup(AbelianGroup):
    """The cyclic group Z_n, with quantum numbers ``0, ..., n-1``."""

    identity = 0

    def __init__(self, n):
        n = _as_int(n)
        if n < 1:
            raise ValueError(f"Order of a cyclic group must be >= 1, got {n}.")
        self.n = n

    def add(self, a, b):
        return (a + b) % self.n

    def neg(self, a):
        return -a % self.n

    def sum(self, qs):
        return sum(qs) % self.n

    def normalize(self, q):
        return _as_int(q) % self.n

    def elements(self):
        return list(range(self.n))

    def _key(self):
        return (self.n,)

    def __repr__(self):
        return f"CyclicGroup({self.n})"


class U1Group(AbelianGroup):
    """The group U(1), with integer quantum numbers."""

    identity = 0

    def add(self, a, b):
        return a + b

    def neg(self, a):
        return -a

    def sum(self, qs):
        return sum(qs)

    def normalize(self, q):
        return _as_int(q)

    def _key(self):
        return ()

    def __repr__(self):
        return "U1Group()"


class ProductGroup(AbelianGroup):
    """A direct product of `CyclicGroup` and `U1Group` factors.

    Quantum numbers are tuples with one component per factor.
    """

    def __init__(self, *factors):
        if len(factors) < 2:
            raise ValueError("ProductGroup needs at least two factors.")
        for f in factors:
            if not isinstance(f, (CyclicGroup, U1Group)):
                raise TypeError(
                    "ProductGroup factors must be CyclicGroup or U1Group, "
                    f"got {f!r}."
                )
        self.factors = tuple(factors)
        self.identity = tuple(f.identity for f in self.factors)

    def add(self, a, b):
        return tuple(f.add(x, y) for f, x, y in zip(self.factors, a, b))

    def neg(self, a):
        return tuple(f.neg(x) for f, x in zip(self.factors, a))

    def normalize(self, q):
        if not isinstance(q, (tuple, list)) or len(q) != len(self.factors):
            raise ValueError(
                f"Quantum number of {self!r} must be a tuple of length "
                f"{len(self.factors)}, got {q!r}."
            )
        return tuple(f.normalize(x) for f, x in zip(self.factors, q))

    def elements(self):
        factor_elements = [f.elements() for f in self.factors]
        if any(e is None for e in factor_elements):
            return None
        return list(itt.product(*factor_elements))

    def _key(self):
        return self.factors

    def __repr__(self):
        return f"ProductGroup({', '.join(map(repr, self.factors))})"
