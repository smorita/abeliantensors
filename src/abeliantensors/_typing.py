"""Type aliases used in the annotations of abeliantensors (not public)."""

from collections.abc import Sequence

# A quantum number: an int for CyclicGroup and U1Group, a tuple of ints for
# ProductGroup.
Qnum = int | tuple[int, ...]
# Dimensions of the symmetry sectors along one index.
Dim = list[int]
# Quantum numbers of the symmetry sectors along one index.
Qim = list[Qnum]
# Index or indices given to dot, svd, eig, split, ...
Indices = int | Sequence[int]
