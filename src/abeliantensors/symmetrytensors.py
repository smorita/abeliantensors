from collections.abc import Iterable

from .abeliantensor import AbelianTensor
from .symmetry import CyclicGroup, ProductGroup, U1Group


class FiniteAbelianTensor(AbelianTensor):
    """Base class for tensors with a fixed, finite abelian symmetry group.

    Subclasses set the class attribute `symmetry` to an `AbelianGroup` with
    finitely many elements. Then `qhape` (or `qim`) can be omitted: an index
    with a single sector gets the identity as its quantum number, and an index
    with one sector per group element gets all the elements in the order of
    ``symmetry.elements()``. See `AbelianTensor` for the details.
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Initializers and factory methods

    def __init__(self, shape, *args, qhape=None, **kwargs):
        if qhape is None:
            qhape = type(self)._shape_to_qhape(shape)
        super().__init__(shape, *args, qhape=qhape, **kwargs)

    @classmethod
    def eye(cls, dim, qim=None, *args, **kwargs):
        """Return the identity matrix of the given dimension `dim`."""
        if qim is None:
            qim = cls._dim_to_qim(dim)
        return super().eye(dim, qim, *args, **kwargs)

    @classmethod
    def initialize_with(cls, numpy_func, shape, *args, qhape=None, **kwargs):
        """Return a tensor of the given `shape`, initialized with
        `numpy_func`.
        """
        if qhape is None:
            qhape = cls._shape_to_qhape(shape)
        return super().initialize_with(
            numpy_func, shape, *args, qhape=qhape, **kwargs
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # To and from normal numpy arrays

    @classmethod
    def from_ndarray(cls, a, *args, shape=None, qhape=None, **kwargs):
        """Build a tensor out of a given NumPy array, using the provided form
        data.

        If `qhape` is not provided, it is generated from `shape` as described
        in the class documentation. See `AbelianTensor.from_ndarray` for more
        documentation.
        """
        if qhape is None:
            qhape = cls._shape_to_qhape(shape)
        return super().from_ndarray(
            a, *args, shape=shape, qhape=qhape, **kwargs
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    @classmethod
    def _dim_to_qim(cls, dim):
        """Given the dimensions of sectors along an index, generate the
        corresponding default quantum numbers.
        """
        if len(dim) == 1:
            return [cls.symmetry.identity]
        elements = cls.symmetry.elements()
        if len(dim) == len(elements):
            return list(elements)
        raise ValueError(
            f"Cannot generate a default qhape for an index with {len(dim)} "
            f"sectors under {cls.symmetry!r}: give qhape explicitly, or use "
            f"1 or {len(elements)} sectors."
        )

    @classmethod
    def _shape_to_qhape(cls, shape):
        """Given the `shape` of a tensor, generate the corresponding default
        quantum numbers.
        """
        return [cls._dim_to_qim(dim) for dim in shape]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def split_indices(self, indices, dims, qims=None, dirs=None):
        """Split indices in the spirit of reshape.

        If `qims` is not provided, it is generated from `dims` as described in
        the class documentation. See `AbelianTensor.split_indices` for more
        documentation.
        """
        if qims is None:
            if isinstance(indices, Iterable):
                qims = [type(self)._shape_to_qhape(dim) for dim in dims]
            else:
                qims = type(self)._shape_to_qhape(dims)
        return super().split_indices(indices, dims, qims=qims, dirs=dirs)


class TensorZN(FiniteAbelianTensor):
    """A symmetric tensor class for the cyclic group of order N.

    Subclasses set ``symmetry = CyclicGroup(N)``. See `FiniteAbelianTensor`
    and `AbelianTensor` for the details.
    """


class TensorZ2(TensorZN):
    """A class for Z2 symmetric tensors.

    See the parent class `AbelianTensor` for details.
    """

    symmetry = CyclicGroup(2)


class TensorZ3(TensorZN):
    """A class for Z3 symmetric tensors.

    See the parent class `AbelianTensor` for details.
    """

    symmetry = CyclicGroup(3)


class TensorU1(AbelianTensor):
    """A class for U(1) symmetric tensors.

    See the parent class `AbelianTensor` for details.
    """

    symmetry = U1Group()


class TensorZ2Z2(FiniteAbelianTensor):
    """A class for Z2 x Z2 symmetric tensors.

    Quantum numbers are pairs ``(a, b)`` with ``a, b`` in ``{0, 1}``. See
    `FiniteAbelianTensor` and `AbelianTensor` for the details.
    """

    symmetry = ProductGroup(CyclicGroup(2), CyclicGroup(2))


class TensorZ2Z3(FiniteAbelianTensor):
    """A class for Z2 x Z3 symmetric tensors.

    Quantum numbers are pairs ``(a, b)`` with ``a`` in ``{0, 1}`` and ``b`` in
    ``{0, 1, 2}``. See `FiniteAbelianTensor` and `AbelianTensor` for the
    details.
    """

    symmetry = ProductGroup(CyclicGroup(2), CyclicGroup(3))
