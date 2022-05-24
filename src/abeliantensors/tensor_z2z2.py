import numpy as np
from .abeliantensor_mod import AbelianTensorMod, Z2Z2
from collections.abc import Iterable


class TensorZ2Z2(AbelianTensorMod):
    """A class for Z2 x Z2 symmetric tensors."""

    def __init__(self, shape, *args, qhape=None, **kwargs):
        if qhape is None:
            qhape = type(self)._shape_to_qhape(shape)
        kwargs["symmetry"] = Z2Z2
        return super(TensorZ2Z2, self).__init__(
            shape, *args, qhape=qhape, **kwargs
        )

    @classmethod
    def eye(cls, dim, qim=None, dtype=np.float_):
        """Return the identity matrix of the given dimension `dim`."""
        if qim is None:
            qim = cls._dim_to_qim(dim)
        return super(TensorZ2Z2, cls).eye(
            dim, qim=qim, dtype=np.float_
        )

    @classmethod
    def initialize_with(
        cls, numpy_func, shape, *args, qhape=None, **kwargs
    ):
        """Return a tensor of the given `shape`, initialized with
        `numpy_func`.
        """
        if qhape is None:
            qhape = cls._shape_to_qhape(shape)
        return super(TensorZ2Z2, cls).initialize_with(
            numpy_func, shape, *args, qhape=qhape, **kwargs
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # To and from normal numpy arrays

    @classmethod
    def from_ndarray(
        cls, a, *args, shape=None, qhape=None, **kwargs
    ):
        """Build a `TensorZN` out of a given NumPy array, using the provided
        form data.

        If `qhape` is not provided, it is automatically generated based on
        `shape` to be ``[0, ..., N]`` for each index. See
        `AbelianTensor.from_ndarray` for more documentation.
        """
        if qhape is None:
            qhape = cls._shape_to_qhape(shape)
        return super(TensorZ2Z2, cls).from_ndarray(
            a, *args, shape=shape, qhape=qhape, **kwargs
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Miscellaneous

    @classmethod
    def _dim_to_qim(cls, dim):
        """Given the dimensions of sectors along an index, generate the the
        corresponding default quantum numbers.
        """
        return [0] if len(dim) == 1 else list(range(4))

    @classmethod
    def _shape_to_qhape(cls, shape):
        """Given the `shape` of a tensor, generate the the corresponding default
        quantum numbers.
        """
        return [cls._dim_to_qim(dim) for dim in shape]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The meat: actual tensor operations

    def split_indices(self, indices, dims, qims=None, dirs=None):
        """Split indices in the spirit of reshape.

        If `qhape` is not provided, it is automatically generated based on
        `shape` to be ``[0, ..., N]`` for each index. See `AbelianTensor.split`
        for more documentation.
        """
        # Buildind qims.
        if qims is None:
            if isinstance(indices, Iterable):
                qims = [type(self)._shape_to_qhape(dim) for dim in dims]
            else:
                qims = type(self)._shape_to_qhape(dims)
        return super(TensorZ2Z2, self).split_indices(
            indices, dims, qims=qims, dirs=dirs
        )
