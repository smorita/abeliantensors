"""Define utility functions and set up pytest fixtures for the test suite."""

import numpy as np
import pytest

from abeliantensors import (
    Tensor,
    TensorU1,
    TensorZ2,
    TensorZ2Z2,
    TensorZ2Z3,
    TensorZ3,
)

TENSORCLASSES = {
    "tensor": Tensor,
    "tensorz2": TensorZ2,
    "tensorz3": TensorZ3,
    "tensoru1": TensorU1,
    "tensorz2z2": TensorZ2Z2,
    "tensorz2z3": TensorZ2Z3,
}


@pytest.fixture(scope="module")
def tensorclass(request):
    """A pytest fixture that returns the tensor class currently being
    tested.
    """
    return request.param


def pytest_addoption(parser):
    """Add command line options for setting the tensorclass(es) to test and
    the number of times to repeat each test.
    """
    parser.addoption(
        "--tensorclass",
        action="append",
        default=[],
        help="Tensor class(es) to run tests on.",
    )
    parser.addoption(
        "--n_iters",
        type=int,
        default=100,
        help="Number of times to run each test on new random input.",
    )


def parse_tensorclass(s):
    """Take a string representing a tensorclass, such as "TensorZ2", and return
    the corresponding class.
    """
    try:
        return TENSORCLASSES[s.lower().strip()]
    except KeyError:
        msg = f"Unknown tensor class name: {s}"
        raise ValueError(msg) from None


def pytest_generate_tests(metafunc):
    """Set up passing the command line arguments for n_iters and for the
    tensorclass fixture, and give the latter's default value.
    """
    default_classes = [
        "Tensor",
        "TensorZ2",
        "TensorU1",
        "TensorZ3",
        "TensorZ2Z2",
        "TensorZ2Z3",
    ]
    tensorclass_opts = (
        metafunc.config.getoption("tensorclass") or default_classes
    )
    tensorclasses = list(map(parse_tensorclass, tensorclass_opts))
    if "tensorclass" in metafunc.fixturenames:
        metafunc.parametrize("tensorclass", tensorclasses, indirect=True)
    n_iters = metafunc.config.getoption("n_iters")
    if "n_iters" in metafunc.fixturenames:
        metafunc.parametrize("n_iters", [n_iters])


def _qnum_elements(tensorclass):
    """Return the list of all possible quantum numbers of the tensorclass, or
    None if there are infinitely many or the class has no symmetry.
    """
    symmetry = getattr(tensorclass, "symmetry", None)
    return None if symmetry is None else symmetry.elements()


@pytest.fixture
def n_qnums(tensorclass):
    """Return the number of different possible quantum numbers for the given
    tensorclass, or None if the answer is infinite or undefined.
    """
    elements = _qnum_elements(tensorclass)
    return None if elements is None else len(elements)


@pytest.fixture
def qid(tensorclass):
    """Return the identity charge of the given tensorclass (0 for `Tensor`
    and U(1)).
    """
    symmetry = getattr(tensorclass, "symmetry", None)
    return 0 if symmetry is None else symmetry.identity


@pytest.fixture
def rshape(n_qnums):
    """Return a function that generates random shapes for symmetric tensors."""

    def _rshape(n=None, chi=None, nlow=0, nhigh=5, chilow=0, chihigh=3):
        """Return a random shape for a symmetric tensor.

        `n` is the number of indices, `chi` is the bond dimension of each
        symmetry sector. If either one is None, they are random generated using
        the bounds `nlow`, `nhigh`, `chilow`, and `chihigh`.
        """
        if n is None:
            n = np.random.randint(nlow, high=nhigh)
        shape = []
        for _i in range(n):
            if n_qnums is None:
                dim_dim = np.random.randint(low=1, high=5)
            elif n_qnums > 3:
                # Using every sector of a larger group (e.g. 6 for Z2 x Z3)
                # makes the dense comparisons in the tests too big, so use 1
                # to 4 sectors as for U(1). rqhape always gives the qhape
                # explicitly, so the default-qhape rules are not involved.
                dim_dim = np.random.randint(low=1, high=5)
            else:
                dim_dim = n_qnums
            if chi is None:
                dim = []
                for _ in range(dim_dim):
                    dim.append(np.random.randint(0, high=chihigh))
            else:
                dim = [chi] * dim_dim
            s = sum(dim)
            if s < chilow:
                j = np.random.randint(0, len(dim))
                dim[j] += chilow - s
            shape.append(dim)
        return shape

    return _rshape


@pytest.fixture
def rqhape(tensorclass, n_qnums):
    """Return a function that generates random `qhape`s for symmetric
    tensors.
    """

    def _rqhape(shape, qlow=-3, qhigh=3):
        """Return a random `qhape` for a symmetric tensor.

        `shape` is the corresponding shape. For finite groups, quantum
        numbers are randomly picked from the group's elements. For `TensorU1`,
        they are randomly picked from the `range(qlow, qhigh+1)`.
        """
        elements = _qnum_elements(tensorclass)
        if elements is not None:
            try:
                qhape = []
                for dim in shape:
                    idx = np.random.choice(
                        len(elements), len(dim), replace=False
                    )
                    qhape.append([elements[i] for i in idx])
            except TypeError:
                # This happens when dim is a single number.
                qhape = None
            return qhape
        try:
            assert all(len(dim) <= qhigh - qlow + 1 for dim in shape)
        except TypeError:
            # This happens when dim is a single number.
            pass
        possible_qnums = range(qlow, qhigh + 1)
        try:
            qhape = [
                list(np.random.choice(possible_qnums, len(dim), replace=False))
                for dim in shape
            ]
        except TypeError:
            # This happens when dim is a single number.
            qhape = None
        return qhape

    return _rqhape


@pytest.fixture
def rdirs():
    """Return a function that generates random `dirs`s for symmetric
    tensors.
    """

    def _rdirs(shape=None, length=None):
        """Return a random `dirs` for a symmetric tensor.

        `shape` is the corresponding shape. Alternatively, just the length of
        of the `dirs` can be passed.
        """
        if shape is not None:
            length = len(shape)
        dirs = np.random.randint(low=0, high=2, size=length)
        dirs = list(2 * dirs - 1)
        return dirs

    return _rdirs


@pytest.fixture
def rcharge(tensorclass):
    """Return a function that generates random charges for symmetric
    tensors.
    """

    def _rcharge(low=0, high=4):
        """Return a random charge for a symmetric tensor: a random element of
        the group for finite groups, otherwise a random integer between `low`
        and `high`.
        """
        elements = _qnum_elements(tensorclass)
        if elements is not None:
            return elements[np.random.randint(len(elements))]
        return np.random.randint(low=low, high=high)

    return _rcharge


@pytest.fixture
def rtensor(tensorclass, n_qnums, rqhape, rshape, rdirs, rcharge):
    """Return a function that generates a random tensor of the given
    tensorclass, using the given fixtures for generating form data.
    """

    def _rtensor(
        shape=None,
        qhape=None,
        n=None,
        chi=None,
        nlow=0,
        nhigh=5,
        chilow=0,
        chihigh=6,
        charge=None,
        dirs=None,
        cmplx=True,
        **kwargs,
    ):
        """Return a random tensor of the given form data.

        Full form data (`shape`, `qhape`, `dirs`, `charge`) can be provided, in
        which case only the elements of the tensor are random. Alternatively,
        random form data can also be generated. In that case, `n` is the number
        of indices (alternatively a random number between `nlow` and `nhigh`)
        and `chi` is the bond dimension of all sectors (alternatively random
        numbers between `chilow` and `chihigh`).

        Partial form data can also be given. For instance, if one only gives
        the `shape` argument, random `qhape`, `dirs`, and `charge` are
        generated that match it.

        `cmplx` sets whether the tensor should be complex instead of
        real, and is by default True.
        """
        if shape is None:
            shape = rshape(
                n=n,
                chi=chi,
                nlow=nlow,
                nhigh=nhigh,
                chilow=chilow,
                chihigh=chihigh,
            )
        if qhape is None:
            qhape = rqhape(shape)
        if dirs is None:
            dirs = rdirs(shape=shape)
        elif dirs == 1:
            dirs = [1] * len(shape)
        if charge is None:
            charge = rcharge()

        real = tensorclass.random(
            shape, qhape=qhape, dirs=dirs, charge=charge, **kwargs
        )
        if cmplx:
            imag = tensorclass.random(
                shape, qhape=qhape, dirs=dirs, charge=charge, **kwargs
            )
            res = real + 1j * imag
        else:
            res = real
        return res

    return _rtensor
