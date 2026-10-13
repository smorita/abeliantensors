"""Time one TRG step for Tensor and TensorZ2 at small bond dimensions.

The full demo in demo/trg_demo.py takes hours; this script is a quick
benchmark for comparing library versions on the same machine. Run from the
repository root:

    uv run --group demo python benchmarks/trg_bench.py
"""

import sys
import timeit
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))

from trg import trg_step  # noqa: E402
from trg_demo import ising_tensor  # noqa: E402

from abeliantensors import Tensor, TensorZ2  # noqa: E402

CHIS = (8, 16, 24)
N_CONVERGE = 14
N_REPEAT = 5
EPS = 1e-7


def time_trg_step(cls, chi, A0_np):
    """Converge TRG at bond dimension `chi`, then return the fastest of
    `N_REPEAT` timings of a single further step.
    """
    A = cls.from_ndarray(
        A0_np, shape=[[1, 1]] * 4, qhape=[[0, 1]] * 4, dirs=[1, -1, 1, -1]
    )
    log_fact = 0
    for _ in range(N_CONVERGE):
        A, log_fact, _err = trg_step(A, log_fact, chi, EPS)
    timings = timeit.repeat(
        lambda: trg_step(A, log_fact, chi, EPS), number=1, repeat=N_REPEAT
    )
    return min(timings)


def main():
    beta = np.log(1 + np.sqrt(2)) / 2
    A0_np = ising_tensor(1.0, 0.0, beta)
    print(f"{'chi':>4} {'Tensor [s]':>12} {'TensorZ2 [s]':>13}")
    for chi in CHIS:
        t_dense = time_trg_step(Tensor, chi, A0_np)
        t_z2 = time_trg_step(TensorZ2, chi, A0_np)
        print(f"{chi:>4} {t_dense:>12.4f} {t_z2:>13.4f}")


if __name__ == "__main__":
    main()
