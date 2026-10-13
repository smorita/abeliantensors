# 一般のアーベル対称性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `AbelianTensor` の対称性を、`qodulus` で決める方式から、群オブジェクト(`CyclicGroup` / `U1Group` / `ProductGroup`)で決める方式に置き換える。あわせて `TensorZ2Z2` と `TensorZ2Z3` を追加する。

**Architecture:** 新しいモジュール `symmetry.py` に群オブジェクトを定義する。`AbelianTensor` は量子数を「中身を知らないハッシュ可能な値」として扱い、演算はすべて `self.symmetry` のメソッドに任せる。有限群の `qhape` を自動で作る処理は `FiniteAbelianTensor` にまとめ、具体的なクラスは `symmetry` のクラス属性を設定するだけで定義する。

**Tech Stack:** Python ≥3.12、numpy 2.x、pytest、uv、ruff。

**Spec:** `docs/superpowers/specs/2026-10-13-general-abelian-symmetry-design.md`

## Global Constraints

- 作業ブランチは `general-abelian`(作成済み)。コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` を付ける。
- 量子数は Python の `int`、または `int` のタプル。numpy の整数型は `normalize` で `int` に変換する。
- 群に実装させるメソッドは `identity`、`add`、`neg`、`normalize`、`elements`。`signed(a, d)`(d は ±1 のみ)と `sum` は基底クラスが用意する。
- `qodulus` は非推奨とし、`DeprecationWarning` を出す。`qodulus` と `symmetry` を両方渡したら `TypeError`。`ProductGroup` の `.qodulus` は `AttributeError`。
- `pyproject.toml` の `filterwarnings = ["error::DeprecationWarning"]` は残す。ライブラリの内部で `qodulus` を使ってはならない(使うとテストがエラーになる)。
- 性能: `benchmarks/trg_bench.py` で測る `TensorZ2` の速度低下が、chi = 8、16、24 のいずれでも 20% 以内であること。
- ruff の設定(79 桁、`ruff format`)に従う。各コミットの前に `uv run ruff check && uv run ruff format --check` を通す。

## Review Focus

1. **旧方式のサブクラス `class TensorZ5(TensorZN): qodulus = 5`**:README で案内してきた書き方。何も言わずに U(1) として動くのではなく、`CyclicGroup(5)` として動き、`DeprecationWarning` が出るべき。→ Task 3 の `test_legacy_subclass_with_qodulus_class_attribute`。
2. **正規化されていない `qhape` を `from_ndarray` に渡す**(Z₃ に `[-1, 0, 1]` など):`to_ndarray` で元の配列に戻るべき。→ Task 3 の `test_from_ndarray_with_unnormalized_qhape_roundtrips`。
3. **charge が異なるタプル charge のテンソル同士の二項演算**:警告文の組み立てで例外にならず、`UserWarning` が出るべき。→ Task 4 の `test_binary_op_with_differing_tuple_charges_warns`。
4. **対称性の違うテンソルを混ぜる**(`CyclicGroup(2)` と `CyclicGroup(3)`):`AssertionError` になるべき。→ Task 3 の `test_mixing_symmetries_is_rejected`。
5. **直積群の量子数をリストで渡す**(JSON から読んだ `[[0, 0], [1, 2]]` など):タプルに正規化され、タプルのキーでブロックにアクセスできるべき。→ Task 4 の `test_product_qnums_given_as_lists_are_normalized`。

---

## File Structure

| ファイル | 作業 | 役割 |
|---|---|---|
| `benchmarks/trg_bench.py` | 新規 | TRG 1 ステップの時間を、小さい chi で測る |
| `src/abeliantensors/symmetry.py` | 新規 | 群オブジェクト `AbelianGroup` / `CyclicGroup` / `U1Group` / `ProductGroup` |
| `src/abeliantensors/abeliantensor.py` | 修正 | `qodulus` から `symmetry` への置き換えと、量子数の演算の委譲 |
| `src/abeliantensors/symmetrytensors.py` | 修正 | `FiniteAbelianTensor` と、具体的なクラス(`TensorZ2Z2` / `TensorZ2Z3` を含む) |
| `src/abeliantensors/__init__.py` | 修正 | 新しいクラスの公開 |
| `tests/test_symmetry.py` | 新規 | 群のテスト |
| `tests/test_symmetry_tensors.py` | 新規 | テンソルの互換性、誤用、直積群に固有の動作のテスト |
| `tests/conftest.py` | 修正 | 乱数生成を群に対応させ、テスト対象のクラスを追加する |
| `tests/test_tensors.py` | 修正 | `charge=0` を `charge=qid` に変える |
| `README.md`, `docs/source/index.rst` | 修正 | 使い方と API リファレンス |
| `docs/superpowers/specs/2026-10-13-general-abelian-symmetry-design.md` | 修正 | 計測結果と、旧方式のサブクラスの扱いを追記する |

---

### Task 1: 計測スクリプトと、変更前の基準値

**Files:**
- Create: `benchmarks/trg_bench.py`
- Modify: `docs/superpowers/specs/2026-10-13-general-abelian-symmetry-design.md`(「計測結果」の節)

**Interfaces:**
- Produces: `uv run --group demo python benchmarks/trg_bench.py` を実行すると、次の形の表を標準出力に出す。
  ```
   chi   Tensor [s]  TensorZ2 [s]
     8       0.0012        0.0034
  ```
  Task 5 でこの出力を比べる。

- [ ] **Step 1: スクリプトを作る**

`benchmarks/trg_bench.py`:

```python
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
```

- [ ] **Step 2: lint を確認する**

Run: `uv run ruff check benchmarks && uv run ruff format --check benchmarks`
Expected: `All checks passed!` / `1 file already formatted`(整形の差分が出た場合は `uv run ruff format benchmarks` で直す)

- [ ] **Step 3: 変更前の基準値を 2 回測る**

ライブラリはまだ変更していない(master と同じ)ので、ここで測った値が基準値になる。

Run: `for i in 1 2; do uv run --group demo python benchmarks/trg_bench.py; done`
Expected: 3 行の表が 2 回出る。数分で終わる。chi=24 で 1 分を超える場合は、それも記録する。

各 chi・各クラスについて、2 回のうち小さいほうを基準値とする。

- [ ] **Step 4: 設計書に基準値を記録する**

設計書の末尾の `## 計測結果` の下にある `(実装後に記入する)` を、次の形に置き換える(数値は Step 3 の実測値)。

```markdown
計測環境: <CPU 名(`lscpu | grep "Model name"`)>, Python 3.12, numpy <版>, `benchmarks/trg_bench.py`(2 回のうち速いほう)

| chi | Tensor 変更前 [s] | TensorZ2 変更前 [s] | Tensor 変更後 [s] | TensorZ2 変更後 [s] | TensorZ2 の変化 |
|---|---|---|---|---|---|
| 8  | <値> | <値> | | | |
| 16 | <値> | <値> | | | |
| 24 | <値> | <値> | | | |
```

- [ ] **Step 5: コミットする**

```bash
git add benchmarks/trg_bench.py docs/superpowers/specs/2026-10-13-general-abelian-symmetry-design.md
git commit -m "Add TRG micro-benchmark and record baseline timings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 群オブジェクト(`symmetry.py`)

**Files:**
- Create: `src/abeliantensors/symmetry.py`
- Create: `tests/test_symmetry.py`
- Modify: `src/abeliantensors/__init__.py`

**Interfaces:**
- Produces(以降のタスクがすべて使う):
  - `AbelianGroup`:属性 `identity`。メソッド `add(a, b)`、`neg(a)`、`signed(a, d)`(d が ±1 以外なら `ValueError`)、`sum(qs)`(`qs` は任意の iterable)、`normalize(q)`(不正な値なら `ValueError`)、`elements()`(`list` または `None`)。`==` と `hash` は値で比べる。`copy` / `deepcopy` は自分自身を返す。
  - `CyclicGroup(n)`:属性 `n`。量子数は `int`。`identity == 0`。
  - `U1Group()`:量子数は `int`。`identity == 0`。`elements()` は `None`。
  - `ProductGroup(*factors)`:属性 `factors`(タプル)。量子数は `int` のタプル。`identity == (0, ..., 0)`。
  - `from abeliantensors import AbelianGroup, CyclicGroup, U1Group, ProductGroup`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_symmetry.py`:

```python
"""Tests for the abelian group objects in abeliantensors.symmetry."""

import copy

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
```

- [ ] **Step 2: 失敗することを確認する**

Run: `uv run pytest tests/test_symmetry.py -q -p no:randomly 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'CyclicGroup' from 'abeliantensors'` で収集エラー。

- [ ] **Step 3: `symmetry.py` を実装する**

`src/abeliantensors/symmetry.py`:

```python
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
```

`numbers.Integral` には `bool` も含まれる。`True` は `1` として扱われるが、害はないのでそのままにする。`np.int64` などの numpy の整数型も `numbers.Integral` に登録されている。

- [ ] **Step 4: `__init__.py` から公開する**

`src/abeliantensors/__init__.py` を次の内容にする。

```python
from .abeliantensor import AbelianTensor
from .symmetry import AbelianGroup, CyclicGroup, ProductGroup, U1Group
from .symmetrytensors import TensorU1, TensorZ2, TensorZ3, TensorZN
from .tensor import Tensor
from .tensorcommon import TensorCommon

__all__ = [
    "AbelianGroup",
    "AbelianTensor",
    "CyclicGroup",
    "ProductGroup",
    "Tensor",
    "TensorCommon",
    "TensorU1",
    "TensorZ2",
    "TensorZ3",
    "TensorZN",
    "U1Group",
]
```

- [ ] **Step 5: テストが通ることを確認する**

Run: `uv run pytest tests/test_symmetry.py -q -p no:randomly 2>&1 | tail -3`
Expected: すべて passed

Run: `uv run pytest -q 2>&1 | tail -1 && uv run ruff check && uv run ruff format --check`
Expected: 既存の 136 件と新しいテストがすべて成功し、ruff も通る。

- [ ] **Step 6: コミットする**

```bash
git add src/abeliantensors/symmetry.py src/abeliantensors/__init__.py tests/test_symmetry.py
git commit -m "Add abelian group objects for quantum number arithmetic

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `AbelianTensor` を群オブジェクトに対応させる

**Files:**
- Modify: `src/abeliantensors/abeliantensor.py`(全体。以下の Step で箇所ごとに指定する)
- Modify: `src/abeliantensors/symmetrytensors.py`(全面的に書き換える)
- Create: `tests/test_symmetry_tensors.py`
- Modify: `docs/superpowers/specs/2026-10-13-general-abelian-symmetry-design.md`(§2 に旧方式のサブクラスの扱いを追記する)

**Interfaces:**
- Consumes: Task 2 の `CyclicGroup`、`U1Group`、`AbelianGroup` の API。
- Produces:
  - `AbelianTensor.symmetry`:クラス属性は既定で `None`。インスタンス属性は `AbelianGroup`。
  - `AbelianTensor(..., symmetry=None)`、`AbelianTensor.eye(dim, qim=None, qodulus=..., dtype=np.float64, symmetry=None)`、`initialize_with(..., charge=None, symmetry=None, **kwargs)`、`from_ndarray(..., charge=None, symmetry=None)`。`charge=None` は単位元の意味。
  - `AbelianTensor.qodulus`:非推奨の読み取り専用プロパティ。
  - `FiniteAbelianTensor`(`symmetrytensors.py`):クラス属性 `symmetry` を設定したサブクラスで、`qhape` / `qim` を省略できる。`_dim_to_qim(dim)` は、セクターが 1 つなら `[identity]`、セクター数が位数と等しければ `elements()` を返し、それ以外は `ValueError`。
  - 旧方式のサブクラス(クラス属性 `qodulus = N` または `None`)は、`__init_subclass__` で `symmetry` に変換される。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_symmetry_tensors.py`:

```python
"""Tests for how AbelianTensor and its subclasses use symmetry groups."""

import numpy as np
import pytest

from abeliantensors import (
    AbelianTensor,
    CyclicGroup,
    TensorU1,
    TensorZ2,
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
        a, shape=[[1, 2, 3]] * 2, qhape=[[-1, 0, 1]] * 2, dirs=[1, 1],
        invar=False,
    )
    assert T.qhape == [[0, 1, 2]] * 2
    assert np.allclose(T.to_ndarray(), a)


def test_numpy_integer_qnums_become_python_ints():
    T = TensorU1.random(
        [[1, 1]], qhape=[[np.int64(0), np.int64(3)]], dirs=[1], charge=np.int64(3)
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
```

`test_from_ndarray_with_unnormalized_qhape_roundtrips` の意味: `qhape=[-1, 0, 1]` と `shape=[1, 2, 3]` は、(量子数, 次元)の組で書くと (2, 1)、(0, 2)、(1, 3) を表す。`from_ndarray` は配列を「正規化した量子数の昇順」で読み、`to_ndarray` も同じ順で書き出すので、元の配列に戻るのが正しい。今の実装は正規化する前の値の順に並べ替えるため、このテストは失敗する。

- [ ] **Step 2: 失敗することを確認する**

Run: `uv run pytest tests/test_symmetry_tensors.py -q -p no:randomly 2>&1 | tail -15`
Expected: ほとんどのテストが失敗する(`AttributeError: ... 'symmetry'`、`TypeError: ... unexpected keyword argument 'symmetry'` など)。

- [ ] **Step 3: import と補助関数を追加する(`abeliantensor.py`)**

`from .tensorcommon import TensorCommon` の直後に次を追加する。

```python
from .symmetry import CyclicGroup, U1Group
```

`_arg_swapper` 関数の定義の後(`class AbelianTensor` の前)に次を追加する。

```python
# Sentinel for detecting whether the deprecated `qodulus` argument was given.
_UNSET = object()


def _resolve_symmetry(cls, symmetry, qodulus):
    """Return the `AbelianGroup` that a new tensor of class `cls` should have.

    `qodulus` is the deprecated way of giving the symmetry: an integer N means
    ``CyclicGroup(N)`` and None means ``U1Group()``. If neither argument is
    given, the class's own `symmetry` is used, or U(1) if the class does not
    fix one. A class that fixes its symmetry does not accept a different one.
    """
    if qodulus is not _UNSET:
        if symmetry is not None:
            raise TypeError("Give either `symmetry` or `qodulus`, not both.")
        warnings.warn(
            "The `qodulus` argument is deprecated. Use "
            "`symmetry=CyclicGroup(n)` or `symmetry=U1Group()` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        symmetry = U1Group() if qodulus is None else CyclicGroup(qodulus)
    fixed = cls.symmetry
    if symmetry is None:
        return fixed if fixed is not None else U1Group()
    if fixed is not None and symmetry != fixed:
        raise ValueError(
            f"{cls.__name__} has symmetry {fixed!r}, cannot create it with "
            f"symmetry {symmetry!r}."
        )
    return symmetry
```

- [ ] **Step 4: クラスの docstring、クラス属性、`__init_subclass__`、`qodulus` プロパティ**

クラスの docstring の冒頭の段落を置き換える。

置き換え前:
```
    This class is meant to be subclassed to implement specific symmetries,
    which can typically be done by simply fixing the `qodulus` of the class
    (see below).
```
置き換え後:
```
    The symmetry group is given by an `AbelianGroup` (see
    `abeliantensors.symmetry`), either per instance with the `symmetry`
    argument or fixed by a subclass with the class attribute `symmetry`.
    Quantum numbers are integers for `CyclicGroup` and `U1Group`, and tuples
    of integers for `ProductGroup`.
```

docstring の `qodulus` の項目を置き換える。

置き換え前:
```
    `qodulus`: An integer or None. If an integer, then all arithmetic on the
    quantum numbers is done modulo `qodulus`. If None then arithmetic on
    `qnums` is just usual integer arithmetic.
```
置き換え後:
```
    `symmetry`: An `AbelianGroup` that defines how quantum numbers combine.
    Defaults to `U1Group()`, i.e. usual integer arithmetic.
```

`charge` と `invar` の項目の中の、次の 2 箇所を置き換える。
- ``` ``sum_i k[i]*dirs[i] % qodulus ==\n    charge`` ``` → ``` ``symmetry.sum(symmetry.signed(k[i], dirs[i])\n    for i) == charge`` ```
- `symmetry determined by \`qodulus\`` → `symmetry determined by \`symmetry\``
- `` `charge`: An integer such that`` → `` `charge`: A quantum number such that``

docstring の直後(`# Methods for creating AbelianTensors.` のコメントの前)に次を追加する。

```python
    # The symmetry group fixed by the class, or None if it is given per
    # instance. Subclasses such as TensorZ2 set this.
    symmetry = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Support subclasses written for the old API, which fixed the symmetry
        # with a class attribute `qodulus = N` (or None for U(1)).
        if "qodulus" in vars(cls) and "symmetry" not in vars(cls):
            qodulus = vars(cls)["qodulus"]
            warnings.warn(
                f"Setting the class attribute `qodulus` on {cls.__name__} is "
                "deprecated. Set `symmetry = CyclicGroup(n)` or "
                "`symmetry = U1Group()` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cls.symmetry = (
                U1Group() if qodulus is None else CyclicGroup(qodulus)
            )

    @property
    def qodulus(self):
        """Deprecated. The order N of a `CyclicGroup` symmetry, or None for
        `U1Group`.
        """
        warnings.warn(
            "`qodulus` is deprecated. Use `symmetry` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(self.symmetry, CyclicGroup):
            return self.symmetry.n
        if isinstance(self.symmetry, U1Group):
            return None
        raise AttributeError(f"qodulus is not defined for {self.symmetry!r}.")
```

- [ ] **Step 5: `__init__` を書き換える**

シグネチャの `qodulus=None,` を `qodulus=_UNSET,` に、`charge=0,` を `charge=None,` に変え、`invar=True,` の後に `symmetry=None,` を追加する。位置引数の順序を保つため、`qodulus` の位置は変えない。

本体の次の部分を置き換える。

置き換え前:
```python
        shape = list(map(list, shape))
        qhape = list(map(list, qhape))
        if qodulus is not None:
            charge %= qodulus
            qhape = [[q % qodulus for q in qim] for qim in qhape]
        assert type(self).check_qhape_shape_match(qhape, shape)
        assert len(shape) == len(dirs)
        if invar and charge != 0:
            assert defval == 0
```
置き換え後:
```python
        symmetry = _resolve_symmetry(type(self), symmetry, qodulus)
        if charge is None:
            charge = symmetry.identity
        charge = symmetry.normalize(charge)
        shape = list(map(list, shape))
        qhape = [[symmetry.normalize(q) for q in qim] for qim in qhape]
        assert type(self).check_qhape_shape_match(qhape, shape)
        assert len(shape) == len(dirs)
        if invar and charge != symmetry.identity:
            assert defval == 0
```

末尾の `self.qodulus = qodulus` を `self.symmetry = symmetry` に置き換える。

- [ ] **Step 6: 内部でテンソルを作る箇所を機械的に置き換える**

Run:
```bash
sed -i 's/qodulus=self\.qodulus,/symmetry=self.symmetry,/' src/abeliantensors/abeliantensor.py
grep -c "symmetry=self.symmetry," src/abeliantensors/abeliantensor.py
sed -i 's/^            charge=0,$/            charge=self.symmetry.identity,/' src/abeliantensors/abeliantensor.py
grep -c "charge=self.symmetry.identity," src/abeliantensors/abeliantensor.py
```
Expected: 1 つ目の `grep -c` は `10`(`diag` ×2、`empty_like`、`matrix_dot` ×2、`matrix_eig` ×2、`matrix_svd` ×3)。2 つ目は `4`(`matrix_eig` の S と U、`matrix_svd` の U と S)。数が違う場合は、`grep -n "qodulus" src/abeliantensors/abeliantensor.py` で残りを確認し、以降の Step で扱う箇所かを判断する。

- [ ] **Step 7: `eye`、`initialize_with`、`from_ndarray`**

`eye` のシグネチャを `def eye(cls, dim, qim=None, qodulus=_UNSET, dtype=np.float64, symmetry=None):` に変える(79 桁を超えるので、ruff format に合わせて引数ごとに改行する)。本体の `res = cls(` の前に `symmetry = _resolve_symmetry(cls, symmetry, qodulus)` を追加し、`qodulus=qodulus,` を `symmetry=symmetry,` に変える。

`initialize_with` のシグネチャで `qodulus=None,` を `qodulus=_UNSET,` に、`charge=0,` を `charge=None,` に変え、`charge` の後に `symmetry=None,` を追加する。本体の次の部分を置き換える。

置き換え前:
```python
        shape = list(map(list, shape))
        qhape = list(map(list, qhape))
        if qodulus is not None:
            charge %= qodulus
            qhape = [[q % qodulus for q in qim] for qim in qhape]
        assert cls.check_qhape_shape_match(qhape, shape)
```
置き換え後:
```python
        symmetry = _resolve_symmetry(cls, symmetry, qodulus)
        if charge is None:
            charge = symmetry.identity
        charge = symmetry.normalize(charge)
        shape = list(map(list, shape))
        qhape = [[symmetry.normalize(q) for q in qim] for qim in qhape]
        assert cls.check_qhape_shape_match(qhape, shape)
```
`opt_args` の `"qodulus": qodulus,` を `"symmetry": symmetry,` に変える。末尾の `if res.charge == 0:` を `if res.charge == symmetry.identity:` に変える。

`from_ndarray` のシグネチャで `qodulus=None,` を `qodulus=_UNSET,` に、`charge=0,` を `charge=None,` に変え、`charge` の後に `symmetry=None,` を追加する。本体の次の部分を置き換える。

置き換え前:
```python
        shape, qhape = cls._sorted_shape_qhape(shape=shape, qhape=qhape)
        res = cls(
            shape,
            qhape=qhape,
            qodulus=qodulus,
```
置き換え後:
```python
        symmetry = _resolve_symmetry(cls, symmetry, qodulus)
        # Normalize before sorting, so that the blocks are read in the order
        # of the canonical quantum numbers, as to_ndarray writes them.
        qhape = [[symmetry.normalize(q) for q in qim] for qim in qhape]
        shape, qhape = cls._sorted_shape_qhape(shape=shape, qhape=qhape)
        res = cls(
            shape,
            qhape=qhape,
            symmetry=symmetry,
```

- [ ] **Step 8: 表示と二項演算**

`__repr__` を次の内容にする。

```python
    def __repr__(self):
        r = (
            "%r(%r, qhape=%r, symmetry=%r, sects=%r, dtype=%r, defval=%r, "
            "invar=%r, charge=%r, dirs=%r"
            % (
                type(self),
                self.shape,
                self.qhape,
                self.symmetry,
                self.sects,
                self.dtype,
                self.defval,
                self.invar,
                self.charge,
                self.dirs,
            )
        )
        return r
```

`__str__` の `f"qhape = {self.qhape}, qodulus = {self.qodulus}, "` を `f"qhape = {self.qhape}, symmetry = {self.symmetry}, "` に変える。

二項演算(`_do_binary_elementwise` 系の関数)と `allclose` の `assert self.qodulus == B.qodulus` を、どちらも `assert self.symmetry == B.symmetry` に変える。コメント `# They should also have the same qodulus.` と `# self and B should be of the same form and have the same qodulus.` の `qodulus` も `symmetry` に変える。

charge が違うときの警告文を置き換える。

置き換え前:
```python
                warnings.warn(
                    "Binary operation called on non-scalar tensors "
                    "with differing charges (%i and %i)."
                    % (self.charge, B.charge),
                    stacklevel=3,
                )
```
置き換え後:
```python
                warnings.warn(
                    "Binary operation called on non-scalar tensors "
                    f"with differing charges ({self.charge} and {B.charge}).",
                    stacklevel=3,
                )
```

- [ ] **Step 9: `conj`、`_qod_func`、`is_valid_key`、`compatible_indices`、`flip_dir`、`expand_dims`**

`conj` の次の部分を置き換える。

置き換え前:
```python
        res.charge = -res.charge
        if self.qodulus is not None:
            res.charge %= res.qodulus
        return res
```
置き換え後:
```python
        res.charge = self.symmetry.neg(res.charge)
        return res
```

`_qod_func` メソッドを丸ごと削除する。

`is_valid_key` の末尾を置き換える。

置き換え前:
```python
        key = map(opr.mul, self.dirs, key)
        s = sum(key)
        if self.qodulus is not None:
            s %= self.qodulus
        return s == self.charge
```
置き換え後:
```python
        g = self.symmetry
        return g.sum(map(g.signed, key, self.dirs)) == self.charge
```

`compatible_indices` の次の部分を置き換える。

置き換え前:
```python
        o_qim = [-1 * s_d * o_d * q for q in other.qhape[j]]
        if other.qodulus is not None:
            o_qim = [q % other.qodulus for q in o_qim]
```
置き換え後:
```python
        g = other.symmetry
        o_qim = [g.signed(q, -1 * s_d * o_d) for q in other.qhape[j]]
```

`flip_dir` の 2 箇所を置き換える。
- `res.qhape[axis] = [self._qod_func(-q) for q in res.qhape[axis]]` → `res.qhape[axis] = [self.symmetry.neg(q) for q in res.qhape[axis]]`
- `kf[axis] = self._qod_func(-kf[axis])` → `kf[axis] = self.symmetry.neg(kf[axis])`

`expand_dims` の量子数 0 を単位元に変える。

置き換え前:
```python
        res = self.empty_like()
        res.shape.insert(axis, [1])
        res.qhape.insert(axis, [0])
        res.dirs.insert(axis, direction)
        if not self.isscalar():
            for k, v in self.sects.items():
                new_k = list(k)
                new_k.insert(axis, 0)
                res[tuple(new_k)] = np.expand_dims(v, axis)
        elif res.charge == 0:
            res[(0,)] = np.array((res.defval,), dtype=res.dtype)
            res.defval = 0
        return res
```
置き換え後:
```python
        identity = self.symmetry.identity
        res = self.empty_like()
        res.shape.insert(axis, [1])
        res.qhape.insert(axis, [identity])
        res.dirs.insert(axis, direction)
        if not self.isscalar():
            for k, v in self.sects.items():
                new_k = list(k)
                new_k.insert(axis, identity)
                res[tuple(new_k)] = np.expand_dims(v, axis)
        elif res.charge == identity:
            res[(identity,)] = np.array((res.defval,), dtype=res.dtype)
            res.defval = 0
        return res
```
docstring の `This new index has only one \`qnum\`, 0,` を `This new index has only one \`qnum\`, the identity of the symmetry group,` に変える。

- [ ] **Step 10: `check_consistency` と `check_form_match`**

`check_consistency` の 2 箇所を置き換える。
- `assert all(q == self._qod_func(q) for q in sum(self.qhape, []))` → `assert all(q == self.symmetry.normalize(q) for q in sum(self.qhape, []))`(79 桁を超えるので、ruff format に任せて改行する)。直後に `assert self.charge == self.symmetry.normalize(self.charge)` を追加する。
- `if self.invar and (self.charge != 0 or not self.isscalar()):` → `if self.invar and (self.charge != self.symmetry.identity or not self.isscalar()):`

`check_form_match` のシグネチャの `qodulus=None,` を `symmetry=None,` に変え、docstring の `` and a `qodulus` can also be given`` を `` and a `symmetry` can also be given (by default the one of `tensor1`)`` に変える。本体の次の部分を置き換える。

置き換え前:
```python
        if tensor1 is not None:
            qhape1 = tensor1.qhape
            shape1 = tensor1.shape
            dirs1 = tensor1.dirs
```
置き換え後:
```python
        if tensor1 is not None:
            qhape1 = tensor1.qhape
            shape1 = tensor1.shape
            dirs1 = tensor1.dirs
            if symmetry is None:
                symmetry = tensor1.symmetry
        if symmetry is None:
            symmetry = U1Group()
```

置き換え前:
```python
            qim2 = [d1 * d2 * q for q in qim2]
            if qodulus is not None:
                qim2 = [q % qodulus for q in qim2]
```
置き換え後:
```python
            qim2 = [symmetry.signed(q, d1 * d2) for q in qim2]
```

(今のコードは、二項演算から呼ばれるときに `qodulus` を渡していないため、Z_N で向きの違うテンソルの形の比較に mod が効いていなかった。`tensor1` の対称性を既定にすることで、これも正しくなる。)

- [ ] **Step 11: `join_indices` と `split_indices`**

`join_indices` のブロックのキーを計算する部分を置き換える。

置き換え前:
```python
                k_part = map(opr.mul, k_part, dir_b)
                new_qnum = self._qod_func(sum(k_part) * dir_new)
```
置き換え後:
```python
                new_qnum = g.signed(g.sum(map(g.signed, k_part, dir_b)), dir_new)
```
そして、この `for k in valid_ks:` ループの直前に `g = self.symmetry` を追加する。

同じ関数の、新しい `qim` を計算する部分を置き換える。

置き換え前:
```python
            qhps = ([q * res.dirs[i] for q in res.qhape[i]] for i in batch)
            cartesian_product_of_qims = itt.product(*tuple(qhps))
            new_qim = map(sum, cartesian_product_of_qims)
            new_qim = (q * new_d for q in new_qim)
            new_qim = list(map(self._qod_func, new_qim))
```
置き換え後:
```python
            qhps = (
                [g.signed(q, res.dirs[i]) for q in res.qhape[i]] for i in batch
            )
            cartesian_product_of_qims = itt.product(*tuple(qhps))
            new_qim = [
                g.signed(g.sum(qcomb), new_d)
                for qcomb in cartesian_product_of_qims
            ]
```

`split_indices` の次の部分を置き換える。

置き換え前:
```python
                qcomb_flipped = tuple(map(opr.mul, qcomb, dir_batch))
                qnum = self._qod_func(sum(qcomb_flipped) * self.dirs[ind])
                if self.qodulus is not None:
                    qnum %= self.qodulus
```
置き換え後:
```python
                qnum = g.signed(
                    g.sum(map(g.signed, qcomb, dir_batch)), self.dirs[ind]
                )
```
そして、この `for ind, dims, qims, dir_batch in zip(` ループの直前に `g = self.symmetry` を追加する。

`split_indices` の引数として渡された `qims` は、利用者からの入力なので正規化する。入力を整形する `if isinstance(indices, Iterable): ... else: ...` のブロックの直後、`if not indices:` の直前に、次を追加する。

```python
        qim_batches = [
            [[self.symmetry.normalize(q) for q in qim] for qim in qim_batch]
            for qim_batch in qim_batches
        ]
```

- [ ] **Step 12: `trace`、`multiply_diag`、`matrix_dot`、`matrix_eig`**

`trace` の次の部分を置き換える。

置き換え前:
```python
            qnum = k[axis1] - k[axis2]
            if self.qodulus is not None:
                qnum %= self.qodulus
            if qnum == 0:
```
置き換え後:
```python
            # dirs of axis1 and axis2 are opposite here, so the block lies on
            # the diagonal iff the two quantum numbers are equal.
            if k[axis1] == k[axis2]:
```
docstring の ``charge != 0`` を ``charge != identity`` に変える。

`multiply_diag` の次の箇所を置き換える。
- `assert diag_vect.qodulus == self.qodulus` → `assert diag_vect.symmetry == self.symmetry`
- `assert diag_vect.charge == 0` → `assert diag_vect.charge == self.symmetry.identity`
- 末尾の 2 つの代入を次に置き換える。

置き換え前:
```python
        res.qhape[axis] = [
            self._qod_func(q + diag_vect.charge) for q in diag_vect.qhape[0]
        ]
        res.charge = self._qod_func(self.charge + diag_vect.charge)
```
置き換え後:
```python
        g = self.symmetry
        res.qhape[axis] = [
            g.add(q, diag_vect.charge) for q in diag_vect.qhape[0]
        ]
        res.charge = g.add(self.charge, diag_vect.charge)
```

`matrix_dot` の次の箇所を置き換える。
- `assert self.qodulus == other.qodulus` → `assert self.symmetry == other.symmetry`、その直後の行に `g = self.symmetry` を追加する。
- `res_charge = self._qod_func(self.charge + other.charge)` → `res_charge = g.add(self.charge, other.charge)`
- ベクトル × 行列:
  ```python
                        b_qnum = g.add(
                            g.signed(sum_qnum, flux),
                            g.signed(other.charge, other.dirs[1]),
                        )
  ```
- 行列 × ベクトル:
  ```python
                        a_qnum = g.add(
                            g.signed(sum_qnum, flux),
                            g.signed(self.charge, self.dirs[0]),
                        )
  ```
- 行列 × 行列:
  ```python
                        a_qnum = g.add(
                            g.signed(sum_qnum, a_flux),
                            g.signed(self.charge, self.dirs[0]),
                        )
                        b_qnum = g.add(
                            g.signed(sum_qnum, b_flux),
                            g.signed(other.charge, other.dirs[1]),
                        )
  ```

(元のコード `sum_qnum * flux + other.dirs[1] * other.charge` と同じ意味。`flux` と `dirs` はどちらも ±1。)

`matrix_eig` の `assert self.charge == 0` を `assert self.charge == self.symmetry.identity` に変える。

- [ ] **Step 13: 置き換え漏れがないことを確認する**

Run: `grep -nE "qodulus|_qod_func|charge == 0|charge != 0" src/abeliantensors/abeliantensor.py`
Expected: 出てくるのは次だけ。`_UNSET` / `_resolve_symmetry` / `__init_subclass__` / `qodulus` プロパティの中、`__init__`・`eye`・`initialize_with`・`from_ndarray` のシグネチャと `_resolve_symmetry` の呼び出し、docstring の非推奨の説明。それ以外に残っていたら、上の Step の対応する置き換えに倣って直す。

- [ ] **Step 14: `symmetrytensors.py` を書き換える**

`src/abeliantensors/symmetrytensors.py` を次の内容にする。

```python
from collections.abc import Iterable

from .abeliantensor import AbelianTensor
from .symmetry import CyclicGroup, U1Group


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
```

`src/abeliantensors/__init__.py` の `from .symmetrytensors import ...` 行に `FiniteAbelianTensor` を加え、`__all__` にも `"FiniteAbelianTensor"` を加える(アルファベット順の位置)。

- [ ] **Step 15: 新しいテストが通ることを確認する**

Run: `uv run pytest tests/test_symmetry_tensors.py -q -p no:randomly 2>&1 | tail -15`
Expected: すべて passed

失敗したら、エラーの内容から、Step 3〜14 のどの置き換えに漏れや誤りがあるかを調べて直す(superpowers:systematic-debugging)。

- [ ] **Step 16: 既存のテストがすべて通ることを確認する**

Run: `uv run pytest -q 2>&1 | tail -5`
Expected: 既存の 136 件と、Task 2・Task 3 で追加したテストがすべて成功し、警告(とくに `DeprecationWarning`)は 0 件。

`DeprecationWarning` がエラーとして出た場合、それはライブラリ内部のどこかに `qodulus=` を渡している箇所が残っていることを意味する。トレースバックからその箇所を見つけ、`symmetry=` に直す。

Run: `for s in 1 2 3; do uv run pytest -q -p randomly --randomly-seed=$s 2>&1 | tail -1; done`
Expected: 3 回とも全件成功

- [ ] **Step 17: 設計書に旧方式のサブクラスの扱いを追記する**

設計書の「### 2. `AbelianTensor` を群オブジェクトに対応させる」の「**`qodulus` の非推奨化**」の箇条書きの末尾に、次を追加する。

```markdown
- クラス属性 `qodulus = N`(または `None`)を持つ旧方式のサブクラスは、`AbelianTensor.__init_subclass__` で `symmetry = CyclicGroup(N)`(または `U1Group()`)に変換し、クラスを定義した時点で `DeprecationWarning` を出す。README が「1 行で新しい巡回群のクラスを作れる」と案内してきた書き方なので、これがエラーにならずに U(1) として動いてしまうことを防ぐ。
- `from_ndarray` は、`qhape` を正規化してから並べ替える。今のコードは正規化する前の値の順に並べ替えるため、Z_N に正規化されていない `qhape`(例: Z₃ の `[-1, 0, 1]`)を渡すと、`to_ndarray` で元の配列に戻らない。
```

- [ ] **Step 18: lint を確認してコミットする**

Run: `uv run ruff check && uv run ruff format && uv run ruff format --check && uv run pytest -q 2>&1 | tail -1`
Expected: すべて成功

```bash
git add src/abeliantensors tests/test_symmetry_tensors.py docs/superpowers/specs/2026-10-13-general-abelian-symmetry-design.md
git commit -m "Delegate quantum number arithmetic to symmetry group objects

AbelianTensor now holds an AbelianGroup in .symmetry instead of an integer
qodulus. The qodulus argument, attribute and class attribute still work but
emit DeprecationWarning. Also normalize qhape before sorting in
from_ndarray, pass dtype through in eye, and fix __repr__ formatting.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `TensorZ2Z2` / `TensorZ2Z3` とテストの拡張

**Files:**
- Modify: `src/abeliantensors/symmetrytensors.py`(末尾にクラスを追加)
- Modify: `src/abeliantensors/__init__.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_tensors.py:427-447`(`test_multiply_diag`)、`tests/test_tensors.py:703-735`(`test_eig`)
- Modify: `tests/test_symmetry_tensors.py`(テストを追加)

**Interfaces:**
- Consumes: Task 3 の `FiniteAbelianTensor` と `ProductGroup`。
- Produces: `TensorZ2Z2`、`TensorZ2Z3`(`from abeliantensors import ...`)。conftest のフィクスチャ `qid`(テスト対象のクラスの単位元。`Tensor` と U(1) では 0)。`--tensorclass TensorZ2Z2` / `TensorZ2Z3` の指定。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_symmetry_tensors.py` の import に `ProductGroup`、`TensorZ2Z2`、`TensorZ2Z3` を加え、末尾に次を追加する。

```python
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
```

- [ ] **Step 2: 失敗することを確認する**

Run: `uv run pytest tests/test_symmetry_tensors.py -q -p no:randomly 2>&1 | tail -3`
Expected: `ImportError: cannot import name 'TensorZ2Z2'` で収集エラー。

- [ ] **Step 3: クラスを追加する**

`src/abeliantensors/symmetrytensors.py` の import を `from .symmetry import CyclicGroup, ProductGroup, U1Group` に変え、ファイルの末尾に次を追加する。

```python
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
```

`src/abeliantensors/__init__.py` の `from .symmetrytensors import ...` に `TensorZ2Z2`、`TensorZ2Z3` を加え、`__all__` にも加える(アルファベット順の位置)。

- [ ] **Step 4: 新しいテストが通ることを確認する**

Run: `uv run pytest tests/test_symmetry_tensors.py -q -p no:randomly 2>&1 | tail -3`
Expected: すべて passed

- [ ] **Step 5: conftest を群に対応させる**

`tests/conftest.py` を次のように変える。

import を次に置き換える。
```python
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
```

`parse_tensorclass` の本体を次に置き換える。
```python
    try:
        return TENSORCLASSES[s.lower().strip()]
    except KeyError:
        msg = f"Unknown tensor class name: {s}"
        raise ValueError(msg) from None
```

`pytest_generate_tests` の既定のクラスの一覧を次に変える。
```python
    default_classes = [
        "Tensor",
        "TensorZ2",
        "TensorU1",
        "TensorZ3",
        "TensorZ2Z2",
        "TensorZ2Z3",
    ]
```

`n_qnums` フィクスチャの前に補助関数を追加し、`n_qnums` を書き換える。
```python
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
```

`rqhape` フィクスチャのシグネチャを `def rqhape(tensorclass, n_qnums):` に変え、内側の `_rqhape` の本体を次に置き換える(docstring はそのまま残し、「For finite groups, quantum numbers are randomly picked from the group's elements.」と書き直す)。
```python
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
```

`rcharge` フィクスチャのシグネチャを `def rcharge(tensorclass):` に変え、内側の `_rcharge` の本体を次に置き換える。
```python
        elements = _qnum_elements(tensorclass)
        if elements is not None:
            return elements[np.random.randint(len(elements))]
        return np.random.randint(low=low, high=high)
```
docstring は「a random element of the group for finite groups, otherwise a random integer between `low` and `high`」と書き直す。

- [ ] **Step 6: `test_tensors.py` の `charge=0` を単位元に変える**

`test_multiply_diag`(427 行目付近)と `test_eig`(703 行目付近)のシグネチャに `qid` を加える(`test_multiply_diag` は `rtensor` の後、`test_eig` は `rtensor,` の行の後)。そのうえで、次の 2 箇所を置き換える。

- `shape=D_shape, qhape=D_qhape, dirs=D_dirs, invar=False, charge=0` → `shape=D_shape, qhape=D_qhape, dirs=D_dirs, invar=False, charge=qid`
- `T = rtensor(shape=shp, qhape=qhp, dirs=dirs, charge=0)` → `T = rtensor(shape=shp, qhape=qhp, dirs=dirs, charge=qid)`

Run: `grep -nE "charge=0|charge == 0" tests/test_tensors.py`
Expected: 出力なし

- [ ] **Step 7: テスト全体を実行する**

Run: `uv run pytest -q -p randomly --randomly-seed=1 2>&1 | tail -5`
Expected: すべて成功し、警告は 0 件。test_tensors のパラメータ化されたテストが、6 クラス分(既存の約 1.5 倍)実行される。

失敗した場合は、`--tensorclass TensorZ2Z3` で失敗しているテストだけを再現し、systematic-debugging で原因を調べる。テストが整数の量子数を前提にした計算をしている場合(`q + 1` など)は、テスト側を群のメソッドで書き直し、ライブラリ側の問題ならライブラリを直す。

Run: `/usr/bin/time -f "%e s" uv run pytest -q 2>&1 | tail -2`
Expected: 全件成功。実行時間を記録する。Task 3 の時点の時間の 3 倍を超える場合は、`rshape` で有限群のセクター数を位数に固定している箇所(`dim_dim = n_qnums`)を、位数が 4 を超える群に限って `np.random.randint(1, n_qnums + 1)` にする(rqhape は常に qhape を明示するので、既定の qhape の規則には影響しない)。この変更をした場合は、進捗の記録に判断として残す。

Run: `for s in 2 3 4; do uv run pytest -q -p randomly --randomly-seed=$s 2>&1 | tail -1; done`
Expected: 3 回とも全件成功

- [ ] **Step 8: lint を確認してコミットする**

Run: `uv run ruff check && uv run ruff format && uv run ruff format --check`
Expected: 成功

```bash
git add src/abeliantensors tests
git commit -m "Add TensorZ2Z2 and TensorZ2Z3; run the tensor test suite on them

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: 変更後の性能を測る

**Files:**
- Modify: `docs/superpowers/specs/2026-10-13-general-abelian-symmetry-design.md`(計測結果の表)
- Modify(20% を超えた場合のみ): `src/abeliantensors/symmetry.py`

**Interfaces:**
- Consumes: Task 1 の `benchmarks/trg_bench.py` と、設計書に記録した基準値。

- [ ] **Step 1: 変更後の値を 2 回測る**

Run: `for i in 1 2; do uv run --group demo python benchmarks/trg_bench.py; done`
Expected: 3 行の表が 2 回出る。各 chi・各クラスについて、2 回のうち小さいほうを採る。

- [ ] **Step 2: 基準値と比べる**

各 chi について `TensorZ2 の変化 = (変更後 / 変更前 - 1) × 100 %` を計算する。

- すべての chi で +20% 以下なら Step 4 へ進む。
- どれかの chi で +20% を超えたら Step 3 へ進む。

- [ ] **Step 3(+20% を超えた場合のみ): プロファイルを取り、速くする**

Run:
```bash
uv run --group demo python -c "
import cProfile, pstats, sys
sys.path.insert(0, 'benchmarks'); sys.path.insert(0, 'demo')
import numpy as np, trg_bench
from trg_demo import ising_tensor
from abeliantensors import TensorZ2
A0 = ising_tensor(1.0, 0.0, np.log(1 + np.sqrt(2)) / 2)
cProfile.run('trg_bench.time_trg_step(TensorZ2, 24, A0)', '/tmp/claude-prof.out')
pstats.Stats('/tmp/claude-prof.out').sort_stats('cumulative').print_stats(25)
" 2>&1 | head -60
```

`symmetry.py` のメソッド(`signed`、`sum`、`normalize`、`add`)が上位に出ている場合は、まず `CyclicGroup` と `U1Group` に `signed` の速い版を追加する。

```python
    def signed(self, a, d):
        if d == 1:
            return a
        if d == -1:
            return -a % self.n
        raise ValueError(f"direction must be +1 or -1, got {d!r}")
```
(`U1Group` では `return -a`。)

さらに、`is_valid_key` が上位にあれば、`CyclicGroup` / `U1Group` に限って向きを掛けた和を一度に計算するメソッドを足すことを検討する。ただし、群の API を変える場合は、設計書の §1 にも追記する。

修正するたびに `uv run pytest -q tests/test_symmetry.py tests/test_symmetry_tensors.py` を実行してから、Step 1 に戻って測り直す。2 回対策をしても 20% を超える場合は、作業を止めて、測定値とプロファイルの上位を添えて利用者に相談する。

- [ ] **Step 4: 設計書に変更後の値を記録する**

設計書の「計測結果」の表の、変更後の列と「TensorZ2 の変化」の列を埋める(例: `+7.3%`)。Step 3 で対策をした場合は、表の下に「対策: …」と 1〜2 行で書く。

- [ ] **Step 5: コミットする**

```bash
git add docs/superpowers/specs/2026-10-13-general-abelian-symmetry-design.md src/abeliantensors/symmetry.py
git commit -m "Record TRG benchmark after the symmetry refactor

TensorZ2 change vs. baseline: chi=8 <+x%>, chi=16 <+y%>, chi=24 <+z%>.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
(`<...>` は Step 2 の実際の値で置き換える。)

---

### Task 6: README とドキュメント

**Files:**
- Modify: `README.md`
- Modify: `docs/source/index.rst`

- [ ] **Step 1: README を更新する**

冒頭の説明文(5 行目付近)の `implements U(1) and ℤₙ symmetry` を `implements abelian symmetry` に変え、その段落の末尾に次の 1 文を加える。
```markdown
Supported symmetry groups include U(1), ℤₙ, and direct products of cyclic groups such as ℤ₂×ℤ₂.
```

48〜50 行目の段落(`abeliantensors exports classes \`TensorU1\`, ...`)を次に置き換える。

````markdown
abeliantensors exports classes `TensorU1`, `TensorZ2`, `TensorZ3`,
`TensorZ2Z2`, and `TensorZ2Z3`. The symmetry of each class is an abelian group
object, and other finite abelian groups can be implemented with one-liners:

```python
from abeliantensors import CyclicGroup, FiniteAbelianTensor, ProductGroup

class TensorZ5(FiniteAbelianTensor):
    symmetry = CyclicGroup(5)

class TensorZ2Z4(FiniteAbelianTensor):
    symmetry = ProductGroup(CyclicGroup(2), CyclicGroup(4))
```

Quantum numbers are integers for `CyclicGroup` and `U1Group`, and tuples such
as `(1, 0)` for `ProductGroup`. The old way of fixing a cyclic group with a
class attribute `qodulus = 5` still works but is deprecated.
````

(その直後の「abeliantensors also exports a class called `Tensor`, ...」以降の文はそのまま残す。)

173〜175 行目の「`TensorU1`, `TensorZ2`, and `TensorZ3` implement symmetric tensors for the three different symmetry groups」を「`TensorU1`, `TensorZ2`, `TensorZ3`, `TensorZ2Z2`, and `TensorZ2Z3` implement symmetric tensors for different symmetry groups」に変える。

README の末尾のテストの節で、`--tensorclass` に指定できるクラスを列挙している箇所があれば、`TensorZ2Z2` と `TensorZ2Z3` を加える。

- [ ] **Step 2: API リファレンスを更新する**

`docs/source/index.rst` の `TensorZN` の `autoclass` ブロックの後に、同じ書式(`:members:`、`:no-undoc-members:`、`:show-inheritance:`)で次を追加する。
- `abeliantensors.symmetrytensors.TensorZ2Z2`
- `abeliantensors.symmetrytensors.TensorZ2Z3`
- `abeliantensors.symmetrytensors.FiniteAbelianTensor`

`AbelianTensor` の `autoclass` ブロックの前に、次の節を追加する。

```rst
Symmetry groups
-------------------------------------

.. autoclass:: abeliantensors.symmetry.AbelianGroup
   :members:
   :no-undoc-members:

.. autoclass:: abeliantensors.symmetry.CyclicGroup
   :show-inheritance:

.. autoclass:: abeliantensors.symmetry.U1Group
   :show-inheritance:

.. autoclass:: abeliantensors.symmetry.ProductGroup
   :show-inheritance:
```

- [ ] **Step 3: ドキュメントをビルドする**

Run: `uv run --group docs sphinx-build -W -b html docs/source docs/build/html 2>&1 | tail -3`
Expected: `build succeeded.`

警告が出た場合は、その内容に従って docstring の RST 書式を直す(例: 二重のバッククォート、箇条書きの前の空行)。

- [ ] **Step 4: 最終確認とコミット**

Run:
```bash
uv run pytest -q 2>&1 | tail -1
uv run ruff check && uv run ruff format --check
uv lock --check
git status --short
```
Expected: テストは全件成功、ruff は成功、ロックファイルは最新。`git status` には README と index.rst だけが出る。

```bash
git add README.md docs/source/index.rst
git commit -m "Document general abelian symmetries and the new tensor classes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
