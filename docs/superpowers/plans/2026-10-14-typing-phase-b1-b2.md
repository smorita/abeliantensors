# 型注釈 フェーズ B-1 + B-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `symmetry.py`、`symmetrytensors.py`、`tensorcommon.py` の公開 API に型注釈を付ける。`svd` / `eig` / `split` / `to_matrix` の戻り値が `@overload` で正しく推論されるようにし、それを `tests/typing/` で確かめる。

**Architecture:** 群のクラスは PEP 695 のジェネリクス(`AbelianGroup[Q]`)にして、量子数の型(`int` またはタプル)をパラメータにする。`TensorCommon` のメソッドは `Self` を返すようにし、フラグで戻り値の形が変わるメソッドは `Literal` による `@overload` で場合分けする。ruff の `ANN` ルールで注釈の網羅を検査し、まだ注釈を付けていない `abeliantensor.py` と `tensor.py` だけを一時的に除外する。

**Tech Stack:** pyright 1.1.414(`"standard"` モード)、ruff(`ANN` ルール)、`typing.Self` / `overload` / `Literal` / `assert_type`、Python 3.12 のジェネリクスの構文。

**Spec:** `docs/superpowers/specs/2026-10-13-typing-design.md`(§4 の B-1、B-2、§5、「進め方」と「型注釈の網羅の検査」の追記)

## Global Constraints

- 作業ブランチは `typing-b`(作成済み)。コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` を付ける。
- 実行時の動作は変えない。公開 API(メソッド名、引数名、戻り値の形)も変えない。
- pyright の検査対象は `src/` と `tests/typing/`。`"standard"` モードでエラー 0 件を保つ。
- 新しい `# pyright: ignore` は原則として使わない。使う必要があると判断したら、理由をコメントに書き、設計書の「無視した箇所」に加え、進捗の記録に判断として残す。
- ruff の `ANN` ルールは `ANN001`、`ANN002`、`ANN003`、`ANN201`、`ANN204`、`ANN206`。`src/` 以外(`tests/`、`demo/`、`benchmarks/`)には適用しない。
- 各コミットの前に `uv run pytest -q`、`uv run ruff check`、`uv run ruff format --check`、`uv run pyright` を通す。

## Review Focus

1. **`TensorCommon` の型チェッカー向けの宣言が実行時に存在してしまう**:型を付けると、サブクラスにしかないメソッド(`sqrt`、`multiply_diag` など)の宣言を `if TYPE_CHECKING:` のブロックに足す必要が出てくる。どれも実行時の `TensorCommon` には存在してはならない(`Tensor` が `np.ndarray` から受け継ぐメソッドを隠すため)。→ Task 2 で、既存のテスト `test_tensorcommon_declarations_are_type_checking_only` の名前の一覧を、宣言した名前すべてに広げる。
2. **フラグを変数で渡す呼び出し**:`T.svd(0, 1, return_rel_err=flag)` のように `bool` 型の変数を渡すコードも、型エラーにならずに通るべき(戻り値は和型になる)。→ Task 3 の `tests/typing/check_tensorcommon.py` に、`bool` の変数で呼ぶ `assert_type` を入れる。
3. **`np.ndarray` のサブクラスである `Tensor` での推論**:`Tensor.random(...)` や `D.svd(...)` が `ndarray` ではなく `Tensor` と推論されるべき。→ Task 2 と Task 3 のチェックファイルに、`Tensor` での `assert_type` を入れる。
4. **群のジェネリクス化が実行時の動作を変える**:`class AbelianGroup[Q]:` は実行時にも型パラメータを持つクラスになる。等価性、ハッシュ、コピー、pickle が今までどおり動くべき。→ Task 1 で、既存の `tests/test_symmetry.py` がすべて通ることに加え、`pickle` で往復させるテストを加える。
5. **`ANN` の検査が実際に働いていない**:設定の誤りで、注釈のない公開関数があっても ruff が何も言わない状態になりうる。→ Task 1 で、注釈のない関数を含む一時ファイルで ruff が指摘することを確かめる。

---

## File Structure

| ファイル | 作業 | 内容 |
|---|---|---|
| `pyproject.toml` | 修正 | ruff の `ANN` ルールと一時的な除外、pyright の `include` に `tests/typing` |
| `src/abeliantensors/py.typed` | 新規 | 空のファイル(PEP 561) |
| `src/abeliantensors/_typing.py` | 新規 | 型の別名 `Qnum`、`Dim`、`Qim`、`Indices` |
| `src/abeliantensors/symmetry.py` | 修正 | ジェネリクス化と型注釈 |
| `src/abeliantensors/symmetrytensors.py` | 修正 | 型注釈 |
| `src/abeliantensors/tensorcommon.py` | 修正 | 型注釈、`@overload`、`TYPE_CHECKING` の宣言の追加 |
| `tests/typing/check_symmetry.py` | 新規 | 群の推論結果の確認 |
| `tests/typing/check_tensorcommon.py` | 新規 | `TensorCommon` のメソッドの推論結果の確認 |
| `tests/test_symmetry.py` | 修正 | pickle の往復のテスト |
| `tests/test_errors.py` | 修正 | 型チェッカー向けの宣言の名前の一覧を広げる |
| `docs/superpowers/specs/2026-10-13-typing-design.md` | 修正 | 群のジェネリクス化、`TYPE_CHECKING` の宣言の方針を追記 |

---

### Task 1: B-1(検査の仕組み、`py.typed`、`_typing.py`、群とテンソルのサブクラスの型注釈)

**Files:**
- Modify: `pyproject.toml`
- Create: `src/abeliantensors/py.typed`、`src/abeliantensors/_typing.py`、`tests/typing/check_symmetry.py`
- Modify: `src/abeliantensors/symmetry.py`、`src/abeliantensors/symmetrytensors.py`、`tests/test_symmetry.py`

**Interfaces:**
- Produces:
  - `abeliantensors._typing`: `Qnum = int | tuple[int, ...]`、`Dim = list[int]`、`Qim = list[Qnum]`、`Indices = int | Sequence[int]`
  - `AbelianGroup[Q]`(`identity: Q`、`add(a: Q, b: Q) -> Q`、`neg(a: Q) -> Q`、`normalize(q: object) -> Q`、`elements() -> list[Q] | None`、`signed(a: Q, d: int) -> Q`、`sum(qs: Iterable[Q]) -> Q`)
  - `CyclicGroup(AbelianGroup[int])`、`U1Group(AbelianGroup[int])`、`ProductGroup(AbelianGroup[tuple[int, ...]])`

- [ ] **Step 1: ruff の `ANN` ルールと、pyright の検査対象を設定する**

`pyproject.toml` の `[tool.ruff.lint]` の `select` を次に変える。

```toml
select = [
    "E", "F", "W", "I", "UP", "NPY", "B",
    # Type annotations on the public API of src/ (see the typing spec).
    "ANN001", "ANN002", "ANN003", "ANN201", "ANN204", "ANN206",
]
```

`[tool.ruff.lint.per-file-ignores]` を次にする。

```toml
[tool.ruff.lint.per-file-ignores]
# `l` is used conventionally for lists of indices/legs.
"src/**" = ["E741"]
"tests/**" = ["E741", "ANN"]
"demo/**" = ["ANN"]
"benchmarks/**" = ["ANN"]
# Not annotated yet: removed in phase B-3 and B-4 of the typing spec.
"src/abeliantensors/abeliantensor.py" = ["E741", "ANN"]
"src/abeliantensors/tensor.py" = ["ANN"]
# Removed in Task 3 of this plan.
"src/abeliantensors/tensorcommon.py" = ["ANN"]
```

(`per-file-ignores` は、ファイルに一致するすべての項目が合わさる。`abeliantensor.py` は `src/**` の `E741` も受けるが、明示のため `E741` も書いておく。)

`[tool.pyright]` の `include` を `include = ["src", "tests/typing"]` にする。

- [ ] **Step 2: `ANN` の検査が働くことを確かめる**

```bash
printf 'def probe(x):\n    return x\n' > src/abeliantensors/_ann_probe.py
uv run ruff check src/abeliantensors/_ann_probe.py; echo "exit=$?"
rm src/abeliantensors/_ann_probe.py
```
Expected: `ANN001`(`x`)と `ANN201`(`probe`)が指摘され、`exit=1`。

Run: `uv run ruff check --statistics 2>&1 | tail -6`
Expected: `symmetry.py` と `symmetrytensors.py` の約 82 件が `ANN` で指摘される(ほかのファイルは除外されている)。

- [ ] **Step 3: 失敗する型チェックを書く**

`tests/typing/check_symmetry.py`:

```python
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
```

Run: `uv run pyright 2>&1 | tail -3`
Expected: `assert_type` の不一致などのエラーが出る(群に型注釈がないため)。

- [ ] **Step 4: pickle の往復のテストを書く**

`tests/test_symmetry.py` の import に `import pickle` を加え、末尾に追加する。

```python
@pytest.mark.parametrize("g", [*FINITE_GROUPS, U1Group()], ids=repr)
def test_pickle_roundtrip(g):
    h = pickle.loads(pickle.dumps(g))
    assert h == g
    assert hash(h) == hash(g)
    assert h.elements() == g.elements()
```

Run: `uv run pytest -q -p no:randomly tests/test_symmetry.py -k pickle 2>&1 | tail -1`
Expected: 5 passed(今の動作を固定するテスト。ジェネリクス化の後も通ることを確かめる)。

- [ ] **Step 5: `py.typed` と `_typing.py` を作る**

```bash
touch src/abeliantensors/py.typed
```

`src/abeliantensors/_typing.py`:

```python
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
```

`uv_build` は `src/abeliantensors/` の中のファイルをそのままパッケージに含めるので、`py.typed` のための設定は不要。

- [ ] **Step 6: `symmetry.py` に型注釈を付ける**

import を次にする。

```python
import itertools as itt
import numbers
from collections.abc import Iterable
from typing import Any, Self
```

次の表のとおり、定義の行を置き換える。docstring と本体は、表の下に書いた 2 箇所を除いて変えない。

| 置き換え前 | 置き換え後 |
|---|---|
| `def _as_int(q):` | `def _as_int(q: object) -> int:` |
| `class AbelianGroup:` | `class AbelianGroup[Q]:` |
| `    identity = None`(`AbelianGroup` の中) | `    identity: Q` |
| `    def add(self, a, b):`(`AbelianGroup` の中) | `    def add(self, a: Q, b: Q) -> Q:` |
| `    def neg(self, a):`(同) | `    def neg(self, a: Q) -> Q:` |
| `    def normalize(self, q):`(同) | `    def normalize(self, q: object) -> Q:` |
| `    def elements(self) -> list \| None:`(同) | `    def elements(self) -> list[Q] \| None:` |
| `    def signed(self, a, d):` | `    def signed(self, a: Q, d: int) -> Q:` |
| `    def sum(self, qs):`(同) | `    def sum(self, qs: Iterable[Q]) -> Q:` |
| `    def _key(self):`(同) | `    def _key(self) -> tuple[Any, ...]:` |
| `    def __eq__(self, other):` | `    def __eq__(self, other: object) -> bool:` |
| `    def __hash__(self):` | `    def __hash__(self) -> int:` |
| `    def __copy__(self):` | `    def __copy__(self) -> Self:` |
| `    def __deepcopy__(self, memo):` | `    def __deepcopy__(self, memo: dict[int, Any]) -> Self:` |
| `class CyclicGroup(AbelianGroup):` | `class CyclicGroup(AbelianGroup[int]):` |
| `    def __init__(self, n):` | `    def __init__(self, n: int) -> None:` |
| `CyclicGroup` / `U1Group` の `def add(self, a, b):` | `def add(self, a: int, b: int) -> int:` |
| `CyclicGroup` / `U1Group` の `def neg(self, a):` | `def neg(self, a: int) -> int:` |
| `CyclicGroup` / `U1Group` の `def sum(self, qs):` | `def sum(self, qs: Iterable[int]) -> int:` |
| `CyclicGroup` / `U1Group` の `def normalize(self, q):` | `def normalize(self, q: object) -> int:` |
| `CyclicGroup` の `def elements(self) -> list \| None:` | `def elements(self) -> list[int] \| None:` |
| `CyclicGroup` / `U1Group` / `ProductGroup` の `def _key(self):` | `def _key(self) -> tuple[Any, ...]:` |
| `CyclicGroup` / `U1Group` / `ProductGroup` の `def __repr__(self):` | `def __repr__(self) -> str:` |
| `class U1Group(AbelianGroup):` | `class U1Group(AbelianGroup[int]):` |
| `class ProductGroup(AbelianGroup):` | `class ProductGroup(AbelianGroup[tuple[int, ...]]):` |
| `    def __init__(self, *factors):` | `    def __init__(self, *factors: "CyclicGroup \| U1Group") -> None:` |
| `ProductGroup` の `def add(self, a, b):` | `def add(self, a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:` |
| `ProductGroup` の `def neg(self, a):` | `def neg(self, a: tuple[int, ...]) -> tuple[int, ...]:` |
| `ProductGroup` の `def normalize(self, q):` | `def normalize(self, q: object) -> tuple[int, ...]:` |
| `ProductGroup` の `def elements(self) -> list \| None:` | `def elements(self) -> list[tuple[int, ...]] \| None:` |

`ProductGroup.__init__` の `self.factors = tuple(factors)` を、型を明示した `self.factors: tuple[CyclicGroup | U1Group, ...] = tuple(factors)` にする。

`ProductGroup.elements` の本体を次に置き換える(`None` の確認を、pyright が型を絞り込める書き方にする。結果は同じ)。

```python
        factor_elements: list[list[int]] = []
        for f in self.factors:
            elements = f.elements()
            if elements is None:
                return None
            factor_elements.append(elements)
        return list(itt.product(*factor_elements))
```

(`AbelianGroup` の `identity = None` を `identity: Q`(値なしの宣言)に変えるので、基底クラスそのものの `identity` は実行時に存在しなくなる。`AbelianGroup` は直接使わない抽象クラスで、サブクラスはすべて `identity` を設定しているので影響はない。)

- [ ] **Step 7: `symmetrytensors.py` に型注釈を付ける**

import を次にする。

```python
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Self

import numpy as np

from ._typing import Indices, Qnum
from .abeliantensor import AbelianTensor
from .symmetry import CyclicGroup, ProductGroup, U1Group
```

(`uv run ruff check --fix` で並び順を直す。)

`FiniteAbelianTensor` の定義の行を次のように置き換える。

| 置き換え前 | 置き換え後 |
|---|---|
| `def __init__(self, shape, *args, qhape=None, **kwargs):` | `def __init__(self, shape: Sequence[Sequence[int]], *args: Any, qhape: Sequence[Sequence[Qnum]] \| None = None, **kwargs: Any) -> None:` |
| `def eye(cls, dim, qim=None, *args, **kwargs):` | `def eye(cls, dim: Sequence[int], qim: Sequence[Qnum] \| None = None, *args: Any, **kwargs: Any) -> Self:` |
| `def initialize_with(cls, numpy_func, shape, *args, qhape=None, **kwargs):` | `def initialize_with(cls, numpy_func: Callable[..., Any], shape: Sequence[Sequence[int]], *args: Any, qhape: Sequence[Sequence[Qnum]] \| None = None, **kwargs: Any) -> Self:` |
| `def from_ndarray(cls, a, *args, shape=None, qhape=None, **kwargs):` | `def from_ndarray(cls, a: np.ndarray, *args: Any, shape: Sequence[Sequence[int]] \| None = None, qhape: Sequence[Sequence[Qnum]] \| None = None, **kwargs: Any) -> Self:` |
| `def _dim_to_qim(cls, dim):` | `def _dim_to_qim(cls, dim: Sequence[int]) -> list[Qnum]:` |
| `def _shape_to_qhape(cls, shape):` | `def _shape_to_qhape(cls, shape: Sequence[Sequence[int]]) -> list[list[Qnum]]:` |
| `def split_indices(self, indices, dims, qims=None, dirs=None):` | `def split_indices(self, indices: Indices, dims: Sequence[Any], qims: Sequence[Any] \| None = None, dirs: Any = None) -> Self:` |

`from_ndarray` の本体の `if qhape is None:` を `if qhape is None and shape is not None:` に変える。`shape` も省略された場合は、親クラスの `from_ndarray` が「shape と qhape が必要」という `TypeError` を出す(今は `_shape_to_qhape(None)` の中で分かりにくい `TypeError` になっている)。

`_dim_to_qim` の本体で、`return list(elements)` の行はそのまま(`elements` は `list[Any]`)。

(長い行は `uv run ruff format` で整形する。)

- [ ] **Step 8: 確認する**

Run: `uv run ruff check --fix -q; uv run ruff format -q; uv run ruff check && uv run pyright 2>&1 | tail -3`
Expected: ruff は `All checks passed!`、pyright は `0 errors`(`check_symmetry.py` の `assert_type` がすべて通る)。

pyright が `symmetrytensors.py` の `return super().eye(...)` などで「`AbelianTensor` は `Self` に代入できない」と指摘した場合は、`abeliantensor.py` の該当する classmethod(`eye`、`initialize_with`、`from_ndarray`)に戻り値の型 `-> Self` だけを先に付けて解決する(`from typing import Self` を追加)。これは B-3 の作業の先取りなので、進捗の記録に判断として残す。

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: すべて成功(Step 4 の pickle のテストを含む)。

- [ ] **Step 9: コミットする**

```bash
git add pyproject.toml src/abeliantensors/py.typed src/abeliantensors/_typing.py src/abeliantensors/symmetry.py src/abeliantensors/symmetrytensors.py tests/typing/check_symmetry.py tests/test_symmetry.py
git commit -m "Annotate symmetry groups and finite symmetric tensor classes

Make AbelianGroup generic in the quantum number type (int for cyclic
groups and U(1), tuple for products), add py.typed and type aliases,
check annotations with ruff ANN rules, and add pyright-checked
assert_type tests under tests/typing.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: B-2(`TensorCommon` の基本的なメソッド)

**Files:**
- Modify: `src/abeliantensors/tensorcommon.py`
- Create: `tests/typing/check_tensorcommon.py`
- Modify: `tests/test_errors.py`

**Interfaces:**
- Consumes: `_typing.Indices`(Task 1)
- Produces:
  - `TensorCommon.empty/zeros/ones/random(cls, *args: Any, **kwargs: Any) -> Self`
  - `form_str(self) -> str`、`flatten_shape(shape: Any) -> Any`、`flatten_dim(dim: Any) -> int`
  - `norm_sq(self) -> float`、`norm(self) -> float`
  - `default_trunc_err_func(cls, S: np.ndarray, chi: int, norm_sq: float | None = None) -> float`
  - `from_matrix(self, left_dims: Sequence[Any], right_dims: Sequence[Any], left_qims: Sequence[Any] | None = None, right_qims: Sequence[Any] | None = None, left_dirs: Sequence[int] | None = None, right_dirs: Sequence[int] | None = None) -> Self`
  - `dot(self, other: "TensorCommon", indices: tuple[Indices, Indices]) -> Self`

- [ ] **Step 1: 失敗する型チェックを書く**

`tests/typing/check_tensorcommon.py`:

```python
"""Type-checking assertions for methods defined in TensorCommon.

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import assert_type

from abeliantensors import Tensor, TensorU1, TensorZ2, TensorZ2Z3

T = TensorZ2.random([[1, 1], [1, 1]], dirs=[1, -1])
assert_type(T, TensorZ2)
assert_type(TensorZ2.zeros([[1, 1]], dirs=[1]), TensorZ2)
assert_type(TensorZ2.ones([[1, 1]], dirs=[1]), TensorZ2)
assert_type(TensorZ2.empty([[1, 1]], dirs=[1]), TensorZ2)
assert_type(TensorU1.random([[1, 1]], qhape=[[0, 1]], dirs=[1]), TensorU1)
assert_type(TensorZ2Z3.random([[1] * 6], dirs=[1]), TensorZ2Z3)
assert_type(T.dot(T, (1, 0)), TensorZ2)
assert_type(T.dot(T, ([1], [0])), TensorZ2)
assert_type(T.norm(), float)
assert_type(T.norm_sq(), float)
assert_type(T.form_str(), str)

D = Tensor.random([2, 2])
assert_type(D, Tensor)
assert_type(D.norm(), float)
```

Run: `uv run pyright 2>&1 | tail -3`
Expected: `assert_type` の不一致のエラーが出る。

- [ ] **Step 2: メソッドに型注釈を付ける**

import を次にする。

```python
import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ._typing import Indices
```

(`Literal` と `overload` は Task 3 で加える。)

次の表のとおり、定義の行を置き換える。

| 置き換え前 | 置き換え後 |
|---|---|
| `def empty(cls, *args, **kwargs):` | `def empty(cls, *args: Any, **kwargs: Any) -> Self:` |
| `def zeros(cls, *args, **kwargs):` | `def zeros(cls, *args: Any, **kwargs: Any) -> Self:` |
| `def ones(cls, *args, **kwargs):` | `def ones(cls, *args: Any, **kwargs: Any) -> Self:` |
| `def random(cls, *args, **kwargs):` | `def random(cls, *args: Any, **kwargs: Any) -> Self:` |
| `def form_str(self):` | `def form_str(self) -> str:` |
| `def flatten_shape(shape):` | `def flatten_shape(shape: Any) -> Any:` |
| `def flatten_dim(dim):` | `def flatten_dim(dim: Any) -> int:` |
| `def norm_sq(self):` | `def norm_sq(self) -> float:` |
| `def norm(self):` | `def norm(self) -> float:` |
| `def default_trunc_err_func(cls, S, chi, norm_sq=None):` | `def default_trunc_err_func(cls, S: np.ndarray, chi: int, norm_sq: float \| None = None) -> float:` |
| `def dot(self, other, indices):` | `def dot(self, other: "TensorCommon", indices: tuple[Indices, Indices]) -> Self:` |
| `def _matrix_decomp_format_chis(self, chis, eps):` | `def _matrix_decomp_format_chis(self, chis: Any, eps: float) -> list[int]:` |

`from_matrix` のシグネチャを次にする。

```python
    def from_matrix(
        self,
        left_dims: Sequence[Any],
        right_dims: Sequence[Any],
        left_qims: Sequence[Any] | None = None,
        right_qims: Sequence[Any] | None = None,
        left_dirs: Sequence[int] | None = None,
        right_dirs: Sequence[int] | None = None,
    ) -> Self:
```

(`flatten_shape` は、`Tensor` の形(整数のタプル)と `AbelianTensor` の形(リストのリスト)の両方を受け取り、それぞれ違う形で返すので、`Any` にする。)

- [ ] **Step 3: pyright の指摘に応じて、型チェッカー向けの宣言を加える**

Run: `uv run pyright 2>&1 | grep "for class \"TensorCommon" | sed 's/.*attribute "\([a-z_]*\)".*/\1/' | sort -u`

出てきた名前(例: `sqrt`、`multiply_diag`、`matrix_dot`、`value`、`swapaxes`)ごとに、`TensorCommon` の `if TYPE_CHECKING:` のブロックに次の形の宣言を加える。`cls.` から呼ばれているものは `@classmethod` にする。

```python
        def sqrt(self, *args: Any, **kwargs: Any) -> Any: ...
```

(引数を `*args: Any, **kwargs: Any`、戻り値を `Any` にするのは、`Tensor` と `AbelianTensor` で引数が違っても互換性のエラーにならないようにするため、また `np.ndarray` にも同名のメソッドがある場合(`swapaxes` など)に、`class Tensor` の行で基底クラスどうしの食い違いのエラーが出ないようにするため。)

`tests/test_errors.py` の `test_tensorcommon_declarations_are_type_checking_only` の名前の一覧に、加えた名前をすべて加える。

- [ ] **Step 4: 確認する**

Run: `uv run ruff check --fix -q; uv run ruff format -q; uv run pyright 2>&1 | tail -3`
Expected: `0 errors`。

まだエラーが残る場合:
- `TensorCommon` のメソッドの本体で、型の付いた値に対する誤り(例: `float` を返すべきところで `np.floating` を返している)が出た場合は、`float(...)` で変換する(`np.float64` は `float` のサブクラスなので、値は変わらない)。
- `class Tensor` の行で、加えた宣言と `np.ndarray` の食い違いのエラーが出た場合は、その名前の宣言の戻り値と引数が `*args: Any, **kwargs: Any) -> Any` になっているかを確認する。それでも出る場合は、Global Constraints に従って判断を記録する。

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: すべて成功

- [ ] **Step 5: コミットする**

```bash
git add src/abeliantensors/tensorcommon.py tests/typing/check_tensorcommon.py tests/test_errors.py
git commit -m "Annotate the basic methods of TensorCommon

Tensor-returning methods return Self, so e.g. TensorZ2.random(...)
and T.dot(...) are inferred as TensorZ2.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: B-2(`to_matrix`、`eig`、`svd`、`split` の `@overload`)

**Files:**
- Modify: `src/abeliantensors/tensorcommon.py`、`pyproject.toml`
- Modify: `tests/typing/check_tensorcommon.py`
- Modify: `docs/superpowers/specs/2026-10-13-typing-design.md`

**Interfaces:**
- Consumes: Task 2 の注釈
- Produces:
  - `to_matrix(left_inds, right_inds, dirs=None, return_transposed_shape_data: Literal[False] = False) -> Self`、`...: Literal[True]` のとき `-> tuple[Self, Any, Any, Any]`
  - `eig(a, b, ...)` → `tuple[Self, Self]`、`return_rel_err=True` のとき `tuple[Self, Self, float]`
  - `svd(a, b, ...)` → `tuple[Self, Self, Self]`、`return_rel_err=True` のとき `tuple[Self, Self, Self, float]`
  - `split(a, b, ...)` → `return_rel_err` と `return_sings` の 4 通り(下の Step 2)

- [ ] **Step 1: 失敗する型チェックを書く**

`tests/typing/check_tensorcommon.py` の末尾に追加する。

```python
# Return shapes that depend on flags (@overload in TensorCommon).
assert_type(T.to_matrix([0], [1]), TensorZ2)
assert_type(
    T.to_matrix([0], [1], return_transposed_shape_data=True),
    tuple[TensorZ2, Any, Any, Any],
)
assert_type(T.svd(0, 1), tuple[TensorZ2, TensorZ2, TensorZ2])
assert_type(
    T.svd(0, 1, return_rel_err=True),
    tuple[TensorZ2, TensorZ2, TensorZ2, float],
)
assert_type(T.eig(0, 1), tuple[TensorZ2, TensorZ2])
assert_type(T.eig(0, 1, return_rel_err=True), tuple[TensorZ2, TensorZ2, float])
assert_type(T.split(0, 1), tuple[TensorZ2, TensorZ2])
assert_type(T.split(0, 1, return_sings=True), tuple[TensorZ2, TensorZ2, TensorZ2])
assert_type(T.split(0, 1, return_rel_err=True), tuple[TensorZ2, TensorZ2, float])
assert_type(
    T.split(0, 1, return_rel_err=True, return_sings=True),
    tuple[TensorZ2, TensorZ2, TensorZ2, float],
)
assert_type(D.svd(0, 1), tuple[Tensor, Tensor, Tensor])
assert_type(D.split([0], [1]), tuple[Tensor, Tensor])


def with_flags(flag: bool) -> None:
    """Flags given as bool variables give the union of the return shapes."""
    assert_type(
        T.svd(0, 1, return_rel_err=flag),
        tuple[TensorZ2, TensorZ2, TensorZ2]
        | tuple[TensorZ2, TensorZ2, TensorZ2, float],
    )
    assert_type(
        T.split(0, 1, return_rel_err=flag, return_sings=flag),
        tuple[TensorZ2, TensorZ2]
        | tuple[TensorZ2, TensorZ2, TensorZ2]
        | tuple[TensorZ2, TensorZ2, float]
        | tuple[TensorZ2, TensorZ2, TensorZ2, float],
    )
```

ファイルの先頭の import を `from typing import Any, assert_type` にする。

Run: `uv run pyright 2>&1 | tail -3`
Expected: `assert_type` の不一致のエラーが出る。

- [ ] **Step 2: `@overload` を書く**

import の `from typing import TYPE_CHECKING, Any, Self` を `from typing import TYPE_CHECKING, Any, Literal, Self, overload` にする。

**`to_matrix`**:既存の定義の直前に追加し、既存の定義のシグネチャを置き換える。

```python
    @overload
    def to_matrix(
        self,
        left_inds: Sequence[int],
        right_inds: Sequence[int],
        dirs: Sequence[int] | None = None,
        return_transposed_shape_data: Literal[False] = False,
    ) -> Self: ...

    @overload
    def to_matrix(
        self,
        left_inds: Sequence[int],
        right_inds: Sequence[int],
        dirs: Sequence[int] | None = None,
        *,
        return_transposed_shape_data: Literal[True],
    ) -> tuple[Self, Any, Any, Any]: ...

    def to_matrix(
        self,
        left_inds: Sequence[int],
        right_inds: Sequence[int],
        dirs: Sequence[int] | None = None,
        return_transposed_shape_data: bool = False,
    ) -> Self | tuple[Self, Any, Any, Any]:
```

(変形前の形のデータ(`shape`、`qhape`、`dirs`)は、`Tensor` と `AbelianTensor` で型が違うので `Any` にする。)

**`eig`**:

```python
    @overload
    def eig(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: Literal[False] = False,
        **kwargs: Any,
    ) -> tuple[Self, Self]: ...

    @overload
    def eig(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: Literal[True],
        **kwargs: Any,
    ) -> tuple[Self, Self, float]: ...

    @overload
    def eig(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: bool,
        **kwargs: Any,
    ) -> tuple[Self, Self] | tuple[Self, Self, float]: ...

    def eig(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: bool = False,
        **kwargs: Any,
    ) -> tuple[Self, Self] | tuple[Self, Self, float]:
```

**`svd`**:`eig` と同じ形で、戻り値を `tuple[Self, Self, Self]`(`Literal[False]`)、`tuple[Self, Self, Self, float]`(`Literal[True]`)、`tuple[Self, Self, Self] | tuple[Self, Self, Self, float]`(`bool` と実装)にする。

```python
    @overload
    def svd(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: Literal[False] = False,
        **kwargs: Any,
    ) -> tuple[Self, Self, Self]: ...

    @overload
    def svd(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: Literal[True],
        **kwargs: Any,
    ) -> tuple[Self, Self, Self, float]: ...

    @overload
    def svd(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: bool,
        **kwargs: Any,
    ) -> tuple[Self, Self, Self] | tuple[Self, Self, Self, float]: ...

    def svd(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: bool = False,
        **kwargs: Any,
    ) -> tuple[Self, Self, Self] | tuple[Self, Self, Self, float]:
```

**`split`**:型の別名を、クラスの外(`class TensorCommon:` の直前)に定義しておくと読みやすい。ただし `Self` はクラスの外で使えないので、戻り値はそれぞれのオーバーロードに直接書く。

```python
    @overload
    def split(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: Literal[False] = False,
        return_sings: Literal[False] = False,
        weight: str = "both",
        **kwargs: Any,
    ) -> tuple[Self, Self]: ...

    @overload
    def split(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: Literal[False] = False,
        return_sings: Literal[True],
        weight: str = "both",
        **kwargs: Any,
    ) -> tuple[Self, Self, Self]: ...

    @overload
    def split(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: Literal[True],
        return_sings: Literal[False] = False,
        weight: str = "both",
        **kwargs: Any,
    ) -> tuple[Self, Self, float]: ...

    @overload
    def split(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: Literal[True],
        return_sings: Literal[True],
        weight: str = "both",
        **kwargs: Any,
    ) -> tuple[Self, Self, Self, float]: ...

    @overload
    def split(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: bool = False,
        return_sings: bool = False,
        weight: str = "both",
        **kwargs: Any,
    ) -> (
        tuple[Self, Self]
        | tuple[Self, Self, Self]
        | tuple[Self, Self, float]
        | tuple[Self, Self, Self, float]
    ): ...

    def split(
        self,
        a: Indices,
        b: Indices,
        *args: Any,
        return_rel_err: bool = False,
        return_sings: bool = False,
        weight: str = "both",
        **kwargs: Any,
    ) -> (
        tuple[Self, Self]
        | tuple[Self, Self, Self]
        | tuple[Self, Self, float]
        | tuple[Self, Self, Self, float]
    ):
```

(既存の `split` の本体は、`ret_val` を `+=` でつなげて組み立てている。pyright が `ret_val` の型を最初の代入から決めてしまい、`+=` でエラーになる場合は、`ret_val: tuple[Any, ...]` と型を明示する。実行時の動作は変わらない。)

- [ ] **Step 3: 確認する**

Run: `uv run ruff check --fix -q; uv run ruff format -q; uv run pyright 2>&1 | tail -3`
Expected: `0 errors`(`check_tensorcommon.py` の `assert_type` がすべて通る)。

本体で新しいエラーが出た場合(戻り値の型を明示したことで、本体の返す値が検査されるようになるため):
- `svd` / `eig` の本体が `ret_val` を条件付きで組み立てている箇所は、`split` と同じく `tuple[Any, ...]` を明示して解決する。
- `TensorCommon` に宣言のないメソッドの指摘は、Task 2 の Step 3 と同じ方法で宣言を加え、`tests/test_errors.py` の一覧にも加える。

- [ ] **Step 4: `tensorcommon.py` の `ANN` の除外を外す**

`pyproject.toml` の `per-file-ignores` から、次の 2 行を削除する。

```toml
# Removed in Task 3 of this plan.
"src/abeliantensors/tensorcommon.py" = ["ANN"]
```

Run: `uv run ruff check 2>&1 | tail -3`
Expected: `All checks passed!`。指摘が残る場合は、その関数に Task 2 の表の方針で注釈を付ける。

Run: `uv run pytest -q 2>&1 | tail -1 && uv run ruff format --check -q && uv run pyright 2>&1 | tail -1`
Expected: すべて成功、`0 errors`

- [ ] **Step 5: 設計書を更新してコミットする**

設計書の §5 の「型の別名」の節の後に、次の 2 段落を追加する。

```markdown
**群のジェネリクス化(B-1 で決定)**

- `AbelianGroup` は、量子数の型をパラメータにしたジェネリクス `AbelianGroup[Q]`(PEP 695 の構文)にする。`CyclicGroup` と `U1Group` は `AbelianGroup[int]`、`ProductGroup` は `AbelianGroup[tuple[int, ...]]`。基底クラスで量子数の型を `Qnum` に固定すると、サブクラスの実装(整数どうしの足し算など)が型エラーになるため。

**`TensorCommon` の型チェッカー向けの宣言(B-2 で決定)**

- `TensorCommon` のメソッドに型を付けると、本体の中でサブクラスにしかないメソッド(`sqrt`、`multiply_diag` など)を呼んでいる箇所が検査されるようになる。それらは `if TYPE_CHECKING:` のブロックに `(*args: Any, **kwargs: Any) -> Any` の形で宣言する。具体的な型は、B-3、B-4 でサブクラスに付ける。
```

```bash
git add src/abeliantensors/tensorcommon.py pyproject.toml tests/typing/check_tensorcommon.py tests/test_errors.py docs/superpowers/specs/2026-10-13-typing-design.md
git commit -m "Overload to_matrix, eig, svd and split on their flags

The number of returned values now follows return_rel_err /
return_sings / return_transposed_shape_data, e.g. T.svd(0, 1) is a
tuple of three tensors of the same class as T.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
