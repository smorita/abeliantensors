# 型注釈 フェーズ B-3(AbelianTensor)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `abeliantensor.py` の公開 API(属性、演算子、約 55 個のメソッド)に型注釈を付け、`ANN` の除外を外す。`TensorZ2 + TensorZ2` や `T.matrix_svd()` などが正しい型で推論されることを `tests/typing/` で確かめる。

**Architecture:** 属性は `AbelianTensor` のクラス本体に注釈だけで宣言する。演算子は、生成関数の戻り値を `Callable[[_AT, Any], _AT]`(`_AT` は `AbelianTensor` を上限とする `TypeVar`)と注釈することで、実行時の定義の仕方を変えずにサブクラスの型を返すようにする。`__eq__` / `__ne__` だけは `object` と食い違うので、`if TYPE_CHECKING:` の中にテンソルを返す宣言を書く。

**Tech Stack:** pyright 1.1.414(`"standard"`)、ruff(`ANN` ルール)、`typing.Self` / `TypeVar` / `overload` / `Literal` / `assert_type`。

**Spec:** `docs/superpowers/specs/2026-10-13-typing-design.md`(§4 の B-3、§5)

## Global Constraints

- 作業ブランチは `typing-b`(B-1 + B-2 のコミットの上で続ける)。コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` を付ける。
- 実行時の動作は変えない。公開 API(メソッド名、引数名、戻り値の形)も変えない。
- pyright(`src/`、`tests/typing/`)は `"standard"` モードでエラー 0 件を保つ。
- 新しい `# pyright: ignore` は、`__eq__` / `__ne__` の 2 件(本計画で決めたもの)以外は原則として使わない。使う場合は理由を書き、設計書の「無視した箇所」と進捗の記録に残す。
- 数値のスカラーを返すメソッド(`value`、`max`、`min`、`sum`、`average`)の戻り値は `Any` とする(実数・複素数・NumPy の数値型のどれにもなるため)。
- 各コミットの前に `uv run pytest -q`、`uv run ruff check`、`uv run ruff format --check`、`uv run pyright` を通す。

## Review Focus

1. **`__eq__` / `__ne__` の実行時の定義が変わる**:型チェッカー向けの宣言を `if TYPE_CHECKING:` の外に書くと、実行時の `==` が要素ごとの比較ではなくなる。→ Task 1 の `test_comparison_operators_stay_elementwise`。
2. **演算子の推論がサブクラスの型にならない**:生成関数の型の付け方を誤ると、`TensorZ2 + TensorZ2` が `AbelianTensor` や `Unknown` になる。→ Task 1 の `tests/typing/check_abeliantensor.py`。
3. **`dirs` の型を `list[int]` にしたことで、正しく動いていたコードが型エラーになる**:実行時にタプルの `dirs` は動かない(テンソルは作れるが、`flip_dir`、`conj`、`dot`、`svd` などで `AttributeError` になる)ことを確認済み。リストで渡す既存のテストと型チェックファイルが通ることで確かめる。→ Task 2 の確認。
4. **属性の型を宣言したことで、実行時の属性が変わる**:注釈だけの宣言(`shape: list[Dim]` など)はクラス属性を作らないので、`vars(AbelianTensor)` に現れないはず。→ Task 1 の `test_attribute_annotations_are_not_class_attributes`。
5. **`join_indices` のフラグの型**:`return_transposed_shape_data` を `bool` の変数で渡したときに和型になるべき。→ Task 3 の `check_abeliantensor.py`。

---

## File Structure

| ファイル | 作業 | 内容 |
|---|---|---|
| `src/abeliantensors/abeliantensor.py` | 修正 | 属性、演算子、メソッドの型注釈 |
| `src/abeliantensors/_typing.py` | 修正なし(`Dim`、`Qim`、`Qnum`、`Indices` を使う) | |
| `tests/typing/check_abeliantensor.py` | 新規 | 推論結果の確認 |
| `tests/test_symmetry_tensors.py` | 修正 | Review Focus 1、4 のテスト |
| `pyproject.toml` | 修正 | `abeliantensor.py` の `ANN` の除外を外す |
| `docs/superpowers/specs/2026-10-13-typing-design.md` | 修正 | 演算子の型の付け方、無視した箇所 |

---

### Task 1: 属性と演算子

**Files:**
- Modify: `src/abeliantensors/abeliantensor.py`
- Create: `tests/typing/check_abeliantensor.py`
- Modify: `tests/test_symmetry_tensors.py`

**Interfaces:**
- Produces: `AbelianTensor` の属性の型(`shape: list[Dim]`、`qhape: list[Qim]`、`dirs: list[int]`、`charge: Qnum`、`sects: dict[tuple[Qnum, ...], np.ndarray]`、`defval: Any`、`invar: bool`、`dtype: DTypeLike`、`symmetry: AbelianGroup[Any]`)。演算子と生成されたメソッドは、受け取ったテンソルと同じクラスを返す。

- [ ] **Step 1: 失敗する型チェックを書く**

`tests/typing/check_abeliantensor.py`:

```python
"""Type-checking assertions for AbelianTensor.

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import Any, assert_type

import numpy as np

from abeliantensors import AbelianGroup, TensorZ2
from abeliantensors._typing import Dim, Qim, Qnum

T = TensorZ2.random([[1, 1], [1, 1]], dirs=[1, -1])

# Attributes
assert_type(T.shape, list[Dim])
assert_type(T.qhape, list[Qim])
assert_type(T.dirs, list[int])
assert_type(T.charge, Qnum)
assert_type(T.sects, dict[tuple[Qnum, ...], np.ndarray])
assert_type(T.symmetry, AbelianGroup[Any])
assert_type(T.invar, bool)

# Operators return the class of the operand.
assert_type(T + T, TensorZ2)
assert_type(T + 1.0, TensorZ2)
assert_type(2.0 * T, TensorZ2)
assert_type(T * T, TensorZ2)
assert_type(T / 2, TensorZ2)
assert_type(T**2, TensorZ2)
assert_type(-T, TensorZ2)
assert_type(abs(T), TensorZ2)
assert_type(T == T, TensorZ2)
assert_type(T != 0, TensorZ2)
assert_type(T < T, TensorZ2)
assert_type(T >= 0, TensorZ2)
assert_type(T.sqrt(), TensorZ2)
assert_type(T.exp(), TensorZ2)
assert_type(T.abs(), TensorZ2)
```

Run: `uv run pyright 2>&1 | grep -c reportAssertTypeFailure; uv run pyright 2>&1 | tail -1`
Expected: `assert_type` の不一致が多数(属性は宣言されておらず、演算子は `Unknown` のため)。

- [ ] **Step 2: 実行時の動作を固定するテストを書く**

`tests/test_symmetry_tensors.py` の末尾に追加する。

```python
def test_comparison_operators_stay_elementwise():
    """__eq__/__ne__ are declared for type checkers only; at runtime they
    must still compare element-wise and return tensors.
    """
    T = TensorZ2.random([[1, 1], [1, 1]], dirs=[1, -1])
    eq = T == T
    ne = T != T
    assert isinstance(eq, TensorZ2)
    assert isinstance(ne, TensorZ2)
    assert eq.all()
    assert not ne.any()
    assert AbelianTensor.__hash__ is None


def test_attribute_annotations_are_not_class_attributes():
    for name in (
        "shape",
        "qhape",
        "dirs",
        "charge",
        "sects",
        "defval",
        "invar",
        "dtype",
    ):
        assert name not in vars(AbelianTensor)
    assert "symmetry" in vars(AbelianTensor)
```

Run: `uv run pytest -q -p no:randomly tests/test_symmetry_tensors.py -k "elementwise or class_attributes" 2>&1 | tail -1`
Expected: 2 passed(今の動作を固定するテスト)

- [ ] **Step 3: 生成関数とモジュールの補助関数に型を付ける**

import を次のようにする(既存のものに加える)。

```python
from collections.abc import Callable, Iterable, Sequence  # Iterable: already imported
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar

from ._typing import Dim, Qim, Qnum
from .symmetry import AbelianGroup, CyclicGroup, U1Group
```

(`Iterable`、`Literal` はすでに import されている。`overload` と `Indices` は Task 3 で使い始めるので、そのときに加える。ruff が未使用と指摘するものは、使うまで入れない。)

生成関数の定義の直前に追加する。

```python
# The tensor type that a generated operator method is bound to, so that e.g.
# TensorZ2 + TensorZ2 is inferred as TensorZ2.
_AT = TypeVar("_AT", bound="AbelianTensor")
```

| 置き換え前 | 置き換え後 |
|---|---|
| `def _generate_binary_deferer(op_func):` | `def _generate_binary_deferer(op_func: Callable[..., Any]) -> Callable[[_AT, Any], _AT]:` |
| (その中の)`def deferer(self, B, *args, **kwargs):` | `def deferer(self: _AT, B: Any, *args: Any, **kwargs: Any) -> _AT:` |
| `def _generate_unary_deferer(op_func):` | `def _generate_unary_deferer(op_func: Callable[..., Any]) -> Callable[[_AT], _AT]:` |
| (その中の)`def deferer(self, *args, **kwargs):` | `def deferer(self: _AT, *args: Any, **kwargs: Any) -> _AT:` |
| `def _arg_swapper(op):` | `def _arg_swapper(op: Callable[..., Any]) -> Callable[..., Any]:` |
| (その中の)`def op_swapped(a, b, *args, **kwargs):` | `def op_swapped(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:` |
| `def _warn_qodulus_deprecated():` | `def _warn_qodulus_deprecated() -> None:` |
| `def _resolve_symmetry(cls, symmetry, qodulus):` | `def _resolve_symmetry(cls: "type[AbelianTensor]", symmetry: AbelianGroup[Any] \| None, qodulus: Any) -> AbelianGroup[Any]:` |
| `def _fixed_symmetry(cls):` | `def _fixed_symmetry(cls: "type[AbelianTensor]") -> AbelianGroup[Any] \| None:` |
| `def _check_same_symmetry(a, b):` | `def _check_same_symmetry(a: "AbelianTensor", b: Any) -> None:` |
| `def _check_form_data(cls, shape, qhape, dirs):` | `def _check_form_data(cls: "type[AbelianTensor]", shape: Sequence[Any], qhape: Sequence[Any], dirs: Sequence[int]) -> None:` |
| `def _require_invariant(tensor, operation):` | `def _require_invariant(tensor: "AbelianTensor", operation: str) -> None:` |

(`_check_same_symmetry` の `b` を `Any` にするのは、`dot` が `TensorCommon` 型の引数をそのまま渡すため。)

- [ ] **Step 4: 属性を宣言する**

クラス本体の `symmetry: AbelianGroup = U1Group()` を次に置き換える。

```python
    symmetry: AbelianGroup[Any] = U1Group()

    # Instance attributes, set in __init__ (see the class docstring).
    shape: list[Dim]
    qhape: list[Qim]
    dirs: list[int]
    charge: Qnum
    sects: dict[tuple[Qnum, ...], np.ndarray]
    defval: Any
    invar: bool
    dtype: DTypeLike
```

(`dtype` は、`np.dtype` のインスタンスではなく、`np.float64` のようなクラスなど、渡された値がそのまま入るので `DTypeLike` にする。)

- [ ] **Step 5: `__eq__` / `__ne__` を型チェッカー向けに宣言する**

`__eq__ = _generate_binary_deferer(opr.eq)` と `__ne__ = _generate_binary_deferer(opr.ne)` の 2 行を、次に置き換える。

```python
    if TYPE_CHECKING:
        # Element-wise comparisons return tensors, unlike object.__eq__ and
        # object.__ne__ (as in numpy's type stubs).
        def __eq__(self, other: object) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]

        def __ne__(self, other: object) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    else:
        __eq__ = _generate_binary_deferer(opr.eq)
        __ne__ = _generate_binary_deferer(opr.ne)
```

(行が長いので ruff の `E501` に当たる場合は、`# noqa: E501` ではなく、ignore のコメントを宣言の直前の行に移して、宣言の行には `# pyright: ignore[...]` だけを残す。ruff format の結果を見て調整する。)

- [ ] **Step 6: 確認する**

Run: `uv run ruff check --fix -q; uv run ruff format -q; uv run pyright 2>&1 | tail -3`
Expected: `0 errors`(`check_abeliantensor.py` の `assert_type` がすべて通る)。

属性の型を宣言したことで、`dtype` への代入や、`allclose` の中の `None` かもしれない値などのエラーが出た場合:
- `dtype` は Step 4 のとおり `DTypeLike` にしてあれば出ない。
- `allclose` の `self.sects.get(k, self.defval)` が `ndarray | None` と推論されて `np.allclose` に渡せないというエラーは、`defval` が `Any` なら出ないはずなので、出た場合は推論の経路を調べ、`np.allclose` の手前で値の型を確かめる書き方にする(動作は変えない)。

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: すべて成功

- [ ] **Step 7: コミットする**

```bash
git add src/abeliantensors/abeliantensor.py tests/typing/check_abeliantensor.py tests/test_symmetry_tensors.py
git commit -m "Annotate AbelianTensor attributes and operators

Generated operator methods are typed to return the class of the
operand (TensorZ2 + TensorZ2 is TensorZ2); __eq__/__ne__ are declared
as returning tensors for type checkers only.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 作成、要素、形のデータに関するメソッド

**Files:**
- Modify: `src/abeliantensors/abeliantensor.py`
- Modify: `tests/typing/check_abeliantensor.py`

- [ ] **Step 1: 失敗する型チェックを書く**

`tests/typing/check_abeliantensor.py` の import に `AbelianTensor`、`CyclicGroup` を加え(`from abeliantensors import AbelianGroup, AbelianTensor, CyclicGroup, TensorZ2`)、末尾に追加する。

```python
# Creation
assert_type(
    AbelianTensor.random(
        [[1, 1]], qhape=[[0, 1]], dirs=[1], symmetry=CyclicGroup(2)
    ),
    AbelianTensor,
)
assert_type(AbelianTensor([[1]], qhape=[[0]], dirs=[1]), AbelianTensor)
assert_type(TensorZ2.eye([1, 1]), TensorZ2)
assert_type(
    TensorZ2.from_ndarray(np.eye(2), shape=[[1, 1], [1, 1]], dirs=[1, -1]),
    TensorZ2,
)
assert_type(T.to_ndarray(), np.ndarray)
assert_type(T.empty_like(), TensorZ2)
assert_type(T.view(), TensorZ2)
assert_type(T.copy(), TensorZ2)

# Elements and values
assert_type(T[(0, 0)], np.ndarray)
assert_type(T.value(), Any)
assert_type(T.sum(), Any)
assert_type(T.max(), Any)
assert_type(T.any(), bool | np.bool_)
assert_type(T.all(), bool | np.bool_)
assert_type(T.allclose(T), bool)
assert_type(T.isscalar(), bool)
assert_type(T.is_full(), bool)
assert_type(len(T), int)
assert_type(bool(T.trace()), bool)

# Element-wise and form operations
assert_type(T.conj(), TensorZ2)
assert_type(T.conjugate(), TensorZ2)
assert_type(T.real(), TensorZ2)
assert_type(T.imag(), TensorZ2)
assert_type(T.astype(np.complex128), TensorZ2)
assert_type(T.diag(), TensorZ2)
assert_type(T.flip_dir(0), TensorZ2)
assert_type(T.expand_dims(0), TensorZ2)
assert_type(T.is_valid_key((0, 0)), bool)
assert_type(T.compatible_indices(T, 0, 1), bool)
assert_type(T.check_consistency(), bool)
assert_type(TensorZ2.check_form_match(tensor1=T, tensor2=T), bool)
assert_type(repr(T), str)
```

Run: `uv run pyright 2>&1 | grep -c reportAssertTypeFailure`
Expected: 0 より大きい(まだ戻り値の型がないため)。

- [ ] **Step 2: メソッドに型注釈を付ける**

次の表のとおり、定義の行(シグネチャ)を置き換える。表にない引数の既定値はそのまま。`list[int] | None` の `dirs` は、実行時にリストでなければ動かない(Review Focus 3)ため。

| メソッド | 新しいシグネチャ(`def` の行) |
|---|---|
| `__init_subclass__` | `def __init_subclass__(cls, **kwargs: Any) -> None:` |
| `qodulus`(property) | `def qodulus(self) -> int \| None:` |
| `__init__` | `def __init__(self, shape: Sequence[Sequence[int]], qhape: Sequence[Sequence[Qnum]] \| None = None, qodulus: Any = _UNSET, sects: dict[tuple[Qnum, ...], np.ndarray] \| None = None, dirs: list[int] \| None = None, dtype: DTypeLike = np.float64, defval: Any = 0, charge: Qnum \| None = None, invar: bool = True, symmetry: AbelianGroup[Any] \| None = None) -> None:` |
| `view` | `def view(self) -> Self:` |
| `diag` | `def diag(self) -> Self:` |
| `eye` | `def eye(cls, dim: Sequence[int], qim: Sequence[Qnum] \| None = None, qodulus: Any = _UNSET, dtype: DTypeLike = np.float64, symmetry: AbelianGroup[Any] \| None = None) -> Self:` |
| `initialize_with` | `def initialize_with(cls, numpy_func: Callable[..., Any], shape: Sequence[Sequence[int]], *args: Any, qhape: Sequence[Sequence[Qnum]] \| None = None, qodulus: Any = _UNSET, dirs: list[int] \| None = None, invar: bool = True, charge: Qnum \| None = None, symmetry: AbelianGroup[Any] \| None = None, **kwargs: Any) -> Self:` |
| `empty_like` | `def empty_like(self) -> Self:` |
| `fill` | `def fill(self, value: Any) -> None:` |
| `__getitem__` | `def __getitem__(self, k: tuple[Qnum, ...]) -> np.ndarray:` |
| `value` | `def value(self) -> Any:` |
| `__setitem__` | `def __setitem__(self, key: tuple[Qnum, ...], value: np.ndarray) -> None:` |
| `__delitem__` | `def __delitem__(self, key: tuple[Qnum, ...]) -> None:` |
| `__repr__` | `def __repr__(self) -> str:` |
| `__str__` | `def __str__(self, *args: Any, **kwargs: Any) -> str:` |
| `conj` | `def conj(self) -> Self:` |
| `astype` | 引数はそのまま、戻り値に `-> Self` を加える |
| `_defer_unary_elementwise` | `def _defer_unary_elementwise(self, op_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Self:` |
| `_defer_binary_elementwise` | `def _defer_binary_elementwise(self, B: Any, op_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Self:` |
| `any` | `def any(self) -> bool \| np.bool_:` |
| `all` | `def all(self) -> bool \| np.bool_:` |
| `allclose` | `def allclose(self, B: "AbelianTensor", rtol: float = 1e-05, atol: float = 1e-08) -> bool:` |
| `max`、`min`、`average`、`sum` | `def max(self) -> Any:` などと同じ形 |
| `real`、`imag` | `def real(self) -> Self:`、`def imag(self) -> Self:` |
| `__len__` | `def __len__(self) -> int:` |
| `__bool__` | `def __bool__(self) -> bool:` |
| `to_ndarray` | `def to_ndarray(self) -> np.ndarray:` |
| `from_ndarray` | `def from_ndarray(cls, a: np.ndarray, shape: Sequence[Sequence[int]] \| None = None, qhape: Sequence[Sequence[Qnum]] \| None = None, dirs: list[int] \| None = None, qodulus: Any = _UNSET, invar: bool = True, charge: Qnum \| None = None, symmetry: AbelianGroup[Any] \| None = None) -> Self:` |
| `isscalar` | `def isscalar(self) -> bool:` |
| `is_valid_key` | `def is_valid_key(self, key: Sequence[Qnum]) -> bool:` |
| `compatible_indices` | `def compatible_indices(self, other: "AbelianTensor", i: int, j: int) -> bool:` |
| `flip_dir` | `def flip_dir(self, axis: int) -> Self:` |
| `expand_dims` | `def expand_dims(self, axis: int, direction: int = 1) -> Self:` |
| `_sorted_shape_qhape` | `def _sorted_shape_qhape(tensor: "AbelianTensor \| None" = None, shape: Sequence[Sequence[int]] \| None = None, qhape: Sequence[Sequence[Qnum]] \| None = None) -> tuple[list[Any], list[Any]]:` |
| `defblock` | `def defblock(self, key: Sequence[Qnum]) -> np.ndarray:` |
| `is_full` | `def is_full(self) -> bool:` |
| `check_consistency` | `def check_consistency(self) -> bool:` |
| `check_qim_dim_match` | `def check_qim_dim_match(cls, qim: Sequence[Any], dim: Sequence[Any]) -> bool:` |
| `check_qhape_shape_match` | `def check_qhape_shape_match(cls, qhape: Sequence[Sequence[Any]], shape: Sequence[Sequence[Any]]) -> bool:` |
| `check_form_match` | `def check_form_match(cls, tensor1: "AbelianTensor \| None" = None, tensor2: "AbelianTensor \| None" = None, qhape1: Sequence[Sequence[Qnum]] \| None = None, shape1: Sequence[Sequence[int]] \| None = None, dirs1: Sequence[int] \| None = None, qhape2: Sequence[Sequence[Qnum]] \| None = None, shape2: Sequence[Sequence[int]] \| None = None, dirs2: Sequence[int] \| None = None, symmetry: AbelianGroup[Any] \| None = None, qodulus: Any = _UNSET) -> bool:` |

(すべて 79 桁を超えるので、`uv run ruff format` で引数ごとに改行される。)

`_defer_unary_elementwise` / `_defer_binary_elementwise` の戻り値を `Self` にしたことで、Task 1 の生成関数の中の `type(self)._defer_binary_elementwise(self, B, op_func, ...)` が `_AT` を返すと推論されるはず。そうならない場合は、`self._defer_binary_elementwise(B, op_func, ...)` の形に書き換える(`type(self).method(self, ...)` と `self.method(...)` は、インスタンス属性で上書きされていない限り同じ動作)。

- [ ] **Step 3: 確認する**

Run: `uv run ruff check --fix -q; uv run ruff format -q; uv run pyright 2>&1 | grep error | sed 's/.*abeliantensors\///' | cut -c1-160; uv run pyright 2>&1 | tail -1`
Expected: `0 errors`。

シグネチャに型を付けたことで本体のエラーが出た場合は、次の方針で直す(実行時の動作は変えない)。
- 戻り値の型と本体が合わない(例: `any` で `np.any(...)` を返す)場合は、シグネチャの型の方が正しいかを確かめ、合わせる。`bool(...)` で値を変換して直すことはしない(戻り値の値が変わるため)。
- `None` かもしれない値の扱いは、フェーズ A と同じく `None` の確認を加える。
- 1 つの箇所で 15 分以上かかる場合は、その箇所の型を `Any` に緩めて先に進み、進捗の記録に判断として残す。

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: すべて成功

- [ ] **Step 4: コミットする**

```bash
git add src/abeliantensors/abeliantensor.py tests/typing/check_abeliantensor.py
git commit -m "Annotate AbelianTensor creation, element and form-data methods

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: 演算(結合・分割・縮約・分解)と `ANN` の除外の解除

**Files:**
- Modify: `src/abeliantensors/abeliantensor.py`、`pyproject.toml`
- Modify: `tests/typing/check_abeliantensor.py`
- Modify: `docs/superpowers/specs/2026-10-13-typing-design.md`

- [ ] **Step 1: 失敗する型チェックを書く**

`tests/typing/check_abeliantensor.py` の末尾に追加する。

```python
# Joining, splitting, contracting and decomposing
assert_type(T.join_indices([0, 1], dirs=[1]), TensorZ2)
assert_type(
    T.join_indices([0, 1], dirs=[1], return_transposed_shape_data=True),
    tuple[TensorZ2, list[Dim], list[Qim], list[int]],
)
assert_type(T.split_indices(0, [[1], [1]], qims=[[0], [0]]), TensorZ2)
assert_type(T.swapaxes(0, 1), TensorZ2)
assert_type(T.transpose(), TensorZ2)
assert_type(T.transpose((1, 0)), TensorZ2)
assert_type(T.trace(), TensorZ2)
assert_type(T.multiply_diag(T.diag(), 1), TensorZ2)
assert_type(T.dot(T, (1, 0)), TensorZ2)
assert_type(T.matrix_dot(T), TensorZ2)
assert_type(T.matrix_eig(), tuple[TensorZ2, TensorZ2, float])
assert_type(T.matrix_svd(), tuple[TensorZ2, TensorZ2, TensorZ2, float])


def join_with_flag(flag: bool) -> None:
    assert_type(
        T.join_indices([0, 1], dirs=[1], return_transposed_shape_data=flag),
        TensorZ2 | tuple[TensorZ2, list[Dim], list[Qim], list[int]],
    )
```

Run: `uv run pyright 2>&1 | grep -c reportAssertTypeFailure`
Expected: 0 より大きい。

- [ ] **Step 2: `join_indices` の `@overload`**

既存の `def join_indices(self, *inds, dirs=None, return_transposed_shape_data=False):` の前に、オーバーロードを追加し、実装のシグネチャを置き換える(`typing` の import に `overload` を、`._typing` の import に `Indices` を加える)。

```python
    @overload
    def join_indices(
        self,
        *inds: Indices,
        dirs: int | Sequence[int] | None = None,
        return_transposed_shape_data: Literal[False] = False,
    ) -> Self: ...

    @overload
    def join_indices(
        self,
        *inds: Indices,
        dirs: int | Sequence[int] | None = None,
        return_transposed_shape_data: Literal[True],
    ) -> tuple[Self, list[Dim], list[Qim], list[int]]: ...

    @overload
    def join_indices(
        self,
        *inds: Indices,
        dirs: int | Sequence[int] | None = None,
        return_transposed_shape_data: bool,
    ) -> Self | tuple[Self, list[Dim], list[Qim], list[int]]: ...

    def join_indices(
        self,
        *inds: Indices,
        dirs: int | Sequence[int] | None = None,
        return_transposed_shape_data: bool = False,
    ) -> Self | tuple[Self, list[Dim], list[Qim], list[int]]:
```

(フラグは `*inds` の後なのでキーワード専用。フェーズ A で、変形前の形のデータを返さない場合は `transposed_shape` などを `None` で初期化しているので、本体の `return res, transposed_shape, ...` で `None` かもしれないというエラーが出た場合は、その `return` の直前に `None` でないことの確認を加えるのではなく、`return_transposed_shape_data` が真の分岐の中でだけ代入されるように書き方を整理する。)

- [ ] **Step 3: 残りのメソッドに型注釈を付ける**

| メソッド | 新しいシグネチャ |
|---|---|
| `_find_trunc_dim` | `def _find_trunc_dim(cls, S: np.ndarray, S_sects: dict[Any, Any], minusabs_next_els: list[Any], dims: dict[Any, int], chis: Sequence[int], eps: float = 0, break_degenerate: bool = False, degeneracy_eps: float = 1e-6, trunc_err_func: Callable[..., float] \| None = None, norm_sq: float \| None = None) -> tuple[int, dict[Any, int], float]:` |
| `split_indices` | `def split_indices(self, indices: Indices, dims: Sequence[Any], qims: Sequence[Any] \| None = None, dirs: Any = None) -> Self:` |
| `swapaxes` | `def swapaxes(self, i: int, j: int) -> Self:` |
| `transpose` | `def transpose(self, p: Sequence[int] = (1, 0)) -> Self:` |
| `trace` | `def trace(self, axis1: int = 0, axis2: int = 1) -> Self:` |
| `multiply_diag` | `def multiply_diag(self, diag_vect: "AbelianTensor", axis: int, direction: str = "r") -> Self:` |
| `dot` | `def dot(self, other: TensorCommon, indices: Sequence[Indices]) -> Self:` |
| `matrix_dot` | `def matrix_dot(self, other: "AbelianTensor") -> Self:` |
| `matrix_eig` | `def matrix_eig(self, chis: Any = None, eps: float = 0, print_errors: str = "deprecated", hermitian: bool = False, break_degenerate: bool = False, degeneracy_eps: float = 1e-6, sparse: bool = False, trunc_err_func: Callable[..., float] \| None = None) -> tuple[Self, Self, float]:` |
| `matrix_svd` | `def matrix_svd(self, chis: Any = None, eps: float = 0, print_errors: str = "deprecated", break_degenerate: bool = False, degeneracy_eps: float = 1e-6, sparse: bool = False, trunc_err_func: Callable[..., float] \| None = None) -> tuple[Self, Self, Self, float]:` |

(`dot` のシグネチャは `TensorCommon.dot` と同じにする。上書きとして互換である必要があるため。)

- [ ] **Step 4: `ANN` の除外を外す**

`pyproject.toml` の `per-file-ignores` の `"src/abeliantensors/abeliantensor.py" = ["E741", "ANN"]` を削除する(`E741` は `"src/**"` の項目で引き続き除外される)。コメント `# Not annotated yet: removed in phase B-3 and B-4 of the typing spec.` は、`tensor.py` の行が残るのでそのままにする。

Run: `uv run ruff check --output-format concise 2>&1 | tail -5`
Expected: `All checks passed!`。指摘が残る場合は、表の方針で注釈を付ける(表から漏れたメソッドなど)。

- [ ] **Step 5: 確認する**

Run: `uv run ruff format -q; uv run pyright 2>&1 | grep error | sed 's/.*abeliantensors\///' | cut -c1-160; uv run pyright 2>&1 | tail -1`
Expected: `0 errors`。エラーが出た場合は Task 2 の Step 3 の方針で直す。

Run: `uv run pytest -q 2>&1 | tail -1 && uv run ruff check && uv run ruff format --check`
Expected: すべて成功

- [ ] **Step 6: 設計書を更新してコミットする**

設計書の §5 の「**演算子**」の節を、次に置き換える。

```markdown
**演算子(B-3 で決定)**

- `_generate_binary_deferer` / `_generate_unary_deferer` の戻り値を `Callable[[_AT, Any], _AT]` / `Callable[[_AT], _AT]`(`_AT` は `AbelianTensor` を上限とする `TypeVar`)と注釈する。生成された関数をクラス属性に代入すると、pyright はそれをメソッドとして扱い、`_AT` を呼び出し元のクラスに束縛するので、`TensorZ2 + TensorZ2` は `TensorZ2` と推論される。実行時の定義の仕方は変えない。
- `__eq__` / `__ne__` は `object` の定義(戻り値 `bool`)と食い違うため、`if TYPE_CHECKING:` の中にテンソルを返す宣言を書き、`# pyright: ignore[reportIncompatibleMethodOverride]` を付ける(NumPy の型定義と同じやり方)。実行時は従来どおり生成した関数を使う。
```

「無視した箇所」の表に次の 2 行を加える。

```markdown
| `abeliantensor.py` `__eq__`(`if TYPE_CHECKING:` の中の宣言) | `reportIncompatibleMethodOverride` | 要素ごとの比較の結果としてテンソルを返す。`object.__eq__` の戻り値 `bool` と食い違う |
| `abeliantensor.py` `__ne__`(同) | `reportIncompatibleMethodOverride` | 同上 |
```

```bash
git add src/abeliantensors/abeliantensor.py pyproject.toml tests/typing/check_abeliantensor.py docs/superpowers/specs/2026-10-13-typing-design.md
git commit -m "Annotate AbelianTensor operations; enforce ANN on abeliantensor.py

join_indices is overloaded on return_transposed_shape_data, and
matrix_eig/matrix_svd return typed tuples.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
