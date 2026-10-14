# 型注釈 フェーズ B-4(Tensor)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tensor.py` の公開 API に型注釈を付け、演算子を含めて `Tensor` が `Tensor` と推論されるようにする(`(D + D).svd(0, 1)` が型エラーにならない)。最後に `src/` の `ANN` の除外をなくす。

**Architecture:** `Tensor` は `np.ndarray` のサブクラスで、演算子と一部のメソッド(`conj`、`transpose` など)は NumPy から受け継ぐ。それらは `if TYPE_CHECKING:` の中で `Self` を返すと宣言する(実行時の動作は変わらない)。NumPy の型定義と食い違うものは、理由付きの `# pyright: ignore` で無視して設計書に記録する。`Tensor` 自身が定義するメソッドには、`AbelianTensor` と揃えた型注釈を付ける。

**Tech Stack:** pyright 1.1.414(`"standard"`)、ruff(`ANN` ルール)、`typing.Self` / `overload` / `Literal` / `assert_type`、`numpy.typing.ArrayLike`。

**Spec:** `docs/superpowers/specs/2026-10-13-typing-design.md`(§4 の B-4、§5 の「`Tensor`」、2026-10-14 の演算子の追記)

## Global Constraints

- 作業ブランチは `typing-b`(B-3 の上で続ける)。コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` を付ける。
- 実行時の動作は変えない。公開 API(メソッド名、引数名、戻り値の形)も変えない。
- pyright(`src/`、`tests/typing/`)は `"standard"` モードでエラー 0 件を保つ。
- `# pyright: ignore[ルール名]` は `np.ndarray` / `object` との食い違いに限り、必ず理由を書き、設計書の「無視した箇所」に記録する。
- `__divmod__` / `__rdivmod__` は宣言しない(実行時に配列のタプルを返すため)。
- 各コミットの前に `uv run pytest -q`、`uv run ruff check`、`uv run ruff format --check`、`uv run pyright` を通す。

## Review Focus

1. **`if TYPE_CHECKING:` の宣言が実行時に存在する**:演算子や `conj` などを実行時の `Tensor` に定義してしまうと、NumPy の実装が上書きされて動作が変わる。→ Task 1 の `test_tensor_type_checking_declarations_are_not_runtime_methods`。
2. **演算の結果を `Tensor` として使い続けるコード**:`(D + D).svd(0, 1)`、`(2.0 * D).join_indices(...)`、`D += 1` のあとの `D.svd(...)` などが型エラーにならず、`Tensor` と推論されるべき。→ Task 1 の `tests/typing/check_tensor.py`。
3. **`divmod`**:`divmod(D, 2)` は実行時にタプルを返すので、型も `Tensor` ではなくタプルと推論されるべき(宣言しないので NumPy の型定義のまま)。→ Task 1 の `check_tensor.py`。
4. **`from_ndarray` にリストやスカラーを渡す**:実行時に `np.array(a)` で受け付けるので、型エラーにならないべき。→ Task 2 の `check_tensor.py`。
5. **`Tensor` と `TensorZ2` の入れ替え**:同じコードを `Tensor` と `TensorZ2` で書いたとき、どちらも型エラーにならないべき。→ Task 2 の `check_tensor.py` の `swap_classes` 関数。

---

## File Structure

| ファイル | 作業 | 内容 |
|---|---|---|
| `src/abeliantensors/tensor.py` | 修正 | 演算子などの型チェッカー向けの宣言、メソッドの型注釈 |
| `tests/typing/check_tensor.py` | 新規 | 推論結果の確認 |
| `tests/test_errors.py` | 修正 | Review Focus 1 のテスト |
| `pyproject.toml` | 修正 | `tensor.py` の `ANN` の除外とそのコメントを削除 |
| `docs/superpowers/specs/2026-10-13-typing-design.md` | 修正 | 無視した箇所の追加、完了の記録 |

---

### Task 1: 演算子と NumPy から受け継ぐメソッドの宣言

**Files:**
- Modify: `src/abeliantensors/tensor.py`
- Create: `tests/typing/check_tensor.py`
- Modify: `tests/test_errors.py`

- [ ] **Step 1: 失敗する型チェックを書く**

`tests/typing/check_tensor.py`:

```python
"""Type-checking assertions for Tensor (a np.ndarray subclass).

Checked by pyright (see [tool.pyright] include); not collected by pytest.
"""

from typing import Any, assert_type

import numpy as np

from abeliantensors import Tensor

D = Tensor.random([2, 2])

# Operators keep the Tensor type (declared under TYPE_CHECKING).
assert_type(D + D, Tensor)
assert_type(D + 1.0, Tensor)
assert_type(2.0 * D, Tensor)
assert_type(D - D, Tensor)
assert_type(1.0 - D, Tensor)
assert_type(D * D, Tensor)
assert_type(D / 2, Tensor)
assert_type(D // 2, Tensor)
assert_type(D % 2, Tensor)
assert_type(D**2, Tensor)
assert_type(-D, Tensor)
assert_type(+D, Tensor)
assert_type(abs(D), Tensor)
assert_type(D == D, Tensor)
assert_type(D != 0, Tensor)
assert_type(D < D, Tensor)
assert_type(D >= 0, Tensor)
assert_type((D + D).svd(0, 1), tuple[Tensor, Tensor, Tensor])

E = Tensor.random([2, 2])
E += 1
assert_type(E, Tensor)
E *= 2.0
assert_type(E.svd(0, 1), tuple[Tensor, Tensor, Tensor])

# Methods inherited from ndarray, declared to return Tensor.
assert_type(D.conj(), Tensor)
assert_type(D.transpose(), Tensor)
assert_type(D.swapaxes(0, 1), Tensor)
assert_type(D.copy(), Tensor)
assert_type(D.astype(np.complex128), Tensor)

# divmod returns a tuple of arrays at runtime, so it is not declared.
quotient, remainder = divmod(D, 2)
assert_type(quotient, Any)
```

(最後の `divmod` の確認は、NumPy の型定義の推論結果に合わせて Step 5 で調整する。目的は「`Tensor` 1 つではなくタプルとして扱われること」なので、`quotient, remainder = ...` という分解が型エラーにならないことが本質。`assert_type` の期待値は、実際の推論結果(`reveal_type` で確認)に合わせて書き換えてよい。)

Run: `uv run pyright 2>&1 | grep -c reportAssertTypeFailure`
Expected: 0 より大きい(演算子は `ndarray`、`conj` は `Any` などと推論されるため)。

- [ ] **Step 2: 実行時の動作を固定するテストを書く**

`tests/test_errors.py` の末尾に追加する。

```python
def test_tensor_type_checking_declarations_are_not_runtime_methods():
    """Tensor's operators, conj, transpose, ... are declared for type
    checkers only; at runtime they must still come from np.ndarray.
    """
    for name in (
        "__add__",
        "__radd__",
        "__iadd__",
        "__sub__",
        "__eq__",
        "__neg__",
        "conj",
        "transpose",
        "swapaxes",
        "copy",
        "astype",
    ):
        assert name not in vars(Tensor), name
        assert getattr(Tensor, name) is getattr(np.ndarray, name), name
    D = Tensor.random([2, 2])
    assert isinstance(D + D, Tensor)
    assert isinstance(D == D, Tensor)
    q, r = divmod(D, 2)
    assert isinstance(q, Tensor)
    assert isinstance(r, Tensor)
```

Run: `uv run pytest -q -p no:randomly tests/test_errors.py -k not_runtime_methods 2>&1 | tail -1`
Expected: 1 passed(今の動作を固定するテスト)。`getattr(Tensor, name) is getattr(np.ndarray, name)` が成り立たない名前(`TensorCommon` が実行時に定義しているものなど)があれば、その名前を一覧から外し、進捗の記録に残す。

- [ ] **Step 3: 宣言を追加する**

`tensor.py` の import を次のようにする。

```python
import functools as fct
import operator as opr
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, overload

import numpy as np
import scipy.sparse.linalg as spsla
from numpy.typing import ArrayLike, DTypeLike

from ._typing import Indices
from .tensorcommon import TensorCommon
```

(この時点で使わないものは ruff が指摘するので、`uv run ruff check --fix` で外し、Task 2 で必要になったときに加える。)

`class Tensor` の docstring の直後(`# Practically all methods of Tensor take keyword argument` のコメントの前)に、次のブロックを追加する。

```python
    if TYPE_CHECKING:
        # Operators and some methods are inherited from np.ndarray, whose type
        # stubs say they return ndarray. At runtime they return Tensor (numpy
        # preserves the subclass), so declare that, for code such as
        # (D + D).svd(0, 1). __divmod__/__rdivmod__ are left to numpy: they
        # return a tuple of arrays.
        def __add__(self, other: Any) -> Self: ...
        def __radd__(self, other: Any) -> Self: ...
        def __iadd__(self, other: Any) -> Self: ...
        def __sub__(self, other: Any) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
        def __rsub__(self, other: Any) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
        def __isub__(self, other: Any) -> Self: ...
        def __mul__(self, other: Any) -> Self: ...
        def __rmul__(self, other: Any) -> Self: ...
        def __imul__(self, other: Any) -> Self: ...
        def __truediv__(self, other: Any) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
        def __rtruediv__(self, other: Any) -> Self: ...
        def __itruediv__(self, other: Any) -> Self: ...
        def __floordiv__(self, other: Any) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
        def __rfloordiv__(self, other: Any) -> Self: ...
        def __ifloordiv__(self, other: Any) -> Self: ...
        def __mod__(self, other: Any) -> Self: ...
        def __rmod__(self, other: Any) -> Self: ...
        def __imod__(self, other: Any) -> Self: ...
        def __pow__(self, other: Any) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
        def __rpow__(self, other: Any) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
        def __ipow__(self, other: Any) -> Self: ...
        def __lshift__(self, other: Any) -> Self: ...
        def __rlshift__(self, other: Any) -> Self: ...
        def __ilshift__(self, other: Any) -> Self: ...
        def __rshift__(self, other: Any) -> Self: ...
        def __rrshift__(self, other: Any) -> Self: ...
        def __irshift__(self, other: Any) -> Self: ...
        def __and__(self, other: Any) -> Self: ...
        def __rand__(self, other: Any) -> Self: ...
        def __iand__(self, other: Any) -> Self: ...
        def __xor__(self, other: Any) -> Self: ...
        def __rxor__(self, other: Any) -> Self: ...
        def __ixor__(self, other: Any) -> Self: ...
        def __or__(self, other: Any) -> Self: ...
        def __ror__(self, other: Any) -> Self: ...
        def __ior__(self, other: Any) -> Self: ...
        def __eq__(self, other: object) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
        def __ne__(self, other: object) -> Self: ...  # pyright: ignore[reportIncompatibleMethodOverride]
        def __lt__(self, other: Any) -> Self: ...
        def __le__(self, other: Any) -> Self: ...
        def __gt__(self, other: Any) -> Self: ...
        def __ge__(self, other: Any) -> Self: ...
        def __neg__(self) -> Self: ...
        def __pos__(self) -> Self: ...
        def __abs__(self) -> Self: ...
        def __invert__(self) -> Self: ...
        def conj(self) -> Self: ...
        def transpose(self, *axes: Any) -> Self: ...
        def swapaxes(self, axis1: int, axis2: int) -> Self: ...
        def copy(self, *args: Any, **kwargs: Any) -> Self: ...
        def astype(self, dtype: DTypeLike, *args: Any, **kwargs: Any) -> Self: ...
```

(ruff format は、この `def ...: ...` の 1 行の定義の間に空行を入れる。それでよい。行が 79 桁を超えて ruff の `E501` に当たる場合は、`# pyright: ignore[...]` のコメントだけで超えているなら、`[tool.ruff.lint]` の `E501` はコメントの長さを除外しないので、その行に `# noqa: E501` を加える代わりに、宣言を `def __sub__(\n            self, other: Any\n        ) -> Self: ...  # pyright: ignore[...]` のように改行して短くする。)

- [ ] **Step 4: pyright の指摘に合わせて無視の指定を調整する**

Run: `uv run ruff check --fix -q; uv run ruff format -q; uv run pyright 2>&1 | grep -E "error" | sed 's/.*abeliantensors\///' | cut -c1-160`

- 「互換性のない上書き」のエラーが出た宣言には、`# pyright: ignore[reportIncompatibleMethodOverride]` を付ける(上のコードで付けたもの以外に、`conj`、`transpose`、`swapaxes`、`copy`、`astype` などで出る可能性がある)。
- 逆に、`# pyright: ignore` を付けたのにエラーが出ない宣言では、pyright が「不要な ignore」を報告しない設定なので、そのままにせず ignore を外す(`uv run pyright --outputjson` で、その行にエラーが出ないことを、ignore を一時的に外して確かめる)。

Expected: `0 errors`

- [ ] **Step 5: 確認する**

Run: `uv run pyright 2>&1 | tail -1`
Expected: `0 errors`(`check_tensor.py` の `assert_type` がすべて通る)。`divmod` の確認が合わない場合は、`reveal_type(divmod(D, 2))` で実際の推論結果を見て、`assert_type` の期待値をそれに合わせる。その結果がタプル型であることを確かめる。

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: すべて成功(Step 2 のテストを含む)

- [ ] **Step 6: コミットする**

```bash
git add src/abeliantensors/tensor.py tests/typing/check_tensor.py tests/test_errors.py
git commit -m "Declare Tensor's ndarray operators as returning Tensor

numpy's stubs type D + D as ndarray although it is a Tensor at runtime;
declare the operators (and conj/transpose/swapaxes/copy/astype) under
TYPE_CHECKING so that e.g. (D + D).svd(0, 1) type-checks.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `Tensor` のメソッドの型注釈と `ANN` の除外の解除

**Files:**
- Modify: `src/abeliantensors/tensor.py`、`pyproject.toml`
- Modify: `tests/typing/check_tensor.py`
- Modify: `docs/superpowers/specs/2026-10-13-typing-design.md`

- [ ] **Step 1: 失敗する型チェックを書く**

`tests/typing/check_tensor.py` の import に `TensorZ2` を加え(`from abeliantensors import Tensor, TensorZ2`)、末尾に追加する。

```python
# Creation
assert_type(Tensor([2, 2]), Tensor)
assert_type(Tensor.zeros([2, 3]), Tensor)
assert_type(Tensor.eye(3), Tensor)
assert_type(Tensor.from_ndarray(np.eye(2)), Tensor)
assert_type(Tensor.from_ndarray([[1.0, 0.0], [0.0, 1.0]]), Tensor)
assert_type(Tensor.from_ndarray(2.5), Tensor)
assert_type(D.to_ndarray(), np.ndarray)

# Methods defined in Tensor
assert_type(D.diag(), Tensor)
assert_type(D.conjugate(), Tensor)
assert_type(D.sqrt(), Tensor)
assert_type(D.exp(), Tensor)
assert_type(D.abs(), Tensor)
assert_type(D.real(), Tensor)
assert_type(D.flip_dir(0), Tensor)
assert_type(D.expand_dims(0), Tensor)
assert_type(D.trace(), Tensor)
assert_type(D.value(), Any)
assert_type(D.sum(), Any)
assert_type(D.isscalar(), bool)
assert_type(D.allclose(D), bool)
assert_type(D.compatible_indices(D, 0, 1), bool)
assert_type(Tensor.check_form_match(tensor1=D, tensor2=D), bool)
assert_type(D.join_indices([0, 1]), Tensor)
assert_type(
    D.join_indices([0, 1], return_transposed_shape_data=True),
    tuple[Tensor, tuple[int, ...], None, None],
)
assert_type(D.join_indices([0, 1]).split_indices(0, [2, 2]), Tensor)
assert_type(D.multiply_diag(D.diag(), 1), Tensor)
assert_type(D.dot(D, (1, 0)), Tensor)
assert_type(D.matrix_dot(D), Tensor)
assert_type(D.matrix_svd(), tuple[Tensor, Tensor, Tensor, float])
assert_type(D.matrix_eig(), tuple[Tensor, Tensor, float])


def swap_classes(T: Tensor | TensorZ2) -> None:
    """The same code type-checks for Tensor and TensorZ2."""
    U, S, V = T.svd(0, 1)
    R = U.dot(S.diag(), (1, 0)).dot(V, (1, 0))
    assert_type(R, Tensor | TensorZ2)
    assert_type((R - T).norm(), float)
```

Run: `uv run pyright 2>&1 | grep -c reportAssertTypeFailure`
Expected: 0 より大きい。

- [ ] **Step 2: メソッドに型注釈を付ける**

次の表のとおり、定義の行を置き換える(複数行の定義は、構文木で範囲を特定して置き換えるとよい)。フェーズ A で付けた `# pyright: ignore[...]` と理由のコメントは残す。

| メソッド | 新しいシグネチャ |
|---|---|
| `__new__` | `def __new__(cls, shape: Any, *args: Any, qhape: Any = None, charge: Any = None, invar: Any = None, dirs: Any = None, **kwargs: Any) -> Self:` |
| `initialize_with` | `def initialize_with(cls, numpy_func: Callable[..., Any], shape: Any, *args: Any, qhape: Any = None, charge: Any = None, invar: Any = None, dirs: Any = None, **kwargs: Any) -> Self:` |
| `eye` | `def eye(cls, dim: Any, qim: Any = None, qodulus: Any = None, *args: Any, **kwargs: Any) -> Self:` |
| `diag` | `def diag(self, **kwargs: Any) -> "Tensor":` |
| `to_ndarray` | `def to_ndarray(self, **kwargs: Any) -> np.ndarray:` |
| `from_ndarray` | `def from_ndarray(cls, a: ArrayLike, **kwargs: Any) -> Self:` |
| `fill` | `def fill(self, value: Any) -> None:` |
| `conjugate` | `def conjugate(self) -> Self:` |
| `abs`、`log`、`exp`、`sqrt`、`sign` | `def abs(self) -> Self:` などと同じ形 |
| `average` | `def average(self) -> Any:` |
| `any`、`all` | `def any(self, *args: Any, **kwargs: Any) -> Any:`(NumPy の `any` / `all` は `axis` の有無で戻り値が配列にもなるため) |
| `allclose` | `def allclose(self, other: Any, *args: Any, **kwargs: Any) -> bool:` |
| `real`、`imag` | `def real(self) -> Self:  # pyright: ignore[...]`(既存の ignore を残す) |
| `sum` | `def sum(self) -> Any:  # pyright: ignore[...]`(同) |
| `isscalar` | `def isscalar(self) -> bool:` |
| `compatible_indices` | `def compatible_indices(self, other: TensorCommon, i: int, j: int) -> bool:` |
| `flip_dir` | `def flip_dir(self, axis: int) -> Self:` |
| `expand_dims` | `def expand_dims(self, axis: int, direction: int = 1) -> Self:` |
| `value` | `def value(self) -> Any:` |
| `check_form_match` | `def check_form_match(cls, tensor1: TensorCommon \| None = None, tensor2: TensorCommon \| None = None, qhape1: Any = None, shape1: Any = None, dirs1: Any = None, qhape2: Any = None, shape2: Any = None, dirs2: Any = None, qodulus: Any = None, symmetry: Any = None) -> bool:` |
| `_find_trunc_dim` | `def _find_trunc_dim(cls, S: np.ndarray, chis: Sequence[int], eps: float = 0, break_degenerate: bool = False, degeneracy_eps: float = 1e-6, trunc_err_func: Callable[..., float] \| None = None, norm_sq: float \| None = None) -> tuple[int, float]:` |
| `split_indices` | `def split_indices(self, indices: Indices, dims: Sequence[Any], qims: Any = None, **kwargs: Any) -> Self:` |
| `multiply_diag` | `def multiply_diag(self, diag_vect: "Tensor", axis: int, *args: Any, **kwargs: Any) -> Self:` |
| `trace` | `def trace(self, axis1: int = 0, axis2: int = 1) -> Self:  # pyright: ignore[...]`(既存の ignore を残す) |
| `dot` | `def dot(self, B: TensorCommon, indices: Sequence[Indices]) -> Self:  # pyright: ignore[...]`(同) |
| `matrix_dot` | `def matrix_dot(self, B: "Tensor") -> Self:` |
| `matrix_eig` | `def matrix_eig(self, chis: Any = None, eps: float = 0, print_errors: str = "deprecated", hermitian: bool = False, break_degenerate: bool = False, degeneracy_eps: float = 1e-6, sparse: bool = False, trunc_err_func: Callable[..., float] \| None = None) -> tuple[Self, Self, float]:` |
| `matrix_svd` | `def matrix_svd(self, chis: Any = None, eps: float = 0, print_errors: str = "deprecated", break_degenerate: bool = False, degeneracy_eps: float = 1e-6, sparse: bool = False, trunc_err_func: Callable[..., float] \| None = None) -> tuple[Self, Self, Self, float]:` |

(`diag` の戻り値を `Self` ではなく `"Tensor"` にするのは、本体が `np.diag(self).view(Tensor)` で、サブクラスでも `Tensor` を返すため。)

**`join_indices`** は `AbelianTensor` と同じく `@overload` を付け、実装のシグネチャは `Any` にする。

```python
    @overload
    def join_indices(
        self,
        *inds: Indices,
        return_transposed_shape_data: Literal[False] = False,
        **kwargs: Any,
    ) -> Self: ...

    @overload
    def join_indices(
        self,
        *inds: Indices,
        return_transposed_shape_data: Literal[True],
        **kwargs: Any,
    ) -> tuple[Self, tuple[int, ...], None, None]: ...

    @overload
    def join_indices(
        self,
        *inds: Indices,
        return_transposed_shape_data: bool,
        **kwargs: Any,
    ) -> Self | tuple[Self, tuple[int, ...], None, None]: ...

    def join_indices(
        self,
        # The overloads above give the public types; the body reshapes inds
        # through several forms, so the implementation takes Any.
        *inds: Any,
        return_transposed_shape_data: bool = False,
        **kwargs: Any,
    ) -> Any:
```

(`return_transposed_shape_data=True` のときの変形前の形は `self.shape`(整数のタプル)、`qhape` と `dirs` は `None`。)

- [ ] **Step 3: 確認する**

Run: `uv run ruff check --fix -q; uv run ruff format -q; uv run pyright 2>&1 | grep error | sed 's/.*abeliantensors\///' | cut -c1-160; uv run pyright 2>&1 | tail -1`
Expected: `0 errors`。

本体のエラーが出た場合の方針は B-3 と同じ(実行時の動作を変えない書き換え、または `Any` に緩めて記録)。シグネチャと `np.ndarray` の定義との食い違い(`fill`、`any`、`all`、`conjugate`、`__new__` など)が出た場合は、理由を書いて `# pyright: ignore[reportIncompatibleMethodOverride]` を付け、設計書に記録する。

- [ ] **Step 4: `ANN` の除外をなくす**

`pyproject.toml` の `per-file-ignores` から、次の 2 行を削除する。

```toml
# Not annotated yet: removed in phase B-3 and B-4 of the typing spec.
"src/abeliantensors/tensor.py" = ["ANN"]
```

Run: `uv run ruff check --output-format concise 2>&1 | tail -5`
Expected: `All checks passed!`。指摘が残る場合は、表の方針で注釈を付ける(メソッドの中の補助関数など)。

Run: `grep -n "ANN" pyproject.toml`
Expected: `select` の行と、`tests/**`、`demo/**`、`benchmarks/**` の除外だけが出る(`src/` の除外は残っていない)。

- [ ] **Step 5: 確認する**

Run: `uv run pytest -q 2>&1 | tail -1 && uv run ruff check && uv run ruff format --check && uv run pyright 2>&1 | tail -1`
Expected: すべて成功、`0 errors`

- [ ] **Step 6: 設計書を更新してコミットする**

設計書の「無視した箇所」の表に、Task 1 と Task 2 で付けた `# pyright: ignore` をすべて加える(`grep -n "pyright: ignore" src/abeliantensors/tensor.py` の出力と一致させる)。演算子の宣言の分は、1 行ずつ書かずに、次のようにまとめてよい。

```markdown
| `tensor.py` `if TYPE_CHECKING:` の中の演算子の宣言(`__sub__`、`__rsub__`、`__truediv__`、`__floordiv__`、`__pow__`、`__rpow__`、`__eq__`、`__ne__` など、実際に付けたものを列挙) | `reportIncompatibleMethodOverride` | NumPy の型定義では、これらの演算子は引数の型ごとに細かく場合分けされているか、`object` の定義(`bool` を返す)と食い違う。実行時の `Tensor` は NumPy の演算子を使い、結果は `Tensor` になるので、`Self` を返すと宣言する |
```

設計書の冒頭の「状態」を `状態: フェーズ A、B 完了(2026-10-14)` に変える。

```bash
git add src/abeliantensors/tensor.py pyproject.toml tests/typing/check_tensor.py docs/superpowers/specs/2026-10-13-typing-design.md
git commit -m "Annotate Tensor; enforce ANN on all of src/

Completes phase B of the typing spec: every public method in src/ is
annotated, and Tensor, TensorZ2 etc. can be swapped in type-checked
code.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
