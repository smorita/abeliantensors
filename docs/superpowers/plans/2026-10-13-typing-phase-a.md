# 型チェック フェーズ A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `uv run pyright` が `src/` でエラー 0 件(`"standard"` モード)になり、それが CI で維持されるようにする。

**Architecture:** pyright の設定を `pyproject.toml` に置き、Pylance と CI で同じ設定を使う。修正は、型注釈、`None` の確認、変数の初期化、`TensorCommon` への(型チェッカー専用の)宣言で行い、実行時の動作は変えない。直せない `np.ndarray` との食い違いだけを、理由付きの `# pyright: ignore` で無視する。

**Tech Stack:** pyright(`dev` 依存グループ、`uv.lock` で固定)、uv、pytest、GitHub Actions。

**Spec:** `docs/superpowers/specs/2026-10-13-typing-design.md`(フェーズ A の部分)

## Global Constraints

- 作業ブランチは `typing`(作成済み)。コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` を付ける。
- 実行時の動作は変えない。公開 API(メソッド名、引数名、戻り値の形)も変えない。例外的に変わってよいのは、もともと `TypeError` などで失敗していた入力に対するエラーメッセージだけ。
- 対象は `src/` のみ。`[tool.pyright]` の `typeCheckingMode = "standard"`。
- `# pyright: ignore[ルール名]` は、`np.ndarray` との食い違いなど、公開 API を変えずには直せない箇所に限って使い、同じ行か直前の行に理由をコメントで書く。使った箇所はすべて設計書の「無視した箇所」に記録する。
- 各コミットの前に `uv run pytest -q`、`uv run ruff check`、`uv run ruff format --check` を通す。

## Review Focus

1. **`TensorCommon` への宣言が実行時のメソッド解決を変える**:`Tensor` は `TensorCommon` → `np.ndarray` の順に継承し、`conj` などを NumPy から受け継いでいる。宣言を `if TYPE_CHECKING:` の外に書くと `Tensor.conj` が壊れる。→ Task 2 の `test_tensorcommon_declarations_are_type_checking_only`。
2. **`join_indices` が呼び出し元の `dirs` リストを書き換える**:今のコードは渡されたリストに `del dirs[i]` をしている。型のために `list(dirs)` でコピーするので、呼び出し元のリストは変わらなくなるべき。→ Task 3 の `test_join_indices_does_not_mutate_dirs_argument`。
3. **必須の引数を省略したときのエラー**:`AbelianTensor.eye`(`qim`)、`initialize_with`(`qhape`)、`from_ndarray`(`shape`、`qhape`)、`split_indices`(`qims`)は、今は `NoneType` に関する分かりにくい `TypeError` で失敗する。`None` の確認を入れるので、何が足りないかを示す `TypeError` になるべき。→ Task 3 の `test_missing_required_form_data_is_type_error`。
4. **`eig` / `svd` / `split` に単一の整数のインデックスを渡す**:`a`、`b` の正規化の書き方を変えるので、`T.svd(0, 1)` のような呼び方が今までどおり動くべき。→ Task 2 の `test_decompositions_accept_single_int_indices`。
5. **pyright のバージョンの違いによる CI の失敗**:`uv.lock` で固定したバージョンを使い、ローカルと CI で同じ結果になるべき。→ Task 1 で `uv run pyright --version` が固定されたバージョンを示すことを確認する。

---

## File Structure

| ファイル | 作業 | 内容 |
|---|---|---|
| `pyproject.toml`、`uv.lock` | 修正 | `[tool.pyright]` と、`dev` グループへの `pyright` の追加 |
| `src/abeliantensors/tensorcommon.py` | 修正 | 型チェッカー専用の宣言、`to_matrix` / `dot` / `eig` / `svd` / `split` の書き方の整理 |
| `src/abeliantensors/abeliantensor.py` | 修正 | 型注釈、`None` の確認、変数の初期化 |
| `src/abeliantensors/tensor.py` | 修正 | 変数の初期化、`np.ndarray` との食い違いの無視 |
| `tests/test_errors.py` | 修正 | Review Focus 2〜4 のテスト |
| `tests/test_tensors.py` または新規テスト | 修正 | Review Focus 1、4 のテスト(Task 2 で `tests/test_errors.py` に置く) |
| `.github/workflows/ci.yml` | 修正 | `uv run pyright` の手順 |
| `docs/superpowers/specs/2026-10-13-typing-design.md` | 修正 | 「無視した箇所」と、`TYPE_CHECKING` による宣言の方針 |

---

### Task 1: pyright の設定と依存関係

**Files:**
- Modify: `pyproject.toml`、`uv.lock`

**Interfaces:**
- Produces: `uv run pyright` で、`src/` を `"standard"` モードで検査できる状態。

- [ ] **Step 1: 依存関係に追加する**

Run: `uv add --dev pyright`
Expected: `pyproject.toml` の `[dependency-groups] dev` に `pyright>=...` が加わり、`uv.lock` が更新される。

- [ ] **Step 2: 設定を追加する**

`pyproject.toml` の `[tool.ruff.lint.per-file-ignores]` の節の後に追加する。

```toml
[tool.pyright]
include = ["src"]
typeCheckingMode = "standard"
pythonVersion = "3.12"
venvPath = "."
venv = ".venv"
```

- [ ] **Step 3: 現状のエラー数を確認する**

Run: `uv run pyright --version && uv run pyright 2>&1 | tail -1`
Expected: バージョンが表示され、`100 errors, 0 warnings, 0 informations`(設計書の調査時点の件数)。数が違う場合は、その数を進捗の記録に書いて先へ進む。

- [ ] **Step 4: コミットする**

```bash
uv lock --check
git add pyproject.toml uv.lock
git commit -m "Add pyright configuration and dev dependency

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `tensorcommon.py`

**Files:**
- Modify: `src/abeliantensors/tensorcommon.py`
- Modify: `tests/test_errors.py`(テストの追加)
- Modify: `docs/superpowers/specs/2026-10-13-typing-design.md`(§2 の表の 1 行目)

**Interfaces:**
- Produces: `TensorCommon` に `if TYPE_CHECKING:` で宣言された `shape`、`qhape`、`dirs`、`charge`、`initialize_with`、`conj`、`join_indices`、`split_indices`、`compatible_indices`(実行時には存在しない)。

- [ ] **Step 1: テストを書く**

`tests/test_errors.py` の import に `TensorCommon` を加え(`from abeliantensors import AbelianTensor, Tensor, TensorCommon, TensorZ2`)、末尾に追加する。

```python
def test_tensorcommon_declarations_are_type_checking_only():
    """Tensor inherits conj etc. from np.ndarray; declarations made in
    TensorCommon for type checkers must not shadow them at runtime.
    """
    for name in (
        "shape",
        "qhape",
        "dirs",
        "charge",
        "initialize_with",
        "conj",
        "join_indices",
        "split_indices",
        "compatible_indices",
    ):
        assert name not in vars(TensorCommon)
    T = Tensor.random([2, 2]) + 1j * Tensor.random([2, 2])
    assert np.allclose(T.conj(), np.conj(np.asarray(T)))


@pytest.mark.parametrize("cls", [Tensor, TensorZ2])
def test_decompositions_accept_single_int_indices(cls):
    if cls is Tensor:
        T = Tensor.random([2, 3])
    else:
        T = TensorZ2.random([[1, 1], [2, 1]], dirs=[1, -1])
    U, S, V = T.svd(0, 1)
    assert np.allclose(
        U.dot(S.diag(), (1, 0)).dot(V, (1, 0)).to_ndarray(), T.to_ndarray()
    )
    US, SV = T.split(0, 1)
    assert np.allclose(US.dot(SV, (1, 0)).to_ndarray(), T.to_ndarray())
    US, SV, rel_err = T.split(0, 1, return_rel_err=True)
    assert rel_err < 1e-12
    M = T.dot(T.conj(), (1, 1))
    S, U = M.eig(0, 1, hermitian=True)
    assert S.shape is not None
```

Run: `uv run pytest -q -p no:randomly tests/test_errors.py -k "type_checking_only or single_int" 2>&1 | tail -2`
Expected: 3 passed(今の動作を固定するテスト。この後の変更で壊れないことを確かめるために使う)。

- [ ] **Step 2: 型チェッカー専用の宣言を追加する**

import を次に置き換える。

```python
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
```

クラスの docstring の直後(`# Initializing tensors` のコメントの前)に追加する。

```python
    if TYPE_CHECKING:
        # Attributes and methods that the subclasses (Tensor, AbelianTensor)
        # provide and that the methods of this class use. They are declared
        # only for type checkers: at runtime they must not exist here, because
        # Tensor inherits some of them (e.g. conj) from np.ndarray, which comes
        # after TensorCommon in its MRO.
        shape: Any
        qhape: Any
        dirs: Any
        charge: Any

        @classmethod
        def initialize_with(cls, *args: Any, **kwargs: Any) -> Any: ...

        def conj(self, *args: Any, **kwargs: Any) -> Any: ...

        def join_indices(self, *args: Any, **kwargs: Any) -> Any: ...

        def split_indices(self, *args: Any, **kwargs: Any) -> Any: ...

        def compatible_indices(self, *args: Any, **kwargs: Any) -> Any: ...
```

(引数を `*args: Any, **kwargs: Any` にするのは、サブクラスごとに引数が違っても「互換性のない上書き」と判定されないようにするため。戻り値を `Any` と明示するのは、`NoReturn` と推論されて呼び出し後のコードが到達不能とみなされるのを防ぐため。)

- [ ] **Step 3: `to_matrix` の変数が未定義になりうる書き方を整理する**

`to_matrix` の本体(`if dirs is None:` のブロックより後、メソッドの末尾まで)を次に置き換える。

```python
        # join_indices does not return a matrix with left_inds as the first
        # index and right_inds as the second, so we may have to transpose.
        swap = bool(left_inds and right_inds and left_inds[0] > right_inds[0])
        if not return_transposed_shape_data:
            result = self.join_indices(left_inds, right_inds, dirs=dirs)
            if swap:
                result = result.swapaxes(1, 0)
            return result

        (
            result,
            transposed_shape,
            transposed_qhape,
            transposed_dirs,
        ) = self.join_indices(
            left_inds,
            right_inds,
            dirs=dirs,
            return_transposed_shape_data=True,
        )
        if swap:
            result = result.swapaxes(1, 0)
            # Do the corresponding swap for the transposed form data as well.
            ts_left = transposed_shape[: len(right_inds)]
            ts_right = transposed_shape[len(right_inds) :]
            transposed_shape = ts_right + ts_left
            if transposed_qhape is not None:
                qs_left = transposed_qhape[: len(right_inds)]
                qs_right = transposed_qhape[len(right_inds) :]
                transposed_qhape = qs_right + qs_left
            if transposed_dirs is not None:
                qs_left = transposed_dirs[: len(right_inds)]
                qs_right = transposed_dirs[len(right_inds) :]
                transposed_dirs = qs_right + qs_left
        return result, transposed_shape, transposed_qhape, transposed_dirs
```

- [ ] **Step 4: `dot` の `None` の扱いを明示する**

`dot` の次の部分を置き換える(`try` / `except TypeError` で `None` を扱っていたのを、`None` の確認に変える。`Tensor` では `qhape` と `dirs` が `None`、`AbelianTensor` では両方とも値があるので、動作は同じ)。

置き換え前:
```python
        try:
            l_qims = s_transposed_qhape[: len(s_open)]
            r_qims = o_transposed_qhape[len(o_sum) :]
        except TypeError:
            l_qims = None
            r_qims = None
        try:
            l_dirs = s_transposed_dirs[: len(s_open)]
            r_dirs = o_transposed_dirs[len(o_sum) :]
        except TypeError:
            l_dirs = None
            r_dirs = None
```
置き換え後:
```python
        if s_transposed_qhape is not None and o_transposed_qhape is not None:
            l_qims = s_transposed_qhape[: len(s_open)]
            r_qims = o_transposed_qhape[len(o_sum) :]
        else:
            l_qims = None
            r_qims = None
        if s_transposed_dirs is not None and o_transposed_dirs is not None:
            l_dirs = s_transposed_dirs[: len(s_open)]
            r_dirs = o_transposed_dirs[len(o_sum) :]
        else:
            l_dirs = None
            r_dirs = None
```

- [ ] **Step 5: `eig` と `svd` のインデックスの正規化**

`eig` と `svd` の両方で、次の 4 行を置き換える。

置き換え前:
```python
        if not isinstance(a, Iterable):
            a = (a,)
        if not isinstance(b, Iterable):
            b = (b,)
```
置き換え後:
```python
        a = tuple(a) if isinstance(a, Iterable) else (a,)
        b = tuple(b) if isinstance(b, Iterable) else (b,)
```

(リストで渡された場合もタプルになるが、`a` と `b` は長さとスライスにしか使わず、`to_matrix` に渡ったあとも `join_indices` が `list(...)` に直すので、結果は変わらない。)

- [ ] **Step 6: `split` の `svd_result[3]`**

`ret_val += (svd_result[3],)` を `ret_val += (svd_result[-1],)` に置き換え、その直前に次のコメントを加える。

```python
            # With return_rel_err the last element of svd_result is rel_err.
```

- [ ] **Step 7: 確認する**

Run: `uv run pyright 2>&1 | grep tensorcommon.py | head; uv run pyright 2>&1 | tail -1`
Expected: `tensorcommon.py` のエラーが 0 件。全体の件数は 100 件から約 40 件減る。

`TYPE_CHECKING` の宣言によって、`tensor.py` に新しいエラー(`np.ndarray` との兼ね合いで「基底クラスどうしの定義が食い違う」など)が出た場合は、Task 4 でまとめて扱う。ここでは件数を記録するだけにする。

Run: `uv run pytest -q 2>&1 | tail -1 && uv run ruff check && uv run ruff format --check`
Expected: すべて成功

- [ ] **Step 8: 設計書を直してコミットする**

設計書の §2 の表の 1 行目(`TensorCommon` がサブクラスの属性を参照している)の「修正の方法」を、次に置き換える。

```
`TensorCommon` の本体に、`if TYPE_CHECKING:` のブロックで、サブクラスが必ず持つ属性とメソッドの型を宣言する。メソッドは `*args: Any, **kwargs: Any` を受け取り `-> Any` を返す形にする(`raise NotImplementedError` だけの定義は `NoReturn` と推論され、呼び出し後のコードが検査されなくなるため使わない)。実行時には存在しないので、`Tensor` が `np.ndarray` から受け継ぐ `conj` などを隠さない
```

```bash
git add src/abeliantensors/tensorcommon.py tests/test_errors.py docs/superpowers/specs/2026-10-13-typing-design.md
git commit -m "Fix pyright errors in tensorcommon.py

Declare the attributes and methods TensorCommon expects from subclasses
under TYPE_CHECKING (so Tensor still inherits conj etc. from np.ndarray),
and restructure to_matrix, dot, eig, svd and split so that no variable
is possibly unbound or None.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `abeliantensor.py`

**Files:**
- Modify: `src/abeliantensors/abeliantensor.py`
- Modify: `tests/test_errors.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_errors.py` の末尾に追加する。

```python
def test_missing_required_form_data_is_type_error():
    with pytest.raises(TypeError, match="qim"):
        AbelianTensor.eye([1, 1])
    with pytest.raises(TypeError, match="qhape"):
        AbelianTensor.random([[1, 1]], dirs=[1])
    with pytest.raises(TypeError, match="shape"):
        AbelianTensor.from_ndarray(np.zeros(2), qhape=[[0, 1]], dirs=[1])
    with pytest.raises(TypeError, match="qhape"):
        AbelianTensor.from_ndarray(np.zeros(2), shape=[[1, 1]], dirs=[1])
    T = AbelianTensor.random([[2, 1]], qhape=[[0, 1]], dirs=[1])
    with pytest.raises(TypeError, match="qims"):
        T.split_indices(0, [[1, 1], [1]])


def test_join_indices_does_not_mutate_dirs_argument():
    T = z2([[1, 1], [1, 1], [1, 1]], [1, 1, 1])
    dirs = [1]
    T.join_indices([0, 1], dirs=dirs)
    assert dirs == [1]
```

Run: `uv run pytest -q -p no:randomly tests/test_errors.py -k "missing_required or mutate_dirs" 2>&1 | grep -E "^E  |passed|failed" | head`
Expected: `test_missing_required_form_data_is_type_error` は失敗する(今は `NoneType` に関するメッセージの `TypeError` なので、`match` に合わない)。`test_join_indices_does_not_mutate_dirs_argument` は失敗する(`dirs` が `[]` になる)。後者が通ってしまう場合は、`join_indices` が `dirs` を消さない経路になっているので、`T.join_indices([0, 1], [2], dirs=[1, 1])` のように 2 つのバッチで試し、`dirs` が書き換わる入力を見つけてテストを直す。

- [ ] **Step 2: 引数の型注釈(dtype、casting)**

`from typing import ...` がなければ、import 群に次を加える。

```python
from typing import Literal

from numpy.typing import DTypeLike
```

(`numpy.typing` はサードパーティの import の位置に置く。ruff の `I` ルールに従い `uv run ruff check --fix` で並べ替える。)

- `__init__` のシグネチャの `dtype=np.float64,` を `dtype: DTypeLike = np.float64,` に変える。
- `eye` のシグネチャの `dtype=np.float64` を `dtype: DTypeLike = np.float64` に変える。
- `astype` のシグネチャを次に変える。

```python
    def astype(
        self,
        dtype: DTypeLike,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "unsafe",
        copy: bool = True,
    ):
```

(これで `abeliantensor.py:667`、`:2036`、`:2150`、`:2309`、`:2318` の 5 件が消える。)

- [ ] **Step 3: 必須の引数の `None` の確認**

- `eye`:`if not cls.check_qim_dim_match(qim, dim):` の直前に追加する。

```python
        if qim is None:
            raise TypeError(f"{cls.__name__}.eye requires qim.")
```

- `initialize_with`:`symmetry = _resolve_symmetry(cls, symmetry, qodulus)` の直前に追加する。

```python
        if qhape is None:
            raise TypeError(f"{cls.__name__} requires a qhape.")
```

- `from_ndarray`:最初の `if dirs is None:` の直前に追加する。

```python
        if shape is None or qhape is None:
            raise TypeError(f"{cls.__name__}.from_ndarray requires shape and qhape.")
```

- `split_indices`:docstring の直後(`# Formatting the input` のコメントの前)に追加する。

```python
        if qims is None:
            raise TypeError(f"{type(self).__name__}.split_indices requires qims.")
```

(行が長いものは `uv run ruff format` で整形する。)

- [ ] **Step 4: `_sorted_shape_qhape` と `check_form_match`**

`_sorted_shape_qhape` の次の 2 行を置き換える。

置き換え前:
```python
        shape = tensor.shape if shape is None else shape
        qhape = tensor.qhape if qhape is None else qhape
```
置き換え後:
```python
        if shape is None or qhape is None:
            if tensor is None:
                raise TypeError("Give either a tensor or both shape and qhape.")
            shape = tensor.shape if shape is None else shape
            qhape = tensor.qhape if qhape is None else qhape
```

`check_form_match` の、`if tensor2 is not None:` のブロックの直後(`if not (` の直前)に追加する。

```python
        if (
            qhape1 is None
            or shape1 is None
            or dirs1 is None
            or qhape2 is None
            or shape2 is None
            or dirs2 is None
        ):
            raise TypeError(
                "Give tensor1 and tensor2, or their qhape, shape and dirs."
            )
```

- [ ] **Step 5: `_find_trunc_dim` の `chis`**

シグネチャの `chis=None,` を `chis,` に変える(呼び出し元はすべて `chis=` を渡している。`grep -n "_find_trunc_dim(" src/abeliantensors/abeliantensor.py` で確認する)。

本体の `# Find the smallest chi for which the error is small enough.` のコメントの直前に追加する。

```python
        if not chis:
            raise ValueError("chis must not be empty.")
        # Overwritten below; initialized so that they are always bound.
        chi = chis[0]
        err = 0
```

- [ ] **Step 6: `join_indices` の `dirs` と、変形前の形のデータ**

`join_indices` の次の部分を置き換える。

置き換え前:
```python
        else:
            if not isinstance(dirs, Iterable):
                dirs = [dirs]
            if len(dirs) != len(index_batches):
```
置き換え後:
```python
        else:
            # Copy, since dirs is modified below.
            dirs = list(dirs) if isinstance(dirs, Iterable) else [dirs]
            if len(dirs) != len(index_batches):
```

同じ関数の `transposed_shape = res.shape.copy()` を含む `if return_transposed_shape_data:` のブロックの直前に追加する。

```python
        transposed_shape = transposed_qhape = transposed_dirs = None
```

- [ ] **Step 7: `split_indices` の入力の整形**

置き換え前:
```python
        if isinstance(indices, Iterable):
            if not len(indices) == len(dims) == len(qims):
                raise ValueError(
                    "indices, dims and qims must have the same length."
                )
            indices = list(indices)
            dim_batches = list(map(list, dims))
            qim_batches = list(map(list, qims))
```
置き換え後:
```python
        if isinstance(indices, Iterable):
            indices = list(indices)
            if not len(indices) == len(dims) == len(qims):
                raise ValueError(
                    "indices, dims and qims must have the same length."
                )
            dim_batches = [list(dim) for dim in dims]
            qim_batches = [list(qim) for qim in qims]
```

- [ ] **Step 8: `matrix_svd` の `svds`**

`u, s, v = spsla.svds(...)` の呼び出しの直後(`order = np.argsort(-s)` の前)に追加する。

```python
                    if u is None or v is None:
                        # Not reached: we ask svds for the singular vectors.
                        raise RuntimeError("svds returned no singular vectors.")
```

- [ ] **Step 9: 確認する**

Run: `uv run ruff check --fix && uv run ruff format && uv run pyright 2>&1 | grep abeliantensor.py; uv run pyright 2>&1 | tail -1`
Expected: `abeliantensor.py` のエラーが 0 件。

残っている場合は、そのエラーの種類に応じて設計書 §2 の表の方法で直す(新しい `# pyright: ignore` は使わない。使う必要があると判断したら、進捗の記録に判断として残す)。

Run: `uv run pytest -q 2>&1 | tail -1`
Expected: すべて成功(Step 1 の 2 件を含む)

- [ ] **Step 10: コミットする**

```bash
git add src/abeliantensors/abeliantensor.py tests/test_errors.py
git commit -m "Fix pyright errors in abeliantensor.py

Annotate dtype/casting parameters, check required form data for None
with clear TypeErrors, make _find_trunc_dim's chis required, and copy
the dirs argument in join_indices instead of mutating the caller's list.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `tensor.py`

**Files:**
- Modify: `src/abeliantensors/tensor.py`
- Modify: `docs/superpowers/specs/2026-10-13-typing-design.md`(「無視した箇所」)

- [ ] **Step 1: `_find_trunc_dim`**

シグネチャの `chis=None,` を `chis,` に変える。本体の `# Find the smallest chi for which the error is small enough.` のコメントの直前に追加する。

```python
        if not chis:
            raise ValueError("chis must not be empty.")
        # Overwritten below; initialized so that they are always bound.
        chi = chis[0]
        err = 0
```

- [ ] **Step 2: `join_indices` と `split_indices`**

`join_indices` の `if return_transposed_shape_data:` / `transposed_shape = self.shape` の直前に `transposed_shape = None` を追加する。

`split_indices` の次の部分を置き換える。

置き換え前:
```python
        if isinstance(indices, Iterable):
            if len(indices) != len(dims):
                raise ValueError("indices and dims must have the same length.")
            indices = list(indices)
            dim_batches = list(map(list, dims))
```
置き換え後:
```python
        if isinstance(indices, Iterable):
            indices = list(indices)
            if len(indices) != len(dims):
                raise ValueError("indices and dims must have the same length.")
            dim_batches = [list(dim) for dim in dims]
```

- [ ] **Step 3: `matrix_svd` の `svds`**

`U, S, V = spsla.svds(self, k=maxchi, return_singular_vectors=True)` の直後に追加する。

```python
            if U is None or V is None:
                # Not reached: we ask svds for the singular vectors.
                raise RuntimeError("svds returned no singular vectors.")
```

- [ ] **Step 4: `np.ndarray` との食い違いを、理由付きで無視する**

次の各行の末尾に `# pyright: ignore[ルール名]` を付け、直前の行に理由のコメントを書く。

| 行 | ルール | 理由のコメント |
|---|---|---|
| `def real(self):` | `reportIncompatibleMethodOverride` | `# A method, not a property as in ndarray, to match AbelianTensor.real().` |
| `return super().real`(`real` の中) | `reportAttributeAccessIssue` | (上と同じ理由なのでコメント不要) |
| `def imag(self):` | `reportIncompatibleMethodOverride` | `# A method, not a property as in ndarray, to match AbelianTensor.imag().` |
| `return super().imag`(`imag` の中) | `reportAttributeAccessIssue` | (同上) |
| `def sum(self):` | `reportIncompatibleMethodOverride` | `# Takes no axis argument and returns a scalar, like AbelianTensor.sum().` |
| `def trace(self, axis1=0, axis2=1):` | `reportIncompatibleMethodOverride` | `# Returns a Tensor and checks compatibility, like AbelianTensor.trace().` |
| `def dot(self, B, indices):` | `reportIncompatibleMethodOverride` | `# tensordot-style signature, unlike ndarray.dot; parameter named B for backward compatibility.` |

Task 2 の `TYPE_CHECKING` の宣言によって、`class Tensor(TensorCommon, np.ndarray):` の行などに新しいエラー(基底クラスどうしの定義の食い違い)が出ている場合は、同じ方法で無視し、表に加える。

- [ ] **Step 5: 確認する**

Run: `uv run pyright 2>&1 | tail -3`
Expected: `0 errors, 0 warnings, 0 informations`

Run: `uv run pytest -q 2>&1 | tail -1 && uv run ruff check && uv run ruff format --check`
Expected: すべて成功

- [ ] **Step 6: 設計書の「無視した箇所」を記録してコミットする**

設計書の末尾の `(実装時に記入する。...)` を、実際に `# pyright: ignore` を付けた箇所の表(ファイル:行、ルール、理由)に置き換える。

Run: `grep -n "pyright: ignore" src/abeliantensors/*.py`
(この出力と表が一致していることを確認する。)

```bash
git add src/abeliantensors/tensor.py docs/superpowers/specs/2026-10-13-typing-design.md
git commit -m "Fix pyright errors in tensor.py; record ignored ndarray overrides

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: CI に pyright を加える

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: 手順を追加する**

`test` ジョブの `- run: uv run ruff format --check` の直後に追加する。

```yaml
      - run: uv run pyright
```

- [ ] **Step 2: 確認する**

Run: `uvx --from check-jsonschema check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/ci.yml`
Expected: `ok -- validation done`

Run: `for v in 3.12 3.13 3.14; do echo "== $v"; uv run --python $v --isolated pyright 2>&1 | tail -1; done`
Expected: 3 つとも `0 errors`

- [ ] **Step 3: 最終確認とコミット**

Run:
```bash
uv run pytest -q 2>&1 | tail -1
uv run ruff check && uv run ruff format --check
uv run --group docs sphinx-build -W -q -b html docs/source docs/build/html
uv lock --check
```
Expected: すべて成功

```bash
git add .github/workflows/ci.yml
git commit -m "Run pyright in CI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
