# 型チェック(pyright)への対応 設計書

- 日付: 2026-10-13
- ブランチ: `typing`
- 状態: フェーズ A、B 完了(2026-10-14)

## 目的

VS Code で `src/` のファイルを開くと、Pylance がエラーを表示する。このエラーをなくし、型チェックを CI で維持する(フェーズ A)。そのうえで、公開 API に型注釈を付け、利用者のコードでも IDE の補完や型の検査が効くようにする(フェーズ B)。

### 前提・制約

- 現状(master の `c805b46` 時点)では、pyright 1.1.414 の `"standard"` モードで、`src/` に 100 件のエラーがある。内訳は次のとおり。

  | 種類 | 件数 |
  |---|---|
  | 引数の型の不一致(`None` を既定値にした引数を、確認せずに使っているなど) | 39 |
  | 代入されていない可能性がある変数 | 23 |
  | 継承して使う前提のクラス(`TensorCommon`)が、サブクラスにしかない属性を参照している | 19 |
  | `None` かもしれない値への添字アクセスや反復 | 12 |
  | `np.ndarray` との兼ね合い(`Tensor` がプロパティをメソッドで上書きしているなど) | 7 |

- 利用者の Pylance の設定は `typeCheckingMode = "basic"` である。
- 実行時の動作は変えない。公開 API(メソッド名、引数、戻り値の形)も変えない。
- 対象は `src/` と、型の検査のために新しく作る `tests/typing/` に限る。既存の `tests/`、`demo/`、`benchmarks/` は対象外とする。

### 完了条件

**フェーズ A**
1. `uv run pyright` が `src/` でエラー 0 件になる。
2. CI の `test` ジョブ(Python 3.12 / 3.13 / 3.14)で `uv run pyright` を実行し、成功する。
3. テストが変更前と同じくすべて成功する。

**フェーズ B**
1. `uv run pyright` が `src/` と `tests/typing/` でエラー 0 件になる。
2. 公開メソッド(先頭が `_` でないもの、特殊メソッドを含む)の引数と戻り値に、すべて型注釈がある。
3. `tests/typing/` で、主要な API(テンソルの作成、`dot`、`svd`、`eig`、`split`、演算子)の推論結果を `assert_type` で確かめている。

## 設計

### 1. 設定(フェーズ A)

`pyproject.toml` に次を追加する。

```toml
[tool.pyright]
include = ["src"]
typeCheckingMode = "standard"
pythonVersion = "3.12"
venvPath = "."
venv = ".venv"
```

- Pylance はリポジトリの `[tool.pyright]` を利用者の設定より優先するので、エディタの表示と CI の結果がそろう。エディタでの検査は、今の `"basic"` より少し厳しい `"standard"` になる。
- pyright は `dev` の依存グループに加え、`uv.lock` でバージョンを固定する。ローカルでも CI でも `uv run pyright` で実行する。
- フェーズ B で `tests/typing/` を作った時点で、`include` に `"tests/typing"` を加える。

### 2. フェーズ A の修正方針

| 種類 | 修正の方法 |
|---|---|
| `TensorCommon` がサブクラスの属性を参照している | `TensorCommon` の本体に、`if TYPE_CHECKING:` のブロックで、サブクラスが必ず持つ属性とメソッドの型を宣言する。メソッドは `*args: Any, **kwargs: Any` を受け取り `-> Any` を返す形にする(`raise NotImplementedError` だけの定義は `NoReturn` と推論され、呼び出し後のコードが検査されなくなるため使わない)。実行時には存在しないので、`Tensor` が `np.ndarray` から受け継ぐ `conj` などを隠さない |
| `None` を既定値にした引数 | 引数に `X \| None` の型注釈を付け、使う前に `None` を確かめる。関数の中で既定値を代入している場合は、その後の型が確定するように書く |
| 代入されていない可能性がある変数 | 分岐の前で初期化するか、分岐の書き方を整理する。動作は変えない |
| `np.ndarray` との兼ね合い | 公開 API を変えずには直せないので、`# pyright: ignore[ルール名]` に理由を付けて、その行だけを無視する |
| 戻り値の要素数がフラグで変わる(`svd_result[3]` など) | 変数への代入の仕方を整理して回避する。`@overload` による本格的な対応はフェーズ B で行う |

- `# pyright: ignore` は、理由をコメントに書けるものに限って使う。無視した箇所の一覧を、本書の末尾の「無視した箇所」に記録する。
- 各コミットで、テストがすべて成功することを確認する。

### 3. CI(フェーズ A)

`.github/workflows/ci.yml` の `test` ジョブの、`uv run ruff format --check` の後に `uv run pyright` を加える。

### 4. フェーズ B の段階

| 段階 | 対象 | 内容 |
|---|---|---|
| B-1 | `symmetry.py`、`symmetrytensors.py` | 型注釈を付ける。`py.typed` を追加する |
| B-2 | `tensorcommon.py`、`_typing.py` | 共通インターフェースの型を決め、`@overload` を書く。`tests/typing/` を作る |
| B-3 | `abeliantensor.py` | 型注釈を付ける。演算子の型を付ける |
| B-4 | `tensor.py` | 型注釈を付ける。`np.ndarray` と食い違う箇所を無視する |

各段階の終わりに、pyright のエラー 0 件を CI で維持する。

**進め方(2026-10-14 追記)**

- 実装計画は段階ごとに作る。1 回目は B-1 + B-2、2 回目は B-3、3 回目は B-4。前の段階で決めた型の書き方(とくに B-2 の `TensorCommon`)を、次の段階の計画に反映するため。
- 各回の終わりに、master へマージできる状態(pyright 0 件、テストすべて成功)にする。

**型注釈の網羅の検査(2026-10-14 追記)**

- 完了条件 2(公開メソッドに型注釈がある)は pyright の `"standard"` モードでは検査されないので、ruff の `ANN` ルール(flake8-annotations)で検査する。対象は `src/` のみ。
  - 有効にするルール: `ANN001`、`ANN002`、`ANN003`(引数)、`ANN201`(公開関数の戻り値)、`ANN204`(特殊メソッドの戻り値)、`ANN205`(静的メソッドの戻り値)、`ANN206`(クラスメソッドの戻り値)
  - `ANN202`(非公開関数の戻り値)は有効にしない。設計の範囲を公開 API に限るため。
  - まだ注釈を付けていないファイルは `per-file-ignores` で一時的に除外し、段階ごとに外していく。フェーズ B の終わりには、`src/` に `ANN` の除外が残っていないこと。

### 5. フェーズ B の型の設計

**型の別名(新規 `src/abeliantensors/_typing.py`、非公開)**

```python
from collections.abc import Sequence

Qnum = int | tuple[int, ...]       # 量子数(ProductGroup ではタプル)
Dim = list[int]                    # 1 本のインデックスのセクターごとの次元
Qim = list[Qnum]                   # 1 本のインデックスの量子数
Indices = int | Sequence[int]      # dot / svd などに渡すインデックスの指定
```

**群のジェネリクス化(B-1 で決定)**

- `AbelianGroup` は、量子数の型をパラメータにしたジェネリクス `AbelianGroup[Q]`(PEP 695 の構文)にする。`CyclicGroup` と `U1Group` は `AbelianGroup[int]`、`ProductGroup` は `AbelianGroup[tuple[int, ...]]`。基底クラスで量子数の型を `Qnum` に固定すると、サブクラスの実装(整数どうしの足し算など)が型エラーになるため。

**`TensorCommon` の型チェッカー向けの宣言(B-2 で決定)**

- `TensorCommon` のメソッドに型を付けると、本体の中でサブクラスにしかないメソッド(`sqrt`、`multiply_diag` など)を呼んでいる箇所が検査されるようになる。それらは `if TYPE_CHECKING:` のブロックに `(*args: Any, **kwargs: Any) -> Any` の形で宣言する。具体的な型は、B-3、B-4 でサブクラスに付ける。

**`TensorCommon` の属性**

- `shape`、`qhape`、`dirs`、`charge` は、`TensorCommon` では `Any` として宣言し、具体的な型を各サブクラスで宣言する。
  - `AbelianTensor`: `shape: list[Dim]`、`qhape: list[Qim]`、`dirs: list[int]`、`charge: Qnum`
  - `Tensor`: `shape` は `np.ndarray` のもの(`tuple[int, ...]`)。`qhape`、`dirs` は `None`、`charge` は `0`
- ジェネリクス(`TensorCommon[ShapeT, ...]`)は使わない。`Tensor` が `np.ndarray` のサブクラスであることと組み合わせると複雑になりすぎるうえ、`TensorCommon` 自身のメソッドは少ないので、型検査の効果はサブクラスで十分に得られるため。

**戻り値**

- テンソルを返すメソッド(`dot`、`conj`、`transpose`、`join_indices`、`split_indices` など)の戻り値は `Self` にする。

**`@overload`(`typing.Literal` による場合分け)**

| メソッド | フラグ | 場合の数 |
|---|---|---|
| `to_matrix` | `return_transposed_shape_data` | 2 |
| `join_indices`(`AbelianTensor`、`Tensor`) | `return_transposed_shape_data` | 2 |
| `eig`、`svd` | `return_rel_err` | 2 |
| `split` | `return_rel_err`、`return_sings` | 4 |

例: `T.svd([0], [1])` は `tuple[Self, Self, Self]`、`T.svd([0], [1], return_rel_err=True)` は `tuple[Self, Self, Self, float]` と推論される。

**演算子(B-3 で決定)**

- `_generate_binary_deferer` / `_generate_unary_deferer` の戻り値を `Callable[[_AT, Any], _AT]` / `Callable[Concatenate[_AT, ...], _AT]`(`_AT` は `AbelianTensor` を上限とする `TypeVar`)と注釈する。生成された関数をクラス属性に代入すると、pyright はそれをメソッドとして扱い、`_AT` を呼び出し元のクラスに束縛するので、`TensorZ2 + TensorZ2` は `TensorZ2` と推論される。実行時の定義の仕方は変えない。
- `__eq__` / `__ne__` は `object` の定義(戻り値 `bool`)と食い違うため、`if TYPE_CHECKING:` の中にテンソルを返す宣言を書き、`# pyright: ignore[reportIncompatibleMethodOverride]` を付ける(NumPy の型定義と同じやり方)。実行時は従来どおり生成した関数を使う。
- `sqrt` は、`TensorCommon` が自身の型検査のために宣言した `sqrt`(戻り値 `Any`)が生成された関数より優先されてしまうため、同じく `if TYPE_CHECKING:` の中で `Self` を返すと宣言する(無視の指定は不要)。
- `join_indices` の実装のシグネチャは、本体で引数の形を何度も作り変えるため `Any` にし、公開される型は `@overload` で与える。

**`Tensor`(`np.ndarray` のサブクラス)**

- 演算子(2026-10-14 追記、B-4 で実施): `Tensor` の演算子は `np.ndarray` から受け継ぐので、NumPy の型定義のままだと `Tensor + Tensor` が `ndarray` と推論され、`(D + D).svd(0, 1)` のように結果に対してこのライブラリのメソッドを呼ぶコードが型エラーになる。`Tensor` は `TensorZ2` などと入れ替えて使うためのクラスなので、算術・ビット演算・比較・単項の演算子と、その右側版・累算代入版を `if TYPE_CHECKING:` の中で `Self` を返すと宣言する。NumPy の型定義と食い違うもの(`__sub__`、`__truediv__` など 8 個)は理由付きで無視する。`__divmod__` / `__rdivmod__` は実行時に配列のタプルを返すので宣言しない(NumPy の型定義のまま)。実行時の動作は変えない。
- 公開 API は変えない。NumPy の型定義と食い違う上書き(`real`、`imag`、`sum`、`dot` など)は、`# pyright: ignore[ルール名]` に理由を付けて無視し、本書の「無視した箇所」に記録する。

**`py.typed`(B-1)**

- `src/abeliantensors/py.typed`(空のファイル)を追加する(PEP 561)。これにより、別のプロジェクトから abeliantensors を使うときにも、Pylance がこのライブラリの型注釈を使う。

**型の検査のテスト(B-2 以降、`tests/typing/`)**

- 推論される型を `typing.assert_type` で確かめるファイルを置き、pyright の検査対象に加える。pytest では収集しない(ファイル名を `check_*.py` にする)。
- 例: `assert_type(T.svd([0], [1]), tuple[TensorZ2, TensorZ2, TensorZ2])`

## 対象外

- strict モードでの型チェック
- `tests/`(`tests/typing/` を除く)、`demo/`、`benchmarks/` の型注釈
- 実行時の型検査(`isinstance` による引数の検査の追加など)

## 無視した箇所

フェーズ A の時点で `# pyright: ignore` を使った箇所。どれも `Tensor` が `np.ndarray` のサブクラスであることによるもので、公開 API を変えずには直せない。

| 箇所 | ルール | 理由 |
|---|---|---|
| `tensor.py` `class Tensor(TensorCommon, np.ndarray)` | `reportIncompatibleMethodOverride` | `TensorCommon` が型チェッカー向けに宣言した `shape: Any` が、`ndarray.shape` プロパティと食い違う。実行時には `ndarray.shape` が使われる |
| `tensor.py` `def real(self)` | `reportIncompatibleMethodOverride` | `ndarray.real` はプロパティだが、`AbelianTensor.real()` に合わせてメソッドにしている |
| `tensor.py` `real` の中の `super().real` | `reportAttributeAccessIssue` | 同上 |
| `tensor.py` `def imag(self)` | `reportIncompatibleMethodOverride` | `ndarray.imag` はプロパティだが、`AbelianTensor.imag()` に合わせてメソッドにしている |
| `tensor.py` `imag` の中の `super().imag` | `reportAttributeAccessIssue` | 同上 |
| `tensor.py` `def sum(self)` | `reportIncompatibleMethodOverride` | `AbelianTensor.sum()` に合わせて、軸の引数を取らずにスカラーを返す |
| `tensor.py` `def trace(self, axis1=0, axis2=1)` | `reportIncompatibleMethodOverride` | `AbelianTensor.trace()` に合わせて、`Tensor` を返し、インデックスの互換性を確かめる |
| `tensor.py` `def dot(self, B, indices)` | `reportIncompatibleMethodOverride` | `ndarray.dot` と違い `tensordot` 形式の引数をとる。また、引数名が `TensorCommon.dot` の `other` と違って `B` だが、後方互換のためそのままにしている |
| `abeliantensor.py` `__eq__`(`if TYPE_CHECKING:` の中の宣言) | `reportIncompatibleMethodOverride` | 要素ごとの比較の結果としてテンソルを返す。`object.__eq__` の戻り値 `bool` と食い違う |
| `abeliantensor.py` `__ne__`(同) | `reportIncompatibleMethodOverride` | 同上 |
| `tensor.py` `if TYPE_CHECKING:` の中の演算子の宣言(`__sub__`、`__rsub__`、`__truediv__`、`__floordiv__`、`__pow__`、`__rpow__`、`__eq__`、`__ne__` の 8 個) | `reportIncompatibleMethodOverride` | NumPy の型定義では、これらの演算子は引数の型ごとに細かく場合分けされているか、`object` の定義(`bool` を返す)と食い違う。実行時の `Tensor` は NumPy の演算子を使い、結果は `Tensor` になるので、`Self` を返すと宣言する。8 個とも、外すと pyright のエラーになることを確認済み |
