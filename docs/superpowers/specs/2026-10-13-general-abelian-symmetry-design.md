# 一般のアーベル対称性への拡張 設計書

- 日付: 2026-10-13
- ブランチ: `general-abelian`
- 状態: レビュー待ち

## 目的

`AbelianTensor` が扱える対称性を、今の Z_N と U(1) から、一般のアーベル群に広げる。
今回の実装で動作を保証するのは、有限アーベル群の直積 Z_{n1} × Z_{n2} × …(例: Z₂×Z₂、Z₂×Z₃)である。
U(1) を含む直積(例: U(1)×U(1)、U(1)×Z₂)は将来対応する予定とし、そのとき API を変えずに追加できる設計にする。

### 前提・制約

- 今の対称性は `qodulus`(整数 N、または None)だけで決まり、量子数の演算はコードの中に「足し算して mod を取る」形で直接書かれている(`abeliantensor.py` で `qodulus` が 55 箇所、`_qod_func` の呼び出しが 13 箇所)。
- `tensor_z2z2` ブランチ(2022 年)では、群オブジェクトという考え方を試していた。ただし `abeliantensor.py` を丸ごと複製しており、さらに `normalize = abs` という Z₂×Z₂ でしか成り立たない書き方を使っていた。今回はこの考え方だけを受け継ぎ、複製はしない。
- 互換性: `TensorZ2`、`TensorZ3`、`TensorU1`、`TensorZN` は引き続き使えるようにする。`qodulus` は非推奨とし、`DeprecationWarning` を出す。pickle などで保存したテンソルとの互換性は保証しない。
- 性能: 既存の `TensorZ2` の TRG 1 ステップの時間が、今の master と比べて 20% を超えて遅くならないこと。

### 完了条件

1. `uv run pytest` が警告なしですべて成功する。テスト対象のクラスは `Tensor`、`TensorZ2`、`TensorZ3`、`TensorU1`、`TensorZ2Z2`、`TensorZ2Z3`。
2. `tests/test_symmetry.py` で、群の公理と `normalize` の入力検証を確認するテストが成功する。
3. `uv run ruff check`、`uv run ruff format --check`、ドキュメントのビルド(`sphinx-build -W`)が成功する。
4. `benchmarks/trg_bench.py` で、`TensorZ2` の速度低下が chi = 8、16、24 のいずれでも 20% 以内である。計測結果を本書の末尾とコミットメッセージに記録する。

## 設計

### 1. 群オブジェクト(新規 `src/abeliantensors/symmetry.py`)

```python
class AbelianGroup:
    """量子数の演算を定義するアーベル群の基底クラス。値オブジェクトとして扱う。"""

    identity                          # 単位元
    def add(self, a, b): ...          # a + b(サブクラスが実装する)
    def neg(self, a): ...             # -a(逆元。サブクラスが実装する)
    def normalize(self, q): ...       # 利用者からの入力を標準形に直す。不正な値なら ValueError
    def elements(self): ...           # 全元のリスト(有限群)、または None

    def signed(self, a, d):           # 基底クラスで実装する。向き d (+1 / -1) を適用する
        if d == 1:
            return a
        if d == -1:
            return self.neg(a)
        raise ValueError(f"direction must be +1 or -1, got {d}")

    def sum(self, qs):                # 基底クラスで実装する。identity から add で畳み込む
        ...
    # __eq__ / __hash__ / __repr__ は値で比べる

class CyclicGroup(AbelianGroup):      # Z_n。量子数は int。0..n-1 に正規化する
class U1Group(AbelianGroup):          # U(1)。量子数は int。正規化しない。elements() は None
class ProductGroup(AbelianGroup):     # 直積。量子数は成分ごとの int のタプル
    def __init__(self, *factors): ... # 成分は CyclicGroup または U1Group(2 つ以上)
```

- 量子数に掛ける整数は、今のコードではすべて向き ±1(またはその積)である。`dirs` は `check_consistency` で ±1 に限られている。そのため一般の n 倍は用意せず、`signed(a, d)` だけを用意する。
- `normalize` は、利用者からの入力(`qhape`、`charge`)を受け取る境界でだけ呼ぶ。内部の演算結果は常に標準形なので、計算のたびに正規化しない。
  - `CyclicGroup(n)`: 整数(numpy の整数型を含む)を受け取り、`int(q) % n` を返す。整数でない値は `ValueError`。
  - `U1Group`: 整数を受け取り、`int(q)` を返す。
  - `ProductGroup`: 長さが成分の数と等しいタプル(またはリスト)だけを受け取り、成分ごとに `normalize` してタプルで返す。長さが違う場合や、整数を 1 つだけ渡された場合は `ValueError`。
- 量子数は常に Python の `int`、または `int` のタプルにする。辞書のキーの比較とハッシュを安定させるため。
- `elements()`: `CyclicGroup(n)` は `list(range(n))` を返す。`ProductGroup` は、すべての成分が有限なら `itertools.product` で作った全元を辞書順に並べて返し、U(1) の成分が 1 つでもあれば `None` を返す。
- 量子数のソートは、整数どうしでもタプルどうしでも可能なので、既存の `sorted(zip(qim, dim))` はそのまま使える。

### 2. `AbelianTensor` を群オブジェクトに対応させる(`abeliantensor.py`)

**属性と引数**
- `self.qodulus` の代わりに `self.symmetry`(`AbelianGroup`)を持たせる。既定値は `U1Group()`。今の `qodulus=None` の動作と同じ。
- テンソルを作るすべての入口で `symmetry=` 引数を受け取る。対象は `__init__`、`initialize_with`(とそれを使う `zeros`、`ones`、`random` など)、`eye`、`from_ndarray`、そして内部でテンソルを作る箇所(`empty_like`、`copy`、`matrix_dot`、`svd`、`eig` など)。
- `charge` 引数の既定値は `0` から `None` に変え、`None` のときは `symmetry.identity` を使う。
- `qhape` と `charge` は、入口で `symmetry.normalize` を通す。

**`qodulus` の非推奨化**
- `qodulus=N` が渡されたら、`DeprecationWarning`(`stacklevel=2`)を出したうえで `CyclicGroup(N)` に変換する。`qodulus=None` は `U1Group()` に変換する。
- `qodulus` と `symmetry` が両方渡されたら `TypeError`。
- インスタンスの `.qodulus` は、`DeprecationWarning` を出す読み取り専用のプロパティにする。`CyclicGroup` なら N、`U1Group` なら None を返し、`ProductGroup` では `AttributeError`。
- サブクラスのクラス属性 `qodulus` は削除し、`symmetry` に置き換える。そのため `TensorZ2.qodulus` のように、クラスから直接 `qodulus` を読む書き方は動かなくなる(リポジトリ内に使っている箇所はない)。

**量子数の演算の置き換え**(`g = self.symmetry`)

| 今のコード | 書き換え後 |
|---|---|
| `sum(map(opr.mul, dirs, key)) % N` | `g.sum(map(g.signed, key, dirs))` |
| `-q`、`q * d`、`x * flux` | `g.neg(q)`、`g.signed(q, d)`、`g.signed(x, flux)` |
| `_qod_func(a + b)` | `g.add(a, b)` |
| `charge != 0`、`qnum == 0` | `!= g.identity`、`== g.identity` |
| `self.qodulus == B.qodulus` | `self.symmetry == B.symmetry` |
| `trace` の `(k[a1] - k[a2]) % N == 0` | `k[a1] == k[a2]` |

- `_qod_func` は削除する。
- `check_consistency` は、`qodulus` で正規化されているかの確認を、`g.normalize(q) == q` の確認に置き換える。
- `__repr__` と `__str__` は `symmetry=` を表示する。あわせて既存の書式の誤り(`dtype=% rdefval=`)も直す。

### 3. テンソルのサブクラス(`symmetrytensors.py`)

```python
class FiniteAbelianTensor(AbelianTensor):     # qhape の自動生成を担う(今の TensorZN の処理を一般化)
class TensorZN(FiniteAbelianTensor)           # 互換のため残す
class TensorZ2(TensorZN):   symmetry = CyclicGroup(2)
class TensorZ3(TensorZN):   symmetry = CyclicGroup(3)
class TensorU1(AbelianTensor):   symmetry = U1Group()
class TensorZ2Z2(FiniteAbelianTensor):   symmetry = ProductGroup(CyclicGroup(2), CyclicGroup(2))
class TensorZ2Z3(FiniteAbelianTensor):   symmetry = ProductGroup(CyclicGroup(2), CyclicGroup(3))
```

- `FiniteAbelianTensor` は、`__init__`、`eye`、`initialize_with`、`from_ndarray`、`split_indices` で、`qhape` が省略されたときに自動で作る。また `symmetry=cls.symmetry` を固定する。
- `_dim_to_qim(dim)` は次のように動く。
  - セクターが 1 つなら `[symmetry.identity]` を返す。
  - セクター数が群の位数と等しければ `symmetry.elements()` を返す。
  - それ以外は `ValueError`(「`qhape` を明示してください」)。
- `eye` は `dtype` 引数を正しく渡すように直す。今は無視して常に `np.float64` を使っている。
- 新しい群のテンソルは、`FiniteAbelianTensor` を継承して `symmetry` を設定するだけで定義できる。この方法を README に書く。
- `__init__.py` から `AbelianGroup`、`CyclicGroup`、`U1Group`、`ProductGroup`、`FiniteAbelianTensor`、`TensorZ2Z2`、`TensorZ2Z3` を公開する。

### 4. テスト

- **新規 `tests/test_symmetry.py`**: 対象は `CyclicGroup(2)`、`CyclicGroup(3)`、`ProductGroup(Z2, Z2)`、`ProductGroup(Z2, Z3)`。
  - 群の公理: 全元について、単位元、結合律、逆元、可換性を確認する。
  - `normalize`: 正常な入力の正規化(`CyclicGroup(3).normalize(-1) == 2` など)と、不正な入力(タプルの長さ違い、直積群に整数を渡す、整数でない値)で `ValueError` になることを確認する。
  - `elements()` の内容と順序、値による等価性とハッシュ、`signed(a, 2)` が `ValueError` になること。
- **`tests/conftest.py`**:
  - テスト対象のクラスの既定値に `TensorZ2Z2` と `TensorZ2Z3` を加え、`--tensorclass` オプションでも指定できるようにする。
  - `n_qnums` は `tensorclass.symmetry.elements()` から求める。
  - `rqhape` と `rcharge` は、有限群なら `elements()` から選び、U(1) なら今までどおり整数を選ぶ。
- **`tests/test_tensors.py`**: charge や量子数を整数として計算している箇所を、群のメソッドに書き直す。テストの意図は変えない。
- **互換性と誤用のテスト**:
  - `qodulus=` を渡すと `DeprecationWarning` が出て、`CyclicGroup` または `U1Group` に変換されること
  - `.qodulus` を読むと警告が出ること
  - `qodulus` と `symmetry` を両方渡すと `TypeError` になること
  - `symmetry` の違うテンソル(`CyclicGroup(2)` と `CyclicGroup(3)`)を混ぜて演算すると `AssertionError` になること
  - `TensorZ2Z3` でセクター数が 1 でも 6 でもないインデックスを作り、`qhape` を省略すると `ValueError` になること
- `pyproject.toml` の `filterwarnings = ["error::DeprecationWarning"]` はそのまま残す。非推奨の動作を確かめるテストでは、`pytest.warns` で警告を受け取る。

### 5. 性能の計測

- 新規 `benchmarks/trg_bench.py` を作る。
  - demo の `trg.trg_step` と `trg_demo.ising_tensor` を import して使う(`demo/` を `sys.path` に追加する)。
  - chi は 8、16、24。chi ごとに 14 ステップ回して収束させたあと、1 ステップの時間を `timeit.repeat(number=1, repeat=5)` で測り、その最小値を取る。
  - 対象は `Tensor` と `TensorZ2`。結果を表にして標準出力に出す。
  - 実行方法は `uv run --group demo python benchmarks/trg_bench.py`。CI では実行しない。
- 変更の前(今の master)と後で、同じマシンで測る。
  - `TensorZ2` の速度低下がどれかの chi で 20% を超えた場合は、プロファイルを取ったうえで速くする対策を入れる(例: `CyclicGroup` の `add` と `neg` を表を引く形にする)。
  - 対策をしても 20% 以内に収まらない場合は、作業を止めて利用者に相談する。

## 作業の順序

1. 計測スクリプトを作り、今の master での基準値を測る(§5)
2. 群オブジェクトとそのテスト(§1、§4 の `test_symmetry.py`)
3. `AbelianTensor` を群オブジェクトに対応させる(§2)。この時点で既存のテストがすべて通ること
4. サブクラスの整理と、`TensorZ2Z2` / `TensorZ2Z3` の追加。テストの拡張(§3、§4)
5. 変更後の性能を測り、必要なら速くする対策を入れる(§5)
6. README とドキュメント

## 対象外

- U(1) を含む直積群の動作確認(API は対応しているが、テストしない)
- pickle などで保存したテンソルとの互換性
- `tensor_z2z2` ブランチの削除や整理
- `tensor_z2z2` ブランチにあった、SVD が失敗したときに gesvd でやり直す処理

## 計測結果

(実装後に記入する)
