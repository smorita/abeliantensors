# abeliantensors モダン化 設計書

- 日付: 2026-10-13
- ブランチ: `modernize`
- 状態: レビュー待ち

## 目的

自分(研究室)用に、abeliantensors を最新の Python 環境で使える状態にする。
PyPI 公開や upstream (mhauru/abeliantensors) への PR は目的としない。

### 前提・制約

- Python 3.12 以上を対象とする。
- NumPy 2.x と最新の scipy に対応する。
- 環境・依存管理には uv を使う。
- ライブラリ本体の数値的な挙動は変えない。

### 完了条件

1. Python 3.12 と最新の依存関係で、`uv sync --locked && uv run pytest` が警告なしで全件成功する。
2. `uv run ruff check` と `uv run ruff format --check` が成功する。
3. GitHub Actions(Python 3.12 / 3.13 / 3.14)でテスト・lint・ドキュメントビルドが成功する。
4. `uv run --group docs make -C docs html` が警告なしでビルドできる。
5. `uv run --group demo python demo/trg_demo.py` が最後まで実行できる。

## 現状(2026-10-13 調査時点)

- パッケージ定義は `setup.py` のみ。`python_requires>=3.6`、依存関係は `scipy>=1.0.0` のみで、numpy が明示されていない。
- CI は `.travis.yml`(Python 3.6–3.8)と codecov で構成されている。
- ドキュメントは Sphinx と `recommonmark`(開発終了)で構成されている。
- `.gitignore` がない。
- 最新の numpy / scipy / ncon / pytest の環境で試した結果は次のとおり。
  - `np.float_` などの NumPy 2 で削除された名前(11 箇所)が原因で、import の時点で失敗する。
  - 上記を `ruff --select NPY201 --fix` で修正すると、129 件が成功し、7 件が失敗する。
    - 4 件は ncon 2.x の仕様変更による失敗(`If non-int indices are used, order must be specified`)。
    - 3 件は `test_svd[{TensorZ2,TensorU1,TensorZ3}-100-False-1e-15]` で、行列の形が 1 つずれる不一致。
  - pytest の非推奨警告が出る(`tests/conftest.py:61` で `parametrize` に `map` を渡している)。
  - `ruff format` を適用すると 9 ファイルが変わる。

## 設計

### 1. パッケージングと uv

- `pyproject.toml` を新しく作り、`setup.py` と `.travis.yml` を削除する。
- ビルドバックエンドは `uv_build`(`src/abeliantensors` レイアウトにそのまま対応している)。
- `uv.lock` と `.python-version`(`3.12`)をコミットする。
- `.gitignore` を新しく作る。対象は `.venv/`、`__pycache__/`、`docs/build/`、`.pytest_cache/`、`.ruff_cache/`、`.coverage`。

```toml
[project]
name = "abeliantensors"
version = "0.2.0"
description = "Library for Abelian symmetry preserving tensors."
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
authors = [{ name = "Markus Hauru", email = "markus@mhauru.org" }]
keywords = ["tensor", "tensor networks", "linear algebra"]
dependencies = ["numpy>=2.0", "scipy>=1.13"]

[build-system]
requires = ["uv_build>=0.11,<0.12"]
build-backend = "uv_build"

[dependency-groups]
dev  = ["pytest", "pytest-randomly", "pytest-cov", "ncon", "ruff"]
docs = ["sphinx", "sphinx-rtd-theme", "myst-parser"]
demo = ["ncon", "matplotlib", "seaborn"]
```

- 旧 `extras_require` のテスト・ドキュメント・デモ用の依存は `dependency-groups` に移す。旧 `demo` extras にあった `timeit` は標準ライブラリなので外す。
- `coverage` と codecov の構成は `pytest-cov` に置き換える。codecov には送信しない。
- バージョンは 0.2.0 とし、作者は原作者(Markus Hauru)のままにする。
- README を更新する。インストールとテストの手順は `uv sync` / `uv run pytest` に書き換える。Travis と codecov のバッジは削除する。

### 2. 最新の NumPy / scipy / ncon / pytest への対応

方針: ライブラリの数値的な挙動は変えない。テストの許容誤差を緩めるだけの対処はしない。

1. **NumPy 2 で削除された名前**: `ruff check --select NPY201 --fix` で置き換え、差分を目視で確認する。
2. **ncon 2.x**: テストと demo の呼び出し側を、整数の添字を使うか `order` を明示するように直す。必要なら `ncon>=2` を下限として指定する。
3. **`test_svd[...-100-False-1e-15]` の失敗**: 修正する前に原因を特定する(systematic-debugging)。
   - ライブラリの切り捨て判定に不具合があった場合は、本体を修正する。
   - 縮退した特異値や丸め誤差のせいで、どちらの結果も正しいと言える場合は、テストの比較方法を修正し、理由をコメントに残す。
   - どちらで対処したかを、コミットメッセージに記録する。
4. **pytest の非推奨警告**: `tests/conftest.py` で `map(...)` を `list(map(...))` に変える。
5. **警告をエラーとして扱う設定**: `[tool.pytest.ini_options]` に次を追加する。
   ```toml
   testpaths = ["tests"]
   filterwarnings = ["error::DeprecationWarning"]
   ```
   テストのデフォルトの反復回数は変えない。

### 3. ruff

```toml
[tool.ruff]
line-length = 79
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "NPY", "B"]
```

- 79 桁は、既存のコード(black の 79 桁)に合わせるため。
- 指摘が多すぎるルールは、理由を書いたうえで `ignore` するか、ファイル単位で無視する。
- コミットは次の 3 つに分ける。
  1. 設定の追加と lint の修正
  2. `ruff format` の一括適用(整形以外の変更は含めない)
  3. 2 のハッシュを `.git-blame-ignore-revs` に登録

### 4. GitHub Actions

`.github/workflows/ci.yml` を作る。

- 実行タイミング: `push` と `pull_request`。
- `test` ジョブ: Python 3.12 / 3.13 / 3.14 の matrix。`astral-sh/setup-uv` を使い、次を実行する。
  - `uv sync --locked`
  - `uv run ruff check`
  - `uv run ruff format --check`
  - `uv run pytest`
- `docs` ジョブ: `uv run --group docs sphinx-build -W -b html docs/source docs/build/html`

### 5. ドキュメント

`docs/source/conf.py` を次のように変更する。

- `recommonmark` を `myst_parser` に置き換える。
- 重複している `source_suffix` を 1 つにまとめる。
- 存在しない `_static` / `_templates` への参照を削除する(または空ディレクトリを作る)。
- `sys.path.insert` を削除し、インストール済みのパッケージを autodoc で参照する。
- copyright の年を更新する。

README にビルド方法(`uv run --group docs make -C docs html`)を書く。

### 6. demo

- `uv run --group demo python demo/trg_demo.py` で最後まで動くように直す。対象は ncon の仕様変更と、非推奨になった API。
- パラメータは変えない。実行時間が長い場合は、その旨を README に書く。
- 図を表示する `plt.show()` が非対話環境で止まる場合でも、今の挙動を基本として保つ。必要な場合だけ、ファイルに保存するオプションを検討する。

## 作業の順序

各テーマを 1 つ以上の独立したコミットにする。

1. パッケージングと uv(§1)
2. NumPy / scipy / ncon / pytest への対応(§2)
3. ruff(§3、コミットは 3 つに分ける)
4. GitHub Actions(§4)
5. ドキュメント(§5)
6. demo(§6)

## 対象外

- 型ヒントの追加、API の変更、性能改善。
- PyPI への公開、upstream への PR。
- codecov などの外部サービスとの連携。
