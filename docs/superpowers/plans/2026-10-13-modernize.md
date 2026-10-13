# abeliantensors モダン化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** abeliantensors を uv 管理・Python ≥3.12・NumPy 2.x / 最新 scipy で動く状態にし、ruff・GitHub Actions・Sphinx(myst)・demo まで整える。

**Architecture:** 1 テーマ = 1 コミット(ruff は 3 コミット)で段階的に進める。ライブラリの数値的な挙動は変えない。テストの許容誤差を緩めるだけの対処はしない。

**Tech Stack:** uv 0.11 / uv_build、Python 3.12+、numpy 2.x、scipy ≥1.13、ncon 2.x、pytest、ruff、Sphinx + myst-parser、GitHub Actions。

**Spec:** `docs/superpowers/specs/2026-10-13-modernize-design.md`

## Global Constraints

- `requires-python = ">=3.12"`。`.python-version` は `3.12`。
- `dependencies = ["numpy>=2.0", "scipy>=1.13"]`。
- `version = "0.2.0"`。authors は `Markus Hauru <markus@mhauru.org>` のまま。
- ビルドバックエンドは `uv_build>=0.11,<0.12`。
- ruff: `line-length = 79`、`target-version = "py312"`、lint ルールは `["E", "F", "W", "I", "UP", "NPY", "B"]`。
- pytest: `filterwarnings = ["error::DeprecationWarning"]`。テストのデフォルトの反復回数は変えない。
- CI の matrix は Python 3.12 / 3.13 / 3.14。
- 作業ブランチは `modernize`(作成済み)。コミットメッセージの末尾に `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` を付ける。
- 対象外: 型ヒント、API の変更、性能改善、PyPI 公開、codecov。

## Review Focus

1. **テストの乱数シード**: pytest-randomly で毎回シードが変わるので、たまたま通っただけのテストを見逃さない。Task 2 の完了確認では、ランダムなシードで 3 回実行し、さらに `--randomly-seed` を固定して 1 回実行する。
2. **Python 3.14 + 最新 numpy の wheel**: ローカルで確認しなかったバージョンで CI が失敗しないようにする。Task 4 で、push する前に `uv run --python 3.14 pytest` をローカルで実行する。
3. **`uv sync --locked` の失敗**: `pyproject.toml` を変えたのに `uv.lock` を更新し忘れると、CI が失敗する。`pyproject.toml` を変更するタスクでは、毎回 `uv lock --check` を確認ステップに入れる。
4. **整形コミットへの意味のある変更の混入**: `ruff format` のコミットに整形以外の変更が入らないようにする。Task 3 で、`ruff format` の前後でテスト結果が同じであることと、`git diff --stat` が format の対象ファイルだけであることを確認する。
5. **demo を `demo/` 以外から実行した場合の失敗**: `from trg import trg_step` が失敗しないようにする。Task 6 で README に「`demo/` で実行する」と書き、その手順どおりに実行して確認する。

---

## File Structure

| ファイル | 作業 | 役割 |
|---|---|---|
| `pyproject.toml` | 新規 | メタデータ、依存関係、ビルド、pytest と ruff の設定 |
| `uv.lock` | 新規(生成) | バージョンの固定 |
| `.python-version` | 新規 | `3.12` |
| `.gitignore` | 新規 | 生成物を除外する |
| `setup.py`, `.travis.yml` | 削除 | — |
| `src/abeliantensors/*.py` | 修正 | NumPy 2 で削除された名前、lint |
| `tests/conftest.py` | 修正 | `parametrize` に list を渡す |
| `tests/test_tensors.py` | 修正 | ncon の添字を `int` にする。SVD テスト(原因次第) |
| `.git-blame-ignore-revs` | 新規 | format コミットを登録する |
| `.github/workflows/ci.yml` | 新規 | CI |
| `docs/source/conf.py` | 修正 | myst、パス、設定の整理 |
| `demo/*.py` | 修正(必要に応じて) | 最新の環境で動かす |
| `README.md` | 修正 | 手順とバッジ |

---

### Task 1: pyproject.toml と uv への移行

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `uv.lock`(生成)
- Delete: `setup.py`, `.travis.yml`
- Modify: `README.md`(バッジとインストール手順)

**Interfaces:**
- Produces: `uv sync` で作れる環境。依存グループは `dev`(既定)、`docs`、`demo`。

- [ ] **Step 1: `pyproject.toml` を作る**

```toml
[project]
name = "abeliantensors"
version = "0.2.0"
description = "Library for Abelian symmetry preserving tensors."
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Markus Hauru", email = "markus@mhauru.org" }]
keywords = ["tensor", "tensor networks", "linear algebra"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
]
dependencies = ["numpy>=2.0", "scipy>=1.13"]

[project.urls]
Homepage = "https://github.com/smorita/abeliantensors"
Upstream = "https://github.com/mhauru/abeliantensors"

[build-system]
requires = ["uv_build>=0.11,<0.12"]
build-backend = "uv_build"

[dependency-groups]
dev = ["pytest", "pytest-randomly", "pytest-cov", "ncon>=2", "ruff"]
docs = ["sphinx", "sphinx-rtd-theme", "myst-parser"]
demo = ["ncon>=2", "matplotlib", "seaborn"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`filterwarnings` は、Task 2 で警告の原因を直してから追加する。この時点で入れると、Task 2 の前に収集エラーになってしまうため。

- [ ] **Step 2: `.python-version` と `.gitignore` を作る**

`.python-version`:
```
3.12
```

`.gitignore`:
```
.venv/
__pycache__/
*.egg-info/
dist/
docs/build/
.pytest_cache/
.ruff_cache/
.coverage
demo/latest_trg_performance.pdf
```

- [ ] **Step 3: `setup.py` と `.travis.yml` を削除する**

```bash
git rm setup.py .travis.yml
```

- [ ] **Step 4: ロックファイルを作り、環境を構築する**

Run: `uv lock && uv sync`
Expected: `uv.lock` が作られ、`.venv` に abeliantensors が editable でインストールされる。

Run: `uv run python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"`
Expected: numpy 2.x、scipy 1.13 以上。

- [ ] **Step 5: 修正前の状態を記録する(失敗することを確認する)**

Run: `uv run pytest -q -x`
Expected: `AttributeError: np.float_ was removed in the NumPy 2.0 release` で収集に失敗する。これは Task 2 で直す。

- [ ] **Step 6: README を更新する**

`README.md` から次を削除する。
- 3〜4 行目の Travis と Codecov のバッジ
- 208〜211 行目の `[travis-img]`、`[travis-url]`、`[codecov-img]`、`[codecov-url]` の定義

インストール節(17 行目と 23 行目の `pip install ...` 周辺)を次に置き換える。

````markdown
This fork is managed with [uv](https://docs.astral.sh/uv/) and requires
Python ≥ 3.12, NumPy ≥ 2.0, and SciPy ≥ 1.13.

```
git clone https://github.com/smorita/abeliantensors
cd abeliantensors
uv sync          # creates .venv with the library and dev tools
uv run pytest    # run the test suite
```

To use it from another uv project:
```
uv add --editable /path/to/abeliantensors
```
````

前後の文章で `pip` や `setup.py` に触れている箇所があれば、上の手順に合わせて直す。

- [ ] **Step 7: ロックファイルの整合性を確認する**

Run: `uv lock --check`
Expected: exit 0

- [ ] **Step 8: コミットする**

```bash
git add pyproject.toml uv.lock .python-version .gitignore README.md
git commit -m "Migrate packaging to pyproject.toml and uv

Drop setup.py and Travis CI. Require Python >=3.12, numpy>=2.0,
scipy>=1.13. Test/doc/demo extras become dependency groups.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: 最新の NumPy / scipy / ncon / pytest への対応

**Files:**
- Modify: `src/abeliantensors/abeliantensor.py`, `src/abeliantensors/symmetrytensors.py`, `src/abeliantensors/tensor.py`, `src/abeliantensors/tensorcommon.py`(NPY201 で修正される箇所)
- Modify: `tests/conftest.py:59`
- Modify: `tests/test_tensors.py:1031-1035`(i_list の型)、SVD テスト(原因次第)
- Modify: `pyproject.toml`(`filterwarnings` の追加)

**Interfaces:**
- Consumes: Task 1 の `uv sync` 環境
- Produces: `uv run pytest` が警告なしで全件成功する状態

- [ ] **Step 1: NumPy 2 で削除された名前を機械的に置き換える**

Run: `uvx ruff check --select NPY201 --fix src tests demo`
Expected: `Found 11 errors (11 fixed, 0 remaining)` 程度。

Run: `git diff`
確認: `np.float_` → `np.float64` のような名前の置き換えだけであること。`np.complex_` → `np.complex128` なども同様。ロジックの変更が含まれていたら差し戻す。

- [ ] **Step 2: import が通ることを確認する**

Run: `uv run python -c "import abeliantensors; print(abeliantensors.TensorZ2)"`
Expected: `<class 'abeliantensors.symmetrytensors.TensorZ2'>`

- [ ] **Step 3: pytest の非推奨警告を直す**

`tests/conftest.py` の `pytest_generate_tests` を次のように変える。

```python
    tensorclasses = list(map(parse_tensorclass, tensorclass_opts))
```

(変更前: `tensorclasses = map(parse_tensorclass, tensorclass_opts)`)

- [ ] **Step 4: 失敗の一覧を取る**

Run: `uv run pytest -q -p no:randomly 2>&1 | tail -15`
Expected: 7 件程度が失敗する。`test_ncon_svd_ncon[*]`(ncon の ValueError)と `test_svd[*-False-1e-15]`(形の不一致)。

- [ ] **Step 5: ncon 2.x の添字の型を直す**

原因: `np.random.choice` は `np.int64` を返す。ncon 2.0 はこれを「int ではない添字」とみなし、`If non-int indices are used, order must be specified` になる。

`tests/test_tensors.py` の `test_ncon_svd_ncon` 内(1031 行目付近)を変える。

```python
            if n_svd_inds:
                i_list = np.random.choice(
                    len(T.shape), size=n_svd_inds, replace=False
                ).tolist()
            else:
                i_list = []
```

`.tolist()` は Python の `int` のリストを返すので、その後の `U_left_inds = [i + 1 for i in i_list]` も `int` になる。`i_list_compl` は `range` から作るので最初から `int`。

`test_svd`(603 行目付近)も、同じ形のコードなので揃えておく。

```python
        if n:
            i_list = np.random.choice(
                len(T.shape), size=n, replace=False
            ).tolist()
        else:
            i_list = []
```

Run: `uv run pytest -q -p no:randomly -k ncon_svd_ncon`
Expected: 4 passed

- [ ] **Step 6: `test_svd[*-False-1e-15]` の失敗を再現して原因を調べる(systematic-debugging)**

**REQUIRED SUB-SKILL:** superpowers:systematic-debugging。修正を先に入れないこと。

再現:
Run: `uv run pytest -q -p no:randomly "tests/test_tensors.py::test_svd" -k "False and 1e-15" 2>&1 | tail -30`
Expected: TensorZ2 / TensorU1 / TensorZ3 で `ValueError: shapes (k,k) and (k-1,k) not aligned`。これは `np.dot(g, S_np_mat)` の行(662 行目付近)で起きる。

調べること:
1. 失敗する反復で、ライブラリの `S` の要素数 `len(S_svd_np)` と、参照実装の `len(S_np_svd)` を表示する。どちらが 1 本多いか。
2. 多い側の最小の特異値の値を見る。0 や 1e-16 レベル(対称性による 0 や丸め誤差)か、意味のある値か。
3. `truncate=False`、`eps=1e-15` のとき、ライブラリ(`src/abeliantensors/abeliantensor.py` の `svd` → `_find_trunc_dim` 系)と参照実装(`tests/ndarray_decomp.py:79-125`)で、残す本数を決める処理の違いを比べる。比較では `err < eps` と `<=` のどちらか、累積誤差の計算順、縮退の扱い(`degeneracy_eps`)を見る。
4. 同じ手順で元の環境(numpy 1.x)でも失敗するかを確かめ、numpy 2 が原因かどうかを切り分ける。
   `uv run --no-project --python 3.12 --with "numpy<2" --with "scipy<1.14" --with "ncon<2" --with pytest env PYTHONPATH=src python -m pytest -q -p no:randomly tests/test_tensors.py::test_svd -k "False and 1e-15"`
   (`np.float_` を置き換えた後のコードを使う。numpy 1.x では `np.float64` も使える)

判断基準:
- **(a) ライブラリの不具合**: 理論的に残すべき本数とライブラリの結果が食い違う場合。たとえば、相対誤差が eps 未満なのに余計に残す、または必要な値を捨てる。この場合はライブラリを最小限修正し、その事象を直接確かめる回帰テストを `tests/test_tensors.py` に追加する。テスト名は `test_svd_truncation_<事象>` とし、失敗することを確認してから修正する。
- **(b) 丸め誤差によるずれで、どちらの結果も正しい場合**: 余分な 1 本の特異値がマシンイプシロン×最大特異値程度のとき。この場合は、テストの NumPy 比較の部分(636〜668 行目)で、両方の結果を共通の本数 `k = min(len(S_np_svd), len(S_svd_np))` に揃えてから比べる。ただし、余分な特異値が `max(S) * 1e-12` 以下であることを assert で確認する。理由をコメントに残す。`rel_err` の比較(668 行目)はそのまま残す。

どちらになったかと根拠(実際に観測した値)を記録し、Step 9 のコミットメッセージに書く。

- [ ] **Step 7: 決めた方針で修正し、全件成功を確認する**

Run: `uv run pytest -q -p no:randomly`
Expected: 全件成功、0 failed

- [ ] **Step 8: 警告をエラーとして扱う設定を入れ、乱数シードを変えても通ることを確認する**

`pyproject.toml` の `[tool.pytest.ini_options]` を次のようにする。

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["error::DeprecationWarning"]
```

Run: `for i in 1 2 3; do uv run pytest -q 2>&1 | tail -1; done`
Expected: 3 回とも `N passed`。failed も warnings も出ない。

Run: `uv run pytest -q -p randomly --randomly-seed=12345 2>&1 | tail -1`
Expected: `N passed`

Run: `uv lock --check`
Expected: exit 0

- [ ] **Step 9: コミットする**

```bash
git add -A src tests pyproject.toml
git commit -m "Support NumPy 2, recent SciPy, ncon 2 and pytest

- Replace aliases removed in NumPy 2 (np.float_ etc.).
- Pass plain ints as ncon indices (ncon 2 rejects np.int64).
- Pass a list, not a map, to metafunc.parametrize.
- test_svd eps=1e-15: <(a) or (b)、観測した値と理由を 1〜3 行で書く>
- Turn DeprecationWarnings into errors in the test suite.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(`<...>` の部分は Step 6 の結論で必ず置き換える)

---

### Task 3: ruff(lint の修正 → format → blame-ignore)

**Files:**
- Modify: `pyproject.toml`、lint が指摘する `src/`、`tests/`、`demo/` のファイル
- Create: `.git-blame-ignore-revs`

**Interfaces:**
- Consumes: Task 2 で全件成功するテスト
- Produces: `uv run ruff check` と `uv run ruff format --check` が成功する状態

- [ ] **Step 1: ruff の設定を追加する**

`pyproject.toml` の末尾に追加する。

```toml
[tool.ruff]
line-length = 79
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "NPY", "B"]
```

- [ ] **Step 2: 現状を数える**

Run: `uv run ruff check --statistics`
指摘をルールごとに数えて記録する。

- [ ] **Step 3: 自動で直せる安全な指摘を直す**

Run: `uv run ruff check --fix`
Run: `git diff --stat && git diff`
確認: import の並び替え、`super(Cls, self)` → `super()`、`"{}".format` → f-string(UP032)などの機械的な変更だけであること。

- [ ] **Step 4: 残った指摘を手で直す。直さないものは理由を書いて無視する**

Run: `uv run ruff check`

方針:
- **F(未使用の変数・import など)と B(バグの可能性)**: 原則としてコードを直す。挙動が変わりそうなものは直さない。その行に `# noqa: <code>` と理由を書く。
- **E501(行が長すぎる)**: docstring や URL などで折り返せない行が 20 箇所を超えるときは、`[tool.ruff.lint] ignore = ["E501"]` を追加し、コメントで「format が折り返さない docstring 行があるため」と書く。20 箇所以下なら手で折り返す。
- **E741(`l` などの曖昧な変数名)**: テンソルの添字の慣習的な名前なので、`[tool.ruff.lint.per-file-ignores]` に `"src/**" = ["E741"]` を入れてよい。理由をコメントに書く。
- **NPY002(旧来の `np.random.*` API)**: テストは pytest-randomly のシード固定に頼っているので、`Generator` には移行しない。`ignore` に `"NPY002"` を追加し、コメントで「tests rely on the legacy global RNG seeded by pytest-randomly」と書く。

Run: `uv run ruff check`
Expected: `All checks passed!`

- [ ] **Step 5: テストを実行する**

Run: `uv run pytest -q -p randomly --randomly-seed=12345 2>&1 | tail -1`
Expected: Task 2 Step 8 と同じ件数が成功

- [ ] **Step 6: lint のコミットをする**

```bash
git add -A pyproject.toml src tests demo
git commit -m "Add ruff configuration and fix lint findings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: format を一括適用する(整形以外の変更は入れない)**

Run: `uv run ruff format`
Run: `git diff --stat`
確認: 変更されたのは `.py` ファイルだけであること。

Run: `uv run pytest -q -p randomly --randomly-seed=12345 2>&1 | tail -1`
Expected: Step 5 と同じ結果

Run: `uv run ruff check`
Expected: `All checks passed!`。format 後に E501 などが新しく出た場合は、ここでは直さない。Step 8 の後で別コミットにする。

```bash
git add -A src tests demo docs/source/conf.py
git commit -m "Apply ruff format

Formatting only; no functional changes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 8: blame-ignore を登録する**

```bash
printf '# ruff format\n%s\n' "$(git rev-parse HEAD)" > .git-blame-ignore-revs
git config blame.ignoreRevsFile .git-blame-ignore-revs
git add .git-blame-ignore-revs
git commit -m "Ignore ruff format commit in git blame

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Run: `uv run ruff format --check && uv run ruff check`
Expected: 両方とも成功

---

### Task 4: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`(CI バッジ)

**Interfaces:**
- Consumes: Task 1〜3 のコマンド(`uv sync --locked`、`uv run ruff ...`、`uv run pytest`)と、Task 5 の docs ビルドコマンド
- Produces: push と PR で動く CI

- [ ] **Step 1: ワークフローを作る**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.12", "3.13", "3.14"]
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with:
          python-version: ${{ matrix.python-version }}
          enable-cache: true
      - run: uv sync --locked
      - run: uv run ruff check
      - run: uv run ruff format --check
      - run: uv run pytest --cov=abeliantensors --cov-report=term

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v7
        with:
          python-version: "3.12"
          enable-cache: true
      - run: uv sync --locked --group docs
      - run: uv run sphinx-build -W -b html docs/source docs/build/html
```

`actions/checkout` と `astral-sh/setup-uv` のメジャーバージョンは、作業時点の最新に合わせる。`gh api repos/astral-sh/setup-uv/releases/latest --jq .tag_name` などで確認する。

- [ ] **Step 2: matrix の各バージョンをローカルで確認する**

Run: `for v in 3.12 3.13 3.14; do echo "== $v"; uv run --python $v --isolated pytest -q -p randomly --randomly-seed=12345 2>&1 | tail -1; done`
Expected: 3 つとも同じ件数が成功。

3.14 で wheel がないなどの理由で失敗した場合は、matrix から外さずにユーザーに報告して判断を仰ぐ。

- [ ] **Step 3: YAML の構文を確認する**

Run: `uvx --from check-jsonschema check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/ci.yml`
Expected: `ok -- validation done`

- [ ] **Step 4: README にバッジを追加する**

`README.md` の先頭のバッジの位置(Task 1 で削除した場所)に追加する。

```markdown
[![CI](https://github.com/smorita/abeliantensors/actions/workflows/ci.yml/badge.svg)](https://github.com/smorita/abeliantensors/actions/workflows/ci.yml)
```

- [ ] **Step 5: コミットする**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "Add GitHub Actions CI using uv

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

CI が実際に動くかの確認は push 後になる。push はユーザーの指示を受けてから行う(finishing-a-development-branch で扱う)。

---

### Task 5: ドキュメント(myst-parser への移行)

**Files:**
- Modify: `docs/source/conf.py`
- Modify: `docs/source/index.rst`(`-W` で警告が出る箇所だけ)
- Modify: `README.md`(ビルド方法)

**Interfaces:**
- Consumes: `docs` 依存グループ(Task 1)
- Produces: `uv run --group docs sphinx-build -W -b html docs/source docs/build/html` が成功する状態(Task 4 の CI が使う)

- [ ] **Step 1: 現状のビルド結果を見る**

Run: `uv sync --group docs && uv run sphinx-build -W -b html docs/source docs/build/html 2>&1 | tail -20`
Expected: `recommonmark` がないため、Extension error で失敗する。

- [ ] **Step 2: `conf.py` を書き換える**

`docs/source/conf.py` を、コメントのボイラープレートも含めて次の内容に置き換える。

```python
# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "abeliantensors"
copyright = "2020-2026, Markus Hauru and contributors"
author = "Markus Hauru"

master_doc = "index"
extensions = ["sphinx.ext.autodoc", "myst_parser"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"
```

削除するもの:
- `sys.path.insert`(`uv sync` でパッケージがインストールされるので不要)
- 存在しない `_templates` / `_static`(参照すると `-W` で警告になる)
- 重複している `source_suffix`

- [ ] **Step 3: 警告なしでビルドできるまで直す**

Run: `uv run sphinx-build -W -b html docs/source docs/build/html 2>&1 | tail -20`

予想される警告と対処:
- `index.rst` の toctree が自分自身(`index`)を参照している警告: toctree ブロックを削除する(ページが 1 つしかないので不要)。
- docstring の RST 書式の警告: 該当する docstring を最小限直す。直したファイルは format 済みの状態を保つ(`uv run ruff format --check`)。

Expected: `build succeeded.`(警告 0)

- [ ] **Step 4: Makefile 経由でもビルドできることを確認する**

Run: `uv run --group docs make -C docs html 2>&1 | tail -3`
Expected: `build succeeded.`

- [ ] **Step 5: README にビルド方法を書く**

README のドキュメントに関する節(なければインストール節の後)に追加する。

````markdown
To build the API documentation locally:
```
uv run --group docs make -C docs html   # output in docs/build/html
```
````

- [ ] **Step 6: コミットする**

```bash
uv lock --check
git add docs/source README.md
git commit -m "Update Sphinx docs: recommonmark -> myst-parser

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: demo の動作確認

**Files:**
- Modify: `demo/trg_demo.py`, `demo/trg.py`(必要な場合だけ)
- Modify: `README.md`(demo の実行方法)

**Interfaces:**
- Consumes: `demo` 依存グループ、Task 2 で修正済みのライブラリ
- Produces: `cd demo && uv run --group demo python trg_demo.py` が最後まで実行でき、`latest_trg_performance.pdf` が作られる状態

- [ ] **Step 1: 短時間で通るかを確認する(パラメータは一時的にだけ変える。コミットしない)**

本番の設定は chi=2〜70、2 クラス、各 5 回の timeit なので時間がかかる。まずは import と 1 周分の処理が通るかを、使い捨てのスクリプトで確かめる。

```bash
cd demo && uv run --group demo python -c "
import numpy as np, trg_demo
from abeliantensors import Tensor, TensorZ2
from trg import trg_step
from ncon import ncon
J, h = 1.0, 0.0
beta = np.log(1 + np.sqrt(2)) / 2
A0 = trg_demo.ising_tensor(J, h, beta)
for cls in (Tensor, TensorZ2):
    A = cls.from_ndarray(A0, shape=[[1, 1]] * 4, qhape=[[0, 1]] * 4, dirs=[1, -1, 1, -1])
    lf = 0
    for i in range(1, 15):
        A, lf, err = trg_step(A, lf, 8, 1e-7)
    Z = ncon(A, [1, 1, 2, 2]).value()
    f = (-(np.log(Z) + lf) / beta) / 2 ** (i + 1)
    print(cls.__name__, (f - trg_demo.ising_exact_f(J, h, beta)) / trg_demo.ising_exact_f(J, h, beta))
"
```

Expected: Tensor と TensorZ2 で同じ相対誤差(1e-5 程度以下)が表示される。

エラーが出た場合は、`demo/*.py` の該当箇所を直す。
- ncon 2: 単一のテンソル `ncon(A, [1, 1, 2, 2])` が受け付けられない場合は `ncon([A], [[1, 1, 2, 2]])` にする。
- 非推奨の API: 警告やエラーの内容に従う。

- [ ] **Step 2: 本番の設定で最後まで実行する(バックグラウンド、上限 60 分)**

Run(`run_in_background`):
```bash
cd demo && /usr/bin/time -v timeout 3600 uv run --group demo python -W error::DeprecationWarning trg_demo.py > ../demo_run.log 2>&1; echo "exit=$?" >> ../demo_run.log
```

Expected: `exit=0` になり、`demo/latest_trg_performance.pdf` が作られる。`Elapsed (wall clock)` を記録する。

`plt.loglog(..., label=cls)` でクラスオブジェクトをラベルに渡していることについて、matplotlib の警告やエラーが出た場合は `label=cls.__name__` にする。

60 分で終わらない場合は、パラメータは変えずに、途中までの進み具合(到達した chi)を記録してユーザーに報告する。

- [ ] **Step 3: README に demo の実行方法と所要時間を書く**

````markdown
### Demo

```
cd demo
uv run --group demo python trg_demo.py   # writes latest_trg_performance.pdf
```
Runs TRG for bond dimensions 2–70 with `Tensor` and `TensorZ2`;
takes about <Step 2 で測った時間> on a laptop.
````

(`<...>` は Step 2 で測った値で必ず置き換える)

- [ ] **Step 4: 後片付けとコミット**

```bash
rm -f demo_run.log
git status --short   # demo/latest_trg_performance.pdf は .gitignore で除外されているはず
git add demo README.md
git commit -m "Make TRG demo run on the modernized stack

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(demo に変更がなく README だけ変えた場合は、メッセージを `Document how to run the TRG demo with uv` にする)

---

### Task 7: 最終確認

- [ ] **Step 1: クリーンな環境で完了条件をすべて確認する**

```bash
rm -rf .venv
uv sync --locked
uv run pytest -q 2>&1 | tail -1
uv run ruff check
uv run ruff format --check
uv run --group docs sphinx-build -W -b html docs/source docs/build/html 2>&1 | tail -1
uv lock --check
git status --short
```

Expected:
- pytest は全件成功し、warnings は 0
- ruff は 2 つとも成功
- `build succeeded.`
- ロックファイルは最新
- working tree はクリーン

- [ ] **Step 2: superpowers:finishing-a-development-branch に進む**

push と GitHub Actions の確認は、そこでユーザーの判断を受けてから行う。
