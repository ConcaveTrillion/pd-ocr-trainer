# CLAUDE — pd-ocr-trainer

DocTR OCR model training pipeline (detection, recognition, typeface
classification) for PGDP data, driven by a NiceGUI training UI. HF dataset
publishing, model export, and metadata sidecars live here — not in the labeler.
Architecture: `docs/ROADMAP.md` (approved plan) + `docs/DATASETS.md` (dataset spec).

## Commands

| target | does |
|---|---|
| `make setup` | install deps + pre-commit hooks (`uv sync --group all-dev`) |
| `make install` | install `pd-ocr-trainer-ui` as a uv tool (puts CLIs on PATH) |
| `make test` | `uv run pytest -n auto -v -ra` |
| `make test-single TEST='...'` | run one pytest node id |
| `make test-k K='pattern'` | run tests by `-k` expression |
| `make lint` / `make lint-fix` | ruff via pre-commit (with auto-fix) |
| `make format` | format then lint |
| `make pre-commit-check` | run all pre-commit hooks |
| `make ci` | install + pre-commit + test + build |
| `make build` | wheel + sdist into `dist/` |
| `make run` | start NiceGUI training UI (`uv run pd-ocr-trainer-ui`) |
| `make export-models` | copy trained artifacts to workspace exports dir |
| `make clean` / `make reset` | remove caches / rebuild venv |
| `make release-patch/minor/major` | bump version, commit, tag |
| `make local-setup` | clone `../pd-book-tools` + install editable |
| `make dev-local` | install `../pd-book-tools` editable in venv |
| `make run-local` | run UI against local editable workspace |

Append `AI=1` to any target for agent-friendly output — verbose output is
captured to `.ci-ai.log`; stdout shows `✅ <target> passed` on success or
filtered failure sections on error. Works for every target: `make ci AI=1`,
`make test AI=1`, etc.

## Rules

- Make targets first; fall back to `uv run …` only when no target exists.
- Never `python -m pytest`. Always `uv run pytest -n auto` or `make test`. Bare `python`/`python3`/`.venv/bin/python` miss the venv.
- HF dataset naming: `<owner>/pd-ocr-<source>-<lang>-<typeface>[-<qualifier>]`.
  - `source`: `real` | `synth` | `eval` | `mix`
  - `lang`: BCP-47 lowercase (`en`, `ga`, `de`, `el`, `grc`, `la`, …)
  - `typeface` (closed enum): `roman`, `italic`, `smallcaps`, `blackletter`, `fraktur`, `clogaelach`, `greek`, `greek-classical`
  - Typeface-classifier datasets use the literal slot `typeface` instead of an enum value.
- `typeface` is a hard filter — no `mixed` at publish time; split into per-typeface repos.
- `italic` and `smallcaps` get no detection/recognition repos; they appear only in classifier datasets.
- Every trained model writes a JSON sidecar (`name`, `task`, `language`, `typeface`, `trained_on`, `doctr_arch`, `trainer_version`, `trained_at`).
- Model artifacts exported here are consumed by `pd-ocr-cli`; never rename exports without coordinating downstream.
- Git LFS required for `.pt`/`.bin` model files — a suspiciously small model file is an LFS pointer, not a real model.
- `make dev-local` installs an editable `pd-book-tools`; `make upgrade-deps` in that state reverts it — use `make upgrade-deps-local` when that target exists (milestone t1).
- Do not kick off long training runs as part of tests; training is operator-driven.
- `model-trainer.ipynb.old` is not the current entry point — use `make run`.

## Key docs

- `docs/ROADMAP.md` — HF datasets integration plan (approved, not yet implemented); milestones a/a.5/b/c/d, glyph milestones g1/g2, tooling t1/t2.
- `docs/DATASETS.md` — dataset shape + card-data spec (parquet/imagefolder, auth, caching).
- `docs/specs/` — per-milestone detail specs (glyph-annotation-eval-slicing, glyph-feature-classifier, dev-local-upgrade-flow, local-mode-port-autoselect).

## Layout

- `src/pd_ocr_trainer/` — installable package; `dataset_store.py` (`DatasetSource`), `train_detect.py`, `train_recog.py`, `dataset_ui.py`, `ui.py`.
- `ml-training/<profile>/{detection,recognition}/` — local training data by profile/group.
- `ml-validation/<profile>/{detection,recognition}/` — local validation data by profile/group.
- `matched-ocr/` — richer page documents; preferred backfill source over `labels.json`.

## Sibling repos

- `../pd-book-tools/` — upstream; pinned in `pyproject.toml`; editable via `make dev-local`.
- `../pd-ocr-labeler/` — labels ground truth; trainer consumes labeled exports; labeler must declare `language` + `typeface` on exports (cross-repo work).
- `../pd-ocr-synth/` — synthetic training data (upstream); stamps `language` + `typeface` from recipes.
- `../pd-ocr-cli/` — consumes exported model artifacts; model-rename in milestone d is a breaking change there.
