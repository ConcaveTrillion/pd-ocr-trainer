# CLAUDE — pd-ocr-trainer

DocTR OCR model training pipeline (detection, recognition, typeface
classification) for PGDP data, driven by a NiceGUI training UI. HF dataset
publishing, model export, and metadata sidecars live here — not in the labeler.
Architecture: `docs/plans/roadmap.md` (approved plan) + `docs/architecture/datasets.md` (dataset spec).

## Commands

| target | does |
|---|---|
| `make setup AI=1` | install deps + pre-commit hooks (`uv sync --group all-dev`) |
| `make install` | install `pd-ocr-trainer-ui` as a uv tool (puts CLIs on PATH) |
| `make test AI=1` | `uv run pytest -n auto -v -ra` |
| `make test-single TEST='...' AI=1` | run one pytest node id |
| `make test-k K='pattern' AI=1` | run tests by `-k` expression |
| `make lint AI=1` / `make lint-fix AI=1` | ruff via pre-commit (with auto-fix) |
| `make format AI=1` | format then lint |
| `make pre-commit-check AI=1` | run all pre-commit hooks |
| `make ci AI=1` | install + pre-commit + test + build |
| `make build AI=1` | wheel + sdist into `dist/` |
| `make run` | start NiceGUI training UI (`uv run pd-ocr-trainer-ui`) |
| `make export-models AI=1` | copy trained artifacts to workspace exports dir |
| `make clean` / `make reset` | remove caches / rebuild venv |
| `make release-patch/minor/major` | bump version, commit, tag |
| `make local-setup` | clone `../pd-book-tools` + install editable |
| `make dev-local` | install `../pd-book-tools` editable in venv |
| `make run-local` | run UI against local editable workspace |

`AI=1` captures verbose output to `.ci-ai.log`; stdout shows `✅` on pass or
filtered failure sections on error. Remove `AI=1` only if you need full verbose
output for debugging.

## Rules

- Always run `make ci AI=1` before committing.
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

- `docs/plans/roadmap.md` — HF datasets integration plan (approved, not yet implemented); milestones a/a.5/b/c/d, glyph milestones g1/g2, tooling t1/t2.
- `docs/architecture/datasets.md` — dataset shape + card-data spec (parquet/imagefolder, auth, caching).
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

## GH issues

Cross-cut work tasks are tracked as GH issues in
**`ConcaveTrillion/ocr-container-meta`** (not in this repo's own tracker).
Plans under `docs/plans/` in the workspace root are synced there
via `/decompose-spec --sync`. Milestone naming: `spec: <plan-basename> (#N)`.

When shipping a plan task:

- Before starting: `gh issue view <N> --repo ConcaveTrillion/ocr-container-meta`
- After completing: `gh issue close <N> --repo ConcaveTrillion/ocr-container-meta`
- List open tasks:
  `gh issue list --repo ConcaveTrillion/ocr-container-meta --milestone "spec: <name> (#N)" --state open`

## docs/ folder

This repo follows the workspace docs/ template — see [`docs/README.md`](docs/README.md). Active
folders: `architecture/`, `decisions/`, `plans/`, `process/`, `research/`,
`runbooks/`, `specs/`, `templates/`, `usage/`, plus parallel `archive/`
subfolders.

**Superpowers redirect.** When a superpowers skill (e.g. `brainstorming`,
`writing-plans`) instructs you to save to `docs/superpowers/specs/<file>.md`
or `docs/superpowers/plans/<file>.md`, save to `docs/specs/<file>.md` or
`docs/plans/<file>.md` instead. There is no `docs/superpowers/` subdirectory
in this repo.

<!-- workspace-process:start -->

## Before coding

These steps are workspace defaults for any coding task. **User-level settings
override them** — a user's own `~/.claude/CLAUDE.md`, `settings.json`, or a
direct instruction in the conversation takes precedence and may waive or
change any step below.

### Working principles

- **Use skills.** Invoke the relevant superpowers skill before starting —
  process skills first (`brainstorming`, `systematic-debugging`,
  `writing-plans`, `test-driven-development`), then implementation skills.
  If a skill applies, using it is not optional.
- **Write clearly.** Follow `docs/process/writing-style.md` for direct user
  updates, handoffs, final summaries, docs, reports, issue text, PR text, and
  user-facing copy. Keep agent communication short, clear, and easy to scan.
- **Delegate by default.** Dispatch subagents for non-trivial work: per-repo
  agents for repo changes, `Explore` for code searches. This keeps large tool
  output out of the parent context.
- **Parallelize.** Run independent tasks as concurrent subagents — multiple
  agent calls in a single message. Set `model: sonnet` on implementers and
  reviewers.

### Steps

1. **Check the working tree.** `git status --short`. Surface or resolve stray
   uncommitted work before starting — don't build on it.
2. **Read repo guidance.** This repo's `CLAUDE.md` and `CONVENTIONS.md` for
   repo-specific rules.
3. **Consult `docs/` for authoritative context** (whichever folders exist):
   `plans/` (the work plan), `specs/` (design specs — follow any `Spec:`
   pointer from the issue), `research/` (prior investigations), `decisions/`
   (ADRs / constraints), `architecture/` (shipped design).
4. **Check live issue status.** `gh issue view <N> --repo <owner/repo>` —
   confirm it isn't already closed; note its milestone.
5. **Check for in-flight work.** Open PRs and existing branches touching the
   same area, to avoid colliding with work-in-progress.
6. **Consult agent memory.** `.claude/agent-memory/<repo>/feedback_*.md` for
   corrections not yet promoted to `CONVENTIONS.md`.
7. **Locate code with `Explore` first.** Use an `Explore` subagent to find
   relevant files before broad `Read`/grep.
8. **Isolate in a worktree.** Never work directly in the interactive checkout
   at `/workspaces/ocr-container/<repo>/`. Use the `using-git-worktrees` skill
   to set up an isolated worktree. When delegating to a full-power
   implementation agent, pass `isolation: "worktree"` on the `Agent` call
   (skip for `-docs` agents and the `driver` agent). When an agent returns a
   worktree path + branch, use the `finishing-a-development-branch` skill to
   decide how to integrate.
9. **TDD.** Write the failing test first where the plan calls for it.
10. **Verify before committing.** Focused verification plus `make ci`.
11. **Commit locally; do not push** without explicit say-so.

<!-- workspace-process:end -->
