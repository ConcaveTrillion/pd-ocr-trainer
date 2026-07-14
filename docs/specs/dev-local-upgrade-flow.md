---
Status: draft
Owner: CT
Created: 2026-05-11
Last verified: 2026-07-14
Kind: spec
Supersedes: N/A
Promotes to: N/A
Disposition: Parked until the CUDA environment restoration contract is chosen.
---

# Spec: dev-local-aware `upgrade-deps`

**Status:** spec only, not implemented.
**Owner repo:** `pd-ocr-trainer`.
**Workspace-wide:** sibling spec must land in every pd-* repo; the
detection contract is anchored in `pdomain-book-tools` and reused here.

## Motivation

`make upgrade-deps` ends with `uv sync --group all-dev`, which resolves
the venv strictly against `pyproject.toml` + `uv.lock`. In **dev-local
mode** — `make dev-local` has installed editable sibling checkouts of
`pdomain-book-tools` and (per `dev-env-setup.sh`) a sibling `doctr` clone,
plus, on this repo, an operator-installed CUDA-built `torch` /
`torchvision` / `torchaudio` wheel — that final `uv sync` silently
reverts every one of those to the canonical published version (and to
the default CPU torch wheel from PyPI). The operator has no warning;
the next training run picks up CPU torch and the wrong sibling code.

Trainer is the most GPU-sensitive repo in the workspace: a silent
revert from a CUDA torch wheel to a CPU one turns a 10-minute training
loop into an unusable one and is invisible until somebody notices the
GPU is idle. This spec captures the workspace-standard fix as it
applies to `pd-ocr-trainer`.

## Required behavior

1. **Detect dev-local vs canonical before any `uv sync`.** `upgrade-deps`
   must branch on the detected mode. Canonical mode is unchanged from
   today; dev-local mode refuses-with-message by default.
2. **Detection cascade** (in order):
   1. Probe `uv pip show pdomain-book-tools` for an
      `Editable project location:` line. This is the cross-repo
      contract anchor — `pdomain-book-tools` is dev-local mode's required
      sibling, so its editable status is the canonical signal.
   2. Fallback: a marker file `.venv/.pd-dev-local` written by
      `make dev-local` on success.
   3. Last resort: env var `PD_DEV_LOCAL=1` (operator override; useful
      in CI shells where `.venv/` may be ephemeral).
   Any one of the three trips dev-local mode.
3. **UX.**
   - Default `make upgrade-deps` in dev-local mode: **refuse and print
     a message** explaining that `uv sync` would revert the editable
     `pdomain-book-tools` (and editable `doctr` if present) and the
     CUDA-built torch wheels, and pointing the operator at
     `make upgrade-deps-local`.
   - New `make upgrade-deps-local` recipe: `uv lock --upgrade`, then
     `uv sync --group all-dev`, then **re-apply the dev-local layer**:
     re-install `-e ../pdomain-book-tools`, re-install `-e` doctr if present
     at `../doctr` or `../doctr_package`, and **re-install the
     CUDA-built torch / torchvision / torchaudio wheels** the operator
     has on file (from a documented pinned source — see "GPU-extras
     restore" below).
   - Both flows preserve `make dev-local`'s post-conditions: a passing
     `make check-local-editable` after the recipe completes.
4. **Canonical mode unchanged.** Operators not in dev-local mode see
   no behavior change — `make upgrade-deps` continues to do
   `uv lock --upgrade && uv sync --group all-dev`.
5. **Cross-platform.** The detection probe and recipes must work on
   Linux dev containers and macOS workstations alike. No `bash`-only
   constructs in the Make recipe; prefer `uv run python -c` for any
   parsing logic so we share one implementation across pd-* repos.

## GPU-extras restore — flag-only, complicates this repo

`pyproject.toml` declares `torch>=2.7.0`, `torchvision>=0.22.0`, and
`torchaudio>=2.7.0` with **no `[tool.uv.sources]` entry pinning a CUDA
index**. Practically, this means:

- Canonical `uv sync` resolves torch from PyPI, which on linux x86_64
  is a CPU-or-detected-default wheel — not the CUDA build a GPU
  workstation needs.
- Operators today get a CUDA torch into the venv by **manually**
  `pip install`ing from `https://download.pytorch.org/whl/cuXXX` after
  `make dev-local` completes. That manual step is undocumented in this
  repo and is exactly what `uv sync` clobbers.
- `dev-env-setup.sh` additionally clones and editable-installs
  `mindee/doctr` from git (`pip install -e .[torch]`); `pyproject.toml`
  pins canonical `python-doctr>=0.11.1a0`. So the dev-local layer for
  this repo is **three** editable/operator-applied artifacts that
  `uv sync` reverts: pdomain-book-tools, doctr, and CUDA torch wheels.

The implementation pass needs to decide how `upgrade-deps-local` discovers
"which CUDA build does this venv want." Options to evaluate:

- A `.venv/.pd-cuda-extras` marker recording the index URL + pinned
  wheel versions, written by a new `make gpu-extras` recipe that
  formalizes today's manual step.
- A repo-level `tools/cuda-extras.sh` that the operator edits once and
  `upgrade-deps-local` re-runs at the end.
- A `[tool.uv.sources]` torch entry conditional on a `gpu` extra,
  installed via `uv sync --extra gpu` (would also fix the canonical
  install for GPU operators, but **must remain PEP 503-compatible** —
  no direct-URL pins; pre-clear with the workspace release-strategy
  memo before adopting).

Picking among these is **out of scope for this spec** — the spec
requires only that `upgrade-deps-local` *restore* GPU extras, and
flags that the restoration mechanism does not exist yet in this repo.

## Test plan

- Canonical mode: `make upgrade-deps` in a venv where
  `uv pip show pdomain-book-tools` shows a registry install — proceeds as
  today.
- Dev-local mode: `make dev-local && make upgrade-deps` — refuses with
  the documented message, exits non-zero, leaves the venv untouched.
- Dev-local mode: `make dev-local && make upgrade-deps-local` —
  upgrades the lock, re-applies editable pdomain-book-tools (and doctr if
  the sibling clone is present), re-applies CUDA torch wheels, and
  passes `make check-local-editable`.
- Override: `PD_DEV_LOCAL=1 make upgrade-deps` in a venv with a
  registry-installed `pdomain-book-tools` — refuses (env var wins).
- Marker fallback: delete `pdomain-book-tools` editable install but keep
  `.venv/.pd-dev-local` — refuses (marker still trips the cascade).

## Out of scope

- Choosing the CUDA-extras restoration mechanism (see above).
- Migrating `dev-env-setup.sh` / `install-training-dependencies.sh`
  off `pip install -e .[torch]` for doctr — separate cleanup.
- Any change to canonical-mode `upgrade-deps` semantics.

## Adversarial Review

Stage: migration-time design review. Source: a read-only analyzer and direct comparison with the Makefile, setup scripts, dependency configuration, tests, and history.

The review accepted the risk that `uv sync` can replace editable or operator-selected dependencies. It changed the result by parking this spec as a draft: the proposed refusal path remains useful, but the document leaves CUDA restoration out of scope even though restoration is required for a complete workflow.

Current practice has `dev-local` and `check-local-editable`, but no `upgrade-deps-local`, mode marker, CUDA-state record, or restoration test. Residual risks are an unverified assumption about current PyTorch wheel selection and the unresolved choice between recorded GPU state, declarative extras, or deliberate exclusion.
