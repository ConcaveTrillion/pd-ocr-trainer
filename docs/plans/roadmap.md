---
Status: active
Owner: CT
Created: 2026-05-11
Last verified: 2026-07-14
Kind: plan
Supersedes: N/A
Promotes to: N/A
Disposition: Approved roadmap; major milestones remain unimplemented.
---

# pd-ocr-trainer roadmap — Hugging Face datasets integration

**Status:** approved plan, not yet implemented.
**Companion spec:** [`../specs/datasets.md`](../specs/datasets.md) — dataset shape and
format spec (parquet/imagefolder layouts, card-data keys, auth, caching).
This file is the *plan*; `docs/specs/datasets.md` is the *spec*.

## Goal

Move the repository from local-only OCR datasets toward versioned Hugging Face datasets while preserving the current training workflow during the transition. The milestones also cover typeface and glyph classifiers, model metadata, and two local-development safeguards.

## Architecture

The target separates dataset contracts from their producers and consumers. `pd-ocr-trainer` loads local or Hub-backed sources, trains task-specific models, and exports model metadata; sibling repositories remain responsible for producing their own source data. This is planned architecture, not current behavior.

## Tech Stack

The plan builds on Python, DocTR, Hugging Face Datasets and Hub storage, NiceGUI, Parquet or ImageFolder datasets, and the existing `uv` and Make workflows.

## Global Constraints

Dataset names, language and typeface metadata, single-typeface publishing, classifier-only italic/small-caps handling, model sidecars, Git LFS artifacts, and downstream export compatibility must remain consistent across milestones. Training remains operator-driven; tests must not launch long training jobs.

## Headline decisions

- **HF dataset publishing lives in `pd-ocr-trainer`**, not in the
  labeler. The labeler labels; the trainer is the consumer of labeled
  data, so it owns packaging, publishing, and versioning.
- Two new attributes are required on every dataset and every model:
  - `language` — BCP-47 lowercase, open string (`en`, `ga`, `de`, `el`,
    `grc`, `la`, …).
  - `typeface` — closed enum (see below).
- **`typeface` is a hard filter, not a tag.** Detection and
  recognition models train on a single typeface variant. There is no
  `mixed` typeface at publish time.
- **A separate typeface classifier model handles word-level style
  detection** (italic, smallcaps). It runs on each word crop after
  detection and emits the typeface enum + confidence as a *style
  annotation* on the final document (italic survives to PGDP markup,
  etc.), not as a router to a different recognizer. **No
  italic-recog or smallcaps-recog models exist** — the roman
  recognizer handles roman+italic+smallcaps word crops as part of one
  trained distribution.

## Repo naming convention

```text
<owner>/pd-ocr-<source>-<lang>-<typeface>[-<qualifier>]
```

| Slot        | Values                                                                                    |
|-------------|-------------------------------------------------------------------------------------------|
| `source`    | `real` \| `synth` \| `eval` \| `mix`                                                      |
| `lang`      | BCP-47 lowercase (`en`, `ga`, `de`, `el`, `grc`, `la`, …)                                 |
| `typeface`  | closed enum (next section) for detection/recognition; literal `typeface` for classifier datasets |
| `qualifier` | optional, free-form (`2026q2`, `pgdp-batch-3`, …)                                         |

Examples:

- `ntw8532/pd-ocr-real-en-roman`
- `ntw8532/pd-ocr-synth-ga-clogaelach`
- `ntw8532/pd-ocr-real-en-typeface` (typeface-classifier dataset)

For dataset-shape and card-data details that go inside each repo, see
[`../specs/datasets.md#dataset-shapes`](../specs/datasets.md#dataset-shapes) and
[`../specs/datasets.md#card-data-block`](../specs/datasets.md#card-data-block).

## Typeface enum

`roman`, `italic`, `smallcaps`, `blackletter`, `fraktur`, `clogaelach`,
`greek`, `greek-classical`.

For **detection and recognition repos**, a dataset declares **a single
value** from this enum. `mixed` is disallowed at publish time —
operators must split mixed data into per-typeface repos. The literal
string `typeface` is also disallowed in this slot.

For **typeface-classifier repos**, the slot is the literal string
`typeface` (signaling "this dataset spans multiple typeface values"),
and the per-row `typeface` column carries the actual enum value.

## Tiering rule for mixed-typeface ground truth

Real-world pages mix typefaces. Resolve by tier:

- **Page-level typefaces** — `roman`, `clogaelach`, `fraktur`,
  `blackletter`, `greek`, `greek-classical`. These get **full
  detection + recognition repos** (a Cló Gaelach page is all Cló
  Gaelach, etc.).
- **Word-level typefaces** — `italic`, `smallcaps`. These get **no
  detection or recognition repos**. They appear only as labels in
  typeface-classifier datasets. Their visual variations are part of
  the training distribution for the page-level recognizer that
  contains them (the roman recognizer handles roman+italic+smallcaps
  word crops as one trained distribution).

Italic and smallcaps drop out of the recognition tier entirely. They
live in classifier-dataset space.

## Milestones (each independently shippable)

### (a) HF read path alongside local

- Add a `DatasetSource` interface in `dataset_store.py`. Two
  implementations: `LocalSource` (current behavior) and
  `HFDatasetSource` (`datasets.load_dataset` plus the DocTR adapter).
- `train_detect.py` and `train_recog.py` gain a `--source` argument;
  profile config grows an optional `sources:` list.
- Set `HF_HOME` to the existing `global-shared-ai-cache` Docker
  volume (see
  [`../specs/datasets.md#caching`](../specs/datasets.md#caching)).
- Trainer UI shows configured sources read-only.
- **Ship criterion:** a profile with a single `hf:` source trains
  end-to-end against a hand-uploaded test dataset.

### (a.5) Typeface classifier path

- Introduce `train_typeface.py` (or equivalent) in the trainer
  alongside `train_detect.py` / `train_recog.py`. Same `--source`
  argument and HF read path as (a); reuses `DatasetSource` /
  `HFDatasetSource` from milestone (a).
- Loader recognizes `pd-ocr-shape: typeface-classification/v1` and
  yields `(image, typeface)` pairs. Small image-classification model
  (architecture TBD, kept deliberately small — runs per word crop at
  inference).
- Profile config grows the same `sources:` list shape; classifier
  profiles point at one or more `pd-ocr-*-typeface` repos.
- **Ship criterion:** a profile with one classifier `hf:` source
  trains end-to-end against a hand-uploaded test dataset.

### (b) Trainer publishes datasets

- New CLI: `pd-ocr-trainer publish-dataset --profile <p> --task <detection|recognition|typeface-classification> --to <repo>`.
- Wraps `huggingface_hub.HfApi.upload_large_folder`.
- Emits imagefolder + `metadata.jsonl` for recognition, parquet for
  detection, imagefolder + `metadata.jsonl` (with a `typeface`
  column) for typeface-classification (see
  [`../specs/datasets.md#dataset-shapes`](../specs/datasets.md#dataset-shapes)).
- Generates a `README.md` with YAML front matter (license, language,
  tags, `pd_ocr_*` keys).
- Idempotent: skips when the local snapshot hash matches the remote
  tip.
- NiceGUI button mirrors the CLI.
- **Ship criterion:** round-trip — publish, then load via (a) and
  train.

### (c) Backfill existing corpora

- One-off backfill of `ml-training/{all,italics}/{detection,recognition}`
  into HF repos.
- Interactive script: prompts the operator for `(language, typeface)`
  per project.
- **Prefer `matched-ocr/` as the source** — it carries the richer
  pdomain-book-tools page document, not the already-derived `labels.json`.
- The labeler's per-word style flags are also backfilled, into a
  real-en-typeface (and per-language) classifier dataset. Same
  backfill pass, one additional output dataset shape.
- Private-by-default; promote to public after license review (see
  [`../specs/datasets.md#authentication`](../specs/datasets.md#authentication)).
- **Ship criterion:** every existing on-disk corpus is reproducible
  from an HF repo with documented `(language, typeface)`, including
  a classifier dataset derived from labeler style flags.

### (d) Cut the local-only path

- Profiles without `sources:` emit a deprecation warning.
- Trainer kanban is replaced by a publish-to-staging-branch flow.
- After one release, remove `LocalSource` as the default.
- Coordinate the model-rename
  `pd-<lang>-<typeface>-<task>-<base>` with `pd-ocr-cli`.
- **Ship criterion:** production training uses only HF sources.

## Glyph-annotation milestones

Workspace-wide feature: each OCR word may carry an optional
`glyph_annotations` sidecar (ct/st ligatures, long-s positions, swash
caps, …). GT text remains canonical; annotations live in parallel.
The data model is owned by `pdomain-book-tools`; training data is produced
by `pd-ocr-synth` (gold) and the labeler (human).

The trainer has two distinct, **independently shippable** jobs.

### (g1) Glyph-annotation eval slicing — *cheap, ships first*

- Eval pipeline reports recognition CER/WER **sliced per glyph
  feature** (one bucket per `LigatureMark.kind`, plus `long_s`,
  `swash`).
- Read-only consumer of `glyph_annotations` — no new model.
- `glyph_annotations is None` words are **excluded** from sliced
  metrics; never silently counted as feature-absent.
- Output: per-feature breakdown table in the trainer UI + JSON
  sidecar consumable by CI for regression alerts.
- Spec: [`../specs/glyph-annotation-eval-slicing.md`](../specs/glyph-annotation-eval-slicing.md).
- **Ship criterion:** running eval against an annotated
  `ml-validation/all/recognition/` set produces overall CER/WER plus a
  per-feature table; features with N(pos) < 30 flagged "low support."
- **Dependency:** `pdomain-book-tools` `GlyphAnnotations` data model
  landed; at least one eval dataset annotated.
- **Not blocked by (g2).**

### (g2) Glyph-feature classifier — *new model, ships after (g1)*

- New training entry point (e.g. `train_glyph.py`) alongside
  `train_detect.py` / `train_recog.py` / `train_typeface.py`.
- Model: small CNN, multi-head sigmoid (one head per binary feature).
  Architecture TBD via experiment (see spec).
- Training data: `pd-ocr-synth` (primary, weight ~0.8) +
  human-labeled crops from `pd-ocr-labeler` (secondary, weight ~0.2,
  upsampled).
- New HF dataset shape `pd-ocr-shape: glyph-classification/v1`;
  repo naming `<owner>/pd-ocr-<source>-<lang>-glyph`.
- Export: same sidecar shape as recognition/detection
  (see [model-metadata-sidecar](#model-metadata-sidecar)) with
  `task: "glyph-classification"` and per-feature `t_auto` /
  `t_suggest` thresholds calibrated on a held-out human-labeled set
  (precision ≥ 0.99 for auto-fill; recall ≥ 0.9 for suggest).
- Eval gated on the **human-labeled held-out set**, not synth.
- Inference consumer: the labeler pre-fills annotation suggestions
  humans accept/reject — closes the loop.
- Spec: [`../specs/glyph-feature-classifier.md`](../specs/glyph-feature-classifier.md).
- **Ship criterion:** classifier exports against a synth + small
  human dataset, labeler loads the sidecar and pre-fills annotations
  on a sample page.
- **Dependencies:** `pdomain-book-tools` `GlyphAnnotations` model;
  `pd-ocr-synth` emits `glyph-classification/v1` datasets;
  human-labeled glyph-annotation pipeline in `pd-ocr-labeler`.

## Tooling milestones

### (t1) dev-local-aware `upgrade-deps` — *workspace-wide hazard*

- `make upgrade-deps` ends with `uv sync --group all-dev`, which
  silently reverts a dev-local venv (editable `pdomain-book-tools`,
  editable sibling `doctr`, and operator-installed CUDA torch wheels)
  back to canonical published / CPU-default builds. Especially
  painful here because trainer is the most GPU-sensitive repo in the
  workspace.
- Detect dev-local via `uv pip show pdomain-book-tools` (Editable project
  location), `.venv/.pd-dev-local` marker, or `PD_DEV_LOCAL=1`.
- Default `upgrade-deps` refuses-with-message in dev-local mode;
  sibling `upgrade-deps-local` recipe does
  `uv lock --upgrade` + `uv sync` + dev-local restore (editable peers
  **and** CUDA torch/torchvision/torchaudio wheels).
- Spec: [`../specs/dev-local-upgrade-flow.md`](../specs/dev-local-upgrade-flow.md).
- **Ship criterion:** `make upgrade-deps` in a dev-local venv leaves
  the venv untouched and prints the redirect; `make upgrade-deps-local`
  upgrades the lock and ends with a passing `check-local-editable` plus
  a CUDA torch wheel restored.
- **Open sub-question:** the GPU-extras restoration mechanism does
  not exist in this repo yet (no `[tool.uv.sources]` torch entry, no
  `make gpu-extras` recipe). Resolve before implementing (t1) end-to-end.

### (t2) Local-mode port auto-select + stable bookmarks + in-UI URL display

- The NiceGUI training UI launched via `make run` currently fails
  loud on port collision and prints the bound URL only to stdout.
  Triggered by a real stale-port collision; trainer is a long-running
  oversight UI, so operators routinely close the launching console.
- Local-mode startup tries persisted-last-port, then default port,
  then `port=0`. Explicit `--port N` disables the fallback and fails
  loud on collision (operator intent honored).
- Persist last successfully-bound port to a small state file so
  bookmarks survive across restarts.
- Surface the bound URL in the running UI itself (footer / header /
  About / copy widget) in addition to the stdout banner.
- Spec: [`../specs/local-mode-port-autoselect.md`](../specs/local-mode-port-autoselect.md).
- **Workspace-wide convergence.** Sibling items in
  `pd-prep-for-pgdp` (commit `b23b913`), `pd-ocr-labeler-spa`
  (commit `b956275`), and `pd-ocr-labeler` (sibling item being added
  in parallel). All four local web apps must implement the same
  behavior surface.
- **Ship criterion:** all six tests in the spec's acceptance section
  pass; in-UI URL is visible after closing the launching console.

## Cross-repo dependencies

- **`pd-ocr-labeler`** — declare `language` + `typeface` on DocTR
  exports. The existing `--style italics` filter should write the
  chosen style as a dataset-level `typeface` in an export manifest.
  Decision needed: per-project, per-page, or derived from per-word
  style flags? Recommendation: **per-export-group**, with a
  sanity-check warning if per-word flags disagree.
  Additionally: **the labeler's per-word style flags are the
  canonical source of training data for the typeface classifier.**
  The labeler export should provide enough metadata for the trainer to emit a classifier
  dataset (`typeface-classification/v1`) in addition to recognition
  crops. Open question on the labeler side: same export pass produces
  both shapes, or two separate passes?
- **`pd-ocr-cli`** — model-naming rename
  `pd-<profile>-<task>-…` → `pd-<lang>-<typeface>-<task>-…` is a
  breaking change. Either keep the old prefix as an alias for one
  release, or do a coordinated point release.
- **`pd-ocr-synth`** — synth exports provide `language` + `typeface`
  from the recipe so the trainer can package them (recipes already
  encode font and language by identity).
- **`pdomain-book-tools` (glyph milestones)** — owns the
  `GlyphAnnotations` data model. (g1) and (g2) both block on it
  landing.
- **`pd-ocr-synth` (glyph milestones)** — emits gold
  `glyph_annotations` on every word, plus a `glyph-classification/v1`
  dataset shape for (g2). Recipes already know which font has which
  ligatures, so annotations are deterministic.
- **`pd-ocr-labeler` (glyph milestones)** — UI to view/edit
  `glyph_annotations` on words; export of human-labeled
  `glyph-classification/v1` datasets; loads the (g2) classifier
  artifact for pre-fill at label time. Closes the loop back into
  training data.

## Open questions blocking forward motion

- **Auth.** Where does `HF_TOKEN` live? Plan: dev container mount from
  `~/.huggingface/token`; CI via GH Actions secret. Confirm the dev
  container actually mounts it and that the trainer fails closed on a
  missing token.
- **Private vs public default.** Plan says private-by-default,
  promote-per-dataset. Confirm.
- **Mixed-typeface resolution.** The page-level / word-level tiering
  rule above is a recommendation — needs explicit user sign-off.
- **Italic glyphs in roman-recog training distribution.** Does the
  roman recognizer need explicit italic samples mixed into its
  training data to handle italic word crops at inference, or does
  generalization carry it? Validate empirically. Collect
  italic-tagged crops via the labeler regardless, so the option to
  inject them into the recognizer's training set stays open.
- **HF LFS quotas** for `ntw8532` against synth dataset sizes
  (millions of crops, tens of GB).
- **License/attribution policy.** Plan: per-row `license` (SPDX) in
  the schema; the labeler refuses export when the source license is
  missing.

## Model metadata sidecar

Every trained model writes a sidecar JSON so consumers (notably
`pd-ocr-cli`) can trace the model back to its source datasets:

```json
{
  "name": "pd-en-roman-detection-2026-05-05",
  "task": "detection",
  "language": "en",
  "typeface": "roman",
  "trained_on": [
    {"repo": "ntw8532/pd-ocr-real-en-roman", "revision": "v2026.05.05", "rows": 4231, "weight": 0.7},
    {"repo": "ntw8532/pd-ocr-synth-en-roman", "revision": "a3f2e1c", "rows": 50000, "weight": 0.3}
  ],
  "doctr_arch": "db_resnet50",
  "trainer_version": "0.x.y",
  "trained_at": "2026-05-05T18:00:00Z"
}
```

A typeface-classifier model writes the same sidecar shape, with
`task: "typeface-classification"` and `typeface: "typeface"` (the
literal string, since the model spans all enum values):

```json
{
  "name": "pd-en-typeface-classification-2026-05-05",
  "task": "typeface-classification",
  "language": "en",
  "typeface": "typeface",
  "trained_on": [
    {"repo": "ntw8532/pd-ocr-real-en-typeface", "revision": "v2026.05.05", "rows": 12000, "weight": 0.6},
    {"repo": "ntw8532/pd-ocr-synth-en-typeface", "revision": "b1d4f7a", "rows": 80000, "weight": 0.4}
  ],
  "trainer_version": "0.x.y",
  "trained_at": "2026-05-05T18:00:00Z"
}
```

See also
[`../specs/datasets.md#model-metadata-sidecar`](../specs/datasets.md#model-metadata-sidecar).

## Future possibility — Modal for trainer GPU compute (separate path)

Recorded 2026-05-06. Not a milestone, not scheduled — a flag for future
planning so the option isn't lost.

Modal is on the table as one possible GPU backend for training runs,
but **as its own dedicated Modal app**, not as a tenant on the shared
scheduled-drain function that `pd-prep-for-pgdp` and (prospectively)
`pd-ocr-labeler-spa` are considering for their bursty preprocess /
OCR-pre-pass workloads. Likely shape here: a long-timeout GPU function
(Modal supports up to 24h) invoked on demand when a training run is
kicked off, with GPU class chosen per run (A10G / L4 / A100 depending
on dataset size and architecture).

Why trainer warrants its own path, not the shared drain:

- **Run length:** hours-to-days end-to-end, not minutes-per-batch.
  Cramming a training run into a 6h cron-fired drain function would
  either time out or starve the prep/labeler queues sharing it.
- **GPU class:** training likely wants more VRAM than a T4
  (A10G/A100 territory); prep/labeler are happy on a T4.
- **Cost profile:** Modal's $30/mo free credit (~50 hr/mo on T4) does
  not cover real training runs. Trainer's Modal usage is a separate
  budget conversation: paid credit, sponsor compute, donated GPU time,
  or lab/university infra.
- **State:** training produces large model artifacts with their own
  storage lifecycle, separate from prep/labeler's per-batch R2
  inbox/outbox keys.

Alternative paths trainer should keep open:

- Local GPU on a contributor's workstation (current development path).
- Donated / lab / university GPU.
- Modal paid (if a sponsor or grant covers it).
- Cloud Run with GPU (newer Google offering, sometimes cheaper than
  Modal per-hour for sustained workloads, more ops overhead).

Out of scope for trainer: the shared scheduled-drain Modal function
used by prep/labeler. Even if those neighbors share a Modal account
someday, trainer should not piggyback on their drain — separate app,
separate budget, separate GPU class.

Cross-reference: `pd-prep-for-pgdp` and `pd-ocr-labeler-spa` are
independently recording their own future-state notes about the shared
scheduled-drain pattern. This trainer note exists specifically to flag
"we considered the shared pattern and rejected it for trainer."

## Relevant files

Primary edit sites for the milestones above:

- `/workspaces/ocr-container/pd-ocr-trainer/src/pd_ocr_trainer/dataset_store.py`
  — primary edit site for milestones (a) and (b).
- `/workspaces/ocr-container/pd-ocr-trainer/src/pd_ocr_trainer/dataset_ui.py`
  — kanban UI; milestones (b) and (d).
- `/workspaces/ocr-container/pd-ocr-trainer/src/pd_ocr_trainer/train_detect.py`
  — gains `--source` in (a). Already has `push_to_hf_hub` for models.
- `/workspaces/ocr-container/pd-ocr-trainer/src/pd_ocr_trainer/train_recog.py`
  — same.
- `/workspaces/ocr-container/pd-ocr-trainer/src/pd_ocr_trainer/train_typeface.py`
  — *new file*, milestone (a.5). Same `--source` / HF read path as
  the others; trains the typeface-classifier model on
  `typeface-classification/v1` datasets. Not yet on disk — to be
  created.
- `/workspaces/ocr-container/pd-ocr-trainer/src/pd_ocr_trainer/ui.py`
  — `_prefixed_model_name` (~line 247), the rename point.
- `/workspaces/ocr-container/pd-ocr-trainer/ml-training/all/{detection,recognition}/labels.json`
  — current label format, backfill source.
- `/workspaces/ocr-container/pd-ocr-trainer/matched-ocr/` — richer
  source documents, **preferred** backfill input.
- `/workspaces/ocr-container/pd-ocr-labeler/pd_ocr_labeler/operations/export/cli.py`
  — needs language/typeface declarations (cross-repo).
- `/workspaces/ocr-container/pd-ocr-labeler/pd_ocr_labeler/constants.py`,
  `state/page_state.py` — per-word style flags (stay per-word).
