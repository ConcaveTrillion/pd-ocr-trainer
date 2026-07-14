---
Status: active
Owner: CT
Created: 2026-05-11
Last verified: 2026-07-14
Kind: spec
Supersedes: N/A
Promotes to: docs/architecture/datasets.md
Disposition: Approved target contract; implementation has not started.
---

# Workspace dataset design — Hugging Face Datasets

## Agent Index

- **Kind:** spec
- **Status:** active
- **Read when:** designing Hugging Face dataset exchange, naming, card data, caching, or trainer dataset integration.
- **Search terms:** Hugging Face datasets, dataset contract, card data, typeface, dataset publishing.

**Status:** specification, not yet implemented.
See [`../plans/roadmap.md`](../plans/roadmap.md) for the milestone-by-milestone
implementation plan.

**Scope:** cross-project. Defines how `pd-ocr-labeler`, `pd-ocr-synth`,
and `pd-ocr-trainer` produce, exchange, and consume datasets via the
Hugging Face Hub.

## Goal

Today every workspace project reads and writes datasets on a local
filesystem laid out per `pd-ocr-trainer/dataset_store.py`. That works
for single-machine workflows but breaks down once:

- Datasets are large enough that re-rendering or re-labeling per machine
  is wasteful.
- Multiple machines participate (labeler on a workstation, training on
  a GPU host).
- Reproducibility requires pinning to a specific dataset version.
- Other projects or collaborators want to consume the same data.

Hugging Face Datasets gives us versioned, addressable, cache-friendly
storage with first-class image and parquet support, and the workspace
already pulls models from HF (`pd-ocr-cli`). This spec extends the same
ecosystem to datasets.

## Roles

| Project           | Produces                                                                                  | Consumes              |
|-------------------|-------------------------------------------------------------------------------------------|-----------------------|
| `pd-ocr-labeler`  | Real labeled OCR exports and typeface-classifier labels | — |
| `pd-ocr-synth`    | Synthetic OCR exports and known typeface labels | — |
| `pd-ocr-trainer`  | Versioned Hub datasets, eval splits, and curated mixes | Labeler and synth exports |

The trainer packages and publishes producer exports to Hub dataset repos.
It reads from local disk or Hub dataset repos transparently. Producer
repositories do not own Hub publishing in this design.

## Dataset shapes

Three shapes; each project must produce or consume one of them.

### `recognition/v1`

Word-crop or line-crop images for recognition training.

- Format: **HF imagefolder with `metadata.jsonl`**.
- One row per cropped image.
- Columns: `image` (Image), `text` (string), and free-form provenance
  fields (`font`, `degradations`, `corpus`, …).

```text
<repo>/
├── data/
│   ├── 0000000.png
│   ├── 0000001.png
│   └── ...
├── metadata.jsonl       # one JSON object per image
├── README.md            # YAML-front-matter dataset card
└── recipe.snapshot.yaml # for synth-produced datasets only
```

### `detection/v1`

Page images plus bounding-box annotations for detection training.

- Format: **parquet** (boxes are awkward in imagefolder).
- One row per page.
- Columns: `image` (Image), `lines` (sequence of `{bbox, text, words}`),
  `size` (sequence of int), plus provenance.

```text
<repo>/
├── data/
│   ├── train-00000-of-00004.parquet
│   ├── train-00001-of-00004.parquet
│   └── ...
├── README.md
└── recipe.snapshot.yaml
```

### `typeface-classification/v1`

Word-crop images labeled with their typeface, for training the
typeface-classifier model. The classifier runs on each word crop after
detection and outputs a typeface enum value (plus confidence) used as
a *style annotation* on the final document — not a router to a
different recognizer.

- Format: **HF imagefolder with `metadata.jsonl`**.
- One row per cropped word image.
- Columns: `image` (Image), `typeface` (enum string — one of the
  values in [Typeface enum and tiering rule](#typeface-enum-and-tiering-rule)),
  plus existing provenance fields (`language`, `source`, `corpus`,
  `license`, …).
- A classifier dataset must contain **at least 2 distinct `typeface`
  values** (a single-valued classifier dataset trains nothing).

```text
<repo>/
├── data/
│   ├── 0000000.png
│   ├── 0000001.png
│   └── ...
├── metadata.jsonl       # one JSON object per image, with `typeface` column
├── README.md            # YAML-front-matter dataset card
└── recipe.snapshot.yaml # for synth-produced datasets only
```

### Card data block

Shape, language, typeface, and provenance are declared in the dataset
card's `card_data`:

```yaml
---
task_categories: [text-recognition]
tags: [ocr, gaelic, irish, pd-ocr]
language: ga                   # REQUIRED — BCP-47 lowercase
typeface: clogaelach           # REQUIRED — single value, no `mixed`
pd-ocr-shape: recognition/v1   # REQUIRED — or detection/v1 or typeface-classification/v1
pd-ocr-source: pd-ocr-synth    # REQUIRED — producer name
pd-ocr-recipe-sha: 2c4f...     # optional, synth datasets only
---
```

For a typeface-classifier dataset, the `typeface` card field is the
literal string `typeface` (signaling "this dataset spans multiple
typeface values"); the per-row `typeface` column carries the actual
enum value:

```yaml
---
task_categories: [image-classification]
tags: [ocr, typeface, pd-ocr]
language: en
typeface: typeface                            # literal string, classifier datasets only
pd-ocr-shape: typeface-classification/v1
pd-ocr-source: pd-ocr-labeler
---
```

Required keys on every dataset:

- `language` — BCP-47 (`en`, `ga`, `de`, `el`, `grc`, `la`, …).
- `typeface` — single value from the enum (next section) for
  detection/recognition datasets; the literal string `typeface` for
  classifier datasets. **Hard filter, not a tag** — `mixed` is
  disallowed at publish time.
- `pd-ocr-shape`, `pd-ocr-source` — as before.

The `pd-ocr-*` keys are conventional, not standardized — the trainer's
loader keys off them so it can tell shape and provenance without
parsing parquet schemas.

## Typeface enum and tiering rule

Allowed values for `typeface`:

`roman`, `italic`, `smallcaps`, `blackletter`, `fraktur`, `clogaelach`,
`greek`, `greek-classical`.

A detection or recognition dataset declares **exactly one** of these
values. `mixed` is rejected at publish time — split mixed data into
per-typeface repos. Classifier datasets are the one exception: their
card-level `typeface` field is the literal string `typeface` (the
per-row column carries the actual enum value).

Real pages mix typefaces. Resolve by tier:

- **Page-level typefaces** — `roman`, `clogaelach`, `fraktur`,
  `blackletter`, `greek`, `greek-classical`. These get **full
  detection + recognition repos** (a Cló Gaelach page is all Cló
  Gaelach, etc.).
- **Word-level typefaces** — `italic`, `smallcaps`. These get **no
  detection or recognition repos**. They appear only as labels in
  typeface-classifier datasets. Their visual variations (italic
  glyphs are slanted latin chars; smallcaps is uppercase at x-height)
  are part of the training distribution for the page-level
  recognizer that contains them — the roman recognizer handles
  roman+italic+smallcaps word crops as one trained distribution. **No
  separate italic or smallcaps recognition models exist.**

Consequence: italic and smallcaps live in classifier-dataset space
only. The classifier output is a *style annotation* on the final
document (italic survives to PGDP markup, etc.), not a router to a
different recognizer.

## Repo naming

One HF repo per dataset. Convention:

```text
<owner>/pd-ocr-<source>-<lang>-<typeface>[-<qualifier>]
```

| Slot        | Values                                                                 |
|-------------|------------------------------------------------------------------------|
| `source`    | `real` \| `synth` \| `eval` \| `mix`                                   |
| `lang`      | BCP-47 lowercase (`en`, `ga`, `de`, `el`, `grc`, `la`, …)              |
| `typeface`  | closed enum (above) for detection/recognition; literal `typeface` for classifier datasets |
| `qualifier` | optional, free-form (`2026q2`, `pgdp-batch-3`, …)                      |

The `typeface` slot accepts a single enum value for detection and
recognition repos. For typeface-classifier datasets, the slot is the
literal string `typeface` — signaling "this dataset spans multiple
typeface values" and keeping the existing 4-slot schema intact (no
separate `<task>` axis).

Examples:

- `ntw8532/pd-ocr-synth-ga-clogaelach` — synthetic Cló Gaelach
  recognition data
- `ntw8532/pd-ocr-synth-de-fraktur` — synthetic German Fraktur
  recognition data
- `ntw8532/pd-ocr-real-en-roman` — labeled real English roman pages
  from the labeler
- `ntw8532/pd-ocr-eval-ga-clogaelach-2026q2` — held-out eval set
- `ntw8532/pd-ocr-real-en-typeface` — real labeler-derived word crops
  with style flags, classifier dataset
- `ntw8532/pd-ocr-synth-en-typeface` — synth-generated word crops
  spanning typefaces (typeface known per font), classifier dataset
- `ntw8532/pd-ocr-real-ga-typeface` — Gaelic classifier dataset

Per-repo over per-config-multi-recipe because:

- Independent versioning (one recipe's update doesn't bump unrelated
  data).
- Independent visibility (private real-data while public synth).
- Cleaner dataset card per recipe.
- Smaller download for trainer profiles that only need one source.

## Versioning

HF dataset repos are git repos under the hood. We use them as such:

| Mechanism                                | Purpose                                                        |
|------------------------------------------|----------------------------------------------------------------|
| `main` branch                            | Latest published version                                       |
| Git tags (`v2026.05.05`, `r-gaelic-1`)   | Pinned releases                                                |
| Commit SHA                               | Deterministic pin in trainer profile config                    |
| `card_data.pd-ocr-recipe-sha`            | Synth: SHA of the recipe snapshot that produced this commit   |

Trainer profile configs pin a SHA or tag, never `main`, so training
runs are reproducible.

## Trainer integration — `pd-ocr-trainer` upgrade

### Current state

`dataset_store.py` reads profiles from
`<root>/ml-training/<profile>/{detection,recognition}` and
`<root>/ml-validation/<profile>/{detection,recognition}`. Profiles are
local directories, full stop.

### Target state

A **DatasetSource** abstraction; profiles become a list of one or more
sources:

```yaml
# pd-ocr-trainer profile config (new schema)
profile: gaelic
sources:
  - hf://ntw8532/pd-ocr-synth-ga-clogaelach@v2026.05.05  # pinned tag
  - hf://ntw8532/pd-ocr-real-ga-clogaelach@a3f2e1        # pinned SHA
  - local:./data/gaelic-corrections                       # local override
weights: [0.7, 0.25, 0.05]   # optional sampling weights
```

Sources are resolved in order. URI schemes:

| Scheme                       | Resolution                                              |
|------------------------------|---------------------------------------------------------|
| `local:<path>`               | Existing behavior; reads `pd-ocr-trainer/v1` layout     |
| `hf:<repo>[@<rev>]`          | `datasets.load_dataset(repo, revision=rev)`             |
| `hf-folder:<repo>[@<rev>]`   | `huggingface_hub.snapshot_download` then read locally   |

The first form preserves existing profiles. The second uses the
streaming-aware `datasets` API (no full download required for very
large sets). The third is an escape hatch when `datasets` can't infer
the schema.

### Refactor plan

1. **Extract** `DatasetSource` interface in `dataset_store.py`. Existing
   local-folder logic becomes `LocalSource`.
2. **Add** `HFDatasetSource` (uses `datasets.load_dataset`).
3. **Add** profile-config parsing for `sources:` lists with backward-
   compat for the implicit-local-folder form.
4. **Cache** HF datasets in the existing `global-shared-ai-cache` Docker
   volume by setting `HF_HOME` there.
5. **Surface** dataset provenance in training logs and saved model
   metadata (so a `.pt` file traces back to its sources).

Backward compatibility: a profile with no `sources:` field continues
to read `<root>/ml-training/<profile>/{detection,recognition}` and
`<root>/ml-validation/<profile>/{detection,recognition}` exactly as today.

### Trainer "create" path (optional)

Beyond consuming, the trainer occasionally needs to *publish*:

- **Eval splits** — a held-out subset chosen for evaluating model
  drift across runs.
- **Curated mixes** — a deduped, weighted blend of upstream sources
  saved as a derived dataset for cheap reuse.

A new `pd-ocr-trainer publish-eval --profile gaelic --to <repo>`
subcommand handles this. It uses `huggingface_hub.upload_large_folder`
under the hood and stamps `pd-ocr-source: pd-ocr-trainer-eval` in
`card_data`.

## Trainer packaging of synth and labeler exports

The trainer grows one packaging command. The profile selects the source
exports, the task selects the dataset shape, and `--to` names the Hub repo:

```bash
pd-ocr-trainer publish-dataset --profile gaelic --task recognition --to ntw8532/pd-ocr-synth-ga-clogaelach
pd-ocr-trainer publish-dataset --profile project-foo --task detection --to ntw8532/pd-ocr-real-en-roman
```

The command wraps `huggingface_hub`'s `HfApi.upload_large_folder` and stamps:

- `language` and `typeface` (required)
- `pd-ocr-source` (which producer)
- `pd-ocr-shape` (recognition/v1 or detection/v1)
- `pd-ocr-recipe-sha` (synth) or `pd-ocr-labeler-session` (labeler)
- A README dataset card with license and provenance

Publish is **idempotent against repo state**: re-running with the same
local snapshot SHA detects "no change" and exits without creating a
new commit.

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

The model name follows `pd-<lang>-<typeface>-<task>-<base>`. Coordinate
renames with `pd-ocr-cli`.

## Tooling

| Tool                              | Why we use it                                                                                  |
|-----------------------------------|------------------------------------------------------------------------------------------------|
| `huggingface_hub` (Python)        | Repo creation, file upload, auth, low-level API. Direct dependency in synth, labeler, trainer. |
| `datasets` (Python)               | High-level loader with streaming and split support. Direct dep in trainer.                     |
| `hf` CLI                          | One-off ops (auth login, repo create, debug uploads). Documented; not a runtime dep.           |
| `HfApi.upload_large_folder`       | The upload primitive — chunks into commits, resumes on failure. Right call for many small files. |
| HF Dataset Viewer                 | Hosted, free for public repos. Renders parquet/imagefolder with previews. Surfaces our `card_data`. |
| Croissant metadata                | `huggingface_hub` writes Croissant JSON-LD automatically; consumers outside HF can pick it up. |
| `datasets-server`                 | Hosted API for stats/preview. Trainer's `validate` step can call it before training.           |

What we don't need:

- A custom uploader. `upload_large_folder` covers everything.
- A custom dataset format. imagefolder + parquet handle both shapes.
- A custom registry. HF repos with our naming convention are the
  registry.

## Authentication

- HF token via `HF_TOKEN` env var (preferred) or `hf auth login` (one-
  time interactive on a dev machine).
- The dev container should expose `HF_TOKEN` as a build-time secret /
  runtime env var, mounted from `~/.huggingface/token` on the host.
- Read-only operations (trainer reading a public dataset) need no token.
- Private datasets and writes need a token with `write` scope.

## Caching

`HF_HOME` is set inside the dev container to a path on the existing
`global-shared-ai-cache` Docker volume so:

- Models pulled by `pd-ocr-cli` and datasets pulled by `pd-ocr-trainer`
  share one cache.
- Multiple containers / sessions don't re-download.
- Disk usage is centrally manageable.

`datasets` streaming bypasses local cache; the trainer config picks
between streaming (large remote sets, single training run) and local
download (smaller sets, repeated reads).

## Migration plan (sequenced)

The detailed milestone-by-milestone breakdown lives in
[`roadmap.md`](../plans/roadmap.md). High-level sequence:

1. **Spec landed** (this doc).
2. **`pd-ocr-trainer` refactor** — `DatasetSource` abstraction,
   `LocalSource`, `HFDatasetSource`, profile-config schema with
   backward-compat.
3. **Producer export metadata** — require synth and labeler exports to
   carry the language, typeface, license, and provenance inputs the trainer needs.
4. **`pd-ocr-trainer publish-dataset`** — package real, synthetic, and
   classifier exports after producer formats stabilize.
5. **`pd-ocr-trainer publish-eval`** — once a real eval split is
   curated.
6. **Documentation** — workspace `README.md` references this doc;
   each project's docs link back here for the contract.

Steps 2–4 are independent and can land in any order after step 1.

## Open questions

1. **Auth.** Where does `HF_TOKEN` live? Plan: dev container mount from
   `~/.huggingface/token`; CI via GH Actions secret. Confirm the dev
   container actually mounts it and that the trainer fails closed on
   a missing token.
2. **Private vs public default.** Synthetic data is fine to share
   publicly (Cló Gaelach licensing already verified). Real labeled
   data depends on the source images' provenance — public-domain
   scans are fine, but anything else needs review. Default to
   private; promote to public per dataset.
3. **Mixed-typeface resolution.** The page-level / word-level tiering
   rule above is the accepted repository contract. Italic and smallcaps
   get no detection or recognition repos; they appear only in classifier
   datasets while their crops may remain in a page-level recognizer's
   training distribution.
4. **HF LFS quotas** for `ntw8532` against synth dataset sizes
   (millions of crops, tens of GB).
5. **License/attribution policy.** Per-row `license` (SPDX) in the
   schema. The labeler should refuse export when the source license
   is missing.
6. **Dataset card automation.** Per-source READMEs by hand are fine
   initially; if the set grows, add a generator that produces the
   card from `recipe.snapshot.yaml` + `manifest.jsonl` stats.
7. **Cross-source dedup.** When a profile mixes synth + real,
   identical crops (rare but possible) inflate sample counts. v1
   ignores this; v2 may add a content-hash dedup at trainer load
   time.
8. **Streaming vs local for training.** Streaming saves disk but slows
   the inner loop on small/medium datasets. Default: download for
   datasets under ~10GB, stream above. Trainer config can override.

## Adversarial Review

Stage: migration-time design review. Source: three read-only migration analyzers plus direct comparison with `src/pd_ocr_trainer/dataset_store.py`, training code, tests, and commit history.

The review accepted that this document is target design, not shipped architecture. The migration moved it from `docs/architecture/` to `docs/specs/`, corrected moved links, and retained the existing local `ExportManager` and profile layout as current behavior. No `DatasetSource`, Hub publisher, card-data validator, or Hugging Face dataset loader exists yet.

The main implementation deviation is that the repository kept its local detection/recognition data path after this design was approved. Residual risks are authentication and cache behavior, empirical italic/small-caps recognition coverage, and compatibility of the proposed schema with current upstream producers. Implementers must revalidate those points instead of treating them as shipped facts.
