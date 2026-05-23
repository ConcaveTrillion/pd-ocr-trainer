# SPEC — Layout Training (End-to-End)

**Status:** ARCHIVED 2026-05-23 — draft spec, never implemented; superseded
by the pd-ocr-trainer retirement plan (training code moves to pd-ocr-training).
Original status: Draft, not yet scheduled.
**Related:** [TOP-50-LABELING-TARGETS](TOP-50-LABELING-TARGETS.md)
(which pages to label first),
[LABELING_STRATEGY](LABELING_STRATEGY.md) (per-day cadence),
[pd-book-tools ROADMAP](../pd-book-tools/docs/ROADMAP.md) (residual
non-training items).
**Supersedes:** the workspace-root `PLAN-layout-aware-ocr.md`
(shipped — see status table in §1.1) and `TODO-layout-training.md`
(rolled into this spec and the pd-book-tools roadmap).

A unified spec covering everything needed to **train and update the
PP-DocLayout fine-tune** from a corpus the labeler captures: (1) the
labeler changes that produce ground truth, (2) the dataset contract
between labeler and trainer, (3) the trainer changes that fine-tune
PP-DocLayout-plus-L, (4) the phasing.

## 1. Goal

Get from "labeler corrects words; trainer fine-tunes detection +
recognition; layout model is zero-shot only" to "labeler captures
block tags **and** the few region types OCR can't see; trainer
fine-tunes the layout model from those; pipeline picks the fine-tune
up automatically."

## 1.1 Status of prior plans

This spec replaces two earlier workspace-root documents:

- `PLAN-layout-aware-ocr.md` — drove the four-phase rollout that
  shipped `pd_book_tools.layout` and the layout-aware reorg
  pipeline. **All four phases shipped.** The few residual
  open-design items from that plan (per-type confidence thresholds,
  decoration-vs-figure post-classification, multi-column reading
  order, detector-failure hardening) belong on the library's own
  roadmap and now live in
  [`pd-book-tools/docs/ROADMAP.md`](../pd-book-tools/docs/ROADMAP.md).
- `TODO-layout-training.md` — was a placeholder for "future training
  workstream." This spec is that workstream. The non-training
  follow-ups it listed (glyph-size sidenote detection, drop-cap
  glyph recognition for cursive caps, multi-column body detection)
  are also in the pd-book-tools roadmap.

What's already in place (so this spec doesn't redo it):

| Pre-requisite | Where | Used by this spec |
|---|---|---|
| `LayoutRegion` / `PageLayout` / `RegionType` data model with `to_dict` / `from_dict` | `pd_book_tools/layout/types.py` | §4.4 storage + §5.2 export |
| `PP_DOCLAYOUT_TO_PGDP` mapping | `pd_book_tools/layout/_mappings.py` | §6 (we add the inverse) |
| `PPDocLayoutPlusLDetector` with `checkpoint_path` | `pd_book_tools/layout/adapters/pp_doclayout.py` | §5.5 — fine-tuned checkpoints load via this without library changes |
| Detector registry + cache + timing wrapper | `pd_book_tools/layout/registry.py` | §5.6 dataset preview, downstream consumers |
| Layout-aware reorg pipeline (`tag_words_with_layout`, `drop_layout_regions`, `bubble_block_roles_from_layout`, `associate_captions`) | `pd_book_tools/ocr/layout_aware_reorg.py` | Picks up improved regions from a fine-tune automatically; no consumer changes needed |
| Fork mirror of base weights at pinned revision (`CT2534/PP-DocLayout_plus-L`) | `pd-book-tools/Makefile` `layout-fork-*` | §5.7 HF publish reuses the same pattern |
| 30-page layout-regression fixture set | `pd-book-tools/tests/fixtures/layout_regression/` | §7 Phase 4 retraining iteration uses these as the eval set |
| `regenerate_layouts.py` fixture-update script | same dir | Phase 3 acceptance: re-running fixtures against a new checkpoint is a one-liner |
| `Page.reorganize_page(layout=…)` consuming `RegionType` | `pd_book_tools/ocr/page.py` | New region types from a fine-tune flow through automatically |
| `pd-ocr-cli --layout-checkpoint <path>` | `pd-ocr-cli/pd_ocr_cli/ocr_to_txt.py` | Fine-tune is consumable from the CLI immediately after Phase 3 ships |
| `ProjectConfig.layout_checkpoint` field | `pd-prep-for-pgdp/specs/01-book-config.md` | Same, for the planned managed-mode app |
| Page-rotation detection | `pd_book_tools/ocr/rotation.py` | Pre-OCR; means upright `Page` reaches the layout detector at training and inference time |

What the model **systematically misses today** — i.e. what a
fine-tune is expected to fix. This is the success bar for Phase 4
(§7.4); it is the same list that lived in `TODO-layout-training.md`
and underwrote `TOP-50-LABELING-TARGETS.md`'s tier ordering:

- **Page numbers** — usually unlabelled, occasionally lumped into
  `header` or `footer`. Fixture: `contents-page` (model returned 0
  regions for the page-number column, breaking leader-dot pairing).
- **Chapter headers / engraved title pieces** — small ornate
  elements at chapter starts; detected as `figure` if at all, never
  with their own category.
- **Drop caps** — currently handled by the geometric reorg's
  drop-cap stitching pass; a layout-derived `decoration` (or future
  `dropcap`) tag would be cleaner. The geometric stitcher fails on
  cursive caps (S/O/A glyphs); plain caps (R) work.
- **Decorative rules / fleurons / printer's marks** — the `seal`
  class catches some but not all. `decorative-headpiece-list-of-illus`
  is detected as a generic `figure`.
- **Multi-line running headers** with a centered title + page number
  on the same line.
- **Footnote markers vs footnote text region** — currently lumped.
- **Marginalia / sidenotes** — the model lumps them into the body's
  wide `text` region. The geometric `detect_geometric_sidenotes`
  fallback catches narrow margin columns but is a band-aid; a
  fine-tuned `sidebar_text` (or future first-class `sidenote`)
  region is cleaner.

These are the categories the labeler invests in tagging (§4.1) and
the categories the regression fixtures already exercise.

## 2. Approach in one paragraph

Two complementary labeler operations, one trainer tab. The OCR pipeline
already produces a hierarchical block tree (BLOCK → PARAGRAPH → LINE
→ WORDS) with bounding boxes at every level. For everything *with*
OCR text — body text, sidenotes, footnotes, captions, headers, page
numbers, blockquotes, formulas, etc. — the labeler **tags the existing
paragraph block** with one or more role labels from
`Block.ALLOWED_BLOCK_ROLE_LABELS`. For the few region types OCR cannot
see — `figure`, `illustration`, `decoration` — the labeler exposes a
narrow **draw-region** mode that creates a typed rectangle. Both
sources are merged at COCO-export time. The fine-tune is standard
`transformers.Trainer` flow against that COCO.

That's the entire idea. No coordinate-system gymnastics, and no
generic "draw any region" UI — drawing is scoped to the three classes
that have no OCR content.

## 3. What stays where it is (existing assets)

Listed because the spec is mostly plumbing and most of the components
already exist:

| Asset | Location | Reuse as |
|---|---|---|
| `Block.block_role_labels` (validated list) | `pd_book_tools/ocr/block.py` | Storage for per-block tags — already on every Block, currently unused by the labeler |
| `Block.ALLOWED_BLOCK_ROLE_LABELS` frozenset (19 values) | same file | Source-of-truth dropdown vocabulary |
| `Block.BLOCK_ROLE_LABEL_ALIASES` | same file | Accept "block quote", "poem", etc. on input |
| `LayoutRegion`, `PageLayout`, `RegionType` | `pd_book_tools/layout/types.py` | On-disk schema for drawn regions; already round-trips via `to_dict` / `from_dict` |
| `PP_DOCLAYOUT_TO_PGDP` mapping | `pd_book_tools/layout/_mappings.py` | Half of the contract; spec adds the inverse `PGDP_TO_PP_DOCLAYOUT` next to it |
| `PPDocLayoutPlusLDetector` adapter | `pd_book_tools/layout/adapters/pp_doclayout.py` | Loads a fine-tune via the existing `checkpoint_path` argument; **no library change required** |
| Detector registry + cache | `pd_book_tools/layout/registry.py` | Same |
| Labeler `ImageTabs` paragraph selection | `pd_ocr_labeler/views/projects/pages/image_tabs.py` | Selection is already there; we attach an Inspector dock to it |
| Labeler word-rebox / word-add drag plumbing | `image_tabs.py` (`_handle_word_rebox_drag`, `_emit_word_rebox_bbox`) | Generalises directly to figure-region drawing |
| Labeler `<page>.json` round-trip | `pd_ocr_labeler/operations/persistence/` | Already serialises `Block` — including `block_role_labels` |
| Trainer profile concept (`all`, `italics`, …) | `pd_ocr_trainer/dataset_store.py` | Adds `"layout"` task alongside `"detection"` and `"recognition"` |
| Trainer detection card pattern | `pd_ocr_trainer/ui.py` | Replicate as a third Layout card |
| Trainer detection script pattern | `pd_ocr_trainer/train_detect.py` | Pattern only — `train_layout.py` is new (RT-DETR via `transformers.Trainer`) |

The on-disk surface this spec adds: one new optional sibling file
(`<page>.layout-truth.json`, only created when the user draws at
least one figure region); one new module in the trainer; one new
exporter that emits a COCO file under the existing
`ml-training/<profile>/layout/` tree.

## 4. Labeler changes

Two affordances. **Tagging** is the primary operation; **drawing** is
the narrow secondary operation for the three classes that have no OCR
content.

### 4.1 Block-role tagging (primary)

In the existing **Paragraph selection** mode of `ImageTabs`:

1. Each paragraph with one or more role labels gets a small badge
   (e.g. `sidenote · footnote`) overlaid at its top-left. No badge
   means untagged.
2. Clicking a paragraph opens (or refocuses) a right-side **Inspector
   dock**.

Dock states:

- **No selection:** page-level summary — count by role, count of
  untagged paragraphs, "missing tag" warnings (e.g. "this page has a
  paragraph at the bottom that looks like a footnote band but isn't
  tagged" — heuristic; suggestion only).
- **Paragraph selected:** a multi-select dropdown bound to
  `block_role_labels`, populated from
  `Block.ALLOWED_BLOCK_ROLE_LABELS`, with aliases accepted on input
  via `BLOCK_ROLE_LABEL_ALIASES`. Save-on-change. Below the dropdown,
  a "Clear all roles" button.
- **Multiple paragraphs selected:** the dropdown shows the
  intersection of currently-set roles; saving applies the change to
  every selected block (bulk-tag operation). Useful for runs of
  consecutive footnote paragraphs.

The dropdown uses NiceGUI `ui.select(multiple=True)` — same widget
the recognition tab already uses.

### 4.2 Figure-region drawing (secondary, narrow)

Some layout categories have no OCR content the labeler can attach
roles to: a copperplate frontispiece, a printer's-mark fleuron, a
multi-figure plate. For those — and only those — the labeler exposes
a **draw-region** mode.

UI:

- A new toolbar button **"Draw figure / illustration / decoration"**
  next to the existing **"Erase Pixels"** button in `ImageTabs`.
- Clicking it enters draw mode (replaces the active selection mode
  for the duration). Cursor is crosshair; drag draws a rectangle.
- On mouse-up, a small inline picker appears next to the rectangle:
  `( ) figure   ( ) illustration   ( ) decoration   [Save] [Cancel]`.
  Save commits the region; Cancel discards.
- Vocabulary is **fixed at exactly three values** —
  `RegionType.figure`, `RegionType.illustration`, `RegionType.decoration`.
  No other types are drawable. Anything else in the layout
  vocabulary should be reachable via block tagging (§4.1); if a
  category truly needs drawing, that's a spec amendment, not a UI
  configuration.
- After committing, the region appears as an outlined rectangle
  styled by type. Selecting it re-opens the picker so the user can
  retype it; pressing Delete removes it.

A new layer toggle **"Show drawn regions"** appears in the Layers
row alongside Paragraphs / Lines / Words. Off by default once the
user has labelled the page so they don't visually clutter when
focused on word-level corrections.

### 4.3 What is deliberately *not* in the labeler UI

- A generic "draw any region type" affordance. Drawing is restricted
  to `figure / illustration / decoration` (§4.2).
- A "draw a sidenote / footnote / caption" mode. These categories
  have OCR content; tag the existing paragraph block instead.
- Live layout-model inference inside the labeler. The model is
  trained in the trainer; the labeler's role is ground truth.

These are the load-bearing constraints. The reasoning: if a category
has OCR text, the OCR-derived block already has the right bounding
box — re-drawing would produce inconsistent bbox styles between
sidebar tags and tagged-paragraph categories. Restricting draw mode
keeps the UI honest about which signal the model is learning from.

### 4.4 Storage

**OCR-block tags** (most categories):
`Block.to_dict()` already serialises `block_role_labels`. The
labeler's existing `Page.to_dict()` round-trip preserves them.
Saving the page through the existing operations writes them to
`<page>.json`. **No schema change.**

**Drawn figure regions**: persisted in a sibling file
`<page>.layout-truth.json` next to the existing `<page>.json`.
Format is a serialised `PageLayout` (already round-trips via
`to_dict` / `from_dict`):

```json
{
  "schema_version": 1,
  "image": "p0010.png",
  "image_sha1": "af1e7bcfccbd557…",
  "image_width": 950,
  "image_height": 1542,
  "regions": [
    {"type": "figure",     "L": 220, "R": 720, "T": 30,  "B": 480,
     "confidence": 1.0, "raw_label": "drawn"},
    {"type": "decoration", "L": 120, "R": 820, "T": 510, "B": 590,
     "confidence": 1.0, "raw_label": "drawn"}
  ],
  "labeller": "ntw8532",
  "updated_at": "2026-05-04T18:42:00Z"
}
```

`raw_label = "drawn"` distinguishes user-drawn from any
auto-generated entries (e.g. backfill), so the trainer's
stratification flag (§5.5) can target one or the other.

`image_sha1` lets the trainer detect when an image has been replaced
under the same name (e.g. re-cropped scan). The labeler refuses to
load a `<page>.layout-truth.json` whose hash doesn't match the
current `<page>.png` and notifies the user.

The file is created lazily — pages with no drawn regions never get
one, keeping the labelled-projects directory clean.

### 4.5 Backfill for the 53 already-labelled pages

The 53 pages currently in `pd-ocr-trainer/matched-ocr/` have zero
roles set. A new `make backfill-block-roles` target on pd-book-tools
walks each `<page>.json`, runs a heuristic
`guess_block_roles(page) -> Page` (already partially exists in
`pd_book_tools/ocr/layout_aware_reorg.py` — header / footer
detection, sidenote-band detection), populates best-guess roles on
each paragraph, and writes the page back. Output is **a starting
point, not truth** — every backfilled tag carries a marker
(`additional_block_attributes["backfill_origin"] = "auto"`) so the
labeler can:

- visually distinguish backfilled tags (italic badge) from
  human-confirmed ones,
- and the trainer can opt to weight or exclude them via
  `--exclude-auto-backfilled` (§5.5).

The first time a labeller opens a backfilled page and saves the
paragraph (any change, including re-confirming the same tag), the
marker drops; the tag becomes "human truth."

Backfill is **for OCR-block tags only**. Drawn figure regions cannot
be backfilled — there's no signal to derive them from. Drawing
remains a manual, per-page operation. (The OCR pipeline does pass
its own zero-shot layout detector; a stretch in §7 Phase 5 imports
those *predictions* as starting-point regions the user accepts /
rejects rather than draws from scratch.)

## 5. Trainer changes — Layout fine-tuning tab

### 5.1 New `"layout"` dataset task

In `pd_ocr_trainer/dataset_store.py`, extend
`DATASET_TASKS = ("detection", "recognition", "layout")`. The
profile-rooted directory becomes
`ml-training/<profile>/layout/labels.coco.json` and
`ml-training/<profile>/layout/images/`. `dataset_ui.py` (the Kanban
view) already iterates `DATASET_TASKS` — no per-tab plumbing.

### 5.2 The COCO export step (labeler-side)

A new exporter on the **labeler** side (under the existing
**Export** dialog) walks every page in the project and merges two
sources of truth:

1. **OCR-block tags from `<page>.json`**. For each paragraph block
   with a non-empty `block_role_labels`, map
   `block_role_labels[0]` (or, if multiple, the most-specific one
   — see §6) through `PGDP_TO_PP_DOCLAYOUT` to a category. Use the
   paragraph's pixel bounding box as the COCO `bbox`.
2. **Drawn regions from `<page>.layout-truth.json`** (if present).
   One COCO annotation per region, with the region's pixel bounding
   box.
3. **Default `text` annotations**. For each paragraph block
   *without* role labels, emit a `text` annotation. Keeps body-text
   positives available even on pages where no labelling has
   happened. See §6 caveat.

Per-annotation `extra` fields:

- `extra.source` ∈ `{"block_tag", "drawn", "default_text"}`
- `extra.backfill_origin` ∈ `{"auto", "human"}` (only on
  `block_tag` annotations)

Output:

```text
ml-training/<profile>/layout/
  ├── images/                # symlinks or copies of source PNGs
  ├── labels.coco.json       # COCO-format object detection
  ├── manifest.json          # split assignments, image-sha1 map,
  │                          # per-source counts, label histogram
  └── README.md              # exported-by, count, label histogram
```

`labels.coco.json` follows the standard COCO schema. `extra` fields
are non-COCO but harmless; the trainer reads them for stratification
(§5.5).

### 5.3 `LayoutTrainingConfig`

Mirrors `DetectionTrainingConfig` in `pd_ocr_trainer/ui.py`:

```python
class LayoutTrainingConfig:
    def __init__(self) -> None:
        self.profile = BASE_OCR_PROFILE
        self.base_checkpoint = "CT2534/PP-DocLayout_plus-L"
        self.base_revision = "32d3ea36944213ce46f157e9255852620e30eeca"
        self.epochs = 30
        self.batch_size = suggest_batch_size("layout")  # see §5.4
        self.learning_rate = 1e-5
        self.weight_decay = 1e-4
        self.warmup_steps = 200
        self.gradient_accumulation = 1
        self.input_size = 800   # RT-DETR default
        self.eval_every_steps = 200
        self.early_stop_patience = 5
        self.workers = _default_worker_count()
        self.aug_random_resize = True
        self.aug_color_jitter = False  # historical books are near-grayscale
        self.aug_horizontal_flip = False  # page geometry is asymmetric
        self.exclude_auto_backfilled = False
        self.exclude_default_text = False  # see §6 caveat
        self.model_name = _prefixed_model_name(
            "layout", f"model-finetuned-{_default_model_timestamp()}"
        )
```

Adaptive defaults:

- `< 50` annotated pages: warn (lower bound for fine-tuning RT-DETR
  cleanly).
- `> 500`: switch default LR to 5e-5 and bump epochs to 60.

### 5.4 `suggest_batch_size("layout")`

Extends the existing `ui.py` helper:

| GPU VRAM | Batch size |
|---|---|
| 8 GB | 1 |
| 12 GB | 2 |
| 16 GB | 4 |
| 24 GB+ | 8 |

CPU-only falls back to batch=1; the UI emits a warning that "layout
training on CPU is impractical (~30 min / step)" — same convention
detection uses.

### 5.5 `train_layout.py`

Public API mirrors `train_detect.main(args, progress_hook)`:

```python
def main(args, progress_hook: ProgressHook | None = None) -> Path:
    """Returns the checkpoint directory."""
```

Inside `main`:

1. Custom data loader reads `labels.coco.json` and yields
   `{"image": PIL.Image, "annotations": {...}}` — `datasets`
   library does not handle COCO out of the box; easier to write a
   ~30-line loader.
2. Filter annotations by the user's flags:
   - `exclude_auto_backfilled` drops annotations with
     `extra.backfill_origin == "auto"`.
   - `exclude_default_text` drops annotations with
     `extra.source == "default_text"` (see §6 caveat).
3. Load `RTDetrForObjectDetection` + `RTDetrImageProcessor` from
   `args.base_checkpoint` (HF repo) or `args.base_checkpoint_path`
   (local dir).
4. Build `transformers.Trainer` with
   **`remove_unused_columns=False`** (gotcha; appendix A).
5. `compute_metrics` uses `pycocotools` for COCO mAP@[0.5:0.95] and
   mAP@0.5 (no built-in detection metric in `transformers`; sketch
   in appendix A).
6. `trainer.train()` with a `TrainerCallback` that emits the same
   progress dict shape as detection (`step`, `epoch`, `train_loss`,
   `val_loss`, `lr`, plus new `mAP_50_95`, `mAP_50`).
7. Save best-by-mAP checkpoint to
   `model_output_dir(profile, "layout") / model_name`. Standard
   `save_pretrained` shape so
   `RTDetrForObjectDetection.from_pretrained(path)` works in the
   adapter directly.

CLI:

```bash
uv run python -m pd_ocr_trainer.train_layout \
  --coco-json ml-training/all/layout/labels.coco.json \
  --coco-images-dir ml-training/all/layout/images \
  --base-checkpoint CT2534/PP-DocLayout_plus-L \
  --epochs 30 --batch-size 2 \
  [--exclude-auto-backfilled] [--exclude-default-text]
```

The script must run independently of the UI so it's testable from a
sample COCO.

### 5.6 UI tab

A third card next to **Detection Configuration** and **Recognition
Configuration** in `ui.py`:

```text
┌──────────────────────────────┐
│  📐  Layout Configuration    │
│  Layout output root: …       │
│  ▸ Configuration             │
│  [Start Layout Training]     │
│  [Publish to HF…] (optional) │
└──────────────────────────────┘
```

A status pill above Start, refreshed when the profile changes:

```text
✓ 47 layout-tagged pages found in profile 'all'
   • 312 block-tag annotations (6 auto-backfill)
   • 18 drawn figure/illustration regions
   • 421 default-text annotations (toggle below to exclude)
```

A read-only **Layout dataset preview** sub-card renders the first 6
page thumbnails with annotation overlays drawn (using
`pd_book_tools.layout.visualize`), letting the user sanity-check
*before* hitting Start. Drawn regions and block-tag annotations are
rendered with distinct outline styles so the user can see what came
from where.

### 5.7 HF publish (optional follow-on)

`[Publish to HF…]` opens a dialog for repo id / commit message /
optional model card; uses `huggingface_hub.upload_folder`; refuses to
push to repos the user doesn't own (call `whoami` and bail if prefix
mismatches). Same pattern as `make layout-fork-update` in
pd-book-tools' Makefile.

## 6. The block-role → COCO category mapping

`pd-book-tools/pd_book_tools/layout/_mappings.py` already defines
`PP_DOCLAYOUT_TO_PGDP`. This spec adds the inverse:

```python
PGDP_TO_PP_DOCLAYOUT: dict[str, str] = {
    "text":         "text",
    "title":        "doc_title",
    "section":      "paragraph_title",
    "list":         "reference",
    "table":        "table",
    "figure":       "image",
    "illustration": "image",     # same native class as figure
    "decoration":   "seal",      # closest semantic neighbour
    "caption":      "figure_title",
    "header":       "header",
    "footer":       "footer",    # page_number folds into footer
    "footnote":     "footnote",
    "formula":      "formula",
    "abandoned":    "aside_text",
    "sidenote":     "aside_text", # PP-DocLayout has no sidenote class
}
```

Per-`block_role_labels`-value mapping that labeler/trainer share:

| `block_role_labels` | source | → `RegionType` | → PP-DocLayout category |
|---|---|---|---|
| `paragraph` | block-tag | `text` | `text` |
| `sidenote` | block-tag | `sidenote` | `aside_text` |
| `page header` | block-tag | `header` | `header` |
| `page footer` | block-tag | `footer` | `footer` |
| `page number` | block-tag | `footer` | `footer` |
| `printers mark` | block-tag | `decoration` | `seal` |
| `blockquote`, `poetry` | block-tag | `text` | `text` (no first-class layout class — see §10) |
| `illustration` | block-tag *or* drawn | `illustration` | `image` |
| `figure` | block-tag *or* drawn | `figure` | `image` |
| `decoration` | block-tag *or* drawn | `decoration` | `seal` |
| `caption` | block-tag | `caption` | `figure_title` |
| `table` | block-tag | `table` | `table` |
| `footnote` | block-tag | `footnote` | `footnote` |
| `title` | block-tag | `title` | `doc_title` |
| `section` | block-tag | `section` | `paragraph_title` |
| `list` | block-tag | `list` | `reference` |
| `formula` | block-tag | `formula` | `formula` |
| `recovered` | block-tag | (skip) | (skip — meta-tag, not a region role) |

A block with **multiple roles** (e.g. `[sidenote, caption]`) emits one
COCO annotation. The picked category is the **most specific** in the
priority order: `caption > footnote > sidenote > illustration > figure
> table > formula > list > decoration > header > footer > title >
section > text`. The priority list is encoded in
`pd_book_tools/layout/_mappings.py` so labeler and trainer agree.

**Caveat — default-`text` annotations:** every untagged paragraph
contributes a `text` annotation in §5.2 step 3. This keeps body-text
positives available even on pages where no labelling has happened
yet. Risk: the model over-fits "text" because untagged paragraphs in
a half-labeled corpus skew the prior. Mitigation: the trainer's
`exclude_default_text` flag drops these annotations entirely; useful
once the corpus is fully reviewed. Default behaviour is to keep
default-`text` ON for the first fine-tune (more positives), OFF
after the corpus is fully reviewed.

## 7. Phased rollout

Six phases; each independently shippable.

### Phase 0 — Library prep (~1 week, library-only)

| Task | Where | Effort |
|---|---|---|
| Add `RegionType.dropcap`, `RegionType.blockquote`, `RegionType.poetry`, `RegionType.illustration` | `pd_book_tools/layout/types.py` | ½ day |
| Add inverse `PGDP_TO_PP_DOCLAYOUT` and the priority list (§6) | `pd_book_tools/layout/_mappings.py` | ½ day |
| Implement `guess_block_roles(page) -> Page` heuristic that promotes existing geometric reorg signals (header band, footer band, sidenote column, footnote rule) into `block_role_labels` | `pd_book_tools/ocr/layout_aware_reorg.py` | 2 days |
| Add `image_sha1` field to `Page.to_dict()` so the labeler can detect image swaps | `pd_book_tools/ocr/page.py` | ½ day |
| Tests for both helpers using the existing 30 layout-regression fixtures | `pd-book-tools/tests/` | 1 day |

**Exit criteria:** `guess_block_roles` produces sensible roles on the
30 fixtures; new `RegionType` values round-trip; CI green.

### Phase 1 — Labeler block-tag editor (~1 week, ships standalone value)

§4.1 + §4.5 implementation. The smallest thing that delivers user
value:

1. Inspector dock plumbed to the existing paragraph-selection mode.
2. Multi-select dropdown bound to `block_role_labels` with alias
   normalisation.
3. Per-paragraph badge overlay.
4. Page-level summary view in the dock when no selection.
5. `make backfill-block-roles` target that calls `guess_block_roles`
   over the existing 53 fixtures.

Block tags improve the OCR-output reorg pipeline immediately
(independent of any layout training). This is the most-bang-for-buck
ship.

**Exit criteria:** the user can tag a paragraph and the role
persists across reload; the 53 fixtures pick up best-guess roles via
the backfill target; existing browser tests pass.

### Phase 2 — Labeler figure-region drawing + COCO export (~1 week)

§4.2 + §4.4 + §5.2 implementation:

1. **Draw region** toolbar button + crosshair drag mode, generalising
   the existing `_handle_word_rebox_drag` path.
2. Three-option type picker on commit (`figure / illustration /
   decoration` only).
3. `<page>.layout-truth.json` read/write with `image_sha1` validation.
4. **Show drawn regions** layer toggle.
5. **Export → Layout dataset (COCO)…** action: merges block tags +
   drawn regions + default-text per §5.2.

**Exit criteria:**

- Drawing a figure rectangle and reloading the page brings it back.
- An image-replacement scenario (overwrite the PNG on disk) causes
  the labeler to refuse to load the truth file and notify the user.
- `pycocotools.coco.COCO(labels.coco.json)` validates the export.
- `manifest.json` records image-sha1, per-source counts, label
  histogram.
- Rerunning the exporter is idempotent.

### Phase 3 — Trainer layout tab (~2 weeks)

§5 implementation:

1. `DATASET_TASKS` extension + `dataset_store` helpers (½ day).
2. `LayoutTrainingConfig` + settings round-trip (½ day).
3. `train_layout.py` end-to-end CLI from a COCO file — independent of
   UI (~3 days).
4. Layout card + status pill + corpus-health readout (~2 days).
5. Read-only dataset preview (~1 day).
6. Optional HF publish (~1 day).

Phase 3 can run **in parallel with Phase 2** once Phase 1 ships —
they share only the COCO format contract; Phase 3 can develop
against a hand-written sample COCO before Phase 2's exporter exists.

**Exit criteria:**

- `make install` pulls `transformers>=4.45`, `pycocotools`,
  `datasets`; existing tests pass.
- A SPEC-exported `labels.coco.json` of ≥30 pages produces a
  checkpoint at `~/.local/share/pd-ml-models/<profile>/layout/<name>/`.
- Loading that path via
  `pd_book_tools.layout.registry.get_detector("pp-doclayout-plus-l", checkpoint_path=…)`
  produces non-empty regions on a held-out page.
- mAP@[0.5:0.95] and mAP@0.5 logged at the end of training.
- The 30-page layout-regression fixtures can be re-run against the
  new checkpoint from the trainer UI.

### Phase 4 — Iterate against the labeling roadmap (4–8 weeks calendar)

Mostly elapsed-time; not engineering.

1. Tag the **TOP-50** pages from `TOP-50-LABELING-TARGETS.md` (in
   the order called out there). Specifically, the Tier 1 Dilke
   chapter openers and the figure-rich Singer plate pages will
   exercise both block tagging and figure drawing.
2. After **page 25**, retrain. Run the 30-page regression fixtures
   before/after; capture the diff.
3. Continue 26–50; retrain.
4. Compare against the cursive-drop-cap and sidenote regression
   fixtures (the systematic-misses list from §1.1) — the measurable
   wins.

Per appendix A §A.7, expected outcome:

- 50 pages → modest measurable improvement on cursive cases; sidenote
  band detection improves only marginally (PP-DocLayout has no
  native `sidenote` class — sidenotes still classify as `aside_text`
  but more accurately).
- 150 pages → clearer gain. Trigger to evaluate the §A.4 option-(B)
  stretch (extending the class head with `sidenote` and `dropcap`).

### Phase 5 — Stretches (no schedule)

Listed for completeness; gated on Phase 4 results.

- Promote `sidenote` and `dropcap` to first-class layout classes by
  re-initialising the RT-DETR class head with two new IDs (appendix
  A §A.4 option B). Triggered when corpus ≥150 labeled pages **and**
  cursive-drop-cap fixtures still fail.
- Read-only detector overlay in the labeler ("show me what the model
  thinks") so labellers can accept or reject predictions instead of
  drawing from scratch — particularly useful for the figure /
  illustration drawing flow.
- "Suggest figure regions" — one-shot run of the current detector on
  the open page that pre-populates `<page>.layout-truth.json` with
  candidate figure / illustration regions the user accepts /
  rejects. This is the "import zero-shot predictions as starting
  point" workflow; complements Phase 1 backfill which is for tags
  only.
- Glyph-size sidenote detection — independent of training; tracked
  in [`pd-book-tools/docs/ROADMAP.md`](../pd-book-tools/docs/ROADMAP.md).

## 8. Acceptance criteria summary

A reviewer can verify each:

- [ ] **Labeler P1.** Tagging a paragraph as `sidenote` saves the
      role on `<page>.json` and renders a badge on the paragraph;
      the role survives reload.
- [ ] **Labeler P1.** Multi-select bulk-tag works on contiguous
      paragraphs.
- [ ] **Backfill P1.** `make backfill-block-roles` populates the 53
      existing fixtures with best-guess roles, marked as
      auto-backfill.
- [ ] **Labeler P2.** Draw-region mode commits a `figure` rectangle
      to `<page>.layout-truth.json`; reloading the page renders it
      back; deleting it removes both the rectangle and the file (if
      no other regions remain).
- [ ] **Labeler P2.** The type picker only offers `figure`,
      `illustration`, `decoration` — no other types.
- [ ] **Labeler P2.** Replacing the underlying PNG (different
      sha1) causes the labeler to refuse to load the existing
      `layout-truth.json` and notify the user.
- [ ] **Export P2.** A re-export of the 53-page corpus produces a
      `labels.coco.json` that validates with `pycocotools` and a
      `manifest.json` with image-sha1s and per-source counts
      (block_tag / drawn / default_text).
- [ ] **Trainer P3.** Layout card shows a corpus-health pill and
      blocks Start until the corpus has ≥10 tagged pages (UI
      guard).
- [ ] **Trainer P3.** Training a corpus of ≥30 tagged pages produces
      a checkpoint loadable via the registry; running it on a
      held-out page produces non-empty regions.
- [ ] **Trainer P3.** mAP metrics are logged and the
      `--exclude-auto-backfilled` / `--exclude-default-text` flags
      change the training set size observably.
- [ ] **Pipeline.** Setting `ProjectConfig.layout_checkpoint` to the
      trained checkpoint in pd-prep-for-pgdp picks up the new model
      without code changes.
- [ ] **No regression** in existing browser tests
      (`tests/browser/test_*` under `pd-ocr-labeler/`) or in
      detection / recognition training in pd-ocr-trainer.

## 9. Risks

| Risk | Phase | Mitigation |
|---|---|---|
| User accidentally bulk-tags an entire page as one role | 1 | Bulk operations show a confirmation pill ("Apply 'sidenote' to 14 paragraphs?"); per-page diff in the Inspector dock; one-click revert |
| Region drawing UI gets messy when many drawn regions overlap | 2 | Z-ordered click-through; `Tab` cycles overlapping regions; ship simple before fancy |
| Drawn region's bbox conflicts with an OCR paragraph's bbox (e.g. a large figure block whose OCR'd caption overlaps) | 2 | Drawn regions and block-tag bboxes are independent — both stay; the layout model handles overlap natively |
| User accidentally draws a sidenote / footnote as a "figure" because the picker is the easiest path | 2 | Picker copy is deliberately blunt: "*Use this only for figures, illustrations, and decorations. Tag the paragraph in selection mode otherwise.*" |
| `<page>.layout-truth.json` drifts out of sync with `<page>.png` after a re-crop | 2 | `image_sha1` mismatch refuses to load |
| RT-DETR upstream API drift in `transformers>=5.0` | 3 | Pin `transformers<5` in trainer's `pyproject.toml`; revisit at 5.0 release |
| HF hub rate-limit / outage | 3 | Pre-cache via `make download-base-checkpoint`; we already host the fork mirror `CT2534/PP-DocLayout_plus-L` |
| `pycocotools` wheel pain on macOS | 3 | Document `brew install`; consider `faster-coco-eval` (pure-Python) as alternative |
| Auto-backfilled tags are wrong; model overfits to noise | 3, 4 | `--exclude-auto-backfilled` defaults ON for the first run; flip OFF only after the corpus is fully reviewed |
| Default-`text` annotations skew the prior | 3, 4 | `--exclude-default-text` flag; default OFF for first run, ON once corpus is fully reviewed |
| User publishes the wrong checkpoint to HF | 5 | `whoami` check before push refuses on prefix mismatch |
| 53-page fine-tune produces a worse-than-zero-shot checkpoint | 3, 4 | Promote a checkpoint only when val mAP ≥ zero-shot baseline; keep zero-shot as the registry default |

## 10. What's deliberately out of scope

- **Generic region-drawing for non-figure categories** in the labeler.
  Drawing is restricted to `figure / illustration / decoration`;
  every other category is captured via OCR-block tagging.
- **Live model inference inside the labeler.** The labeler captures
  truth; the trainer fine-tunes; the pipeline consumes — three
  separate concerns. Read-only inference overlay is a Phase 5
  stretch.
- **Training PP-DocLayout from scratch.** Fine-tune-only — same
  position the (now-archived) PLAN-layout-aware-ocr took.
- **Custom architectures (DocLayout-YOLO, DocLayNet-DETR, DiT).**
  Same.
- **Cross-book transfer learning.** PP-DocLayout is already
  cross-book by pretraining.
- **Multi-page region modelling.** PGDP corpus does not appear to
  need it.
- **Adding `dropcap` / `sidenote` as first-class label IDs.** Phase 5;
  triggered by corpus-size threshold.

---

## Appendix A — PP-DocLayout fine-tune research notes

Reference notes behind the §5 trainer plumbing. Not a spec.

### A.1 The model

| Fact | Source | Why it matters |
|---|---|---|
| **PP-DocLayout-plus-L** is RT-DETR with 20 document-layout classes | `model.config.id2label` in adapter | Determines fine-tune class — `RTDetrForObjectDetection` |
| Loaded via `transformers.RTDetrForObjectDetection` and `RTDetrImageProcessor` | `pd_book_tools/layout/adapters/pp_doclayout.py` | Trainer can use standard `transformers.Trainer` API |
| Pinned to `CT2534/PP-DocLayout_plus-L` revision `32d3ea3…` (mirror of `PaddlePaddle/PP-DocLayout_plus-L_safetensors`) | adapter `_DEFAULT_REPO` / `_DEFAULT_REVISION` | Stable diff base |
| 20 native labels: `paragraph_title, image, text, abstract, content, figure_title, formula, formula_number, table, reference, doc_title, footnote, header, algorithm, footer, seal, chart, number, aside_text, reference_content` | live `id2label` JSON | Decides COCO category list |
| Default input size 800×800 | RT-DETR convention | VRAM (§A.5) and aspect-ratio handling (§A.3) |

### A.2 Why fine-tuning RT-DETR is plain `Trainer` flow

```python
from transformers import (
    RTDetrForObjectDetection, RTDetrImageProcessor,
    Trainer, TrainingArguments,
)

processor = RTDetrImageProcessor.from_pretrained(BASE_REPO)
model = RTDetrForObjectDetection.from_pretrained(BASE_REPO)

args = TrainingArguments(
    output_dir="…",
    num_train_epochs=30,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    weight_decay=1e-4,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="mAP_50_95",
    greater_is_better=True,
    remove_unused_columns=False,    # critical — see §A.3
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("…")
```

The model's only quirk is the 20-class output head, irrelevant unless
extending (§A.4).

### A.3 Gotchas the code must handle

**`remove_unused_columns=False`** — `TrainingArguments` defaults to
`True`, which drops any dataset column the model's `forward`
signature doesn't mention. RT-DETR's `forward` doesn't declare
`annotations`, but the collator needs them. Silent fail mode: every
batch arrives without labels, loss stays at NaN. Always set `False`.

**Aspect ratio.** PGDP scans are tall (~1.5:1). `RTDetrImageProcessor`
square-pads by default — boxes get rescaled but interior structure
preserved. Recommended. If first pass plateaus, try
`do_resize=False, do_pad=True` (more VRAM; slower).

**Augmentation.** `aug_random_resize=True`,
`aug_color_jitter=False` (historical books are near-grayscale),
`aug_horizontal_flip=False` (page geometry is asymmetric — left vs
right page running headers). The processor already does
normalisation; don't double up.

### A.4 Class-vocabulary strategies

In increasing capability and risk:

**(A) Strict-mapping (default; §6).** Map PGDP `RegionType` to native
PP-DocLayout labels at load time. Class head untouched. Sidenotes /
drop caps collapse into `aside_text` / `seal`. Cheapest; ship first.

**(B) Re-initialise class head with extended vocabulary.**
`from_pretrained(..., num_labels=22, id2label=extended,
ignore_mismatched_sizes=True)` discards the trained class head and
randomises with extra slots (`sidenote`, `dropcap`). All other
weights survive. Empirically needs ≥100 instances per new class;
with our corpus that's ~120 pages once we're labelling specifically
for the new class. Phase 5 stretch.

**(C) Add new heads, freeze the old.** Same as (B) but freeze the 20
original outputs; train only the 2 new ones. Avoids forgetting at
representation cost. Requires custom training-loop code (no longer
pure Trainer). Not recommended until evidence (B) doesn't work.

### A.5 Hardware footprint

Measured on RT-DETR-resnet50 at 800×800 (PP-DocLayout-plus-L same
family, slightly larger backbone):

| GPU | Batch 1 | Batch 2 | Batch 4 | Batch 8 |
|---|---|---|---|---|
| 8 GB | ~6.2 GB | OOM | OOM | OOM |
| 12 GB | 6.5 GB | ~10.2 GB | OOM | OOM |
| 16 GB | 6.5 GB | ~10.2 GB | ~14.8 GB | OOM |
| 24 GB | ~6.5 GB | ~10.5 GB | ~14.8 GB | ~22.6 GB |

Step time on a single 12 GB card (RTX 3060 / 4060): ~0.4 s/step at
batch 2. With 50 pages × 20 epochs × ~25 steps/epoch ≈ **10 min** for
a baseline run; **~30 min** at 500 pages × 60 epochs.

CPU is ~80× slower. Trainer warns but allows.

### A.6 Evaluation metric

`transformers` does not ship a built-in detection metric. Compute
COCO mAP@[0.5:0.95] and mAP@0.5 via `pycocotools` (also used by
labeler to validate exports). Sketch:

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def compute_metrics(eval_pred, *, val_coco_path, val_image_id_map):
    preds, _ = eval_pred
    results = _to_coco_results(preds, val_image_id_map)
    coco_gt = COCO(val_coco_path)
    coco_dt = coco_gt.loadRes(results) if results else None
    if coco_dt is None:
        return {"mAP_50_95": 0.0, "mAP_50": 0.0}  # early training
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return {
        "mAP_50_95": float(evaluator.stats[0]),
        "mAP_50":    float(evaluator.stats[1]),
        "mAP_75":    float(evaluator.stats[2]),
        "mAR_100":   float(evaluator.stats[8]),
    }
```

`coco_dt.loadRes` throws on empty result lists — wrap.

### A.7 Dataset-size sanity check

Public RT-DETR fine-tune signals on small custom datasets:

- ~30 images: barely better than zero-shot.
- ~150 images: 8–15 mAP-point lift.
- ~500 images: plateaus (further gains need new classes, not data).

PGDP corpus targets per `TOP-50-LABELING-TARGETS.md`:

- 50 pages reachable in 3 weeks → modest improvement on cursive drop
  caps, decorative headpieces, multi-line running headers.
- 150 pages reachable in 8–10 weeks → clear measurable improvement;
  threshold for considering option (B) extension.

### A.8 Things to avoid

- Train PP-DocLayout from scratch — needs weeks of GPU time and a
  corpus an order of magnitude larger than ours.
- Backbone-only fine-tuning. Trainer-default uniform LR is correct.
- Mixup / cutmix — confuses small-text-region detectors.
- Cosine-with-restarts LR — RT-DETR overshoots; linear warmup →
  constant or single warmup → cosine is what converges.
- Evaluating on the train split — refuse to run if `val_dataset` is
  empty.

### A.9 Reading list (audit trail)

- *DETRs Beat YOLOs on Real-time Object Detection* (Zhao et al.,
  2024) — RT-DETR paper; §3 covers architecture.
- `transformers` docs for `RTDetrForObjectDetection` /
  `RTDetrImageProcessor`. The "Custom dataset" example on the model
  card is the closest minimal recipe.
- HuggingFace blog: *Fine-tune DETR on a custom dataset* (2023). Old
  but the COCO-loading idioms still apply.
- PaddlePaddle PP-DocLayout-plus-L model card (HF). Confirms the
  Apache-2.0 licence and the 20 native labels.
- pd-book-tools internals already in this repo:
  `pd_book_tools/layout/adapters/pp_doclayout.py`,
  `pd_book_tools/layout/_mappings.py`,
  `pd_book_tools/layout/registry.py`,
  and the regression fixtures under
  `tests/fixtures/layout_regression/`.
