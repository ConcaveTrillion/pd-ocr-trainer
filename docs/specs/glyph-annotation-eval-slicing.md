---
Status: draft
Owner: CT
Created: 2026-05-11
Last verified: 2026-07-14
Kind: spec
Supersedes: N/A
Promotes to: N/A
Disposition: Parked until the metric and upstream annotation contracts are confirmed.
---

# Spec: glyph-annotation eval slicing

**Status:** spec only, not implemented.
**Owner repo:** `pd-ocr-trainer`.
**Companion spec:** [`./glyph-feature-classifier.md`](./glyph-feature-classifier.md).
**Cross-repo deps:** `pdomain-book-tools` defines
the `GlyphAnnotations` data model; `pd-ocr-synth`
emits annotated training/eval data with annotations as gold.

## Motivation

DocTR recognition CER/WER is currently a single scalar. It hides
regressions on rare typographic forms (ct/st ligatures, long-s, swash
caps, …) — a model can lose ground on long-s while overall CER
*improves* because long-s is rare. We want sliced metrics so a per-form
regression is visible at eval time.

The trainer is a **read-only consumer** of annotations here. It does
not produce annotations and does not train against them in this spec —
that is [`./glyph-feature-classifier.md`](./glyph-feature-classifier.md).

## Shared data model (recap from pdomain-book-tools)

```text
GlyphAnnotations:
    ligatures: list[LigatureMark]          # {kind: "CT"|"ST"|"FI"|..., char_span: [start, end) | None}
    long_s_positions: list[int]            # char indices into the canonical GT string
    swash: bool
```

The GT string remains canonical / "perfect" (no `ſ`, no ligature
codepoints). `glyph_annotations` is a parallel, optional sidecar.

`glyph_annotations` is **`Optional`**:

| Value                                | Meaning                                         |
|--------------------------------------|-------------------------------------------------|
| `None`                               | not labeled — annotation status unknown         |
| empty (`ligatures=[]`, `long_s_positions=[]`, `swash=False`) | labeled and confirmed feature-free |

This distinction is load-bearing for slicing — see below.

## Eval slicing

For each binary glyph-feature `f` in
`{ligature:CT, ligature:ST, ligature:FI, …, long_s, swash}`:

- **positive set** = words where `glyph_annotations is not None` AND `f` is present.
- **negative set** = words where `glyph_annotations is not None` AND `f` is absent.
- **excluded** = words where `glyph_annotations is None` (unknown status).

Compute CER and WER independently for each set. The denominator
**never includes excluded words** — unlabeled words are not silently
counted as "feature absent." That is the eval-correctness invariant.

For ligatures, slice **per `kind`**, not lumped — a model can be fine
on FI and broken on CT, and a single "ligatures-present" bucket would
hide it.

## Reporting format

Eval emits a per-feature breakdown table alongside the existing
overall CER/WER. Suggested shape (markdown rendered in the trainer UI
and JSON for machine consumption):

| Feature        | N (pos) | N (neg) | N (excluded) | CER pos | CER neg | WER pos | WER neg | Δ CER (pos−neg) |
|----------------|---------|---------|--------------|---------|---------|---------|---------|------------------|
| ligature:CT    | 142     | 18 433  | 5 102        | 0.081   | 0.034   | 0.21    | 0.09    | +0.047           |
| ligature:ST    | 88      | 18 487  | 5 102        | 0.063   | 0.034   | 0.18    | 0.09    | +0.029           |
| long_s         | 412     | 18 163  | 5 102        | 0.142   | 0.033   | 0.34    | 0.09    | +0.109           |
| swash          | 27      | 18 548  | 5 102        | 0.052   | 0.034   | 0.15    | 0.09    | +0.018           |

JSON sidecar: same data, one object per feature, machine-grepped by
CI / the trainer UI for regression alerts. Exact key names TBD; align
with the existing eval JSON shape.

A feature with `N(pos) < 30` is flagged "low support" — report the
numbers but do not gate releases on them.

## Integration points

- **Eval entry points:** `ml-validation/{all,italics}/{detection,recognition}/`
  drive the current eval pass via `src/pd_ocr_trainer/train_recog.py`
  and `train_detect.py`. Slicing only matters for **recognition**;
  detection eval (IoU-based) does not consume word text.
- **Loader:** wherever the eval loader yields `(crop, gt_text)`, it
  must additionally yield `glyph_annotations` (None-able). The
  pdomain-book-tools page-document loader is the right plumbing point — it
  already carries the per-word object that will gain
  `glyph_annotations`.
- **Metric aggregator:** add a per-feature accumulator alongside the
  overall CER/WER one. Same edit-distance numerator, narrower
  denominator.
- **UI:** trainer NiceGUI eval panel renders the breakdown table
  underneath the overall scalar. No new page needed.
- **Dataset prerequisite:** at least one eval dataset
  (`ml-validation/all/recognition/` is the obvious first target) needs
  `glyph_annotations` populated on enough words to give meaningful
  positive support per feature. Synth eval data has it for free;
  real-data eval depends on labeler annotation work.

## Edge cases

- **Word with multiple features.** Counted in every relevant positive
  bucket. Buckets are not mutually exclusive — that is by design.
- **Ligature with `char_span = None`.** Treat as feature-present for
  slicing; do not require the span to be filled.
- **Empty annotations on a synth crop.** Counted as negative for every
  feature — synth data carries gold annotations, so absence is real.
- **Per-row `language` / `typeface` in HF datasets.** Slicing is
  orthogonal to the language/typeface split from
  [`roadmap.md`](../plans/roadmap.md). Glyph-slice within an
  already-filtered (lang, typeface) eval set, not across.

## Non-goals

- Training the recognizer to be aware of annotations (out of scope —
  the recognizer's GT is canonical text).
- Producing annotations from the recognizer (that is the classifier
  spec).
- Detection-side slicing.

## Open questions

- Should "low support" (N < 30) be configurable per profile, or fixed?
- Do we want a single summary "annotation-aware CER" (weighted average
  across feature buckets) or only per-feature numbers? Recommendation:
  per-feature only; a weighted scalar is too easy to misread.

## Adversarial Review

Stage: migration-time design review. Source: a read-only analyzer and direct comparison with recognition evaluation code, dataset batches, tests, upstream assumptions, and history.

The review retained the rule that unknown annotations must never become negative examples. It changed the result by parking the design as a draft because the trainer reports DocTR exact and partial text matches, not the CER/WER accumulator assumed here, and recognition batches carry no annotation sidecar.

No glyph-sliced metric, fixture, loader, UI table, or JSON sidecar has shipped. Residual risks are the choice between adding true CER/WER or slicing current metrics, the upstream annotation schema, and the absence of an annotated evaluation fixture.
