# Spec: glyph-feature classifier

**Status:** spec only, not implemented.
**Owner repo:** `pd-ocr-trainer`.
**Companion spec:** [`./glyph-annotation-eval-slicing.md`](./glyph-annotation-eval-slicing.md).
**Cross-repo deps:**

- [`pd-book-tools`](../../../pd-book-tools/) — defines
  `GlyphAnnotations` (the model's output schema target).
- [`pd-ocr-synth`](../../../pd-ocr-synth/) — primary training-data
  source; annotations are gold by construction.
- `pd-ocr-labeler` — secondary training-data source (human-labeled)
  AND the inference consumer (model pre-fills annotation suggestions
  that humans accept/reject).

## Motivation

GT text is canonical and throws away typographic features (`ſ` →
`s`, `ﬅ` → `st`, etc.). We still want those features tracked
per-word as `GlyphAnnotations`. Asking a human to annotate every word
is too expensive. A small model can do a first pass; a human ratifies.

This is a **separate model** from the DocTR recognizer. The recognizer
emits canonical text; the classifier emits glyph features for the same
word crop. They run in parallel at inference time.

## Inputs / outputs

- **Input:** a single word-crop image (the same crop the recognizer
  consumes — already cropped by detection).
- **Output:** a fixed-length vector of independent sigmoid
  probabilities, one per binary feature:
  - `lig_ct`, `lig_st`, `lig_fi`, `lig_fl`, `lig_ffi`, `lig_ffl`,
    `lig_st_round`, … (closed enum, mirrors `LigatureMark.kind`)
  - `long_s`
  - `swash`
- **Mapped to `GlyphAnnotations`** at the consumer side:
  - per-ligature-kind probability above threshold → append a
    `LigatureMark(kind=…, char_span=None)`. The classifier does not
    localize `char_span`; that stays None and may be filled later by a
    different stage.
  - `long_s` above threshold → emit `long_s_positions=[]` placeholder
    plus a flag that long-s is *present somewhere* in the word. **The
    classifier does not localize positions.** A second pass (TBD; see
    Open questions) does position localization. For now, an empty
    `long_s_positions` list combined with the binary present-flag is
    the contract.
  - `swash` above threshold → `swash=True`.

This means `GlyphAnnotations` as emitted by the classifier alone is
incomplete-but-honest: present/absent is reliable, span/positions are
not populated. The labeler accepts/refines it.

## Architecture (proposed; mark TBD via experiment)

- Backbone: small CNN — MobileNetV3-Small or even a 4-block plain
  conv stack. Word crops are small (typically < 256 px wide); a heavy
  backbone is wasted.
- Input: resized to fixed height (e.g. 32 px) preserving aspect, padded
  to a max width (e.g. 256 px). Same preprocessing the recognizer uses
  is the natural starting point — share the pipeline.
- Head: single linear layer → N independent logits → sigmoid.
- Loss: per-feature `BCEWithLogitsLoss`, summed. Class imbalance is
  severe (ct ligatures are rare), so apply per-feature `pos_weight`
  computed from training-set frequency.
- Training: minutes-scale on a single GPU against synth data;
  fine-tune-scale against the smaller human-labeled set.

**Architecture is TBD via experiment.** The above is a starting point;
ship criterion is "matches or beats a logistic-regression-on-pixels
baseline by enough margin to be worth the complexity." If the simple
CNN underperforms, revisit (e.g. tiny ViT, CRNN feature pooling).

## Training data

| Source                                    | Role          | Weight (initial) | Notes                                              |
|-------------------------------------------|---------------|------------------|----------------------------------------------------|
| `pd-ocr-synth` output                     | primary       | 0.8              | annotations gold by construction                   |
| Human-labeled crops via `pd-ocr-labeler`  | secondary     | 0.2              | smaller, real-distribution, anchors against synth-domain drift |

Initial weighting is a starting point; tune empirically. Human-labeled
data should be **upsampled** despite its smaller absolute count —
otherwise the model learns synth fonts' idea of a ct ligature, not the
real-book distribution.

Training datasets follow the HF dataset convention from
[`../ROADMAP.md`](../ROADMAP.md), with a new `pd-ocr-shape`:
`glyph-classification/v1`. Per-row schema:

- `image` (the word crop, PNG bytes)
- one boolean column per feature (`lig_ct: bool`, `lig_st: bool`, …,
  `long_s: bool`, `swash: bool`)
- standard metadata columns (`language`, `typeface`, `license`)

Repo naming: `<owner>/pd-ocr-<source>-<lang>-glyph` — the literal
string `glyph` in the typeface slot, paralleling the `typeface` slot
literal for the typeface classifier.

## Calibration: auto-fill vs suggest-only

Per-feature thresholds, calibrated on the held-out human-labeled set:

- `T_auto` — above this, the labeler **auto-fills** the annotation;
  human still sees and can override.
- `T_suggest` — above this but below `T_auto`, the labeler shows a
  suggestion (e.g. greyed-out chip the human clicks to accept).
- below `T_suggest` — no suggestion shown.

`T_auto` is chosen per feature for **precision ≥ 0.99** on the held-out
set (auto-fill must rarely be wrong; humans trust it). `T_suggest` is
chosen for **recall ≥ 0.9** (catch most positives, false positives are
cheap because the human ignores them).

Thresholds are stored in the model sidecar (see Export below) so the
labeler doesn't hard-code them.

## Eval

Held-out human-labeled set (never used in training). Per-feature:

- precision, recall, F1 at `T_auto` and `T_suggest`
- precision-recall curve (saved as artifact)
- support (positive count) — flag features with N < 50 as low-support

Synth-only eval is **also reported but not the gating metric** — synth
distribution drift means synth metrics overstate real-world accuracy.

## Export format

Match the existing DocTR model export shape from
[`../ROADMAP.md#model-metadata-sidecar`](../ROADMAP.md). One model
artifact + one JSON sidecar:

```json
{
  "name": "pd-en-glyph-classifier-2026-05-05",
  "task": "glyph-classification",
  "language": "en",
  "typeface": "glyph",
  "features": ["lig_ct", "lig_st", "lig_fi", "long_s", "swash"],
  "thresholds": {
    "lig_ct":  {"t_auto": 0.92, "t_suggest": 0.55},
    "lig_st":  {"t_auto": 0.90, "t_suggest": 0.50},
    "long_s":  {"t_auto": 0.95, "t_suggest": 0.60},
    "swash":   {"t_auto": 0.88, "t_suggest": 0.45}
  },
  "trained_on": [
    {"repo": "ntw8532/pd-ocr-synth-en-glyph", "revision": "...", "rows": 80000, "weight": 0.8},
    {"repo": "ntw8532/pd-ocr-real-en-glyph",  "revision": "...", "rows": 4000,  "weight": 0.2}
  ],
  "trainer_version": "0.x.y",
  "trained_at": "2026-05-05T18:00:00Z"
}
```

Same sidecar shape as recognition/detection — labeler/CLI loader code
generalizes. The `task: "glyph-classification"` discriminator selects
the loader. **No new export format.** If experimentation finds the
DocTR pickle/onnx pipeline doesn't fit (e.g. we want a `.onnx` for
labeler-side JS inference), document the divergence here at that
point — until then, match.

## Closed-loop integration

- pd-ocr-synth emits `glyph-classification/v1` datasets alongside its
  recognition/detection output. Annotations are gold.
- pd-ocr-trainer trains the classifier on synth + human data, exports
  artifact + sidecar.
- pd-ocr-labeler loads the artifact, runs inference per word crop,
  pre-fills `GlyphAnnotations` using `T_auto` / `T_suggest`.
- Human accepts/rejects. Accepted/edited annotations flow back as
  human-labeled training data → next training pass.

## Non-goals

- Localizing `LigatureMark.char_span` or `long_s_positions`. Out of
  scope; those are TBD second-pass work.
- Replacing the human in the loop. Auto-fill ≠ skip review.
- Predicting non-binary features (e.g. typeface enum) — that's the
  separate typeface classifier in [`../ROADMAP.md`](../ROADMAP.md).

## Open questions

- Position localization for `long_s_positions` and ligature
  `char_span` — separate model, attention map over CTC alignment, or
  punt to humans? Recommendation: punt for v1, revisit when the binary
  classifier is shipping.
- Do we need a per-typeface classifier (one model per typeface enum
  value), or one model spanning typefaces? Recommendation: one model,
  with `typeface` as an input feature or implicit from training-data
  mix. Validate.
- `T_auto` precision target — is 0.99 right, or stricter (0.999)?
  Depends on labeler UX cost of a wrong auto-fill vs the cost of
  showing nothing.
