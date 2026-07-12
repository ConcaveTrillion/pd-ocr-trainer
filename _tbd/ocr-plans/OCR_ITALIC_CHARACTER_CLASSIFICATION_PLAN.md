# OCR Split + Italic Character Classification Plan

## Goal
Build a robust pipeline that:
1. Splits word OCR results into character candidates.
2. Predicts italic vs non-italic at character level.
3. Aggregates character predictions back to word-level style scope (`whole` or `part`).
4. Sends ambiguous cases to human review.

## High-Level Architecture

### Stage 1: Deterministic Character Split
- Input: OCR word (`text`, `bounding_box`, image context).
- Use existing geometric split (`split_into_characters_from_whitespace`) as the canonical splitter.
- Output: ordered character objects with crops and coordinates.

### Stage 2: Character-Level Style Inference
- Input per character: crop + neighboring context (left/right characters or small word window).
- Model output: probability `p_italic` for each character.
- Baseline model options:
  - Small CNN (fast and easy to train).
  - Tiny ViT (better robustness if enough data).

### Stage 3: Sequence Smoothing / Decoding
- Apply a sequence decoder over per-character probabilities.
- Practical options:
  - HMM/CRF with transition penalties.
  - Simpler dynamic-programming smoother.
- Goal: avoid noisy alternating labels (`italic`, `non-italic`, `italic`, ...).

### Stage 4: Aggregate to Word-Level Scope
Given decoded character labels:
- If all chars non-italic -> no italic label.
- If all chars italic -> word has `italics` with scope `whole`.
- If strict subset italic -> word has `italics` with scope `part`.

## Human-in-the-Loop Policy
- Route samples for review when:
  - Mean confidence near threshold (for example 0.4 to 0.6).
  - Decoder changes many raw model decisions.
  - Predicted italic span starts/ends at punctuation boundaries (common failure case).
- Reviewer corrects character labels and/or span boundaries.
- Feed reviewed samples back into training (active learning loop).

## Data and Label Schema

### Training record (character-level)
- `page_id`
- `word_id`
- `char_index`
- `char_text`
- `char_bbox`
- `crop_path` or inlined tensor reference
- `label_italic` (0/1)
- `source` (`model`, `human`, `synthetic`)

### Optional span-level annotation
- `word_id`
- `italic_spans`: list of `[start_idx, end_idx)` ranges

## Suggested Training Strategy
1. Start with a balanced binary dataset (italic/non-italic chars).
2. Add synthetic perturbations:
   - Blur, noise, low contrast, scan artifacts, slight geometric distortion.
3. Include punctuation-heavy endings (`—`, `;`, `:`) in validation.
4. Calibrate probabilities (temperature scaling) after training.
5. Re-train periodically with hard samples from review queue.

## Metrics
Track both character and span behavior:
- Character metrics:
  - Precision, recall, F1 for italic class.
  - Expected calibration error (ECE).
- Span/word metrics:
  - Span boundary F1 or IoU.
  - Scope accuracy (`whole` vs `part`).
  - Review rate (% requiring human intervention).

## Inference Contract (Recommended)

### Character output
- `is_italic_pred`: bool
- `italic_confidence`: float in [0, 1]

### Word output
- `text_style_labels`: includes `italics` if any italic chars exist
- `text_style_label_scopes["italics"]`:
  - `whole` if all chars italic
  - `part` if some chars italic
- Optional:
  - `italic_char_mask`: list[bool]
  - `italic_spans`: list of contiguous index ranges

## Rollout Plan
1. Phase 1: Offline experiment
- Train/evaluate model on curated dataset.

2. Phase 2: Shadow mode
- Run model in parallel with current heuristic pipeline.
- Compare outputs and collect disagreements.

3. Phase 3: Assisted production
- Enable model output + human review for uncertain cases.

4. Phase 4: Full production with active learning
- Keep review path for low-confidence edge cases.

## Minimal First Implementation
1. Keep current split logic unchanged.
2. Implement character classifier with per-character confidence output.
3. Add simple smoothing (neighbor agreement or DP).
4. Add aggregation function for word scope (`whole`/`part`).
5. Add review queue trigger by confidence threshold.

## Risks and Mitigations
- Risk: split errors dominate classifier quality.
  - Mitigation: track segmentation quality separately; add split fallback diagnostics.
- Risk: domain shift across scans/fonts.
  - Mitigation: augment aggressively and sample diverse historical print styles.
- Risk: punctuation attachment confusion.
  - Mitigation: include punctuation-heavy examples and explicit post-processing checks.
