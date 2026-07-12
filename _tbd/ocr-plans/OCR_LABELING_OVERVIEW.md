# OCR Labeling and Classification Overview

This document bundle summarizes a practical approach for OCR style + structure labeling with a multi-stage ML pipeline.

## Files in this bundle

- `OCR_LABELING_OVERVIEW.md`: High-level architecture and principles.
- `OCR_LABEL_TAXONOMY.md`: Canonical labels, aliases, and modeling guidance.
- `OCR_PIPELINE_AND_ML_PLAN.md`: End-to-end multi-stage ML plan and tooling recommendations.
- `OCR_SOLO_EXECUTION_PLAN.md`: Part-time, solo-friendly execution roadmap.

## Core decisions

1. Keep labels canonical and normalized at write-time.
2. Use multi-label tagging for style and layout roles.
3. Separate role labels from position/layout labels.
4. Use OCR bboxes as first-class signals for geometry and reading order.
5. Use staged models instead of one monolithic model.

## Data model decisions already implemented in pd-book-tools

- Word-level style labels are represented as normalized multi-select labels.
- Block now includes separate line-level and block-level classification labels.
- Label aliases are normalized (spaces/hyphens/underscores/compact variants).

## Why staged inference

1. OCR extraction and text recognition solve a different problem than style/structure semantics.
2. Word style is local visual classification; page structure is global + contextual.
3. Column inference can be mostly geometric and should feed downstream order/rules.
4. Human review should focus on low-confidence and high-impact disagreements.

## Recommended output schema shape (conceptual)

- Word:
  - text, bbox, confidence
  - text_style_labels
- Line/Block:
  - block_role_labels
  - block_position_labels
  - line_role_labels
  - line_position_labels
  - optional column metadata: column_count, column_index, column_span, reading_order_index

## Practical success metrics

- Manual correction minutes per 100 pages
- Auto-accept rate above confidence thresholds
- Per-label F1 (macro + rare labels)
- Reading-order accuracy on multi-column pages
- Disagreement rate between rules and models
