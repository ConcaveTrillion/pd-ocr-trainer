---
Status: active
Owner: CT
Created: 2026-07-14
Last verified: 2026-07-14
Kind: context
---

# Intent Map

## Agent Index

- **Kind:** context
- **Status:** active
- **Read when:** choosing work, checking deferred intent, or reviewing migration classifications.
- **Search terms:** active bets, deferred work, rejected directions, owner decisions, legacy sweep.

## Active bets

- Implement the approved Hugging Face dataset roadmap and target contract in [`docs/plans/roadmap.md`](../plans/roadmap.md) and [`docs/specs/datasets.md`](../specs/datasets.md).
- Build the glyph-feature classifier after its upstream annotation and dataset contracts are available.
- Add local-mode port collision fallback, stable port persistence, explicit-port failure, and in-UI URL display after defining local mode and bound-port discovery.

## Deferred work

- Rework glyph-aware evaluation after choosing whether to add true CER/WER or slice the current DocTR exact and partial text metrics, and after confirming the upstream annotation schema and fixture.
- Make dependency upgrades dev-local aware after choosing a reproducible CUDA environment restoration contract.
- Re-triage the still-observable defects from the retired code review: hard-coded `.cuda()` calls, shared logger state, recursive dataset scans, pair-based count increments, unbounded UI log growth, and session-scoped mutable browser data.
- In the successor layout-training repository, consider narrow figure-region drawing, image-SHA validation, explicit block-role-to-COCO mappings, exclusion controls for auto-filled labels, a zero-shot baseline gate, and RT-DETR training safeguards.
- Revalidate the former top-50 corpus against current data before using its typography, layout, foreign-script, footnote, and running-header coverage as a sampling strategy.

## Rejected directions

- Do not present approved target designs as shipped architecture.
- Do not keep retired execution checklists and point-in-time review transcripts in live retrieval after their current truth and residual intent are preserved.

## Blocked (waiting on)

- Glyph evaluation waits on a metric contract, upstream `GlyphAnnotations`, and an annotated evaluation fixture.
- Dev-local dependency upgrade safety waits on a CUDA environment restoration contract.

## Needs owner decision

- Decide whether glyph slicing should add CER/WER or slice the existing exact and partial text metrics.
- Decide whether dev-local GPU state is recorded, selected through declarative extras, or deliberately excluded from automated restoration.

## Legacy-unverified sweep

- **Still active:** the Hugging Face roadmap, dataset design, glyph-feature classifier, local-port design, and writing-style process.
- **Parked drafts:** the glyph-evaluation and dev-local upgrade designs remain useful but need the owner decisions listed above.
- **Promoted current truth:** local dataset, UI, and training behavior now lives in [`docs/architecture/training-pipeline.md`](../architecture/training-pipeline.md).
- **Retired and removed:** the fixed top-50 labeling plan, point-in-time code review, and superseded layout-training design. Their durable ideas and provenance are retained here and in [`docs/context/decisions.md`](decisions.md).
