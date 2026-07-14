---
Status: active
Owner: CT
Created: 2026-07-14
Last verified: 2026-07-14
Kind: context
---

# Decisions

## Agent Index

- **Kind:** context
- **Status:** active
- **Read when:** checking durable repository decisions, implementation deviations, or documentation tombstones.
- **Search terms:** decisions, rationale, retirement, tombstone, docgraph, deviations.

### 2026-07-14 — Adopt Docgraph governance

- **Context:** The repository used a folder taxonomy without declared lifecycle metadata.
- **Decision:** Govern repository Markdown with Docgraph, authored context, explicit lifecycle statuses, workflow enforcement, and legacy-unverified detection on `master`.
- **Rationale:** Retrieval must distinguish current truth from unimplemented, stale, superseded, and retired work.
- **Evidence:** `docgraph.toml`, `DOCGRAPH.md`, and the initial 17-node, 40-edge index with 86 checker issues.
- **Remaining work:** Keep strict checks at zero as documentation changes.

### 2026-07-14 — Keep target dataset design out of architecture

- **Context:** `docs/architecture/datasets.md` explicitly said it was not implemented, while inference classified its folder as built architecture.
- **Decision:** Move it to `docs/specs/datasets.md` and publish current local behavior separately in `docs/architecture/training-pipeline.md`.
- **Rationale:** Unimplemented target behavior must not be retrieved as shipped truth.
- **Evidence:** `src/pd_ocr_trainer/dataset_store.py`, training code, tests, commit `759b5751`, and the absence of Hub dataset sources or publishers.
- **Remaining work:** Implement or revise the active roadmap.

### 2026-07-14 — Record the book-tools namespace migration deviation

- **Context:** Commit `fe94ed9e` changed the dependency to `pdomain-book-tools`, but UI imports retained the removed `pd_book_tools` namespace.
- **Decision:** Use the installed `pdomain_book_tools` namespace.
- **Rationale:** A clean repository setup must start the application and pass browser tests without an undeclared sibling installation.
- **Evidence:** `pyproject.toml`, `src/pd_ocr_trainer/ui.py`, the initial browser-test failure, and the passing focused and full CI gates after correction.
- **Remaining work:** none

### 2026-07-14 — Retired: Top 50 Pages to Label/Train Next

- **Old path:** `docs/archive/plans/top-50-labeling-targets.md`
- **Outcome:** superseded
- **Superseded by:** `docs/plans/roadmap.md` and current dataset intent
- **Removal commit:** This Docgraph migration commit.
- **Rationale kept:** Commit `daaf5641`, the sampling themes in `docs/context/intent-map.md`, and git history preserve the useful provenance.
- **Remaining work:** Revalidate any future sample shortlist against the current corpus.

### 2026-07-14 — Retired the point-in-time deep code review

- **Old path:** `docs/archive/research/code-review.md`
- **Outcome:** stale research snapshot removed after re-triage
- **Superseded by:** `docs/architecture/training-pipeline.md`, current code, tests, and `docs/context/intent-map.md`
- **Removal commit:** This Docgraph migration commit.
- **Rationale kept:** Confirmed unresolved findings are listed as deferred work; commit `cad57972` preserves the full review.
- **Remaining work:** Convert confirmed defects into focused issues and verify each independently before fixing it.

### 2026-07-14 — Retired: Layout Training (End-to-End)

- **Old path:** `docs/archive/specs/layout-training.md`
- **Outcome:** superseded without implementation in this repository
- **Superseded by:** training ownership in `pd-ocr-training`; current trainer truth is `docs/architecture/training-pipeline.md`
- **Removal commit:** This Docgraph migration commit.
- **Rationale kept:** Commit `daaf5641`, the implementation deviation in current architecture, and residual layout ideas in `docs/context/intent-map.md`.
- **Remaining work:** Revalidate the retained ideas in the successor repository before adoption.
