---
Status: active
Owner: CT
Created: 2026-07-14
Last verified: 2026-07-14
Kind: context
---

# Current State

## Agent Index

- **Kind:** context
- **Status:** active
- **Read when:** starting repository work or checking current documentation and test state.
- **Search terms:** current state, priorities, risks, in-flight work, tests.

## What matters now

The shipped system is the local DocTR detection and recognition pipeline described in [`docs/architecture/training-pipeline.md`](../architecture/training-pipeline.md). The Hugging Face dataset roadmap and its five detailed specs remain unimplemented intent, not current architecture.

## In-flight work

No documentation migration remains in flight. New implementation work should start from the active roadmap or from an explicit owner decision recorded in the intent map.

## Test state

`make ci AI=1` passes after correcting the two UI imports from the removed `pd_book_tools` namespace to the installed `pdomain_book_tools` namespace. The baseline failure was introduced when commit `fe94ed9e` renamed the dependency without changing its imports.

## Risks

Several source defects from the retired point-in-time code review remain observable and need fresh issue-level triage. The active designs also depend on unresolved upstream contracts and have not started implementation.
