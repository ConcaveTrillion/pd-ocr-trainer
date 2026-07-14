---
Status: built
Owner: CT
Created: 2026-07-14
Last verified: 2026-07-14
Kind: architecture
---

# Local training pipeline

## Agent Index

- **Kind:** architecture
- **Status:** built
- **Read when:** changing dataset ingestion, the training UI, detection or recognition training, model outputs, or local runtime configuration.
- **Search terms:** local training, ExportManager, dataset profiles, detection, recognition, NiceGUI, model sidecars.

## Current behavior

The repository trains DocTR detection and recognition models from local, profile-scoped directories. `ExportManager` scans labeler DocTR exports, lets operators assign whole projects or individual pages to training or validation, and copies detection and recognition data into `ml-training/<profile>/` or `ml-validation/<profile>/`. Startup migrates the former ungrouped layout and the legacy `base-ocr` profile into the canonical `all` profile.

The NiceGUI application owns operator configuration and invokes the detection and recognition training functions. Training is operator-driven. The application does not start long training runs during tests.

Detection training writes an architecture sidecar. Recognition training writes vocabulary and architecture sidecars. These are the sidecars currently implemented; the richer model metadata contract in [`docs/specs/datasets.md`](../specs/datasets.md) remains target design.

## Runtime and local development

The UI reads its host, port, browser, and reload settings from `PD_OCR_TRAINER_HOST`, `PD_OCR_TRAINER_PORT`, `PD_OCR_TRAINER_SHOW_BROWSER`, and `PD_OCR_TRAINER_RELOAD`. It passes one explicit port to NiceGUI. Port fallback, persistence, and in-UI URL display remain unimplemented in [`docs/specs/local-mode-port-autoselect.md`](../specs/local-mode-port-autoselect.md).

`make dev-local` installs the sibling book-tools checkout as an editable dependency after a normal `uv sync`. The standard dependency-upgrade target can still replace that editable install; the safer upgrade workflow remains a parked design in [`docs/specs/dev-local-upgrade-flow.md`](../specs/dev-local-upgrade-flow.md).

## Implementation deviations

The approved Hugging Face roadmap did not replace the local data path. No Hub dataset source, publisher, card-data validator, typeface trainer, glyph trainer, glyph-aware evaluation path, or layout trainer exists in this repository.

The archived layout-training design also changed ownership: commit `daaf5641` records that layout training moved to `pd-ocr-training`. Its useful dataset validation and evaluation ideas remain in [`docs/context/intent-map.md`](../context/intent-map.md), not as current trainer architecture.

## Evidence

- Code: `src/pd_ocr_trainer/dataset_store.py`, `src/pd_ocr_trainer/dataset_ui.py`, `src/pd_ocr_trainer/ui.py`, `src/pd_ocr_trainer/train_detect.py`, and `src/pd_ocr_trainer/train_recog.py`.
- Tests: the three browser drag tests under `tests/browser/` verify page and project assignment behavior; other training and dataset-store behavior currently lacks focused tests.
- History: commit `759b5751` established the current local export/profile direction; commit `daaf5641` archived the layout-training direction.
- Verified: 2026-07-14 by repository inspection and `make ci AI=1`.

## Residual risks

The retired code review still matched several source-level defects during this migration. The confirmed items are tracked as deferred work in [`docs/context/intent-map.md`](../context/intent-map.md). Browser tests cover only a narrow part of the dataset-management surface.
