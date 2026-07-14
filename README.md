---
Status: active
Owner: CT
Created: 2026-05-11
Last verified: 2026-07-14
Kind: usage
---

# pd-ocr-trainer

`pd-ocr-trainer` trains DocTR detection and recognition models from PGDP data through a NiceGUI application. It also manages local dataset profiles, validation data, model exports, and model metadata.

## Setup

Install the repository-declared dependencies and hooks:

```bash
make setup
```

Run the application:

```bash
make run
```

Run the full verification gate before committing:

```bash
make ci
```

Git LFS is required for `.pt` and `.bin` model artifacts. See the [Git LFS installation guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

## Dataset layout

Training and validation data is grouped by profile:

```text
ml-training/<profile>/detection
ml-training/<profile>/recognition
ml-validation/<profile>/detection
ml-validation/<profile>/recognition
```

The application migrates the former ungrouped detection and recognition directories, and the legacy `base-ocr` profile, into the canonical `all` profile. Additional labeler exports can use separate profile directories.

## Current and planned documentation

- [`docs/README.md`](docs/README.md) explains the documentation layout and links current entry points.
- [`docs/plans/roadmap.md`](docs/plans/roadmap.md) is the approved, unimplemented Hugging Face dataset roadmap.
- [`docs/specs/datasets.md`](docs/specs/datasets.md) defines the target dataset contract; it is not shipped architecture.
- [`docs/context/current-state.md`](docs/context/current-state.md) records the current operational state.
- [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md), [`CONVENTIONS.md`](CONVENTIONS.md), and [`DOCGRAPH.md`](DOCGRAPH.md) contain contributor and agent guidance.

The old notebook and a patched sibling DocTR checkout are not the current entry point. Use `make run` and the package-managed dependencies.

## License

The package declares the Unlicense in `pyproject.toml`.
