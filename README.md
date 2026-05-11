# pd-book-tools

Python tools for training DocTR OCR models on PGDP Data

## Installation

Install the 'uv' tooling to manage project dependencies:

[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

I used: `pipx install uv` (you will need `pipx` to do this). Upgrade
with `pipx upgrade uv`.

Then run `uv venv` to create a venv.

Deactivate any current venv (`deactivate`), then activate the venv with
`source .venv/bin/activate`.

Install dependencies.

```bash
uv sync
```

Check pre-commit

```bash
pre-commit
```

To get the model files, you need to use git lfs.

[Git LFS installation guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

## Usage

`voila data-labeler.ipynb` will run the labeling notebook web server

### Dataset Layout By Group/Profile

Training and validation datasets are now organized by profile (group), for example:

- `ml-training/base-ocr/detection`
- `ml-training/base-ocr/recognition`
- `ml-validation/base-ocr/detection`
- `ml-validation/base-ocr/recognition`

Additional export groups from the labeler (for example `italics`,
`small-caps`) are saved in their own profile folders under both split
roots.

Legacy datasets that used the old layout (`ml-training/detection`,
`ml-validation/recognition`, etc.) are automatically migrated to
`base-ocr`.

### `model-trainer.ipynb`

To use the trainer, you have to pull down the doctr git repo because the
scripts are not in the PyPI doctr toolset.

Install it in the PARENT directory of this repo (`../doctr`)

e.g.

```bash
cd ..
gh repo clone mindee/doctr
```

or

```bash
cd ..
git clone https://github.com/mindee/doctr.git doctr
```

You need to modify one file in this repo to add logic to allow use of custom
vocabulary.

In file: doctr/references/recognition/train_pytorch.py

Where you find

```python
    vocab = VOCABS[args.vocab]
```

Change this to

```python
    if args.vocab.startswith("CUSTOM:"):
        # Custom vocab
        custom_vocab = args.vocab.split(":", 1)[1]
        if not custom_vocab:
            raise ValueError("Custom vocab cannot be empty")
        vocab = "".join(sorted(set([char for char in custom_vocab])))
    else:
        vocab = VOCABS[args.vocab]
```

Once you've done this, you can run the model training notebook.

## Roadmap

The Hugging Face datasets integration design — milestones, repo
naming, typeface enum, model-metadata sidecar — lives in
[`docs/ROADMAP.md`](docs/ROADMAP.md). The dataset shape and format
spec it builds on is [`docs/DATASETS.md`](docs/DATASETS.md).

Other near-term work:

- **Mac / Apple Silicon (MPS) support** — test and validate model training and
  inference on Apple Silicon via PyTorch MPS backend; the doctr training scripts
  currently target CUDA, MPS compatibility needs investigation (some ops may fall
  back to CPU)

## License

See LICENSE file.
