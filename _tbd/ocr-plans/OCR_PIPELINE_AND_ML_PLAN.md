# Multi-Stage OCR Labeling Pipeline and ML Plan

This plan assumes OCR extraction exists (for example DocTR) and you want stage-2+ semantic labeling.

## Stage 0: Ontology and contracts

1. Freeze canonical labels and aliases.
2. Define confidence policy (`unknown`, `low-confidence`, `not-applicable`).
3. Define data contracts and version them.

## Stage 1: OCR extraction

Inputs:
- page image

Outputs:
- words (bbox, text, confidence)
- lines and blocks (bbox hierarchy)

Requirements:
- normalized coordinate conventions
- stable IDs for word/line/block entities

## Stage 2: Word style classification

Goal:
- Predict `text_style_labels` per word crop (multi-label)

Recommended model:
- ConvNeXt-Tiny or EfficientNet (pretrained)

Features:
- word crop image
- optional context margin around crop
- optional OCR text features (lightweight)

Training:
- loss: BCEWithLogitsLoss
- per-label threshold tuning on validation
- macro + per-label F1 reporting

Rules that can prefill high-precision labels:
- all caps from OCR text pattern
- marker candidates from superscript-like geometry and punctuation

## Stage 3: Line and block classification

Goal:
- Predict line role/position labels and block role/position labels

Inputs:
- line/block image crops
- geometry features (relative x/y/width/height)
- aggregated word style predictions from Stage 2

Model options:
- lightweight classifier per level (line model + block model)
- separate heads for role and position labels

## Stage 4: Column + reading order inference

Goal:
- infer `column_count`, `column_index`, `column_span`, `reading_order_index`

Method:
1. infer columns from line/block x-center clustering
2. assign each line/block to column index
3. detect span elements by width and overlap heuristics
4. compute reading order by column then top-to-bottom

Notes:
- bbox-only inference works well on clean layouts
- keep override path for edge cases (sidenotes, ornaments, mixed layouts)

## Stage 5: Fusion and consistency checks

Combine:
- rules
- model predictions
- geometric constraints

Examples:
- page header should usually be near top
- page footer near bottom
- page number lines often short and isolated

Arbitration:
- if rule confidence is very high, optionally override model
- otherwise preserve model + flag low-confidence for review

## Stage 6: Human-in-the-loop

Use annotation queueing to prioritize:
1. high uncertainty
2. model-rule disagreement
3. rare labels
4. historically hard negatives

Collect corrections and retrain in short cycles.

## Stage 7: Deployment and MLOps

Track versions:
- OCR model version
- style model version
- structure model version
- label schema version

Monitor drift:
- scan quality drift
- label distribution drift
- performance drift on sentinel validation set

## Tool recommendations

Core ML:
- PyTorch
- timm
- scikit-learn (thresholding/calibration)
- Albumentations

Data and workflow:
- parquet manifests (pandas/pyarrow)
- experiment tracking (MLflow or W&B)
- optional orchestration (Prefect/Airflow) if scale grows

Annotation and QA:
- existing custom labeling tool
- canonical-label validation at write-time
- review views with crop + bbox + context + hierarchy

## NLP + ML guidance

Use NLP as secondary, not primary.

- Vision should drive style/layout labels.
- NLP helps disambiguate semantics (headings, references, footnotes, page numbers).
- Best results come from fused multimodal features: visual + geometry + text.
