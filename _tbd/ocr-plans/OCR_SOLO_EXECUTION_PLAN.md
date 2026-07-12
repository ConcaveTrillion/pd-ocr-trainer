# Solo Part-Time Execution Plan (You + AI)

This is a practical low-overhead plan designed for part-time progress.

## Guiding principles

1. Prefer compounding improvements over big rewrites.
2. Optimize for reducing manual labeling time, not perfect architecture.
3. Keep each stage independently trainable and replaceable.
4. Keep reproducibility simple and explicit.

## Suggested 8-week roadmap

### Week 1

- finalize label schema + alias normalization
- standardize export manifest format
- add confidence/provenance fields (`manual`, `rule`, `model`, `fused`)

### Week 2

- implement rule baseline for easy labels
- run auto-label pass over partially labeled data
- produce first disagreement report

### Week 3

- train first word style classifier
- tune per-label thresholds
- evaluate on fixed validation subset

### Week 4

- activate uncertainty-based sampling
- review only top uncertain/disagreement items
- add corrected items to gold set

### Week 5

- train first line/block classifier
- include geometry + Stage 2 features
- baseline report for line/block labels

### Week 6

- add column and reading-order inference
- generate reading-order evaluation examples

### Week 7

- add fusion layer and confidence gates
- auto-accept high-confidence predictions
- queue low-confidence items for manual review

### Week 8

- stabilize v1 pipeline
- document failure modes
- freeze repeatable training/eval commands

## Weekly cadence (part-time)

- 2 sessions: labeling and QA
- 1 session: training and evaluation
- 1 session: pipeline cleanup and reporting

## Minimal stack

- PyTorch + timm
- pandas + parquet manifests
- simple experiment tracker (MLflow/W&B or CSV if needed)
- your labeling tool as source of truth for gold labels

## Gold vs silver strategy

- Gold: manually verified labels
- Silver: auto/rule/model labels

Training recommendation:
- train on both, but weight gold higher
- oversample hard negatives and rare labels

## Core KPIs

- manual minutes per 100 pages
- auto-accept rate above confidence threshold
- per-label F1 (especially rare labels)
- disagreement rate (rule vs model)
- relabel rate after human review

## Active learning queue policy

Priority order:
1. highest uncertainty
2. disagreement between rule and model
3. rare classes
4. repeated failure patterns

## Maintenance checklist

Before each retrain:
1. schema version unchanged or migrated
2. no broken aliases
3. train/val/test split reproducibility verified

After each retrain:
1. compare KPIs against previous run
2. inspect top 20 regressions
3. publish model + threshold + schema versions

## What AI should handle for you

- script generation and refactoring
- evaluation and confusion analysis
- active-learning candidate selection
- consistency checks in schema and aliases
- docs and reproducibility hygiene
