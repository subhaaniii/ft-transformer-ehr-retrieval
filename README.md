# FT-Transformer EHR Retrieval

This repository is a method-focused study of using an FT-Transformer-style encoder for tabular EHR representation in multimodal retrieval.

## Research Question

Can a transformer-based tabular encoder improve cross-modal retrieval compared with a baseline MLP encoder?

## Method Tested

This project compares:

- MLP EHR encoder
- FT-Transformer EHR encoder

Both are used inside a two-tower contrastive learning setup with a CXR encoder and an EHR encoder.

## Dataset Setup

This public repository is designed for synthetic/demo data. It does not include restricted clinical records, patient-level data, or medical images.

The code can be adapted to authorized datasets by passing custom file paths through CLI arguments.

## Metrics

The training script reports:

- Recall@1
- Recall@5
- Recall@10
- Recall@50
- Lift over random baseline
- Positive-pair cosine similarity
- Training loss

## Experiment Matrix

| Variable changed | Values tested | Purpose |
|---|---|---|
| EHR encoder | MLP, FT-Transformer | Compare tabular representation capacity |
| Pairing setup | demo paired samples | Validate retrieval pipeline |
| Learning rate schedule | flat LR, cosine LR | Test convergence stability |
| Retrieval K | 1, 5, 10, 50 | Measure ranking quality |

## How to Run

```bash
pip install -r requirements.txt
python src/train.py --ehr-encoder ftt
python src/train.py --ehr-encoder mlp

## Documentation

- [Method overview](docs/method_overview.md)
- [Experiment design](docs/experiment_design.md)
- [Metric explanation](docs/metric_explanation.md)
- [Reproducibility notes](docs/reproducibility_notes.md)
- [Controlled experiment results](experiments/results_summary.md)