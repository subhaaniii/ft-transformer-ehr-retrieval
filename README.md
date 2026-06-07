# FT-Transformer EHR Retrieval

A controlled method-study comparing an FT-Transformer-style tabular encoder with a baseline MLP encoder for multimodal retrieval.

The project is inspired by multimodal medical representation learning, but the public version uses controlled synthetic data instead of restricted clinical data. The goal is to study method behavior under known data-generating conditions, not to make clinical claims.

## Research Report

A paper-style summary of this project is available here:

[Read the paper-style report](docs/paper_style_report.md)

## Research Question

> Can an FT-Transformer-style tabular encoder improve retrieval alignment compared with a simpler MLP encoder when the data contains feature interactions, noisy pairings, or larger sample sizes?

## What This Repository Shows

This repository demonstrates:

- how to build a two-tower contrastive retrieval pipeline
- how to compare MLP and FT-Transformer tabular encoders
- how Recall@K and lift-over-random can be used for retrieval evaluation
- how model behavior changes under linear, interaction-based, and noisy-pair synthetic setups
- why higher embedding similarity does not always guarantee better top-k retrieval

## Methods Compared

This benchmark compares a baseline MLP EHR encoder with an FT-Transformer-style EHR encoder inside the same contrastive retrieval pipeline.

For the full method description, including encoder design and comparison setup, see the [paper-style report](docs/paper_style_report.md#4-method).

## Related Work

This project is inspired by FT-Transformer-style tabular encoding, TabTransformer, and contrastive retrieval for multimodal representation learning.

For the full related-work discussion and references, see the [paper-style report](docs/paper_style_report.md#13-related-work).

## Dataset Setup

The public repository uses controlled synthetic multimodal data.

Three dataset modes are included:

| Setup | Purpose |
|---|---|
| Linear | Tests whether a simple MLP is already sufficient when the signal is mostly linear |
| Interaction | Tests whether FT-Transformer helps when feature interactions matter |
| Noisy | Tests robustness when a fraction of image-EHR pairings are intentionally corrupted |

The generated images are controlled synthetic visual patterns rather than clinical medical images. This makes the benchmark reproducible, privacy-safe, and suitable for method-behavior analysis.
For full setup details, see the [paper-style report](docs/paper_style_report.md#5-dataset-and-experimental-setup).

## Experiments

The final benchmark compares:

- encoder type: MLP vs FT-Transformer
- dataset mode: linear, interaction, noisy
- sample size: 500 pilot, 1000, 2000
- training duration: 10 epochs for the main benchmark

Metrics reported:

- Recall@1
- Recall@5
- Recall@10
- Recall@50
- lift over random baseline
- positive-pair cosine similarity
- training loss

---

## Demo-Space Visualization

The plots below show the synthetic input spaces used in this repository. These are **pre-training diagnostics**, not learned FT-Transformer embeddings.

<table>
  <tr>
    <th>EHR profile space</th>
    <th>EHR train/eval split</th>
    <th>Synthetic CXR image space</th>
  </tr>
  <tr>
    <td width="33%">
      <a href="figures/ehr_profile_space.png">
        <img src="figures/ehr_profile_space.png" alt="EHR profile space" width="100%">
      </a>
    </td>
    <td width="33%">
      <a href="figures/ehr_pool_split_space.png">
        <img src="figures/ehr_pool_split_space.png" alt="EHR train/eval split" width="100%">
      </a>
    </td>
    <td width="33%">
      <a href="figures/cxr_image_space.png">
        <img src="figures/cxr_image_space.png" alt="Synthetic CXR image space" width="100%">
      </a>
    </td>
  </tr>
</table>

The first two panels use the same EHR PCA coordinates but highlight different information: the first shows clinical feature structure, while the second shows whether Pool A and Pool B are spread across the same EHR space.

| Panel | What to notice |
|---|---|
| **EHR profile space** | Nearby points represent structurally similar synthetic patient profiles. Color indicates hypertension status, and marker shape indicates sex. |
| **EHR train/eval split** | Pool A and Pool B should be mixed across the same broad EHR space, rather than forming isolated train/eval regions. This helps check that the split is not visually biased. |
| **Synthetic CXR image space** | The image-side inputs show visible structure before training, giving the retrieval model meaningful variation to learn from. |

These plots are qualitative diagnostics. The main conclusions should be based on retrieval results and controlled experiments rather than visual inspection alone.

---

## Key Findings

FT-Transformer did not universally outperform the MLP. Its behavior depended on the data setup.

Main observations:

- In linear settings, the MLP was already highly competitive.
- In the 1000-sample interaction setup, FT-Transformer improved retrieval over MLP.
- In noisy-pair settings, both models degraded, showing that better architecture cannot fully compensate for weak pair quality.
- FT-Transformer often improved positive-pair similarity, but higher similarity did not always translate into better top-k retrieval.

For the full findings and interpretation, see the [paper-style report](docs/paper_style_report.md#9-key-findings).

## Repository Structure

```text
ft-transformer-ehr-retrieval/
│
├── src/
│   ├── train.py
│   ├── make_demo_data.py
│   ├── model.py
│   ├── cxr_transforms.py
│   └── ehr_transformer.py
│
├── data_demo/
│   ├── demo_cxr_samples.csv
│   ├── demo_ehr_profiles.csv
│   ├── train_feature_schema.json
│   └── train_transform_stats.json
│
├── docs/
│   ├── method_overview.md
│   ├── experiment_design.md
│   ├── metric_explanation.md
│   └── reproducibility_notes.md
│
├── experiments/
│   └── results_summary.md
│
├── requirements.txt
└── README.md
```

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Generate synthetic data:

```powershell
python src/make_demo_data.py --mode interaction --n-samples 1000
```

Train the MLP baseline:

```powershell
python src/train.py --ehr-encoder mlp --epochs 10 --batch-size 32 --num-workers 0 --output-dir outputs/example_mlp --checkpoint-dir checkpoints/example_mlp
```

Train the FT-Transformer model:

```powershell
python src/train.py --ehr-encoder ftt --epochs 10 --batch-size 32 --num-workers 0 --output-dir outputs/example_ftt --checkpoint-dir checkpoints/example_ftt
```

## Documentation

- [Method overview](docs/method_overview.md)
- [Experiment design](docs/experiment_design.md)
- [Metric explanation](docs/metric_explanation.md)
- [Reproducibility notes](docs/reproducibility_notes.md)
- [Controlled experiment results](experiments/results_summary.md)

---

## References

References are included in the [paper-style report](docs/paper_style_report.md).

## Limitations

This repository is a controlled method-analysis project rather than a clinical deployment study. The benchmark uses synthetic multimodal data so that data structure, pairing quality, feature interactions, and sample-size effects can be studied under known conditions.

For the full limitations and future work discussion, see the [paper-style report](docs/paper_style_report.md#10-limitations).

A stronger follow-up would include repeated random seeds, longer training, additional sample-size sweeps, and evaluation on authorized real-world datasets under proper data-use restrictions.
