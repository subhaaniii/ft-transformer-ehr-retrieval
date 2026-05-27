# FT-Transformer EHR Retrieval: Controlled Experiment Summary

## 1. What method did I test?

This project tests an FT-Transformer-style tabular EHR encoder against a baseline MLP EHR encoder in a two-tower multimodal retrieval setup.

The goal is to study whether a transformer-based tabular encoder improves retrieval alignment compared with a simpler MLP encoder.

Both models use the same retrieval framework:

- CXR branch: image encoder
- EHR branch: MLP or FT-Transformer encoder
- Loss: symmetric InfoNCE contrastive loss
- Task: retrieve the matching EHR sample for a given CXR sample

## 2. What dataset setup did I use?

This repository uses controlled synthetic multimodal data, not real clinical data.

Three synthetic dataset modes were tested:

| Setup | Description |
|---|---|
| Linear | Image patterns are generated from mostly linear combinations of tabular features |
| Interaction | Image patterns depend more strongly on feature interactions |
| Noisy | A fraction of image-EHR pairings are intentionally corrupted |

The purpose of this setup is to test method behavior under controlled conditions without using restricted patient data.

## 3. What metric did I measure?

The experiments report:

- Recall@1
- Recall@5
- Recall@10
- Recall@50
- Lift over random baseline
- Positive-pair cosine similarity
- Training loss

Lift over random is useful because retrieval difficulty changes with candidate pool size.

## 4. What changed across experiments?

The controlled variables were:

| Variable | Values tested |
|---|---|
| Encoder type | MLP, FT-Transformer |
| Data pattern | Linear, interaction, noisy |
| Pairing quality | Clean pairs, 25% noisy/corrupted pairs |
| Sample size | 500, 1000 |
| Training duration | 5 epochs |

## 5. Results

### Summary Table

| Setup | Samples | Encoder | R@1 | R@5 | R@10 | R@50 | Lift@50 | Pos Sim | Train Loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Linear | 500 | MLP | 0.04 | 0.15 | 0.34 | 0.94 | 1.88x | 0.3630 | 3.0041 |
| Linear | 500 | FT-Transformer | 0.04 | 0.17 | 0.33 | 0.97 | 1.94x | 0.4992 | 2.8632 |
| Interaction | 500 | MLP | 0.03 | 0.13 | 0.26 | 0.83 | 1.66x | 0.2264 | 3.1512 |
| Interaction | 500 | FT-Transformer | 0.03 | 0.09 | 0.24 | 0.90 | 1.80x | 0.2902 | 2.9687 |
| Noisy | 500 | MLP | 0.02 | 0.08 | 0.22 | 0.74 | 1.48x | 0.1293 | 3.2743 |
| Noisy | 500 | FT-Transformer | 0.03 | 0.11 | 0.20 | 0.69 | 1.38x | 0.2104 | 3.2186 |
| Interaction | 1000 | MLP | 0.025 | 0.105 | 0.210 | 0.685 | 2.74x | 0.4580 | 2.9626 |
| Interaction | 1000 | FT-Transformer | 0.040 | 0.130 | 0.235 | 0.795 | 3.18x | 0.5265 | 2.7065 |

## 6. What did I learn?

### Linear setup

In the linear setup, the FT-Transformer and MLP performed similarly in retrieval. The FT-Transformer achieved higher positive-pair similarity and slightly better R@5/R@50, but the improvement was not dramatic.

This suggests that when the tabular signal is simple, a well-designed MLP can already capture much of the useful structure.

### Interaction setup with 500 samples

In the interaction setup, the FT-Transformer achieved higher positive-pair similarity and better R@50, but the MLP was slightly better at R@5 and R@10.

This suggests that the FT-Transformer may learn smoother global alignment, but it does not automatically dominate top-rank retrieval in short training.

### Noisy setup

With noisy pairings, both models degraded. FT-Transformer improved R@1/R@5 and positive similarity, but MLP achieved better R@10/R@50.

This shows that better architecture alone cannot fully compensate for weak or corrupted pairing quality.

### Interaction setup with 1000 samples

When the sample size increased from 500 to 1000 in the interaction setup, FT-Transformer showed a clearer advantage:

- R@1 improved from 0.025 to 0.040 compared with MLP
- R@5 improved from 0.105 to 0.130
- R@10 improved from 0.210 to 0.235
- R@50 improved from 0.685 to 0.795
- Lift@50 improved from 2.74x to 3.18x
- Positive-pair similarity improved from 0.4580 to 0.5265

This suggests that FT-Transformer benefits more when there are enough samples and when the data contains feature-interaction structure.

## Main conclusion

FT-Transformer did not universally outperform the MLP in every condition. Its advantage became clearer when the dataset contained interaction-based structure and more samples were available.

The most important lesson is that encoder architecture matters, but it interacts strongly with data structure, sample size, and pairing quality.

## Limitations

- The data is synthetic and does not represent clinical performance.
- The generated images are artificial patterns, not medical images.
- Experiments were limited to 5 epochs.
- The goal is method behavior analysis, not clinical diagnosis or deployment.