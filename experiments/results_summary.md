# FT-Transformer EHR Retrieval: Controlled Experiment Summary

## 1. What method did I test?

This project tests an FT-Transformer-style tabular EHR encoder against a baseline MLP EHR encoder in a two-tower multimodal retrieval setup.

The goal is to study whether a transformer-based tabular encoder improves retrieval alignment compared with a simpler MLP encoder.

Both models use the same retrieval framework:

- CXR branch: synthetic image encoder input
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

The purpose is to evaluate method behavior under controlled conditions without using restricted patient data.

## 3. What metric did I measure?

The experiments report:

- Recall@1
- Recall@5
- Recall@10
- Recall@50
- Lift over random baseline
- Positive-pair cosine similarity
- Training loss

Lift over random is reported because retrieval difficulty changes with candidate pool size.

## 4. What changed across experiments?

| Variable | Values tested |
|---|---|
| Encoder type | MLP, FT-Transformer |
| Data pattern | Linear, interaction, noisy |
| Pairing quality | Clean pairs, 25% noisy/corrupted pairs |
| Sample size | 1000, 2000 |
| Training duration | 10 epochs |

## 5. Results

### Final 10-Epoch Results

| Setup | Samples | Encoder | R@1 | R@5 | R@10 | R@50 | Lift@50 | Pos Sim | Train Loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Linear | 1000 | MLP | 0.025 | 0.125 | 0.270 | 0.925 | 3.70x | 0.7758 | 2.6304 |
| Linear | 1000 | FT-Transformer | 0.030 | 0.135 | 0.305 | 0.900 | 3.60x | 0.8360 | 2.5074 |
| Linear | 2000 | MLP | 0.025 | 0.100 | 0.195 | 0.7475 | 5.98x | 0.8304 | 2.4916 |
| Linear | 2000 | FT-Transformer | 0.015 | 0.1025 | 0.1825 | 0.6800 | 5.44x | 0.8958 | 2.3236 |
| Interaction | 1000 | MLP | 0.025 | 0.105 | 0.210 | 0.685 | 2.74x | 0.4580 | 2.9626 |
| Interaction | 1000 | FT-Transformer | 0.040 | 0.130 | 0.235 | 0.795 | 3.18x | 0.5265 | 2.7065 |
| Interaction | 2000 | MLP | 0.025 | 0.155 | 0.250 | 0.8150 | 6.52x | 0.7708 | 2.3643 |
| Interaction | 2000 | FT-Transformer | 0.010 | 0.0925 | 0.2175 | 0.6725 | 5.38x | 0.7972 | 2.1039 |
| Noisy | 1000 | MLP | 0.030 | 0.090 | 0.200 | 0.685 | 2.74x | 0.4477 | 3.0976 |
| Noisy | 1000 | FT-Transformer | 0.020 | 0.110 | 0.190 | 0.660 | 2.64x | 0.2625 | 3.0341 |
| Noisy | 2000 | MLP | 0.010 | 0.055 | 0.1075 | 0.5125 | 4.10x | 0.4121 | 3.0226 |
| Noisy | 2000 | FT-Transformer | 0.0175 | 0.0925 | 0.1575 | 0.5775 | 4.62x | 0.3136 | 2.9554 |

## 6. What did I learn?

### Linear setup

In the linear setup, the MLP was already highly competitive. FT-Transformer achieved higher positive-pair similarity and lower training loss, but it did not consistently improve retrieval ranking.

This suggests that for simple linear tabular-image relationships, a well-tuned MLP can be sufficient.

### Interaction setup

At 1000 samples, FT-Transformer improved all main retrieval metrics compared with MLP:

- R@1 improved from 0.025 to 0.040
- R@5 improved from 0.105 to 0.130
- R@10 improved from 0.210 to 0.235
- R@50 improved from 0.685 to 0.795
- Lift@50 improved from 2.74x to 3.18x

However, at 2000 samples, MLP achieved stronger retrieval ranking even though FT-Transformer still had higher positive similarity and lower training loss.

This shows that higher embedding similarity does not always translate into better top-k retrieval.

### Noisy setup

With corrupted pairings, both models degraded. FT-Transformer improved several top-k metrics at 2000 samples, but MLP remained competitive at 1000 samples.

This supports an important conclusion: better encoder architecture cannot fully compensate for noisy or weak pairing quality.

### Main conclusion

FT-Transformer did not universally outperform MLP. Its advantage depended on the data pattern, sample size, and pairing quality.

The strongest lesson from this experiment is that representation learning performance is controlled by both architecture and data construction. A more expressive encoder can improve alignment, but retrieval quality still depends heavily on clean pairing and evaluation design.

## 7. Limitations

- The dataset is synthetic and does not represent clinical performance.
- The generated images are artificial patterns, not medical images.
- Experiments were limited to 10 epochs.
- The goal is method behavior analysis, not clinical diagnosis or deployment.
- More seeds should be tested before making strong claims about model superiority.