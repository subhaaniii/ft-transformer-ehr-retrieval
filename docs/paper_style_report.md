# Paper-Style Report: FT-Transformer EHR Retrieval

## Abstract

This project studies whether an FT-Transformer-style tabular encoder improves multimodal retrieval alignment compared with a simpler MLP encoder. The benchmark uses controlled synthetic EHR-like tabular features and synthetic CXR-like visual patterns instead of restricted clinical data. The goal is not to make clinical claims, but to study method behavior under known data-generating conditions.

The project compares MLP and FT-Transformer EHR encoders inside the same two-tower contrastive retrieval pipeline. Experiments vary dataset structure, sample size, and pair quality using linear, interaction-based, and noisy-pair settings. The results show that FT-Transformer does not universally outperform the MLP. Its usefulness depends on data structure, sample size, and pair quality. In some interaction settings, FT-Transformer improves retrieval, but in noisy-pair settings both models degrade, showing that better architecture cannot fully compensate for weak supervision.

---

## 1. Motivation

Electronic health record data is often represented as structured tabular data. Unlike images or free text, tabular clinical data contains heterogeneous features such as continuous measurements, binary indicators, categorical variables, and derived medical attributes.

In multimodal medical AI, tabular EHR features may need to be aligned with another modality, such as medical images or reports. A common question is whether more expressive tabular encoders, such as transformer-style models, improve representation learning compared with simpler MLP encoders.

This project studies that question in a controlled retrieval setting.

The public version uses synthetic data so that the data-generating process, feature interactions, pair quality, and evaluation setup are known and reproducible.

---

## 2. Research Question

The main research question is:

> Can an FT-Transformer-style tabular encoder improve retrieval alignment compared with a simpler MLP encoder when the data contains feature interactions, noisy pairings, or larger sample sizes?

The project studies this question by varying:

- encoder architecture
- dataset mode
- sample size
- pair quality
- retrieval behavior
- positive-pair similarity

---

## 3. Background

### 3.1 Tabular EHR Representation Learning

Structured EHR data is often tabular: each patient or admission can be represented by a row of clinical variables. These variables may include demographics, diagnoses, measurements, medications, laboratory-style features, or synthetic proxies for those concepts.

A simple MLP can model nonlinear relationships after preprocessing, but it treats the feature vector as a flat input.

Transformer-style tabular models instead tokenize features and use attention across feature tokens. This can help when interactions between features matter.

### 3.2 FT-Transformer-Style Encoding

The FT-Transformer idea is based on representing tabular features as tokens and applying Transformer layers over those feature tokens. In this project, the FT-Transformer-style encoder is used as the EHR-side encoder inside a two-tower retrieval model.

The goal is not to reproduce FT-Transformer exactly for supervised tabular classification. Instead, the goal is to test a transformer-style tabular encoder inside a contrastive retrieval pipeline.

### 3.3 Contrastive Retrieval

The model uses a two-tower setup:

- one encoder processes synthetic CXR-like visual inputs
- one encoder processes synthetic EHR-like tabular inputs

The two encoders produce embeddings in a shared space. A symmetric InfoNCE-style contrastive loss pulls matched image-EHR pairs together and pushes non-matching pairs apart.

---

## 4. Method

The benchmark compares two EHR encoder settings:

| Method | Description |
|---|---|
| MLP EHR Encoder | A baseline feed-forward encoder for transformed tabular features |
| FT-Transformer EHR Encoder | A feature-tokenizer plus Transformer encoder that applies self-attention over tabular feature tokens |

Both methods use:

- the same synthetic data generator
- the same image-side encoder
- the same contrastive objective
- the same retrieval metrics
- the same train/evaluation protocol

This isolates the main comparison: whether the EHR encoder architecture changes retrieval behavior.

---

## 5. Dataset and Experimental Setup

The public repository uses controlled synthetic multimodal data.

Each sample contains:

| Component | Description |
|---|---|
| Synthetic EHR profile | Tabular feature vector representing structured clinical-style information |
| Synthetic CXR-like input | Controlled visual pattern associated with the EHR profile |
| Pair ID | Known image-EHR match used for retrieval evaluation |
| Dataset mode | Controls whether the signal is linear, interaction-based, or noisy |

Three dataset modes are included:

| Setup | Purpose |
|---|---|
| Linear | Tests whether a simple MLP is already sufficient when the signal is mostly linear |
| Interaction | Tests whether FT-Transformer helps when feature interactions matter |
| Noisy | Tests robustness when a fraction of image-EHR pairings are intentionally corrupted |

The visual inputs are controlled synthetic patterns, not real clinical images. This makes the benchmark privacy-safe and reproducible.

---

## 6. Experiment Matrix

The final benchmark varies:

| Factor | Values |
|---|---|
| EHR encoder | MLP, FT-Transformer |
| Dataset mode | Linear, interaction, noisy |
| Sample size | 500 pilot, 1000, 2000 |
| Training duration | 10 epochs |

The benchmark is designed to test whether architecture improvements appear consistently or only under specific data conditions.

---

## 7. Evaluation Metrics

The project evaluates retrieval quality and embedding behavior.

| Metric | Meaning |
|---|---|
| Recall@1 | Whether the correct match is ranked first |
| Recall@5 | Whether the correct match appears in the top 5 |
| Recall@10 | Whether the correct match appears in the top 10 |
| Recall@50 | Whether the correct match appears in the top 50 |
| Lift@K | Improvement over random retrieval |
| Positive-pair cosine similarity | Average similarity between true image-EHR pairs |
| Training loss | Final contrastive optimization loss |

This metric set is important because positive-pair similarity alone does not guarantee strong top-k retrieval. A model may pull positive pairs closer on average while still failing to rank the exact match highly among many candidates.

---

## 8. Results

The results show that FT-Transformer does not universally outperform the MLP encoder.

Instead, model behavior depends on the dataset condition.

### 8.1 Linear settings

In mostly linear settings, the MLP is already highly competitive. This suggests that a stronger architecture is not automatically useful when the data structure is simple enough for a baseline encoder.

### 8.2 Interaction settings

In interaction-based settings, FT-Transformer can improve retrieval in some sample-size regimes. This supports the idea that attention over feature tokens may help when feature interactions are important.

However, the improvement is not uniform across all settings. This means the architecture should be evaluated under controlled conditions rather than assumed to be better.

### 8.3 Noisy-pair settings

In noisy-pair settings, both MLP and FT-Transformer degrade. This shows that architecture alone cannot fix weak or corrupted supervision.

This is an important result for multimodal learning: pair quality can dominate model architecture.

---

## 9. Key Findings

### 9.1 FT-Transformer is not automatically better

The FT-Transformer-style encoder does not universally outperform the simpler MLP baseline. Its benefit depends on the data structure.

### 9.2 MLP is strong in simple settings

When the signal is mostly linear or easy to model, the MLP baseline can already perform well.

### 9.3 FT-Transformer can help when feature interactions matter

In interaction-based synthetic settings, FT-Transformer can improve retrieval behavior, especially when the data contains relationships that may benefit from feature-token attention.

### 9.4 Pair quality matters as much as architecture

When pair assignments are noisy, both models degrade. Better architecture cannot fully compensate for weak image-EHR pairing.

### 9.5 Similarity and retrieval are not identical

FT-Transformer may improve positive-pair similarity without always improving top-k retrieval. This shows why retrieval evaluation requires ranking metrics such as Recall@K and Lift@K, not only cosine similarity.

---

## 10. Limitations

This repository is a controlled method-analysis project, not a clinical deployment study.

The data is synthetic, and the visual inputs are controlled patterns rather than real chest X-ray images. Therefore, the results should be interpreted as evidence of experimental design, retrieval-evaluation reasoning, and model-behavior analysis, not as clinical performance.

The benchmark also uses a limited number of sample sizes, dataset modes, and training epochs. Stronger evidence would require repeated random seeds, larger sample-size sweeps, longer training, additional baselines, and authorized real-world datasets.

---

## 11. Future Work

Possible extensions include:

- repeating all experiments across multiple random seeds
- testing larger sample sizes
- training for longer durations
- adding stronger tabular baselines
- testing additional transformer-style tabular encoders
- adding pair-noise sweeps
- analyzing embedding geometry after training
- applying the pipeline to authorized real multimodal datasets
- comparing retrieval behavior under different contrastive loss variants

---

## 12. What I Learned

This project taught me that stronger architecture does not automatically mean stronger retrieval.

The most important lesson is:

> Architecture, data structure, sample size, and pair quality must be evaluated together.

FT-Transformer-style encoders can be useful when feature interactions matter, but they are not magic. If the data is simple, an MLP may be enough. If the pairs are noisy, both models can fail.

This changed how I think about multimodal retrieval. A serious benchmark should not only compare models. It should also test when a model helps, when it fails, and what data conditions make the difference.

---

## 13. Related Work

This project is inspired by deep learning methods for tabular data and contrastive retrieval.

The FT-Transformer-style encoder is motivated by work on feature tokenization and Transformer layers for tabular prediction tasks. The broader idea of applying self-attention to tabular features is also related to TabTransformer, which uses contextual embeddings for categorical tabular features.

This repository does not reproduce those papers directly. Instead, it adapts the idea of transformer-style tabular encoding to a controlled multimodal retrieval setting.

---

## 14. References

- Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. *Revisiting Deep Learning Models for Tabular Data*. NeurIPS, 2021.
- Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin. *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*. arXiv, 2020.
