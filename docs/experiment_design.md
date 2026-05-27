# Experiment Design

## Five Questions

Each experiment in this repository is designed around five questions:

1. What method did I test?
2. What dataset setup did I use?
3. What metric did I measure?
4. What changed when I changed sample size, loss, pairing, or split?
5. What did I learn?

## Dataset Modes

The repository uses controlled synthetic data rather than restricted clinical data.

### Linear Mode

The image pattern is generated mostly from linear combinations of tabular features.

Purpose:

- Test whether a simple MLP is already sufficient.
- Check if FT-Transformer provides any benefit when the signal is simple.

### Interaction Mode

The image pattern depends more strongly on feature interactions.

Purpose:

- Test whether self-attention over tabular feature tokens helps when feature combinations matter.
- Compare whether FT-Transformer improves retrieval more clearly than MLP.

### Noisy Mode

A fraction of image-EHR pairings is intentionally corrupted.

Purpose:

- Test robustness under weak or noisy supervision.
- Check whether architecture can compensate for poor pair quality.

## Sample Sizes

The experiments include:

- 500 samples as a pilot study
- 1000 samples for medium-scale comparison
- 2000 samples for stronger sample-size analysis

## Train/Evaluation Split

The synthetic generator creates separate train and evaluation sample lists:

- `poolB_train_samples.csv`
- `poolA_eval_samples.csv`

The model trains on the training split and reports retrieval metrics on the held-out evaluation split.

## Why Synthetic Data?

The goal is not clinical performance. The goal is controlled method analysis.

Synthetic data allows:

- Reproducibility
- Privacy-safe public sharing
- Explicit control over feature interactions
- Explicit control over noisy pairings
- Clear comparison between MLP and FT-Transformer behavior