# Method Overview

## Goal

This project studies whether an FT-Transformer-style tabular encoder can improve multimodal retrieval compared with a baseline MLP encoder.

The task is cross-modal retrieval: given a synthetic CXR-like image sample, retrieve the matching EHR-style tabular sample from a candidate pool.

## Compared Methods

### MLP EHR Encoder

The MLP encoder uses fully connected layers to map transformed tabular features into a shared embedding space.

It is simple, efficient, and often strong when the relationship between features and target signal is mostly linear or smooth.

### FT-Transformer EHR Encoder

The FT-Transformer treats each tabular feature as a token and applies self-attention across feature tokens.

The motivation is that attention may better capture feature interactions, such as cases where one feature only becomes informative when combined with another.

## Shared Training Setup

Both models use the same two-tower contrastive framework:

- Image branch: encodes synthetic visual patterns
- EHR branch: encodes tabular features
- Projection space: shared embedding space
- Loss: symmetric InfoNCE
- Evaluation: retrieval using cosine similarity

## Main Research Question

Can the FT-Transformer improve retrieval when the data contains feature interactions, noisy pairings, or larger sample sizes?

## Paper Inspiration

The FT-Transformer encoder in this repository is inspired by the tabular Transformer approach studied in:

> Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021.

That work revisits deep learning for tabular data and compares tabular neural architectures under more consistent experimental protocols. A useful lesson from the paper is that strong baselines matter: transformer-based tabular models should be compared against simple but competitive models rather than assumed to be better by default.

This repository follows that idea by comparing FT-Transformer against an MLP baseline instead of reporting FT-Transformer results alone.

## Additional Context

Other tabular deep learning work, such as SAINT, has explored attention-based tabular modeling and contrastive pretraining. This repository does not implement SAINT, but it is conceptually related because the experiment also studies representation learning for structured tabular data.