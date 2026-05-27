# Metric Explanation

## Recall@K

Recall@K measures whether the correct matching EHR sample appears in the top K retrieved candidates for a given image sample.

Examples:

- Recall@1: correct match is ranked first
- Recall@5: correct match appears in the top 5
- Recall@10: correct match appears in the top 10
- Recall@50: correct match appears in the top 50

Higher Recall@K means better retrieval.

## Lift Over Random

Lift compares model retrieval against random retrieval.

For example, if the candidate pool has 200 EHR samples, random Recall@50 is:

```text
50 / 200 = 0.25
```
If the model gets Recall@50 = 0.80, then:

```text
Lift@50 = 0.80 / 0.25 = 3.2x
```

This means the model is 3.2 times better than random retrieval at K=50.

## Positive-Pair Similarity

Positive-pair similarity is the average cosine similarity between the matched image and EHR embeddings.

A higher value usually means the model is pulling matched pairs closer in embedding space.

However, higher positive similarity does not always guarantee better top-k retrieval. Retrieval also depends on how well the model separates the correct match from many competing candidates.

## Training Loss

The training loss is the symmetric InfoNCE loss.

Lower loss generally indicates better optimization, but it should be interpreted together with retrieval metrics.