from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled synthetic demo data for multimodal retrieval."
    )
    parser.add_argument(
        "--mode",
        choices=["linear", "interaction", "noisy"],
        default="linear",
        help="Synthetic data pattern to generate.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of paired synthetic samples.",
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.25,
        help="Fraction of image pairings to corrupt in noisy mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def make_synthetic_image(
    rng: np.random.Generator,
    signal: float,
    interaction_signal: float,
    mode: str,
) -> np.ndarray:
    """
    Create a synthetic grayscale image with controlled visual patterns.

    This is not a medical image. It is only a reproducible toy image that
    encodes tabular signal into visual intensity/shape patterns.
    """
    h, w = 224, 224

    base = rng.normal(90, 18, size=(h, w))

    yy, xx = np.mgrid[0:h, 0:w]

    # Smooth central blob controlled by the main signal.
    cx, cy = 112, 112
    radius = 36 + 18 * signal
    blob = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius**2
    base[blob] += 45 + 55 * signal

    # Diagonal stripe controlled by linear signal.
    stripe = np.abs((yy - xx) - int(30 * signal)) < 4
    base[stripe] += 30 * signal

    if mode == "interaction":
        # Interaction-specific visual cue: appears only when feature
        # combinations are active.
        square_size = int(24 + 45 * interaction_signal)
        x0 = 150 - square_size // 2
        y0 = 60 - square_size // 2
        base[y0 : y0 + square_size, x0 : x0 + square_size] += 80 * interaction_signal

        # Small opposite-corner marker for additional nonlinearity.
        marker = ((xx - 55) ** 2 + (yy - 165) ** 2) < (12 + 20 * interaction_signal) ** 2
        base[marker] += 60 * interaction_signal

    arr = np.clip(base, 0, 255).astype(np.uint8)
    return arr


def build_ehr_dataframe(rng: np.random.Generator, sample_ids: np.ndarray) -> pd.DataFrame:
    n_samples = len(sample_ids)

    age = rng.normal(60, 12, n_samples).clip(18, 95)
    num_admissions = rng.poisson(2, n_samples)
    num_labs = rng.poisson(20, n_samples)
    has_diabetes = rng.binomial(1, 0.35, n_samples)
    has_hypertension = rng.binomial(1, 0.45, n_samples)
    sex = rng.choice(["F", "M"], n_samples)

    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "age": age,
            "num_admissions": num_admissions,
            "num_labs": num_labs,
            "has_diabetes": has_diabetes,
            "has_hypertension": has_hypertension,
            "sex": sex,
        }
    )


def compute_signals(ehr_df: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    age_norm = (ehr_df["age"].to_numpy() - 18) / (95 - 18)
    labs_norm = ehr_df["num_labs"].to_numpy() / max(ehr_df["num_labs"].max(), 1)
    adm_norm = ehr_df["num_admissions"].to_numpy() / max(ehr_df["num_admissions"].max(), 1)

    diabetes = ehr_df["has_diabetes"].to_numpy()
    hypertension = ehr_df["has_hypertension"].to_numpy()
    sex_m = (ehr_df["sex"].to_numpy() == "M").astype(float)

    if mode in ["linear", "noisy"]:
        raw_signal = (
            0.35 * age_norm
            + 0.20 * labs_norm
            + 0.15 * adm_norm
            + 0.15 * diabetes
            + 0.10 * hypertension
            + 0.05 * sex_m
        )
        interaction_signal = np.zeros_like(raw_signal)

    elif mode == "interaction":
        interaction_signal = (
            (diabetes * (labs_norm > np.median(labs_norm))).astype(float)
            + (hypertension * (age_norm > np.median(age_norm))).astype(float)
            + ((adm_norm > np.median(adm_norm)) * sex_m).astype(float)
        )
        interaction_signal = interaction_signal / max(interaction_signal.max(), 1)

        raw_signal = (
            0.15 * age_norm
            + 0.10 * labs_norm
            + 0.10 * diabetes
            + 0.65 * interaction_signal
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize to [0, 1]
    signal = (raw_signal - raw_signal.min()) / (raw_signal.max() - raw_signal.min() + 1e-8)
    interaction_signal = (
        (interaction_signal - interaction_signal.min())
        / (interaction_signal.max() - interaction_signal.min() + 1e-8)
    )

    return signal, interaction_signal


def write_transform_files(ehr_df: pd.DataFrame, data_dir: Path) -> None:
    numeric_cols = ["age", "num_admissions", "num_labs"]
    binary_cols = ["has_diabetes", "has_hypertension"]
    categorical_cols = ["sex"]

    schema = {
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
        "categorical_cols": categorical_cols,
    }

    stats = {
        "numeric": {},
        "binary": {},
        "categorical": {},
    }

    for col in numeric_cols:
        values = ehr_df[col].astype(float)
        transform_type = "log1p" if col.startswith("num_") else "zscore"

        if transform_type == "log1p":
            transformed = np.log1p(np.maximum(values, 0.0))
            fill_value = float(values.median())
        else:
            transformed = values
            fill_value = float(values.median())

        stats["numeric"][col] = {
            "missing_fill": fill_value,
            "mean": float(transformed.mean()),
            "std": float(transformed.std() + 1e-6),
            "transform_type": transform_type,
        }

    for col in binary_cols:
        stats["binary"][col] = {
            "missing_fill": 0.0,
        }

    for col in categorical_cols:
        vocab = sorted(ehr_df[col].dropna().astype(str).unique().tolist())
        stats["categorical"][col] = {
            "missing_fill": vocab[0],
            "vocab": vocab,
        }

    with open(data_dir / "train_feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    with open(data_dir / "train_transform_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def write_split_files(rng: np.random.Generator, sample_ids: np.ndarray, data_dir: Path) -> None:
    shuffled = sample_ids.copy()
    rng.shuffle(shuffled)

    cut = int(0.8 * len(shuffled))
    train_ids = shuffled[:cut]
    eval_ids = shuffled[cut:]

    pd.DataFrame({"sample_id": train_ids}).to_csv(
        data_dir / "poolB_train_samples.csv", index=False
    )
    pd.DataFrame({"sample_id": eval_ids}).to_csv(
        data_dir / "poolA_eval_samples.csv", index=False
    )


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.noise_rate <= 1.0:
        raise ValueError("--noise-rate must be between 0 and 1.")

    rng = np.random.default_rng(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data_demo"
    image_dir = data_dir / "demo_images"

    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = np.arange(1, args.n_samples + 1)
    ehr_df = build_ehr_dataframe(rng, sample_ids)
    signal, interaction_signal = compute_signals(ehr_df, args.mode)

    image_paths_by_sample: dict[int, Path] = {}

    for i, sample_id in enumerate(sample_ids):
        image_path = image_dir / f"demo_{args.mode}_{sample_id:05d}.png"
        arr = make_synthetic_image(
            rng=rng,
            signal=float(signal[i]),
            interaction_signal=float(interaction_signal[i]),
            mode=args.mode,
        )
        Image.fromarray(arr, mode="L").convert("RGB").save(image_path)
        image_paths_by_sample[int(sample_id)] = image_path

    paired_image_source_ids = sample_ids.copy()

    if args.mode == "noisy":
        n_noisy = int(round(args.noise_rate * args.n_samples))
        corrupt_positions = rng.choice(args.n_samples, size=n_noisy, replace=False)
        shuffled_sources = paired_image_source_ids[corrupt_positions].copy()
        rng.shuffle(shuffled_sources)
        paired_image_source_ids[corrupt_positions] = shuffled_sources

    cxr_rows = []
    for row_idx, sample_id in enumerate(sample_ids):
        image_source_id = int(paired_image_source_ids[row_idx])
        image_path = image_paths_by_sample[image_source_id]

        # Store relative paths so the repo is portable across PCs.
        relative_image_path = image_path.relative_to(repo_root)

        cxr_rows.append(
            {
                "sample_id": int(sample_id),
                "image_path": str(relative_image_path),
                "image_source_id": image_source_id,
                "is_noisy_pair": int(image_source_id != int(sample_id)),
            }
        )

    cxr_df = pd.DataFrame(cxr_rows)

    ehr_df.to_csv(data_dir / "demo_ehr_profiles.csv", index=False)
    cxr_df.to_csv(data_dir / "demo_cxr_samples.csv", index=False)

    write_transform_files(ehr_df, data_dir)
    write_split_files(rng, sample_ids, data_dir)

    metadata = {
        "mode": args.mode,
        "n_samples": args.n_samples,
        "noise_rate": args.noise_rate if args.mode == "noisy" else 0.0,
        "seed": args.seed,
        "description": (
            "Synthetic controlled data for method comparison. "
            "Images are generated patterns/noise, not medical images."
        ),
    }

    with open(data_dir / "demo_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Demo data written to: {data_dir}")
    print(f"Mode: {args.mode}")
    print(f"EHR rows: {len(ehr_df)}")
    print(f"CXR rows: {len(cxr_df)}")
    print(f"Noisy pairs: {int(cxr_df['is_noisy_pair'].sum())}")
    print("Note: demo images are synthetic patterns, not medical images.")


if __name__ == "__main__":
    main()