from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    rng = np.random.default_rng(42)

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data_demo"
    image_dir = data_dir / "demo_images"

    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 200

    sample_ids = np.arange(1, n_samples + 1)

    ehr_df = pd.DataFrame({
        "sample_id": sample_ids,
        "age": rng.normal(60, 12, n_samples).clip(18, 95),
        "num_admissions": rng.poisson(2, n_samples),
        "num_labs": rng.poisson(20, n_samples),
        "has_diabetes": rng.integers(0, 2, n_samples),
        "has_hypertension": rng.integers(0, 2, n_samples),
        "sex": rng.choice(["F", "M"], n_samples),
    })

    cxr_rows = []
    for sample_id in sample_ids:
        image_path = image_dir / f"demo_cxr_{sample_id:04d}.png"

        # Synthetic grayscale image, not a medical image.
        arr = rng.normal(128, 35, size=(224, 224)).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").convert("RGB").save(image_path)

        cxr_rows.append({
            "sample_id": sample_id,
            "image_path": str(image_path),
        })

    cxr_df = pd.DataFrame(cxr_rows)

    ehr_df.to_csv(data_dir / "demo_ehr_profiles.csv", index=False)
    cxr_df.to_csv(data_dir / "demo_cxr_samples.csv", index=False)

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

    print(f"Demo data written to: {data_dir}")
    print(f"EHR rows: {len(ehr_df)}")
    print(f"CXR rows: {len(cxr_df)}")
    print("Note: demo images are random synthetic noise, not medical images.")


if __name__ == "__main__":
    main()