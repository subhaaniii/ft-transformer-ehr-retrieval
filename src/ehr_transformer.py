from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


class EHRTransformer:
    def __init__(self, stats: Dict, schema: Dict):
        self.stats = stats
        self.schema = schema

        self.numeric_cols: List[str] = schema["numeric_cols"]
        self.binary_cols: List[str] = schema["binary_cols"]
        self.categorical_cols: List[str] = schema["categorical_cols"]

    @classmethod
    def from_files(cls, stats_path: str | Path, schema_path: str | Path) -> "EHRTransformer":
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        return cls(stats=stats, schema=schema)

    def transform_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Returns:
            transformed_df: contains subject_id + transformed features
            feature_cols: ordered feature column names
        """
        out = pd.DataFrame()
        if "subject_id" in df.columns:
            out["subject_id"] = df["subject_id"]

        feature_cols: List[str] = []

        # -------------------------
        # Numeric -> log1p (count/LOS features) or z-score (continuous features)
        # transform_type is stored in stats by fit_train_transforms.py
        # -------------------------
        for col in self.numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan] * len(df))

            col_stats = self.stats["numeric"][col]
            fill_val = col_stats["missing_fill"]
            mean = col_stats["mean"]
            std = col_stats["std"]
            transform_type = col_stats.get("transform_type", "zscore")

            s = s.fillna(fill_val).astype(float)

            if transform_type == "log1p":
                # fill_val for log1p cols is the original median (pre-transform);
                # use 0 as the safe floor before log1p
                s = np.log1p(np.maximum(s, 0.0))

            z = (s - mean) / std

            new_col = f"num__{col}"
            out[new_col] = z
            feature_cols.append(new_col)

        # -------------------------
        # Binary -> float 0/1
        # -------------------------
        for col in self.binary_cols:
            s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan] * len(df))

            fill_val = self.stats["binary"][col]["missing_fill"]
            s = s.fillna(fill_val).astype(float)

            new_col = f"bin__{col}"
            out[new_col] = s
            feature_cols.append(new_col)

        # -------------------------
        # Categorical -> one-hot
        # -------------------------
        for col in self.categorical_cols:
            s = df[col].astype(str) if col in df.columns else pd.Series(["UNK"] * len(df))
            s = s.fillna("UNK").astype(str)

            vocab = self.stats["categorical"][col]["vocab"]
            fill_val = self.stats["categorical"][col]["missing_fill"]

            s = s.replace("nan", fill_val)
            s = s.where(s.isin(vocab), fill_val)

            for category in vocab:
                new_col = f"cat__{col}__{category}"
                out[new_col] = (s == category).astype(float)
                feature_cols.append(new_col)

        return out, feature_cols