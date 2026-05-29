from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data_demo"
FIG_DIR = REPO_ROOT / "figures"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_image_path(path_str: str) -> Path:
    # Handles Windows-style backslashes stored in CSV
    normalized = str(path_str).replace("\\", os.sep).replace("/", os.sep)
    return REPO_ROOT / normalized


def load_pool_membership():
    pool_a = set(pd.read_csv(DATA_DIR / "poolA_eval_samples.csv")["sample_id"].tolist())
    pool_b = set(pd.read_csv(DATA_DIR / "poolB_train_samples.csv")["sample_id"].tolist())
    return pool_a, pool_b


def assign_pool_label(sample_ids, pool_a, pool_b):
    labels = []
    for sid in sample_ids:
        if sid in pool_a:
            labels.append("Pool A (eval)")
        elif sid in pool_b:
            labels.append("Pool B (train)")
        else:
            labels.append("Other")
    return labels


def load_ehr_space():
    ehr = pd.read_csv(DATA_DIR / "demo_ehr_profiles.csv")

    with open(DATA_DIR / "train_feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    numeric_cols = schema.get("numeric_cols", [])
    binary_cols = schema.get("binary_cols", [])
    categorical_cols = schema.get("categorical_cols", [])

    numeric_df = ehr[numeric_cols + binary_cols].copy()
    categorical_df = pd.get_dummies(ehr[categorical_cols].astype(str), drop_first=False)

    feature_df = pd.concat([numeric_df, categorical_df], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    out = ehr.copy()
    out["pca1"] = coords[:, 0]
    out["pca2"] = coords[:, 1]
    return out


def load_cxr_space(image_size=32):
    cxr = pd.read_csv(DATA_DIR / "demo_cxr_samples.csv")

    features = []
    for rel_path in cxr["image_path"]:
        img_path = resolve_image_path(rel_path)
        img = Image.open(img_path).convert("L").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        features.append(arr.flatten())

    X = np.vstack(features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    out = cxr.copy()
    out["pca1"] = coords[:, 0]
    out["pca2"] = coords[:, 1]
    return out


def scatter_by_pool(ax, df, title):
    colors = {
        "Pool A (eval)": "#1f77b4",
        "Pool B (train)": "#ff7f0e",
        "Other": "#7f7f7f",
    }

    for pool_name, subdf in df.groupby("pool_label"):
        ax.scatter(
            subdf["pca1"],
            subdf["pca2"],
            s=18,
            alpha=0.75,
            label=pool_name,
            c=colors.get(pool_name, "#7f7f7f"),
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def scatter_ehr_clinical(ax, ehr_df):
    # Color by hypertension, marker by sex
    color_map = {0: "#4c78a8", 1: "#e45756"}
    marker_map = {"F": "o", "M": "x"}

    for sex_value in sorted(ehr_df["sex"].astype(str).unique()):
        for htn_value in sorted(ehr_df["has_hypertension"].unique()):
            subdf = ehr_df[
                (ehr_df["sex"].astype(str) == sex_value)
                & (ehr_df["has_hypertension"] == htn_value)
            ]
            if len(subdf) == 0:
                continue

            ax.scatter(
                subdf["pca1"],
                subdf["pca2"],
                s=18,
                alpha=0.75,
                marker=marker_map.get(sex_value, "o"),
                c=color_map.get(htn_value, "#7f7f7f"),
            )

    ax.set_title("EHR profile space\n(color = hypertension, marker = sex)", fontsize=11)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.25)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Female", markerfacecolor="black", markersize=7),
        Line2D([0], [0], marker="x", color="black", label="Male", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Hypertension = 0", markerfacecolor="#4c78a8", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="Hypertension = 1", markerfacecolor="#e45756", markersize=7),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="best")


def save_individual_plot(df, title, filename):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter_by_pool(ax, df, title)
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(FIG_DIR)

    pool_a, pool_b = load_pool_membership()

    ehr_df = load_ehr_space()
    cxr_df = load_cxr_space(image_size=32)

    ehr_df["pool_label"] = assign_pool_label(ehr_df["sample_id"], pool_a, pool_b)
    cxr_df["pool_label"] = assign_pool_label(cxr_df["sample_id"], pool_a, pool_b)

    # Individual plots
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter_ehr_clinical(ax, ehr_df)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ehr_profile_space.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    save_individual_plot(
        ehr_df,
        "EHR profile space by retrieval split",
        "ehr_pool_split_space.png"
    )

    save_individual_plot(
        cxr_df,
        "Synthetic CXR image space by retrieval split",
        "cxr_image_space.png"
    )

    # Combined figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    scatter_ehr_clinical(axes[0], ehr_df)
    scatter_by_pool(axes[1], ehr_df, "EHR profile space by retrieval split")
    scatter_by_pool(axes[2], cxr_df, "Synthetic CXR image space by retrieval split")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "ft_transformer_demo_spaces.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Saved figures:")
    print(FIG_DIR / "ehr_profile_space.png")
    print(FIG_DIR / "ehr_pool_split_space.png")
    print(FIG_DIR / "cxr_image_space.png")
    print(FIG_DIR / "ft_transformer_demo_spaces.png")


if __name__ == "__main__":
    main()