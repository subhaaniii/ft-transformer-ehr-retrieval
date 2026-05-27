"""
FT-Transformer EHR Encoder for Multimodal Retrieval

This script compares a baseline MLP tabular encoder with an FT-Transformer-style
EHR encoder in a two-tower contrastive retrieval setup.

The public version is designed for synthetic/demo data and does not include
restricted clinical records or patient-level data.

Main question:
    Can a transformer-based tabular encoder improve cross-modal retrieval
    compared with a simple MLP encoder?

Example:
    python src/train.py --ehr-encoder ftt
    python src/train.py --ehr-encoder mlp
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Suppress PyTorch nested-tensor warning from TransformerEncoder + norm_first=True.
# The optimisation is disabled but the model is functionally correct.
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True",
    category=UserWarning,
    module="torch.nn.modules.transformer",
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
for _p in [REPO_ROOT, PROJECT_ROOT / "utils"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cxr_transforms import get_cxr_transforms
from model import DualEncoder
from ehr_transformer import EHRTransformer


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def symmetric_infonce_loss(
    z_cxr: torch.Tensor,
    z_ehr: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    B      = z_cxr.size(0)
    logits = z_cxr @ z_ehr.T / temperature
    labels = torch.arange(B, device=z_cxr.device)
    loss   = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    with torch.no_grad():
        pos_sim  = logits.diagonal().mean().item() * temperature
        neg_mask = ~torch.eye(B, dtype=torch.bool, device=z_cxr.device)
        neg_sim  = logits[neg_mask].mean().item() * temperature
    return loss, {"pos_sim": pos_sim, "neg_sim": neg_sim}


_FALLBACK_IMAGE_TENSOR: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    seed:                            int   = 42
    epochs:                          int   = 20
    batch_size:                      int   = 128
    eval_cxr_batch_size:             int   = 128
    eval_ehr_batch_size:             int   = 512
    num_workers:                     int   = 0
    lr:                              float = 1e-4
    weight_decay:                    float = 1e-3
    temperature:                     float = 0.10
    embed_dim:                       int   = 128
    grad_clip:                       float = 1.0
    k_values:                        tuple = (1, 5, 10, 50)
    train_one_cxr_per_sample:        bool = True
    ehr_encoder_type:                str   = "ftt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FT-Transformer EHR encoder for multimodal retrieval")
    p.add_argument("--project-root",      type=Path, default=None)
    p.add_argument("--processed-dir",     type=Path, default=None)
    p.add_argument("--cxr-csv", type=Path, default=Path("data_demo/demo_cxr_samples.csv"))
    p.add_argument("--ehr-csv", type=Path, default=Path("data_demo/demo_ehr_profiles.csv"))
    p.add_argument("--image-path-column", type=str,  default="image_path")
    p.add_argument("--id-column", type=str, default="sample_id")
    p.add_argument("--cxr-cache-dir",     type=Path, default=None)
    p.add_argument("--output-dir",        type=Path, default=None)
    p.add_argument("--checkpoint-dir",    type=Path, default=None)
    p.add_argument("--epochs",       type=int,   default=RunConfig.epochs)
    p.add_argument("--batch-size",   type=int,   default=RunConfig.batch_size)
    p.add_argument("--num-workers",  type=int,   default=RunConfig.num_workers)
    p.add_argument("--lr",           type=float, default=RunConfig.lr)
    p.add_argument("--weight-decay", type=float, default=RunConfig.weight_decay)
    p.add_argument("--temperature",  type=float, default=RunConfig.temperature)
    p.add_argument("--seed",         type=int,   default=RunConfig.seed)
    p.add_argument(
        "--ehr-encoder", choices=["mlp", "ftt"], default="ftt",
        help="EHR encoder architecture: 'ftt' = FTTransformer (default), 'mlp' = baseline MLP.",
    )
    p.add_argument(
        "--use-all-train-cxr", action="store_true",
        help="Use every train CXR row. Default: one per sample.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from last.pt in the checkpoint dir. "
             "--epochs is the TOTAL epoch target (not additional epochs). "
             "If last.pt is epoch 20 and --epochs 35, runs 15 more epochs.",
    )
    p.add_argument(
        "--cosine-lr", action="store_true",
        help="Use cosine annealing LR schedule (lr -> eta-min over --epochs). "
             "Recommended for clean convergence. Without this, flat LR causes "
             "lift@50 to oscillate at the plateau.",
    )
    p.add_argument(
        "--eta-min", type=float, default=1e-6,
        help="Minimum LR for cosine annealing (default 1e-6).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_project_root(arg_root: Path | None, cxr_csv_arg: Path | None = None) -> Path:
    if arg_root is not None:
        return arg_root
    return PROJECT_ROOT


def resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing columns {missing}. Available: {df.columns.tolist()}")


def remap_to_cache(df: pd.DataFrame, cache_dir: Path | None) -> pd.DataFrame:
    if cache_dir is None:
        return df
    df = df.copy()
    df["image_path"] = df["image_path"].apply(lambda p: str(cache_dir / Path(p).name))
    return df


def load_sample_ids(path: Path) -> set[int]:
    return set(pd.read_csv(path, usecols=["sample_id"])["sample_id"].astype(int))


def build_sample_split(processed_dir, paired_samples, seed, train_fraction=0.8):
    poolb = processed_dir / "poolB_train_samples.csv"
    poola = processed_dir / "poolA_eval_samples.csv"
    if poolb.exists() and poola.exists():
        tr = load_sample_ids(poolb) & paired_samples
        va = load_sample_ids(poola) & paired_samples
        if tr and va:
            return tr, va, "poolB_train_vs_poolA_eval"
    samples = np.array(sorted(paired_samples), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    cut = min(max(int(round(len(samples) * train_fraction)), 1), len(samples) - 1)
    return set(samples[:cut].tolist()), set(samples[cut:].tolist()), "random_sample_split"


def transform_ehr_profiles(ehr_df, processed_dir):
    stats_path  = processed_dir / "train_transform_stats.json"
    schema_path = processed_dir / "train_feature_schema.json"
    if stats_path.exists() and schema_path.exists():
        transformer = EHRTransformer.from_files(stats_path, schema_path)
        transformed, feature_cols = transformer.transform_dataframe(ehr_df)
        return transformed, feature_cols, "EHRTransformer"
    feat_path = processed_dir / "ehr_feature_columns.json"
    if feat_path.exists():
        with open(feat_path, encoding="utf-8") as f:
            feature_cols = json.load(f)
        if all(c in ehr_df.columns for c in feature_cols):
            return ehr_df[["sample_id"] + feature_cols].copy(), feature_cols, "ehr_feature_columns.json"
    raise RuntimeError(
        "Cannot build EHR features. Need train_transform_stats.json + "
        "train_feature_schema.json, or ehr_feature_columns.json."
    )


def build_train_pairs(cxr_df, train_ids, image_path_column, one_per_subject, seed):
    tr = cxr_df[cxr_df["sample_id"].astype(int).isin(train_ids)].copy()
    tr = tr.dropna(subset=[image_path_column])
    if one_per_subject:
        tr = (tr.sample(frac=1.0, random_state=seed)
                .drop_duplicates("sample_id")
                .sort_values("sample_id")
                .reset_index(drop=True))
    return tr


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PairedCxrEhrDataset(Dataset):
    def __init__(self, cxr_df, ehr_features, feature_cols, image_path_column, transform):
        require_columns(cxr_df, ["sample_id", image_path_column], "CXR dataframe")
        require_columns(ehr_features, ["sample_id"] + feature_cols, "EHR features")
        self.feature_cols      = feature_cols
        self.image_path_column = image_path_column
        self.transform         = transform
        self.fallback_count    = 0
        ehr_features = ehr_features.drop_duplicates("sample_id")
        self.ehr_by_sample = {
            int(r["sample_id"]): r[feature_cols].to_numpy(dtype=np.float32)
            for _, r in ehr_features.iterrows()
        }
        self.cxr_df = (
            cxr_df[cxr_df["sample_id"].astype(int).isin(self.ehr_by_sample)]
            .reset_index(drop=True)
        )
        if len(self.cxr_df) == 0:
            raise RuntimeError("No CXR rows match transformed EHR features.")

    def __len__(self):
        return len(self.cxr_df)

    def __getitem__(self, idx):
        row        = self.cxr_df.iloc[idx]
        sample_id = int(row["sample_id"])
        image      = self._load_image(str(row[self.image_path_column]))
        ehr        = torch.from_numpy(self.ehr_by_sample[sample_id].copy())
        return image, ehr, sample_id

    def _load_image(self, path):
        global _FALLBACK_IMAGE_TENSOR
        try:
            with Image.open(path) as img:
                return self.transform(img.convert("RGB"))
        except Exception:
            self.fallback_count += 1
            if _FALLBACK_IMAGE_TENSOR is None:
                _FALLBACK_IMAGE_TENSOR = torch.zeros(3, 224, 224)
            return _FALLBACK_IMAGE_TENSOR.clone()


class CxrEvalDataset(Dataset):
    def __init__(self, cxr_df, image_path_column, transform):
        require_columns(cxr_df, ["sample_id", image_path_column], "CXR eval dataframe")
        self.df                = cxr_df.dropna(subset=[image_path_column]).reset_index(drop=True)
        self.image_path_column = image_path_column
        self.transform         = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        global _FALLBACK_IMAGE_TENSOR
        row = self.df.iloc[idx]
        try:
            with Image.open(str(row[self.image_path_column])) as img:
                return self.transform(img.convert("RGB")), int(row["sample_id"])
        except Exception:
            if _FALLBACK_IMAGE_TENSOR is None:
                _FALLBACK_IMAGE_TENSOR = torch.zeros(3, 224, 224)
            return _FALLBACK_IMAGE_TENSOR.clone(), int(row["sample_id"])


class EhrEvalDataset(Dataset):
    def __init__(self, ehr_features, feature_cols):
        require_columns(ehr_features, ["sample_id"] + feature_cols, "EHR eval dataframe")
        df = ehr_features.drop_duplicates("sample_id").reset_index(drop=True)
        self.sample_ids = df["sample_id"].astype(int).to_numpy()
        self.matrix      = df[feature_cols].astype(np.float32).to_numpy()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return torch.from_numpy(self.matrix[idx].copy()), int(self.sample_ids[idx])


def collate_train(batch):
    return (
        torch.stack([x[0] for x in batch]),
        torch.stack([x[1] for x in batch]),
        torch.tensor([x[2] for x in batch], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Embedding + retrieval evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_cxr(model, loader, device):
    model.eval()
    embeds, sample_ids = [], []
    use_amp = device.type == "cuda"

    for images, ids in tqdm(loader, desc="embed_cxr", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            z = model.encode_cxr(images)
        embeds.append(z.float().cpu())
        sample_ids.extend(int(x) for x in ids)

    return torch.cat(embeds), np.asarray(sample_ids, dtype=np.int64)


@torch.no_grad()
def embed_ehr(model, loader, device):
    model.eval()
    embeds, sample_ids = [], []
    use_amp = device.type == "cuda"

    for feats, ids in tqdm(loader, desc="embed_ehr", leave=False):
        feats = feats.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            z = model.encode_ehr(feats)
        embeds.append(z.float().cpu())
        sample_ids.extend(int(x) for x in ids)

    return torch.cat(embeds), np.asarray(sample_ids, dtype=np.int64)


@torch.no_grad()
def evaluate_retrieval(model, cxr_loader, ehr_loader, k_values, device) -> dict[str, float]:
    cxr_mat, cxr_sample_ids = embed_cxr(model, cxr_loader, device)
    ehr_mat, ehr_sample_ids = embed_ehr(model, ehr_loader, device)

    sim = (cxr_mat @ ehr_mat.T).numpy()
    n_cxr, n_ehr = sim.shape

    valid_mask = np.isin(cxr_sample_ids, ehr_sample_ids)
    total = int(valid_mask.sum())
    if total == 0:
        raise RuntimeError("No eval CXR rows have a matching EHR sample.")

    max_k = min(max(k_values), n_ehr)
    top_part = np.argpartition(sim, -max_k, axis=1)[:, -max_k:]
    top_sc = sim[np.arange(n_cxr)[:, None], top_part]
    order = np.argsort(top_sc, axis=1)[:, ::-1]
    top_sort = top_part[np.arange(n_cxr)[:, None], order]

    metrics: dict[str, float] = {}

    for k_req in k_values:
        k = min(k_req, n_ehr)
        top_k_sample_ids = ehr_sample_ids[top_sort[:, :k]]
        hit_mask = (top_k_sample_ids == cxr_sample_ids[:, None]).any(axis=1)
        recall = float((hit_mask & valid_mask).sum() / total)
        random_recall = float(min(k / n_ehr, 1.0))

        metrics[f"recall@{k_req}"] = recall
        metrics[f"lift@{k_req}"] = float(recall / random_recall) if random_recall > 0 else math.inf

    ehr_sample_id_to_idx = {int(sample_id): i for i, sample_id in enumerate(ehr_sample_ids)}
    valid_idxs = np.where(valid_mask)[0]
    partner_idxs = np.array([
        ehr_sample_id_to_idx[int(cxr_sample_ids[i])] for i in valid_idxs
    ])

    pos_sims = sim[valid_idxs, partner_idxs]
    metrics["pos_sim_mean"] = float(pos_sims.mean())
    metrics["n_ehr_pool"] = float(n_ehr)

    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, temperature, grad_clip):
    model.train()
    use_amp = device.type == "cuda"
    running_loss = running_pos_sim = 0.0
    steps = 0

    for images, ehr, _sample_ids in tqdm(loader, desc="train", leave=False):
        if images.size(0) < 2:
            continue
        images = images.to(device, non_blocking=True)
        ehr    = ehr.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            z_cxr, z_ehr = model(images, ehr)
            loss, lm     = symmetric_infonce_loss(z_cxr, z_ehr, temperature)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss    += float(loss.detach())
        running_pos_sim += lm["pos_sim"]
        steps += 1

    if steps == 0:
        raise RuntimeError("No training batches with >= 2 samples.")

    fb = getattr(loader.dataset, "fallback_count", 0)
    if fb > 0:
        print(f"  [WARN] {fb} zero-image fallbacks this epoch.")
        loader.dataset.fallback_count = 0

    return {"loss": running_loss / steps, "pos_sim": running_pos_sim / steps}


def save_checkpoint(path, model, optimizer, epoch, metrics, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":             metrics,
        "config":              config,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    project_root  = resolve_project_root(args.project_root, args.cxr_csv)
    processed_dir = args.processed_dir or (project_root / "data_demo")
    cxr_path      = resolve_path(args.cxr_csv, project_root)

    run_tag = f"ft_transformer_{args.ehr_encoder}"
    output_dir = args.output_dir    or (project_root / "outputs"     / run_tag)
    ckpt_dir   = args.checkpoint_dir or (project_root / "checkpoints" / run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        seed                    = args.seed,
        epochs                  = args.epochs,
        batch_size              = args.batch_size,
        num_workers             = args.num_workers,
        lr                      = args.lr,
        weight_decay            = args.weight_decay,
        temperature             = args.temperature,
        train_one_cxr_per_sample = not args.use_all_train_cxr,
        ehr_encoder_type        = args.ehr_encoder,
    )
    set_seed(cfg.seed)

    ehr_path = resolve_path(args.ehr_csv, project_root)
    for p in [cxr_path, ehr_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print(f"Project root    : {project_root}")
    print(f"EHR encoder     : {cfg.ehr_encoder_type.upper()}")
    print(f"Output dir      : {output_dir}")
    print(f"Checkpoints     : {ckpt_dir}")
    print(f"Config          : {cfg}")

    cxr_df = pd.read_csv(cxr_path)
    ehr_df = pd.read_csv(ehr_path)

    if args.id_column not in cxr_df.columns:
        raise RuntimeError(
            f"CXR CSV missing ID column '{args.id_column}'. "
            f"Available columns: {cxr_df.columns.tolist()}"
        )

    if args.id_column not in ehr_df.columns:
        raise RuntimeError(
            f"EHR CSV missing ID column '{args.id_column}'. "
            f"Available columns: {ehr_df.columns.tolist()}"
        )

    if args.id_column != "sample_id":
        cxr_df = cxr_df.rename(columns={args.id_column: "sample_id"})
        ehr_df = ehr_df.rename(columns={args.id_column: "sample_id"})

    cxr_df["sample_id"] = cxr_df["sample_id"].astype(int)
    ehr_df["sample_id"] = ehr_df["sample_id"].astype(int)
    paired_samples = set(cxr_df["sample_id"]) & set(ehr_df["sample_id"])

    train_ids, val_ids, split_src = build_sample_split(processed_dir, paired_samples, cfg.seed)
    if train_ids & val_ids:
        raise RuntimeError(f"Sample leakage: {len(train_ids & val_ids)} samples in both splits.")

    ehr_features, feature_cols, ehr_src = transform_ehr_profiles(ehr_df, processed_dir)
    train_cxr = build_train_pairs(
        cxr_df, train_ids, args.image_path_column, cfg.train_one_cxr_per_sample, cfg.seed
    )
    val_cxr = (
        cxr_df[cxr_df["sample_id"].isin(val_ids)]
        .dropna(subset=[args.image_path_column])
        .reset_index(drop=True)
    )
    val_ehr = ehr_features[ehr_features["sample_id"].isin(val_ids)].reset_index(drop=True)

    if args.cxr_cache_dir is not None:
        print(f"CXR cache dir   : {args.cxr_cache_dir}")
        train_cxr = remap_to_cache(train_cxr, args.cxr_cache_dir)
        val_cxr   = remap_to_cache(val_cxr,   args.cxr_cache_dir)

    print(f"Split source    : {split_src}")
    print(f"EHR transform   : {ehr_src}")
    print(f"Train pairs     : {len(train_cxr):,}  |  Val CXR: {len(val_cxr):,}  |  Val EHR: {len(val_ehr):,}")
    print(f"EHR feature dim : {len(feature_cols)}")

    train_tfm = get_cxr_transforms("train", backbone="xrv")
    val_tfm   = get_cxr_transforms("val",   backbone="xrv")

    train_ds   = PairedCxrEhrDataset(
        train_cxr, ehr_features, feature_cols, args.image_path_column, train_tfm
    )
    val_cxr_ds = CxrEvalDataset(val_cxr, args.image_path_column, val_tfm)
    val_ehr_ds = EhrEvalDataset(val_ehr, feature_cols)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True, collate_fn=collate_train,
    )
    val_cxr_loader = DataLoader(val_cxr_ds, batch_size=cfg.eval_cxr_batch_size,
                                shuffle=False, num_workers=cfg.num_workers)
    val_ehr_loader = DataLoader(val_ehr_ds, batch_size=cfg.eval_ehr_batch_size,
                                shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DualEncoder(
        ehr_input_dim    = len(feature_cols),
        embed_dim        = cfg.embed_dim,
        pretrained       = True,
        ehr_encoder_type = cfg.ehr_encoder_type,
    ).to(device)

    # Log param counts for the EHR encoder specifically
    ehr_params = sum(p.numel() for p in model.ehr_encoder.parameters())
    cxr_params = sum(p.numel() for p in model.cxr_encoder.parameters())
    print(f"\nDevice          : {device}")
    print(f"EHR encoder params : {ehr_params:,}")
    print(f"CXR encoder params : {cxr_params:,}  (frozen backbone)")
    print(f"Steps/epoch     : {len(train_loader)}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler       = GradScaler(enabled=device.type == "cuda")
    metrics_path = output_dir / "metrics.csv"

    # Cosine annealing: decays lr -> eta_min over T_max epochs.
    # T_max = total epochs so the schedule spans the full training run.
    # Cosine LR can make convergence smoother than a flat learning rate.
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min
        )
        print(f"LR schedule     : cosine  {cfg.lr:.0e} -> {args.eta_min:.0e}  over {args.epochs} epochs")
    else:
        scheduler = None

    # -- Resume from last.pt ------------------------------------------------
    start_epoch = 0
    history: list[dict] = []
    best_lift50 = -math.inf

    if args.resume:
        last_ckpt = ckpt_dir / "last.pt"
        if not last_ckpt.exists():
            print(f"  [WARN] --resume set but no last.pt found at {last_ckpt}. Starting from scratch.")
        else:
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"  Resumed from last.pt  (epoch {start_epoch})")
            # Restore best_lift50 from metrics.csv if it exists
            if metrics_path.exists():
                prev = pd.read_csv(metrics_path)
                history = prev.to_dict("records")
                if "val_lift@50" in prev.columns:
                    best_lift50 = float(prev["val_lift@50"].max())
                    print(f"  Restored best lift@50={best_lift50:.2f}x from metrics.csv")
            if start_epoch >= args.epochs:
                print(f"  [WARN] last.pt epoch ({start_epoch}) >= --epochs ({args.epochs}). "
                      f"Nothing to do. Increase --epochs.")
                return

    config_dump = {
        **asdict(cfg),
        "project_root":      str(project_root),
        "processed_dir":     str(processed_dir),
        "cxr_samples_csv":   str(cxr_path),
        "image_path_column": args.image_path_column,
        "id_column": args.id_column,
        "split_source":      split_src,
        "ehr_transform":     ehr_src,
        "ehr_feature_cols":  feature_cols,
        "device":            str(device),
        "ehr_encoder_params": ehr_params,
        "baseline_reference": "Compare FT-Transformer against MLP encoder",
        "cosine_lr":         args.cosine_lr,
        "eta_min":           args.eta_min,
    }
    for d in [output_dir, ckpt_dir]:
        with open(d / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dump, f, indent=2)

    print(f"\n{'='*60}")
    if start_epoch == 0:
        print("Epoch 0 -- random init baseline")
    else:
        print(f"Resuming from epoch {start_epoch} -> training to epoch {args.epochs}")
    if start_epoch == 0:
        vm0 = evaluate_retrieval(model, val_cxr_loader, val_ehr_loader, cfg.k_values, device)
        print(
            f"  lift@50={vm0['lift@50']:.2f}x  lift@10={vm0['lift@10']:.2f}x  "
            f"  lift@1={vm0['lift@1']:.2f}x  pos_sim={vm0['pos_sim_mean']:.4f}"
        )
    print("Baseline reference: compare against --ehr-encoder mlp")
    print(f"{'='*60}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg.temperature, cfg.grad_clip)
        vm = evaluate_retrieval(model, val_cxr_loader, val_ehr_loader, cfg.k_values, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": float(epoch), "train_loss": tr["loss"],
            "train_pos_sim": tr["pos_sim"], "lr": current_lr,
            **{f"val_{k}": v for k, v in vm.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False)

        print(
            f"  loss={tr['loss']:.4f}  pos_sim={tr['pos_sim']:.4f}  "
            f"lift@50={vm['lift@50']:.2f}x  lift@10={vm['lift@10']:.2f}x  "
            f"lift@1={vm['lift@1']:.2f}x  lr={current_lr:.2e}"
        )

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, row, config_dump)

        if vm["lift@50"] > best_lift50:
            best_lift50 = vm["lift@50"]
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, row, config_dump)
            print(f"  -> New best  lift@50={best_lift50:.2f}x")

    print(f"\nDone.")
    print(f"Best lift@50    : {best_lift50:.2f}x")
    print("Baseline reference: compare against the MLP encoder run.")
    print(f"Metrics         : {metrics_path}")
    print(f"Best checkpoint : {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
