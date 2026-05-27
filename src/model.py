"""
model.py

FTTransformerEHREncoder:
    Feature Tokenizer + Transformer for tabular EHR data.
    Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021).
    Drop-in replacement for EHREncoder in DualEncoder.
    See FTTransformerEHREncoder docstring for architecture and usage notes.

model.py

Dual-encoder architecture for cross-modal CXR ↔ EHR contrastive learning.

CXR encoder:
    DenseNet-121 pretrained on chest X-ray data via torchxrayvision
    → feature extractor output (B, 1024, 7, 7)
    → global average pool → (B, 1024)
    → ProjectionHead(1024, hidden=512, out=128) → L2 normalize

EHR encoder:
    MLP(input_dim → 256) → ProjectionHead(256, hidden=256, out=128) → L2 normalize

Both encoders produce L2-normalized 128-dim embeddings.
Cosine similarity is then a dot product between embeddings ∈ [-1, 1].

──────────────────────────────────────────────────────────────────────────────
CXR encoder:
    DenseNet-121 pretrained on chest X-ray data via torchxrayvision
    -> feature extractor -> global average pooling -> projection head

EHR encoder:
    MLP or FT-Transformer encoder for transformed tabular features

Both encoders produce L2-normalized embeddings for contrastive retrieval.


Input normalization contract:
    The transform pipeline (cxr_transforms.py with backbone="xrv") maps
    [0, 1] PIL pixel values → [-1024, 1024] via (2·x - 1)·1024.
    This matches the Hounsfield-equivalent scale used during MIMIC-CXR
    pretraining in torchxrayvision.
    CXR images stored as RGB have identical channels (grayscale stored as RGB).
    CXREncoder.forward averages the 3 channels to 1 before passing to the
    single-channel DenseNet backbone.

Usage:
    model = DualEncoder(ehr_input_dim=29)

    cxr_embed = model.encode_cxr(cxr_tensor)    # (B, 128)
    ehr_embed = model.encode_ehr(ehr_tensor)    # (B, 128)
    sim = cxr_embed @ ehr_embed.T               # (B, B) cosine similarity matrix
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head, following SimCLR / CLIP conventions.
    Input → Linear(hidden) → LayerNorm → ReLU → [Dropout] → Linear(out_dim)

    Uses LayerNorm instead of BatchNorm1d so the head works at any batch size,
    including batch_size=1 at inference time.

    dropout > 0 inserts a Dropout layer between ReLU and the final linear.
    Use for the CXR projection head (default 0.1) — the EHR encoder trunk
    already has Dropout(0.3) so its projection head uses dropout=0.0.

    Output is NOT normalized here — normalization happens in the encoder.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Text projection head (ClinicalBERT → embedding space)
# ---------------------------------------------------------------------------

class TextProjectionHead(nn.Module):
    # Projects pre-computed ClinicalBERT CLS embeddings (dim=768) into the
    # shared contrastive space (dim=embed_dim, default 128).
    #
    # Architecture: Linear(768→hidden) → LayerNorm → ReLU → Linear(hidden→out)
    # No dropout — ClinicalBERT embeddings are already regularized (frozen
    # encoder, deterministic CLS pooling). Adding dropout here hurts alignment.
    #
    # Output is L2-normalized so cosine similarity == dot product in the loss.
    #
    # This module is intentionally shallow (2 linear layers).
    # ClinicalBERT's 768-dim CLS already encodes rich clinical semantics.
    # Deep projection on top of a frozen encoder typically degrades performance
    # (Wang & Isola 2020, CLIP ablations). The role of this head is purely
    # dimensional: 768 → embed_dim, not semantic transformation.
    def __init__(
        self,
        in_dim: int = 768,       # ClinicalBERT hidden size
        hidden_dim: int = 256,
        out_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 768) float32
        # returns: (B, out_dim) L2-normalized
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# CXR encoder
# ---------------------------------------------------------------------------

# torchxrayvision weights available for the CXR backbone.
# "densenet121-res224-mimic_nb" = trained on MIMIC-CXR No-Finding + CheXpert labels
# "densenet121-res224-all"      = trained on all TXV datasets combined
# "densenet121-res224-chex"     = trained on CheXpert only
XRV_WEIGHTS_DEFAULT = "densenet121-res224-mimic_nb"


class CXREncoder(nn.Module):
    """
    DenseNet-121 backbone pretrained on MIMIC-CXR via torchxrayvision.

    The final classifier head is discarded. Only the DenseNet feature extractor
    (.features) is retained, producing a (B, 1024, 7, 7) spatial feature map
    that is pooled to (B, 1024) before the projection head.

    Input contract:
        x: (B, 3, 224, 224) — RGB tensor where all channels are identical
           (grayscale CXR stored as RGB). Pixel values in [-1024, 1024]
           (torchxrayvision normalization scale).
           The forward pass averages channels to (B, 1, 224, 224) before
           passing to the single-channel DenseNet backbone.
    """
    def __init__(
        self,
        pretrained: bool = True,
        xrv_weights: str = XRV_WEIGHTS_DEFAULT,
        proj_hidden: int = 512,
        embed_dim: int = 128,
    ):
        super().__init__()
        try:
            # torchxrayvision's jfhealthcare baseline model contains a
            # broken absolute import:
            #   from model.utils import get_norm   (vgg.py, line 3)
            # It expects a `model` *package* (directory) but our scripts/
            # directory contains model.py (a module), so Python raises:
            #   ModuleNotFoundError: 'model' is not a package
            #
            # Fix: inject a stub model.utils into sys.modules so the broken
            # import resolves to a no-op instead of crashing. The jfhealthcare
            # model is an optional baseline we never use; the stub is harmless.
            import sys as _sys, types as _types
            if "model.utils" not in _sys.modules:
                _stub_utils = _types.ModuleType("model.utils")
                _stub_utils.get_norm = lambda *a, **kw: None   # never called
                _sys.modules["model.utils"] = _stub_utils
                # also attach as attribute so `import model; model.utils` works
                _sys.modules[__name__].utils = _stub_utils

            import torchxrayvision as xrv
        except ImportError:
            raise ImportError(
                "torchxrayvision is required for the CXR backbone.\n"
                "Install with:  pip install torchxrayvision"
            )

        if pretrained:
            densenet = xrv.models.DenseNet(weights=xrv_weights)
        else:
            # No pretrained weights — used when restoring from a training
            # checkpoint (load_state_dict overwrites these random init weights).
            densenet = xrv.models.DenseNet(weights=None)

        # Discard the pathology classifier head; keep the feature extractor.
        # densenet.features: (B, 1, H, W) → (B, 1024, H/32, W/32)
        self.backbone = densenet.features
        backbone_out_dim = 1024

        # dropout=0.1: regularises the projection head during noisy pseudo-pair
        # training. If freeze_cxr_proj=True in CONFIG this head is excluded from
        # the optimizer entirely and dropout has no effect (eval mode = pass-through).
        self.proj = ProjectionHead(
            backbone_out_dim,
            hidden_dim=proj_hidden,
            out_dim=embed_dim,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float tensor in [-1024, 1024]
               CXR images are grayscale stored as RGB — all 3 channels identical.

        Returns:
            embed: (B, embed_dim) L2-normalized embedding
        """
        # Grayscale conversion: average identical RGB channels → single channel
        x = x.mean(dim=1, keepdim=True)         # (B, 1, H, W)

        feat = self.backbone(x)                   # (B, 1024, H', W')
        feat = F.adaptive_avg_pool2d(feat, 1)     # (B, 1024, 1, 1)
        feat = feat.flatten(1)                    # (B, 1024)
        proj = self.proj(feat)                    # (B, embed_dim)
        return F.normalize(proj, dim=1)           # L2 normalize


# ---------------------------------------------------------------------------
# EHR encoder
# ---------------------------------------------------------------------------

class EHREncoder(nn.Module):
    """
    Tabular MLP encoder for the 29-dim transformed EHR feature vector.

    Architecture:
        Linear(input_dim → 256) → LayerNorm → ReLU → Dropout(0.3)
        → Linear(256 → 256) → LayerNorm → ReLU → Dropout(0.3)
        → ProjectionHead(256, hidden=256, out=embed_dim)
        → L2 normalize

    Two hidden layers are sufficient for this feature count.
    LayerNorm handles the mix of z-scored numerics, 0/1 binaries,
    and one-hot categoricals without requiring separate normalization.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        proj_hidden: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.proj = ProjectionHead(hidden_dim, hidden_dim=proj_hidden, out_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) float tensor

        Returns:
            embed: (B, embed_dim) L2-normalized embedding
        """
        feat = self.trunk(x)             # (B, hidden_dim)
        proj = self.proj(feat)           # (B, embed_dim)
        return F.normalize(proj, dim=1)


# ---------------------------------------------------------------------------
# FTTransformer EHR encoder
# ---------------------------------------------------------------------------

class FTTransformerEHREncoder(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular EHR data.
    Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021).

    Drop-in replacement for EHREncoder. Output contract is identical:
        (B, embed_dim) L2-normalized embedding.

    Architecture:
        Feature Tokenizer:
            Each of the n_features scalar inputs gets its own learnable weight
            vector and bias vector, mapping it to a d_token-dim token:
                T[:, j, :] = x[:, j:j+1] * W[j] + b[j]
            where W ∈ R^{n_features × d_token}, b ∈ R^{n_features × d_token}.

            All 29 EHR features are treated as continuous — they are already
            fully numeric after EHRTransformer preprocessing (z-scored numerics,
            binary 0/1, one-hot category indicators).

        CLS token prepended → (B, n_features+1, d_token)

        Pre-LayerNorm Transformer blocks × n_layers
            Pre-LN (norm_first=True) is more stable than Post-LN at small d_token
            and does not require learning-rate warmup.

        CLS output → LayerNorm → Linear(d_token, embed_dim) → L2 normalize

    Size configs:
        Small (default):  d_token=64,  n_heads=4, n_layers=2, ffn_hidden=128  (~78K params)
        Medium:           d_token=128, n_heads=8, n_layers=3, ffn_hidden=256  (~390K params)

    Compared to EHREncoder (MLP, ~170K params), the Small config is leaner.
    Start with Small — with 46K training pairs and 29 features, the Medium config
    risks overfitting before a meaningful comparison can be made.

    Requires PyTorch >= 1.11 (norm_first parameter in TransformerEncoderLayer).
    """

    def __init__(
        self,
        n_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_hidden: int = 128,
        dropout: float = 0.1,
        embed_dim: int = 128,
    ):
        super().__init__()

        if d_token % n_heads != 0:
            raise ValueError(
                f"d_token ({d_token}) must be divisible by n_heads ({n_heads})."
            )

        # ── Feature tokenizer ──────────────────────────────────────────────
        # Per-feature learnable weight W[j] and bias b[j], each of size d_token.
        # Initialized with trunc-normal (std=0.02) matching BERT-style init —
        # safer than kaiming for scalar-input projections where fan_in=1.
        self.token_weight = nn.Parameter(torch.empty(n_features, d_token))
        self.token_bias   = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.trunc_normal_(self.token_weight, std=0.02)

        # ── CLS token ──────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Transformer encoder (Pre-LN) ───────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_token,
            nhead          = n_heads,
            dim_feedforward = ffn_hidden,
            dropout        = dropout,
            activation     = "relu",
            norm_first     = True,   # Pre-LN for stability
            batch_first    = True,   # (B, seq, d_model) convention
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Output projection ──────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(d_token)
        self.proj     = nn.Linear(d_token, embed_dim, bias=False)

        # Store for repr / param count logging
        self.n_features = n_features
        self.d_token    = d_token
        self.n_heads    = n_heads
        self.n_layers   = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_features) float tensor — all features continuous after preprocessing.

        Returns:
            embed: (B, embed_dim) L2-normalized embedding.
        """
        B = x.size(0)

        # ── Tokenize: (B, n_features) → (B, n_features, d_token) ──────────
        # Broadcast: x is (B, n_features, 1), token_weight is (1, n_features, d_token)
        tokens = x.unsqueeze(-1) * self.token_weight.unsqueeze(0) \
                 + self.token_bias.unsqueeze(0)                     # (B, F, d_token)

        # ── Prepend CLS token → (B, F+1, d_token) ─────────────────────────
        cls    = self.cls_token.expand(B, -1, -1)                   # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)                    # (B, F+1, d_token)

        # ── Transformer ────────────────────────────────────────────────────
        out = self.transformer(tokens)                              # (B, F+1, d_token)

        # ── CLS output → project → L2 normalize ───────────────────────────
        cls_out = self.out_norm(out[:, 0, :])                       # (B, d_token)
        proj    = self.proj(cls_out)                                # (B, embed_dim)
        return F.normalize(proj, dim=1)


# ---------------------------------------------------------------------------
# Combined dual encoder
# ---------------------------------------------------------------------------

class DualEncoder(nn.Module):
    # Wraps CXREncoder, EHREncoder, and optional text projection heads.
    # and two TextProjectionHeads (ClinicalBERT 768 -> embed_dim).
    #
    # Optional text-pivot forward returns 4 embeddings:
    #   z_cxr     (B, embed_dim) -- CXR image → DenseNet → projection → L2-norm
    #   z_report  (B, embed_dim) -- CXR radiology report → ClinicalBERT CLS → text proj → L2-norm
    #   z_ehr     (B, embed_dim) -- EHR features → MLP → projection → L2-norm
    #   z_summary (B, embed_dim) -- EHR LLM summary → ClinicalBERT CLS → text proj → L2-norm
    #
    # Args:
    #   ehr_input_dim:    number of EHR feature columns (e.g., 29)
    #   embed_dim:        shared contrastive embedding dimension (default 128)
    #   pretrained:       load torchxrayvision chest X-ray weights for CXR backbone.
    #                     Set False when restoring from a full training checkpoint.
    #   xrv_weights:      torchxrayvision weight string (default: mimic_nb).
    #   text_hidden:      hidden dim for TextProjectionHead (default 256).
    #                     ClinicalBERT is always frozen (embeddings pre-computed offline).
    #   ehr_encoder_type: "mlp" (baseline MLP encoder) or "ftt" (FTTransformer).
    #                     "mlp"  → EHREncoder (3-layer MLP, ~170K params)
    #                     "ftt"  → FTTransformerEHREncoder (d_token=64, 2 layers, ~78K params)
    #                     NOTE: "ftt" and "mlp" checkpoints are NOT compatible —
    #                     state_dict keys differ. Do not load an MLP checkpoint into
    #                     a FTT model or vice versa.
    def __init__(
        self,
        ehr_input_dim: int,
        embed_dim: int = 128,
        pretrained: bool = True,
        xrv_weights: str = XRV_WEIGHTS_DEFAULT,
        text_hidden: int = 256,
        ehr_encoder_type: str = "mlp",
    ):
        super().__init__()
        self.cxr_encoder = CXREncoder(
            pretrained  = pretrained,
            xrv_weights = xrv_weights,
            proj_hidden = 512,
            embed_dim   = embed_dim,
        )
        if ehr_encoder_type == "ftt":
            self.ehr_encoder = FTTransformerEHREncoder(
                n_features = ehr_input_dim,
                d_token    = 64,
                n_heads    = 4,
                n_layers   = 2,
                ffn_hidden = 128,
                dropout    = 0.1,
                embed_dim  = embed_dim,
            )
        elif ehr_encoder_type == "mlp":
            self.ehr_encoder = EHREncoder(
                input_dim   = ehr_input_dim,
                hidden_dim  = 256,
                proj_hidden = 256,
                embed_dim   = embed_dim,
            )
        else:
            raise ValueError(
                f"Unknown ehr_encoder_type '{ehr_encoder_type}'. "
                f"Choose 'mlp' or 'ftt'."
            )
        # Text towers: project pre-computed ClinicalBERT CLS (768) → embed_dim.
        # One head per modality so each can specialize to its text domain
        # (radiology report language vs. EHR template language).
        self.report_proj  = TextProjectionHead(768, text_hidden, embed_dim)
        self.summary_proj = TextProjectionHead(768, text_hidden, embed_dim)

    def encode_cxr(self, cxr: torch.Tensor) -> torch.Tensor:
        return self.cxr_encoder(cxr)

    def encode_ehr(self, ehr: torch.Tensor) -> torch.Tensor:
        return self.ehr_encoder(ehr)

    def forward(
        self,
        cxr: torch.Tensor,
        ehr: torch.Tensor,
        report_emb:  torch.Tensor | None = None,
        summary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        # Text-pivot mode: all 4 args provided.
        #   Returns (z_cxr, z_report, z_ehr, z_summary) -- 4-tuple.
        # Standard image-EHR mode: report_emb and summary_emb are None.
        #   Returns (z_cxr, z_ehr) -- 2-tuple (backward compat with InfoNCELoss).
        z_cxr = self.encode_cxr(cxr)
        z_ehr = self.encode_ehr(ehr)
        if report_emb is not None and summary_emb is not None:
            z_report  = self.report_proj(report_emb)    # (B, embed_dim) L2-normed
            z_summary = self.summary_proj(summary_emb)  # (B, embed_dim) L2-normed
            return z_cxr, z_report, z_ehr, z_summary
        return z_cxr, z_ehr

    def freeze_backbone(self) -> None:
        """Freeze DenseNet-121 backbone weights (train projection head only)."""
        for param in self.cxr_encoder.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze DenseNet-121 backbone for end-to-end fine-tuning."""
        for param in self.cxr_encoder.backbone.parameters():
            param.requires_grad = True