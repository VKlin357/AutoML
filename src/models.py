"""
Model zoo for tabular NAS.

All models share the same forward signature:
    model(x_num: [B, F_num], x_cat: [B, F_cat])  ->  logits

and the same constructor protocol via ``make_model(arch_cfg, ...)``.

The five families (mlp, resmlp, ft_transformer, gated_tab, autoint) cover
the bulk of the design space published in tabular DL literature, while
staying small enough that all of them fit in this file.

Implementation notes:

- For MLP / ResMLP / GatedTab the categoricals go through learned
  embeddings (per-column dim ~ card^0.25), then are concatenated with
  the numeric features.

- For FT-Transformer the inputs are *tokenized*: every numeric feature
  becomes a token of dim d_token via a per-feature scale+shift, and every
  categorical feature gets a per-column lookup table of size [card, d_token].
  A learnable [CLS] token is prepended; its final representation is used
  for the prediction head. This is the original FT-Transformer recipe
  (Gorishniy et al., NeurIPS 2021), simplified.

- AutoInt is a lightweight feature-interaction self-attention model
  (Song et al., CIKM 2019) using the same tokenizer as FT-Transformer
  but pooling tokens by mean for the head.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _make_act(name: str) -> nn.Module:
    name = (name or "relu").lower()
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
    }.get(name, nn.ReLU())


def _make_norm(name: str, dim: int) -> nn.Module:
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    return nn.Identity()


def _emb_dim(card: int, cap: int) -> int:
    """Per-column embedding dim: ~card^0.25 * 4, capped by cap."""
    return min(cap, max(2, int(round(card ** 0.25 * 4))))


class _CatEmbeddings(nn.Module):
    """Concatenated per-column embeddings -> [B, sum(emb_dims)]."""

    def __init__(self, cat_cardinalities: List[int], embedding_dim_cap: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, _emb_dim(card, embedding_dim_cap)) for card in cat_cardinalities]
        )
        self.total_dim = sum(emb.embedding_dim for emb in self.embeddings)
        self.n_cat = len(cat_cardinalities)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if self.n_cat == 0:
            return x_cat.new_zeros((x_cat.shape[0], 0), dtype=torch.float32)
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)


# ---------------------------------------------------------------------------
# 1) TabularMLP
# ---------------------------------------------------------------------------

class TabularMLP(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        out_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        dropout: float = 0.1,
        normalization: str = "batchnorm",
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.cat_emb = _CatEmbeddings(cat_cardinalities, embedding_dim)
        in_dim = n_num + self.cat_emb.total_dim
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(_make_norm(normalization, h))
            layers.append(_make_act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.cat_emb(x_cat)], dim=1) if self.cat_emb.n_cat else x_num
        return self.net(x)


# ---------------------------------------------------------------------------
# 2) ResMLP — residual MLP with pre-norm
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, dim: int, activation: str, dropout: float, normalization: str):
        super().__init__()
        self.norm = _make_norm(normalization, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = _make_act(activation)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class ResMLP(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        out_dim: int,
        n_blocks: int,
        block_width: int,
        activation: str,
        dropout: float,
        normalization: str,
        embedding_dim: int,
    ):
        super().__init__()
        self.cat_emb = _CatEmbeddings(cat_cardinalities, embedding_dim)
        in_dim = n_num + self.cat_emb.total_dim
        self.input_proj = nn.Linear(in_dim, block_width)
        self.blocks = nn.ModuleList(
            [_ResBlock(block_width, activation, dropout, normalization) for _ in range(n_blocks)]
        )
        self.out_norm = _make_norm(normalization, block_width)
        self.head = nn.Linear(block_width, out_dim)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.cat_emb(x_cat)], dim=1) if self.cat_emb.n_cat else x_num
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# 3) GatedTab — GLU-style gated MLP (a la Gated Linear Units)
# ---------------------------------------------------------------------------

class _GatedBlock(nn.Module):
    def __init__(self, dim: int, activation: str, dropout: float, normalization: str):
        super().__init__()
        self.norm = _make_norm(normalization, dim)
        self.fc_value = nn.Linear(dim, dim)
        self.fc_gate = nn.Linear(dim, dim)
        self.act = _make_act(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.norm(x)
        v = self.act(self.fc_value(h))
        g = torch.sigmoid(self.fc_gate(h))
        h = self.dropout(v * g)
        h = self.proj(h)
        return x + h


class GatedTab(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        out_dim: int,
        n_blocks: int,
        block_width: int,
        activation: str,
        dropout: float,
        normalization: str,
        embedding_dim: int,
    ):
        super().__init__()
        self.cat_emb = _CatEmbeddings(cat_cardinalities, embedding_dim)
        in_dim = n_num + self.cat_emb.total_dim
        self.input_proj = nn.Linear(in_dim, block_width)
        self.blocks = nn.ModuleList(
            [_GatedBlock(block_width, activation, dropout, normalization) for _ in range(n_blocks)]
        )
        self.out_norm = _make_norm(normalization, block_width)
        self.head = nn.Linear(block_width, out_dim)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.cat_emb(x_cat)], dim=1) if self.cat_emb.n_cat else x_num
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_norm(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Tokenizer for transformer-style models
# ---------------------------------------------------------------------------

class _FeatureTokenizer(nn.Module):
    """Per-feature tokenization to dim d_token.

    Numeric features:  x_i  ->  x_i * w_i + b_i,  with w_i, b_i in R^{d_token}
    Categorical:       x_i  ->  Embedding[card_i, d_token]
    Output:            [B, n_num + n_cat, d_token]
    """

    def __init__(self, n_num: int, cat_cardinalities: List[int], d_token: int):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token
        if n_num > 0:
            self.num_w = nn.Parameter(torch.empty(n_num, d_token))
            self.num_b = nn.Parameter(torch.empty(n_num, d_token))
            nn.init.kaiming_uniform_(self.num_w, a=math.sqrt(5))
            nn.init.zeros_(self.num_b)
        else:
            self.register_parameter("num_w", None)
            self.register_parameter("num_b", None)
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        toks = []
        if self.n_num:
            # x_num: [B, n_num] -> [B, n_num, d_token]
            toks.append(x_num.unsqueeze(-1) * self.num_w + self.num_b)
        if self.n_cat:
            cat_toks = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_emb)],
                dim=1,
            )
            toks.append(cat_toks)
        return torch.cat(toks, dim=1) if toks else x_num.new_zeros((x_num.shape[0], 0, self.d_token))


# ---------------------------------------------------------------------------
# 4) FT-Transformer
# ---------------------------------------------------------------------------

class _FTBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, ffn_factor: float,
                 attn_dropout: float, ffn_dropout: float, residual_dropout: float,
                 activation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_dropout, batch_first=True)
        self.res_drop1 = nn.Dropout(residual_dropout)
        ffn_hidden = int(round(d_token * ffn_factor))
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_hidden),
            _make_act(activation),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden, d_token),
        )
        self.res_drop2 = nn.Dropout(residual_dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.res_drop1(a)
        h = self.norm2(x)
        f = self.ffn(h)
        x = x + self.res_drop2(f)
        return x


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        out_dim: int,
        n_blocks: int,
        d_token: int,
        n_heads: int,
        ffn_factor: float,
        attn_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
    ):
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_num, cat_cardinalities, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([
            _FTBlock(d_token, n_heads, ffn_factor, attn_dropout, ffn_dropout,
                     residual_dropout, activation)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, out_dim)

    def forward(self, x_num, x_cat):
        toks = self.tokenizer(x_num, x_cat)            # [B, T, d]
        cls = self.cls.expand(toks.shape[0], -1, -1)   # [B, 1, d]
        x = torch.cat([cls, toks], dim=1)
        for blk in self.blocks:
            x = blk(x)
        h = self.norm(x[:, 0])
        return self.head(h)


# ---------------------------------------------------------------------------
# 5) AutoInt (lightweight)
# ---------------------------------------------------------------------------

class _AutoIntBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, attn_dropout: float,
                 ffn_factor: float, ffn_dropout: float, activation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_dropout, batch_first=True)
        ffn_hidden = int(round(d_token * ffn_factor))
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_hidden),
            _make_act(activation),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden, d_token),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        h = self.norm2(x)
        return x + self.ffn(h)


class AutoInt(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        out_dim: int,
        n_blocks: int,
        d_token: int,
        n_heads: int,
        attn_dropout: float,
        ffn_factor: float,
        ffn_dropout: float,
        activation: str,
    ):
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_num, cat_cardinalities, d_token)
        self.blocks = nn.ModuleList([
            _AutoIntBlock(d_token, n_heads, attn_dropout, ffn_factor, ffn_dropout, activation)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, out_dim)

    def forward(self, x_num, x_cat):
        toks = self.tokenizer(x_num, x_cat)   # [B, T, d]
        for blk in self.blocks:
            toks = blk(toks)
        toks = self.norm(toks)
        h = toks.mean(dim=1)
        return self.head(h)


# ---------------------------------------------------------------------------
# 6) TabM — shared trunk + K independent heads (parameter-efficient ensemble)
#
# Each sample goes through one shared MLP trunk, then K parallel heads vote.
# Averaging K predictions improves calibration and reduces variance without
# the cost of training K separate models.
# Reference: Gorishniy et al. "TabM: Advancing Tabular Deep Learning
# Insights and Solutions", 2024.
# ---------------------------------------------------------------------------

class TabM(nn.Module):
    """Shared trunk + K independent prediction heads (internal ensemble).

    Each head receives a slightly different view of the trunk output via
    per-head affine perturbation (learnable scale + bias) and head-specific
    dropout. This gives real diversity between heads while keeping the trunk
    shared for efficiency — inspired by the BatchEnsemble / TabM paper.

    Architecture:
        trunk: n_blocks × (Linear → Norm → Act → Dropout)
        head_i: affine(h) → Linear(width, out_dim)
        output: mean over K head logits
    """
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        out_dim: int,
        n_blocks: int = 3,
        width: int = 256,
        k: int = 8,
        activation: str = "gelu",
        dropout: float = 0.15,
        head_dropout: float = 0.10,
        normalization: str = "layernorm",
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.k = k
        self.head_dropout = head_dropout
        self.cat_emb = _CatEmbeddings(cat_cardinalities, embedding_dim)
        in_dim = n_num + self.cat_emb.total_dim

        # Shared trunk
        trunk_layers: List[nn.Module] = []
        prev = in_dim
        for _ in range(n_blocks):
            trunk_layers.append(nn.Linear(prev, width))
            trunk_layers.append(_make_norm(normalization, width))
            trunk_layers.append(_make_act(activation))
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
            prev = width
        self.trunk = nn.Sequential(*trunk_layers)

        # Per-head affine perturbation: each head learns its own scale and bias
        # applied to the trunk output before the linear classifier.
        # Initialised to identity (scale=1, bias=0) so training starts stable.
        self.head_scale = nn.Parameter(torch.ones(k, width))
        self.head_bias  = nn.Parameter(torch.zeros(k, width))

        # K independent prediction heads
        self.heads = nn.ModuleList([nn.Linear(width, out_dim) for _ in range(k)])

    def forward(self, x_num, x_cat):
        import torch.nn.functional as F
        x = torch.cat([x_num, self.cat_emb(x_cat)], dim=1) if self.cat_emb.n_cat else x_num
        h = self.trunk(x)                                        # [B, width]

        logits_list = []
        for i, head in enumerate(self.heads):
            # Affine perturbation makes each head see a slightly rotated representation
            h_i = h * self.head_scale[i] + self.head_bias[i]   # [B, width]
            # Head-specific dropout for additional diversity at training time
            h_i = F.dropout(h_i, p=self.head_dropout, training=self.training)
            logits_list.append(head(h_i))                        # [B, out_dim]

        logits = torch.stack(logits_list, dim=1)                 # [B, K, out_dim]
        return logits.mean(dim=1)                                # [B, out_dim]


# ---------------------------------------------------------------------------
# Factory + convenience
# ---------------------------------------------------------------------------

def output_dim_for_task(task: str, n_classes: int) -> int:
    if task in ("binary", "regression"):
        return 1
    return n_classes  # multiclass


def make_model(
    arch_cfg: Dict,
    *,
    n_num: int,
    cat_cardinalities: List[int],
    n_classes: int,
    task: str,
) -> nn.Module:
    out_dim = output_dim_for_task(task, n_classes)
    fam = arch_cfg["family"]
    if fam == "mlp":
        return TabularMLP(
            n_num=n_num, cat_cardinalities=cat_cardinalities, out_dim=out_dim,
            hidden_dims=arch_cfg["hidden_dims"], activation=arch_cfg["activation"],
            dropout=arch_cfg["dropout"], normalization=arch_cfg["normalization"],
            embedding_dim=arch_cfg["embedding_dim"],
        )
    if fam == "resmlp":
        return ResMLP(
            n_num=n_num, cat_cardinalities=cat_cardinalities, out_dim=out_dim,
            n_blocks=arch_cfg["n_blocks"], block_width=arch_cfg["block_width"],
            activation=arch_cfg["activation"], dropout=arch_cfg["dropout"],
            normalization=arch_cfg["normalization"], embedding_dim=arch_cfg["embedding_dim"],
        )
    if fam == "gated_tab":
        return GatedTab(
            n_num=n_num, cat_cardinalities=cat_cardinalities, out_dim=out_dim,
            n_blocks=arch_cfg["n_blocks"], block_width=arch_cfg["block_width"],
            activation=arch_cfg["activation"], dropout=arch_cfg["dropout"],
            normalization=arch_cfg["normalization"], embedding_dim=arch_cfg["embedding_dim"],
        )
    if fam == "ft_transformer":
        return FTTransformer(
            n_num=n_num, cat_cardinalities=cat_cardinalities, out_dim=out_dim,
            n_blocks=arch_cfg["n_blocks"], d_token=arch_cfg["d_token"],
            n_heads=arch_cfg["n_heads"], ffn_factor=arch_cfg["ffn_factor"],
            attn_dropout=arch_cfg["attn_dropout"], ffn_dropout=arch_cfg["ffn_dropout"],
            residual_dropout=arch_cfg["residual_dropout"], activation=arch_cfg["activation"],
        )
    if fam == "autoint":
        return AutoInt(
            n_num=n_num, cat_cardinalities=cat_cardinalities, out_dim=out_dim,
            n_blocks=arch_cfg["n_blocks"], d_token=arch_cfg["d_token"],
            n_heads=arch_cfg["n_heads"], attn_dropout=arch_cfg["attn_dropout"],
            ffn_factor=arch_cfg["ffn_factor"], ffn_dropout=arch_cfg["ffn_dropout"],
            activation=arch_cfg["activation"],
        )
    if fam == "tabm":
        return TabM(
            n_num=n_num, cat_cardinalities=cat_cardinalities, out_dim=out_dim,
            n_blocks=arch_cfg["n_blocks"], width=arch_cfg["width"],
            k=arch_cfg.get("k", 8),
            activation=arch_cfg["activation"], dropout=arch_cfg["dropout"],
            head_dropout=arch_cfg.get("head_dropout", 0.10),
            normalization=arch_cfg["normalization"], embedding_dim=arch_cfg["embedding_dim"],
        )
    raise ValueError(f"Unknown arch family: {fam}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
