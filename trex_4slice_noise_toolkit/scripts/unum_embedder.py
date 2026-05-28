#!/usr/bin/env python3
"""UNUM-style latent network-state embedder for the proxy QoS testbed.

This is a deliberately small but faithful adaptation of the UNUM idea
(Chen et al., NSDI'26) to the four-slice / shared-uplink proxy setting:

  * Tokenizer + quantile **bucketization** of *observable* signals (no raw
    floats reach the controller, and the hidden noise / capacity are NEVER
    features).
  * An encoder-decoder **Transformer** with causal attention trained by
    **windowed next-token prediction** (predict a *window* of future tokens,
    not a single step) using a **per-feature classification head**.
  * The encoder output is the **embedding** the DQN consumes; the decoder's
    future-bucket predictions drive an optional **guardrail**.

Separation principle: the embedder is pretrained *offline* on diverse traffic
and policies (see pretrain_embedder.py), then used **frozen** by the online
controller. Its training objective (predict the future) is independent of the
control reward.

The token feature set is the single source of truth shared by pretraining and
online inference (`OBS_FEATURES` / `build_token`). All of these are quantities
the agent already observes in the *raw* baseline (offered visible loads + the
previous step's proxy consequences), so the comparison stays apples-to-apples:
only the *representation* changes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "unum_embedder requires PyTorch. Install with:\n"
        "  python3 -m pip install torch\n"
        f"Original error: {exc}"
    )


# ----------------------------------------------------------------------------
# Token definition (single source of truth)
# ----------------------------------------------------------------------------

# Observable signals only. Order is fixed and must match between pretrain and
# online inference. The hidden noise offered load and the latent capacity are
# intentionally absent.
OBS_FEATURES: Tuple[str, ...] = (
    "offered_gold_mbps",
    "offered_silver_mbps",
    "offered_bronze_mbps",
    "uplink_pressure",
    "uplink_utilization",
    "gold_shortage_mbps",
    "available_for_gold_mbps",
)
D = len(OBS_FEATURES)


def build_token(observable: Dict[str, float]) -> np.ndarray:
    """Build one raw (pre-bucketization) token from an observable dict.

    `observable` must contain all OBS_FEATURES keys. Returns float32 (D,).
    """
    return np.array([float(observable[k]) for k in OBS_FEATURES], dtype=np.float32)


# ----------------------------------------------------------------------------
# Bucketizer (quantile-based, per feature)
# ----------------------------------------------------------------------------


class Bucketizer:
    """Per-feature quantile bucketization (UNUM Quantile-K)."""

    def __init__(self, n_buckets: int = 32):
        self.n_buckets = int(n_buckets)
        # boundaries[f] is an array of inner edges (n_buckets-1,)
        self.boundaries: List[np.ndarray] = []
        self.midpoints: List[np.ndarray] = []  # (n_buckets,) representative value per bucket

    def fit(self, tokens: np.ndarray) -> "Bucketizer":
        """tokens: (N, D) raw token matrix."""
        assert tokens.ndim == 2 and tokens.shape[1] == D, f"expected (N,{D}) got {tokens.shape}"
        self.boundaries = []
        self.midpoints = []
        qs = np.linspace(0.0, 1.0, self.n_buckets + 1)
        for f in range(D):
            col = tokens[:, f]
            edges = np.quantile(col, qs)
            # Make strictly increasing to avoid empty/degenerate buckets
            edges = np.maximum.accumulate(edges + 1e-9 * np.arange(edges.size))
            inner = edges[1:-1].astype(np.float32)
            self.boundaries.append(inner)
            mids = 0.5 * (edges[:-1] + edges[1:])
            self.midpoints.append(mids.astype(np.float32))
        return self

    def transform(self, tokens: np.ndarray) -> np.ndarray:
        """tokens: (N, D) -> bucket indices (N, D) int64."""
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        out = np.empty(tokens.shape, dtype=np.int64)
        for f in range(D):
            out[:, f] = np.digitize(tokens[:, f], self.boundaries[f])
        np.clip(out, 0, self.n_buckets - 1, out=out)
        return out

    def midpoint(self, feature_idx: int, bucket_idx: int) -> float:
        return float(self.midpoints[feature_idx][int(bucket_idx)])

    # serialization helpers
    def state(self) -> Dict[str, object]:
        return {
            "n_buckets": self.n_buckets,
            "boundaries": [b.tolist() for b in self.boundaries],
            "midpoints": [m.tolist() for m in self.midpoints],
        }

    @classmethod
    def from_state(cls, st: Dict[str, object]) -> "Bucketizer":
        b = cls(n_buckets=int(st["n_buckets"]))
        b.boundaries = [np.array(x, dtype=np.float32) for x in st["boundaries"]]  # type: ignore[index]
        b.midpoints = [np.array(x, dtype=np.float32) for x in st["midpoints"]]    # type: ignore[index]
        return b


# ----------------------------------------------------------------------------
# Model config + Transformer
# ----------------------------------------------------------------------------


@dataclass
class EmbedderConfig:
    n_buckets: int = 32
    context_w: int = 16        # history tokens fed to the encoder
    horizon_h: int = 4         # future tokens predicted (windowed next-token)
    emb_per_feat: int = 8      # per-feature bucket embedding size
    d_model: int = 32
    nhead: int = 2
    n_encoder_layers: int = 2
    ff_dim: int = 64
    dropout: float = 0.1
    embedding_dim: int = 16    # the latent the controller consumes (encoder proj)


class WindowedTransformer(nn.Module):
    """Encoder + windowed multi-head classification decoder.

    Encoder: causal self-attention over W tokens -> embedding (last position).
    Decoder: from the embedding, predict H future tokens' buckets, with one
    classification head per (horizon, feature).
    """

    def __init__(self, cfg: EmbedderConfig):
        super().__init__()
        self.cfg = cfg
        # Per-feature bucket embedding tables
        self.feat_emb = nn.ModuleList(
            [nn.Embedding(cfg.n_buckets, cfg.emb_per_feat) for _ in range(D)]
        )
        self.in_proj = nn.Linear(D * cfg.emb_per_feat, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.context_w, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_encoder_layers)
        self.embed_proj = nn.Linear(cfg.d_model, cfg.embedding_dim)

        # Windowed prediction heads: H * D heads, each -> n_buckets logits
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.embedding_dim, cfg.n_buckets) for _ in range(cfg.horizon_h * D)]
        )

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, D) int64 -> (B, W, d_model)
        parts = [self.feat_emb[f](x[:, :, f]) for f in range(D)]
        h = torch.cat(parts, dim=-1)             # (B, W, D*emb_per_feat)
        h = self.in_proj(h)                      # (B, W, d_model)
        h = h + self.pos_emb[:, : h.size(1), :]
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, D) int64 -> embedding (B, embedding_dim)."""
        h = self._embed_tokens(x)
        w = h.size(1)
        mask = torch.triu(torch.ones(w, w, device=h.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask)           # causal
        last = h[:, -1, :]                        # encoder output at most recent step
        return self.embed_proj(last)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (embedding, list of H*D logit tensors each (B, n_buckets))."""
        emb = self.encode(x)
        logits = [head(emb) for head in self.heads]
        return emb, logits


# ----------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------


def save_embedder(path: str, model: WindowedTransformer, bucketizer: Bucketizer) -> None:
    torch.save(
        {
            "config": asdict(model.cfg),
            "model_state": model.state_dict(),
            "bucketizer": bucketizer.state(),
            "obs_features": list(OBS_FEATURES),
        },
        path,
    )


def load_embedder(path: str, device: str = "cpu") -> Tuple[WindowedTransformer, Bucketizer, EmbedderConfig]:
    blob = torch.load(path, map_location=device, weights_only=False)
    if list(blob.get("obs_features", OBS_FEATURES)) != list(OBS_FEATURES):
        raise ValueError(
            "Embedder was trained with a different OBS_FEATURES set than the "
            "current unum_embedder.py. Re-pretrain or align the feature list."
        )
    cfg = EmbedderConfig(**blob["config"])
    model = WindowedTransformer(cfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    bucketizer = Bucketizer.from_state(blob["bucketizer"])
    return model, bucketizer, cfg


# ----------------------------------------------------------------------------
# Online runtime (rolling window, frozen inference, guardrail signal)
# ----------------------------------------------------------------------------


class EmbedderRuntime:
    """Maintains the rolling token window and serves frozen embeddings.

    Usage per control step:
        rt.push(observable_dict)          # append the latest completed window
        z = rt.embedding()                # (embedding_dim,) np.float32
        div = rt.divergence(observable)   # guardrail signal (bucket distance)
    """

    def __init__(self, path: str, device: str = "cpu"):
        self.device = device
        self.model, self.bucketizer, self.cfg = load_embedder(path, device=device)
        self.window: Deque[np.ndarray] = deque(maxlen=self.cfg.context_w)  # bucket tokens (D,)
        self._last_pred_buckets: Optional[np.ndarray] = None  # (H, D)
        self.embedding_dim = self.cfg.embedding_dim

    def reset(self) -> None:
        self.window.clear()
        self._last_pred_buckets = None

    def _bucketize(self, observable: Dict[str, float]) -> np.ndarray:
        raw = build_token(observable)            # (D,)
        return self.bucketizer.transform(raw)[0] # (D,)

    def push(self, observable: Dict[str, float]) -> None:
        self.window.append(self._bucketize(observable))

    def _window_tensor(self) -> torch.Tensor:
        toks = list(self.window)
        if len(toks) < self.cfg.context_w:
            pad = [toks[0] if toks else np.zeros(D, dtype=np.int64)] * (self.cfg.context_w - len(toks))
            toks = pad + toks
        arr = np.stack(toks, axis=0)[None, :, :]  # (1, W, D)
        return torch.tensor(arr, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def embedding(self) -> np.ndarray:
        x = self._window_tensor()
        emb, logits = self.model(x)
        # cache predicted buckets (argmax) for guardrail: shape (H, D)
        H, n_b = self.cfg.horizon_h, self.cfg.n_buckets
        preds = np.empty((H, D), dtype=np.int64)
        for h in range(H):
            for f in range(D):
                preds[h, f] = int(torch.argmax(logits[h * D + f], dim=-1).item())
        self._last_pred_buckets = preds
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def divergence(self, observed: Dict[str, float]) -> float:
        """Mean bucket-index distance between the previously predicted next
        token (horizon 0) and the now-observed token. Used by the guardrail.
        Returns NaN until a prediction exists.
        """
        if self._last_pred_buckets is None:
            return float("nan")
        obs_b = self._bucketize(observed)         # (D,)
        pred0 = self._last_pred_buckets[0]        # (D,)
        return float(np.mean(np.abs(pred0.astype(np.int64) - obs_b.astype(np.int64))))


if __name__ == "__main__":
    # Tiny smoke test of shapes (random data, no training).
    cfg = EmbedderConfig()
    model = WindowedTransformer(cfg)
    b = Bucketizer(cfg.n_buckets).fit(np.random.randn(2000, D).astype(np.float32) * 50 + 100)
    x = torch.randint(0, cfg.n_buckets, (4, cfg.context_w, D))
    emb, logits = model(x)
    print("embedding:", tuple(emb.shape), "| n heads:", len(logits), "| head0:", tuple(logits[0].shape))
    assert emb.shape == (4, cfg.embedding_dim)
    assert len(logits) == cfg.horizon_h * D
    print("unum_embedder smoke test OK")
