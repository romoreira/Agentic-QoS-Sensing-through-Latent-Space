#!/usr/bin/env python3
"""Offline pretraining of the UNUM-style embedder. No TRex required.

Faithful to UNUM's "separation principle": the embedder is trained on a
*cross-environment, cross-policy* dataset with a self-supervised objective
(windowed next-token prediction), independent of the control reward. The
resulting frozen embedder is then loaded by the online controller.

Data collection
---------------
For each (traffic_model, seed, collection_policy) we roll out the generator,
choose an MBR per step with the collection policy, compute the shared-uplink
proxy (with the *per-step* capacity when the model provides one), and record the
observable token (see unum_embedder.OBS_FEATURES). Diverse policies
(random / reactive / fixed) give the cross-policy coverage UNUM relies on.

The hidden noise offered load and the latent capacity are NEVER tokens.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn

from traffic_model import make_traffic_model
from traffic_model_enriched import make_enriched_traffic_model

from online_dqn_proxy_train import (
    ACTIONS,
    ACTION_MBR,
    DEFAULT_MBR,
    shared_uplink_proxy,
)
from online_proxy_baseline import choose_action

from unum_embedder import (
    OBS_FEATURES,
    D,
    Bucketizer,
    EmbedderConfig,
    WindowedTransformer,
    build_token,
    save_embedder,
)

ENRICHED_KINDS = {"latent_rich", "latent_mild"}


def make_any_traffic_model(kind: str, dt_s: float, seed: int):
    if kind in ENRICHED_KINDS:
        return make_enriched_traffic_model(kind, dt_s=dt_s, seed=seed)
    return make_traffic_model(kind, dt_s=dt_s, seed=seed)


def sample_capacity(sample, default_cap: float) -> float:
    return float(getattr(sample, "capacity_mbps", default_cap))


def observable_from(offered: Dict[str, float], pm: Dict[str, float]) -> Dict[str, float]:
    return {
        "offered_gold_mbps": float(offered.get("gold", 0.0)),
        "offered_silver_mbps": float(offered.get("silver", 0.0)),
        "offered_bronze_mbps": float(offered.get("bronze", 0.0)),
        "uplink_pressure": float(pm["uplink_pressure"]),
        "uplink_utilization": float(pm["uplink_utilization"]),
        "gold_shortage_mbps": float(pm["gold_shortage_mbps"]),
        "available_for_gold_mbps": float(pm["available_for_gold_mbps"]),
    }


def collect_one_rollout(
    kind: str, seed: int, policy: str, duration: float, dt: float, default_cap: float, rng: random.Random
) -> np.ndarray:
    """Returns a (T, D) raw-token sequence for one rollout (one contiguous seq)."""
    model = make_any_traffic_model(kind, dt_s=dt, seed=seed)
    active_mbr = DEFAULT_MBR.copy()

    prev_visible = np.zeros(3, dtype=np.float32)
    prev_pressure = 0.0
    prev_shortage = 0.0

    tokens: List[np.ndarray] = []
    for sample in model.iter_duration(duration):
        # choose MBR for this step
        if policy == "random":
            action = rng.randrange(len(ACTIONS))
        elif policy == "reactive":
            action = choose_action("reactive_threshold", prev_visible, prev_pressure, prev_shortage)
        else:  # fixed
            action = 0
        active_mbr = ACTION_MBR[action].copy()

        offered = {k: float(v) for k, v in sample.mbps.items()}
        cap = sample_capacity(sample, default_cap)
        pm = shared_uplink_proxy(offered, active_mbr, cap)

        obs = observable_from(offered, pm)
        tokens.append(build_token(obs))

        prev_visible = np.array(
            [offered.get("gold", 0.0), offered.get("silver", 0.0), offered.get("bronze", 0.0)],
            dtype=np.float32,
        )
        prev_pressure = pm["uplink_pressure"]
        prev_shortage = pm["gold_shortage_mbps"]

    return np.stack(tokens, axis=0).astype(np.float32)


def build_windows(seq_buckets: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """From one bucketized sequence (T, D) build (X, Y).

    X: (n, w, D) history; Y: (n, h, D) future buckets.
    """
    T = seq_buckets.shape[0]
    xs, ys = [], []
    for t in range(w, T - h + 1):
        xs.append(seq_buckets[t - w:t])
        ys.append(seq_buckets[t:t + h])
    if not xs:
        return np.zeros((0, w, D), np.int64), np.zeros((0, h, D), np.int64)
    return np.stack(xs).astype(np.int64), np.stack(ys).astype(np.int64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output embedder .pt path")
    p.add_argument("--log-csv", default=None, help="optional per-epoch loss CSV")
    p.add_argument(
        "--traffic-models",
        nargs="+",
        default=["latent_rich", "sine_gold_safe_overload", "sine_overload"],
        help="environments to collect from (mix for cross-env coverage)",
    )
    p.add_argument("--collect-policies", nargs="+", default=["random", "reactive", "fixed"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(0, 12)))
    p.add_argument("--duration", type=float, default=1800.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--uplink-capacity-mbps", type=float, default=200.0,
                   help="fallback capacity for models without a per-step capacity")
    # embedder config
    p.add_argument("--n-buckets", type=int, default=32)
    p.add_argument("--context-w", type=int, default=16)
    p.add_argument("--horizon-h", type=int, default=4)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--nhead", type=int, default=2)
    p.add_argument("--n-encoder-layers", type=int, default=2)
    p.add_argument("--ff-dim", type=int, default=64)
    # training
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = random.Random(args.seed)

    # ---- collect ----
    t0 = time.time()
    raw_seqs: List[np.ndarray] = []
    n_rollouts = 0
    for kind in args.traffic_models:
        for seed in args.seeds:
            for policy in args.collect_policies:
                seq = collect_one_rollout(
                    kind, seed, policy, args.duration, args.dt, args.uplink_capacity_mbps, rng
                )
                if seq.shape[0] >= args.context_w + args.horizon_h:
                    raw_seqs.append(seq)
                    n_rollouts += 1
    all_raw = np.concatenate(raw_seqs, axis=0)
    print(f"[collect] {n_rollouts} rollouts, {all_raw.shape[0]} tokens, "
          f"{len(args.traffic_models)} envs x {len(args.seeds)} seeds x {len(args.collect_policies)} policies "
          f"in {time.time()-t0:.1f}s")

    # ---- fit bucketizer (on all tokens) ----
    bucketizer = Bucketizer(n_buckets=args.n_buckets).fit(all_raw)

    # ---- build windows per sequence (no cross-sequence leakage) ----
    Xs, Ys = [], []
    for seq in raw_seqs:
        bseq = bucketizer.transform(seq)
        X, Y = build_windows(bseq, args.context_w, args.horizon_h)
        if X.shape[0] > 0:
            Xs.append(X)
            Ys.append(Y)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    print(f"[windows] X={X.shape} Y={Y.shape}")

    # shuffle + split
    idx = np.arange(X.shape[0])
    rng.shuffle(idx.tolist()) if False else np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    n_val = int(args.val_frac * X.shape[0])
    Xtr, Ytr = X[n_val:], Y[n_val:]
    Xva, Yva = X[:n_val], Y[:n_val]

    Xtr_t = torch.tensor(Xtr, dtype=torch.long)
    Ytr_t = torch.tensor(Ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.long, device=device)
    Yva_t = torch.tensor(Yva, dtype=torch.long, device=device)

    # ---- model ----
    cfg = EmbedderConfig(
        n_buckets=args.n_buckets,
        context_w=args.context_w,
        horizon_h=args.horizon_h,
        d_model=args.d_model,
        nhead=args.nhead,
        n_encoder_layers=args.n_encoder_layers,
        ff_dim=args.ff_dim,
        embedding_dim=args.embedding_dim,
    )
    model = WindowedTransformer(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    H = cfg.horizon_h

    def batch_loss(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        _, logits = model(xb)                    # list of H*D, each (B, n_buckets)
        loss = 0.0
        for h in range(H):
            for f in range(D):
                loss = loss + ce(logits[h * D + f], yb[:, h, f])
        return loss / (H * D)

    @torch.no_grad()
    def val_metrics() -> Tuple[float, float]:
        model.eval()
        _, logits = model(Xva_t)
        loss = 0.0
        within1 = 0.0
        for h in range(H):
            for f in range(D):
                lg = logits[h * D + f]
                tgt = Yva_t[:, h, f]
                loss += ce(lg, tgt).item()
                pred = torch.argmax(lg, dim=-1)
                within1 += (torch.abs(pred - tgt) <= 1).float().mean().item()
        model.train()
        return loss / (H * D), within1 / (H * D)

    log_rows = []
    n = Xtr_t.shape[0]
    bs = args.batch_size
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            xb = Xtr_t[sel].to(device)
            yb = Ytr_t[sel].to(device)
            loss = batch_loss(xb, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        vloss, vwithin1 = val_metrics()
        print(f"[epoch {epoch:02d}] train_loss={ep_loss/max(n_batches,1):.4f} "
              f"val_loss={vloss:.4f} val_within1_acc={vwithin1:.3f}")
        log_rows.append({
            "epoch": epoch,
            "train_loss": ep_loss / max(n_batches, 1),
            "val_loss": vloss,
            "val_within1_acc": vwithin1,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_embedder(args.out, model, bucketizer)
    print(f"[save] embedder -> {args.out}")

    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_within1_acc"])
            w.writeheader()
            for r in log_rows:
                w.writerow(r)
        print(f"[save] pretrain log -> {args.log_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
