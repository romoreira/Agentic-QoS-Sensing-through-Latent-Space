#!/usr/bin/env python3
"""Unified online experiment harness for the proxy QoS testbed.

ONE harness runs every condition so the comparison is apples-to-apples
(identical traffic, proxy, reward, action space, PFCP mechanism). The only
thing that changes between conditions is the *policy* and the *state
representation*:

  --policy fixed_mbr           static MBR baseline      (no learning)
  --policy reactive_threshold  rule-based baseline       (no learning)
  --policy dqn --obs-mode raw      pure DQN, 11-dim instantaneous state
  --policy dqn --obs-mode latent   DQN over a frozen UNUM embedding (+anchors)

Environments:
  --env original   uses traffic_model.make_traffic_model (fixed capacity)
  --env enriched   uses traffic_model_enriched (3-state hidden noise,
                   drifting latent capacity, long Gold envelope)

The harness imports the proxy / reward / action / PFCP pieces from the existing
online_dqn_proxy_train.py and online_proxy_baseline.py, so its behavior is
identical to your current scripts apart from the additions above.

Use --no-trex (and --no-apply) to smoke-test the full pipeline without the
testbed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from traffic_model import make_traffic_model
from traffic_model_enriched import make_enriched_traffic_model

from online_dqn_proxy_train import (
    ACTIONS,
    ACTION_MBR,
    DEFAULT_MBR,
    NOISE_MBR_KBPS,
    QNet,
    apply_mbr,
    compute_reward,
    get_total_rx_tx,
    json_safe,
    make_obs,                 # exact 11-dim raw observation (apples-to-apples)
    shared_uplink_proxy,
)
from online_proxy_baseline import choose_action

ENRICHED_KINDS = {"latent_rich", "latent_mild"}


# ----------------------------------------------------------------------------
# Replay (same as the pure-DQN script)
# ----------------------------------------------------------------------------
class Replay:
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def add(self, item):
        self.buf.append(item)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def parse_args():
    p = argparse.ArgumentParser()
    # what to run
    p.add_argument("--policy", required=True,
                   choices=["fixed_mbr", "reactive_threshold", "dqn"])
    p.add_argument("--obs-mode", default="raw", choices=["raw", "latent"],
                   help="only used when --policy dqn")
    p.add_argument("--env", default="enriched", choices=["original", "enriched"])
    p.add_argument("--traffic-model", default="sine_gold_safe_overload",
                   choices=["stochastic", "sine_abrupt", "sine_stress",
                            "sine_overload", "sine_gold_safe_overload"],
                   help="used when --env original")
    p.add_argument("--enriched-kind", default="latent_rich",
                   choices=["latent_rich", "latent_mild"],
                   help="used when --env enriched")

    # latent embedder
    p.add_argument("--embedder", default=None, help="path to pretrained embedder .pt (latent mode)")
    p.add_argument("--guardrail", action="store_true",
                   help="fall back to reactive_threshold when embedder predictions diverge")
    p.add_argument("--guardrail-bucket-thresh", type=float, default=5.0)
    p.add_argument("--guardrail-window", type=int, default=10)

    # TRex / testbed
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--sync-port", type=int, default=4501)
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--rx-port", type=int, default=1)
    p.add_argument("--wire-overhead-bytes", type=int, default=20)
    p.add_argument("--no-trex", action="store_true", help="skip TRex (pure proxy sim)")
    p.add_argument("--no-apply", action="store_true", help="skip pfcpctl session modify")
    p.add_argument("--force", action="store_true")

    # experiment
    p.add_argument("--duration", type=float, default=1800.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--episode-steps", type=int, default=60)
    p.add_argument("--uplink-capacity-mbps", type=float, default=200.0,
                   help="fixed capacity (original env) / fallback for enriched")
    p.add_argument("--seed", type=int, default=7)

    # DQN hyperparams (defaults match online_dqn_proxy_train.py)
    p.add_argument("--epsilon-start", type=float, default=0.35)
    p.add_argument("--epsilon-end", type=float, default=0.03)
    p.add_argument("--epsilon-decay-steps", type=int, default=1200)
    p.add_argument("--gamma", type=float, default=0.90)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--replay-size", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=32)
    p.add_argument("--target-sync-steps", type=int, default=25)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # outputs
    p.add_argument("--csv", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--episodes-csv", required=True)
    p.add_argument("--tb", required=True)
    return p.parse_args()


def epsilon_by_step(args, step: int) -> float:
    frac = min(1.0, step / max(args.epsilon_decay_steps, 1))
    return args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)


def make_env_model(args):
    if args.env == "enriched":
        return make_enriched_traffic_model(args.enriched_kind, dt_s=args.dt, seed=args.seed)
    return make_traffic_model(args.traffic_model, dt_s=args.dt, seed=args.seed)


def step_capacity(sample, default_cap: float) -> float:
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


def latent_obs(embedding: np.ndarray, active_mbr_kbps: np.ndarray,
               prev_pressure: float, prev_shortage: float) -> np.ndarray:
    anchors = np.concatenate([
        active_mbr_kbps / 300000.0,
        np.array([prev_pressure, prev_shortage / 200.0], dtype=np.float32),
    ])
    return np.concatenate([embedding.astype(np.float32), anchors.astype(np.float32)]).astype(np.float32)


def main() -> int:
    args = parse_args()

    is_dqn = args.policy == "dqn"
    is_latent = is_dqn and args.obs_mode == "latent"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.episodes_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tb).mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(args.tb) if SummaryWriter is not None else None

    model = make_env_model(args)

    # ---- embedder runtime (latent only) ----
    runtime = None
    if is_latent:
        if not args.embedder:
            raise SystemExit("--obs-mode latent requires --embedder <path to .pt>")
        from unum_embedder import EmbedderRuntime
        runtime = EmbedderRuntime(args.embedder, device=device)
        print(f"[latent] loaded embedder {args.embedder} (embedding_dim={runtime.embedding_dim})")

    # ---- TRex ----
    api = None
    client = None
    if not args.no_trex:
        from run_trex_dynamic_4slice import import_trex, build_streams
        api = import_trex()
        STLClient = api[0]
        client = STLClient(server=args.host, sync_port=args.sync_port)
        client.connect()
        client.acquire(ports=[args.port, args.rx_port], force=args.force)

    # ---- agent ----
    q = target = opt = replay = None
    obs_dim = 11
    if is_dqn:
        if is_latent:
            obs_dim = runtime.embedding_dim + 5  # embedding + [mbr(3), pressure, shortage]
        else:
            obs_dim = 11
        q = QNet(obs_dim=obs_dim, n_actions=len(ACTIONS)).to(device)
        target = QNet(obs_dim=obs_dim, n_actions=len(ACTIONS)).to(device)
        target.load_state_dict(q.state_dict())
        opt = optim.Adam(q.parameters(), lr=args.lr)
        replay = Replay(args.replay_size)

    prev_mbr = DEFAULT_MBR.copy()
    active_mbr = DEFAULT_MBR.copy()

    if not args.no_apply:
        rc, out = apply_mbr(active_mbr)
        print(f"Initial MBR apply rc={rc}: {out}")

    # raw-obs running state
    prev_visible = np.zeros(3, dtype=np.float32)
    prevprev_visible = np.zeros(3, dtype=np.float32)
    prev_uplink_pressure = 0.0
    prev_gold_shortage = 0.0

    # initial state
    if is_latent:
        state = latent_obs(runtime.embedding(), active_mbr, prev_uplink_pressure, prev_gold_shortage)
    else:
        state = make_obs(prev_visible, prevprev_visible, active_mbr,
                         prev_uplink_pressure, prev_gold_shortage)

    # guardrail bookkeeping
    guardrail_bad_count = 0

    fieldnames = [
        "policy", "obs_mode", "env",
        "step", "episode", "episode_step", "t_s", "regime",
        "action_id", "action_name", "epsilon", "reward", "train_loss",
        "capacity_mbps", "guardrail_active", "embedder_divergence",
        "offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps",
        "hidden_burst_mbps",
        "mbr_active_gold_kbps", "mbr_active_silver_kbps", "mbr_active_bronze_kbps",
        "after_mbr_gold_mbps", "after_mbr_silver_mbps", "after_mbr_bronze_mbps", "after_mbr_hidden_mbps",
        "mbr_drop_gold_mbps", "mbr_drop_silver_mbps", "mbr_drop_bronze_mbps",
        "available_for_gold_mbps", "gold_rx_proxy_mbps", "gold_shortage_mbps",
        "total_after_mbr_mbps", "uplink_rx_proxy_mbps", "uplink_pressure", "uplink_utilization",
        "proxy_loss_gold", "proxy_loss_silver", "proxy_loss_bronze",
        "trex_total_tx_pkts", "trex_total_rx_pkts",
        "pfcp_modify_rc", "pfcp_modify_output",
    ]
    ep_fieldnames = ["policy", "obs_mode", "episode", "start_step", "end_step",
                     "reward_sum", "reward_mean", "loss_mean", "gold_shortage_sum"]

    episode_rewards: List[float] = []
    episode_losses: List[float] = []
    episode_gold_shortage: List[float] = []
    episode_start = 0

    obs_mode_tag = args.obs_mode if is_dqn else "n/a"

    try:
        with open(args.csv, "w", newline="") as f_csv, \
             open(args.jsonl, "w") as f_jsonl, \
             open(args.episodes_csv, "w", newline="") as f_ep:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            ep_writer = csv.DictWriter(f_ep, fieldnames=ep_fieldnames, extrasaction="ignore")
            ep_writer.writeheader()

            for sample in model.iter_duration(args.duration):
                step = int(sample.step)
                episode = step // max(args.episode_steps, 1)
                episode_step = step % max(args.episode_steps, 1)

                eps = math.nan
                guardrail_active = 0

                # ---- choose action ----
                if not is_dqn:
                    action = choose_action(args.policy, prev_visible,
                                           prev_uplink_pressure, prev_gold_shortage)
                else:
                    eps = epsilon_by_step(args, step)
                    use_guardrail = (
                        is_latent and args.guardrail
                        and guardrail_bad_count >= args.guardrail_window
                    )
                    if use_guardrail:
                        action = choose_action("reactive_threshold", prev_visible,
                                               prev_uplink_pressure, prev_gold_shortage)
                        guardrail_active = 1
                    elif random.random() < eps:
                        action = random.randrange(len(ACTIONS))
                    else:
                        with torch.no_grad():
                            sx = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                            action = int(torch.argmax(q(sx), dim=1).item())

                prev_mbr = active_mbr.copy()
                active_mbr = ACTION_MBR[action].copy()

                # ---- apply MBR ----
                rc, out = 0, "no_apply"
                if not args.no_apply:
                    if is_dqn:
                        rc, out = apply_mbr(active_mbr)
                    else:
                        if step == 0 or not np.array_equal(active_mbr, prev_mbr):
                            rc, out = apply_mbr(active_mbr)
                        else:
                            out = "unchanged"

                offered = {k: float(v) for k, v in sample.mbps.items()}
                capacity = step_capacity(sample, args.uplink_capacity_mbps)

                # ---- TRex traffic (optional) ----
                total_tx_pkts = total_rx_pkts = 0.0
                if not args.no_trex:
                    client.stop(ports=[args.port])
                    client.reset(ports=[args.port])
                    streams, _, _ = build_streams(api, offered, args.wire_overhead_bytes)
                    client.add_streams(streams=streams, ports=[args.port])
                    client.clear_stats()
                    client.start(ports=[args.port], duration=args.dt)
                    time.sleep(max(args.dt, 0.05))
                    stats = client.get_stats()
                    total_tx_pkts, total_rx_pkts = get_total_rx_tx(stats)

                # ---- proxy + reward (time-varying capacity) ----
                pm = shared_uplink_proxy(offered, active_mbr, capacity)
                reward = compute_reward(pm, active_mbr, prev_mbr)

                current_visible = np.array([
                    offered.get("gold", 0.0),
                    offered.get("silver", 0.0),
                    offered.get("bronze", 0.0),
                ], dtype=np.float32)

                # ---- update latent runtime + guardrail signal ----
                divergence = math.nan
                if is_latent:
                    obs_dict = observable_from(offered, pm)
                    divergence = runtime.divergence(obs_dict)  # pred(t) vs observed(t)
                    runtime.push(obs_dict)
                    if args.guardrail and not math.isnan(divergence):
                        if divergence > args.guardrail_bucket_thresh:
                            guardrail_bad_count += 1
                        else:
                            guardrail_bad_count = 0

                # ---- build next state ----
                if is_latent:
                    next_state = latent_obs(runtime.embedding(), active_mbr,
                                            pm["uplink_pressure"], pm["gold_shortage_mbps"])
                else:
                    next_state = make_obs(current_visible, prev_visible, active_mbr,
                                          pm["uplink_pressure"], pm["gold_shortage_mbps"])

                # ---- learn (dqn only) ----
                train_loss = math.nan
                if is_dqn:
                    done = 1.0 if episode_step == args.episode_steps - 1 else 0.0
                    replay.add((state, action, reward, next_state, done))
                    if len(replay) >= max(args.batch_size, args.warmup_steps):
                        s, a, r, ns, d = replay.sample(args.batch_size)
                        s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)
                        qsa = q(s).gather(1, a.view(-1, 1)).squeeze(1)
                        with torch.no_grad():
                            y = r + args.gamma * (1.0 - d) * target(ns).max(dim=1).values
                        loss = nn.functional.smooth_l1_loss(qsa, y)
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                        opt.step()
                        train_loss = float(loss.item())
                    if step > 0 and step % args.target_sync_steps == 0:
                        target.load_state_dict(q.state_dict())

                # ---- log row ----
                row = {
                    "policy": args.policy,
                    "obs_mode": obs_mode_tag,
                    "env": args.env,
                    "step": step,
                    "episode": episode,
                    "episode_step": episode_step,
                    "t_s": float(sample.t_s),
                    "regime": sample.regime,
                    "action_id": action,
                    "action_name": ACTIONS[action][0],
                    "epsilon": eps,
                    "reward": reward,
                    "train_loss": train_loss,
                    "capacity_mbps": capacity,
                    "guardrail_active": guardrail_active,
                    "embedder_divergence": divergence,
                    "offered_gold_mbps": offered.get("gold", 0.0),
                    "offered_silver_mbps": offered.get("silver", 0.0),
                    "offered_bronze_mbps": offered.get("bronze", 0.0),
                    "mbr_active_gold_kbps": active_mbr[0],
                    "mbr_active_silver_kbps": active_mbr[1],
                    "mbr_active_bronze_kbps": active_mbr[2],
                    "trex_total_tx_pkts": total_tx_pkts,
                    "trex_total_rx_pkts": total_rx_pkts,
                    "pfcp_modify_rc": rc,
                    "pfcp_modify_output": out,
                    **pm,
                }
                writer.writerow(json_safe(row))
                f_jsonl.write(json.dumps(json_safe(row)) + "\n")
                f_csv.flush()
                f_jsonl.flush()

                episode_rewards.append(reward)
                episode_gold_shortage.append(pm["gold_shortage_mbps"])
                if not math.isnan(train_loss):
                    episode_losses.append(train_loss)

                if episode_step == args.episode_steps - 1:
                    ep_row = {
                        "policy": args.policy,
                        "obs_mode": obs_mode_tag,
                        "episode": episode,
                        "start_step": episode_start,
                        "end_step": step,
                        "reward_sum": float(np.sum(episode_rewards)),
                        "reward_mean": float(np.mean(episode_rewards)),
                        "loss_mean": float(np.mean(episode_losses)) if episode_losses else math.nan,
                        "gold_shortage_sum": float(np.sum(episode_gold_shortage)),
                    }
                    ep_writer.writerow(json_safe(ep_row))
                    f_ep.flush()
                    episode_rewards.clear()
                    episode_losses.clear()
                    episode_gold_shortage.clear()
                    episode_start = step + 1

                if tb_writer is not None:
                    tb_writer.add_scalar("reward/step", reward, step)
                    if not math.isnan(train_loss):
                        tb_writer.add_scalar("loss/train", train_loss, step)
                    tb_writer.add_scalar("gold/shortage_mbps", pm["gold_shortage_mbps"], step)
                    tb_writer.add_scalar("gold/available_for_gold_mbps", pm["available_for_gold_mbps"], step)
                    tb_writer.add_scalar("uplink/pressure", pm["uplink_pressure"], step)
                    tb_writer.add_scalar("uplink/utilization", pm["uplink_utilization"], step)
                    tb_writer.add_scalar("uplink/capacity_mbps", capacity, step)
                    tb_writer.add_scalar("traffic/offered_gold_mbps", offered.get("gold", 0.0), step)
                    tb_writer.add_scalar("traffic/hidden_burst_mbps", pm["hidden_burst_mbps"], step)
                    tb_writer.add_scalar("mbr/gold_kbps", active_mbr[0], step)
                    tb_writer.add_scalar("mbr/silver_kbps", active_mbr[1], step)
                    tb_writer.add_scalar("mbr/bronze_kbps", active_mbr[2], step)
                    if is_latent:
                        tb_writer.add_scalar("guardrail/active", guardrail_active, step)
                        if not math.isnan(divergence):
                            tb_writer.add_scalar("guardrail/divergence", divergence, step)

                print(
                    f"[{args.policy}/{obs_mode_tag}] step={step:04d} ep={episode:03d} "
                    f"action={action:02d} {ACTIONS[action][0]} "
                    f"offered=({offered.get('gold',0.0):.1f},{offered.get('silver',0.0):.1f},{offered.get('bronze',0.0):.1f}) "
                    f"cap={capacity:.0f} hidden={pm['hidden_burst_mbps']:.1f} "
                    f"shortage={pm['gold_shortage_mbps']:.1f} reward={reward:.2f} "
                    f"gr={guardrail_active} loss={train_loss if not math.isnan(train_loss) else 'nan'}"
                )

                # ---- roll forward ----
                state = next_state
                prevprev_visible = prev_visible.copy()
                prev_visible = current_visible.copy()
                prev_uplink_pressure = pm["uplink_pressure"]
                prev_gold_shortage = pm["gold_shortage_mbps"]

    finally:
        if client is not None:
            try:
                client.stop(ports=[args.port])
            except Exception:
                pass
            client.disconnect()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote JSONL: {args.jsonl}")
    print(f"Wrote episodes CSV: {args.episodes_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
