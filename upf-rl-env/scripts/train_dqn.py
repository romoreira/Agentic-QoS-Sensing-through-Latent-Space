"""Train a DQN agent on SliceQoS-v0.

GPU-oriented run:
    python scripts/train_dqn.py \
      --device cuda \
      --total-steps 200000 \
      --save-dir checkpoints/dqn_raw \
      --tb-dir runs/tensorboard/dqn_raw
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover, handled at runtime
    SummaryWriter = None  # type: ignore[assignment]

import upf_rl_env  # registers env
from upf_rl_env.config import DQNConfig
from upf_rl_env.sources.real_csv import RealCsvMetricSource
from upf_rl_agents.dqn import DQNAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=DQNConfig.total_steps)
    p.add_argument("--eval-every", type=int, default=DQNConfig.eval_every)
    p.add_argument("--eval-episodes", type=int, default=DQNConfig.eval_episodes)
    p.add_argument("--save-every", type=int, default=DQNConfig.save_every)
    p.add_argument("--save-dir", type=str, default="checkpoints/dqn_raw")
    p.add_argument("--log-csv-dir", type=str, default="runs/dqn_raw")
    p.add_argument("--tb-dir", type=str, default="runs/tensorboard/dqn_raw")
    p.add_argument("--tb-log-every", type=int, default=10)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--seed", type=int, default=DQNConfig.seed)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--lr", type=float, default=DQNConfig.lr)
    p.add_argument("--batch-size", type=int, default=DQNConfig.batch_size)
    p.add_argument("--gamma", type=float, default=DQNConfig.gamma)
    p.add_argument("--metric-source", type=str, default="mock", choices=["mock", "real_csv"],
                   help="Telemetry source for the environment. real_csv replays a CSV from real_telemetry_probe.py.")
    p.add_argument("--csv", type=str, default=None,
                   help="Path to real telemetry CSV when --metric-source real_csv is used.")
    p.add_argument("--csv-no-loop", action="store_true",
                   help="Do not loop the real CSV at EOF, keep replaying the final row.")
    return p.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False. Use --device cpu only for a CPU run.")
    return requested


def _make_writer(tb_dir: str, disabled: bool) -> Optional["SummaryWriter"]:
    if disabled:
        return None
    if SummaryWriter is None:
        raise SystemExit("TensorBoard is not installed. Install project deps or run with --no-tensorboard.")
    os.makedirs(tb_dir, exist_ok=True)
    return SummaryWriter(log_dir=tb_dir)  # type: ignore[operator]


def _tb_log_step(writer: Optional["SummaryWriter"], step: int, reward: float, action: int, info: dict, train_metrics: dict) -> None:
    if writer is None:
        return

    metrics = info.get("metrics", {})
    reward_components = info.get("reward_components", {})
    hidden = info.get("hidden_state", {})

    writer.add_scalar("env/reward", reward, step)
    writer.add_scalar("env/action", action, step)

    for name in ("gold", "silver", "bronze"):
        writer.add_scalar(f"traffic/offered_{name}_mbps", metrics.get(f"offered_{name}_mbps", 0.0), step)
        writer.add_scalar(f"traffic/rx_{name}_mbps", metrics.get(f"rx_{name}_mbps", 0.0), step)
        writer.add_scalar(f"traffic/loss_{name}", metrics.get(f"loss_{name}", 0.0), step)
        writer.add_scalar(f"traffic/qer_red_{name}", metrics.get(f"qer_red_{name}", 0.0), step)
        writer.add_scalar(f"mbr/{name}_kbps", metrics.get(f"mbr_{name}_kbps", 0.0), step)

    writer.add_scalar("traffic/n6_tx_mbps", metrics.get("n6_tx_mbps", 0.0), step)
    writer.add_scalar("traffic/n6_utilization", metrics.get("n6_utilization", 0.0), step)

    if hidden:
        writer.add_scalar("hidden/regime", hidden.get("regime", 0), step)
        writer.add_scalar("hidden/noise_offered_mbps", hidden.get("hidden_offered_mbps", hidden.get("noise_offered_mbps", 0.0)), step)
        writer.add_scalar("hidden/noise_rx_mbps", hidden.get("noise_rx_mbps", 0.0), step)
        writer.add_scalar("hidden/noise_loss", hidden.get("noise_loss", 0.0), step)
        writer.add_scalar("hidden/pfcp_sessions", hidden.get("pfcp_sessions", 0.0), step)
        writer.add_scalar("hidden/trex_total_tx_pkts", hidden.get("trex_total_tx_pkts", 0.0), step)
        writer.add_scalar("hidden/trex_total_rx_pkts", hidden.get("trex_total_rx_pkts", 0.0), step)
        writer.add_scalar("hidden/n6_capacity_mbps", hidden.get("n6_capacity_mbps", 0.0), step)

    for k, v in reward_components.items():
        writer.add_scalar(f"reward_components/{k}", v, step)

    if train_metrics:
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"dqn/{k}", v, step)


def evaluate(env: gym.Env, agent: DQNAgent, episodes: int) -> dict:
    rewards, gold_loss_means, sla_viol_rates, utils = [], [], [], []
    silver_loss_means, bronze_loss_means = [], []
    for ep in range(episodes):
        obs, _info = env.reset(seed=10_000 + ep)
        done = False
        ep_r = 0.0
        ep_gloss, ep_sloss, ep_bloss, ep_sla, ep_util = [], [], [], [], []
        while not done:
            a = agent.act(obs, eval_mode=True)
            obs, r, terminated, truncated, info = env.step(a)
            ep_r += r
            m = info["metrics"]
            ep_gloss.append(m["loss_gold"])
            ep_sloss.append(m["loss_silver"])
            ep_bloss.append(m["loss_bronze"])
            ep_sla.append(1.0 if m["loss_gold"] > 0.01 else 0.0)
            ep_util.append(m["n6_utilization"])
            done = terminated or truncated
        rewards.append(ep_r)
        gold_loss_means.append(float(np.mean(ep_gloss)))
        silver_loss_means.append(float(np.mean(ep_sloss)))
        bronze_loss_means.append(float(np.mean(ep_bloss)))
        sla_viol_rates.append(float(np.mean(ep_sla)))
        utils.append(float(np.mean(ep_util)))
    return {
        "eval/return_mean": float(np.mean(rewards)),
        "eval/return_std": float(np.std(rewards)),
        "eval/gold_loss_mean": float(np.mean(gold_loss_means)),
        "eval/silver_loss_mean": float(np.mean(silver_loss_means)),
        "eval/bronze_loss_mean": float(np.mean(bronze_loss_means)),
        "eval/sla_viol_rate": float(np.mean(sla_viol_rates)),
        "eval/n6_util_mean": float(np.mean(utils)),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("train")

    device = _resolve_device(args.device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_csv_dir, exist_ok=True)
    writer = _make_writer(args.tb_dir, args.no_tensorboard)

    cfg_dqn = DQNConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        save_every=args.save_every,
        save_dir=args.save_dir,
        seed=args.seed,
        device=device,
    )

    if args.metric_source == "real_csv":
        if not args.csv:
            raise SystemExit("--csv is required when --metric-source real_csv is used.")
        train_source = RealCsvMetricSource(args.csv, loop=not args.csv_no_loop, use_action_mbr=True)
        eval_source = RealCsvMetricSource(args.csv, loop=True, use_action_mbr=True)
        train_env = gym.make("upf-rl/SliceQoS-v0", metric_source=train_source)
        eval_env = gym.make("upf-rl/SliceQoS-v0", metric_source=eval_source)
        log.info("metric_source=real_csv csv=%s rows=%d", args.csv, len(train_source.rows))
        log.warning("real_csv is offline replay: actions do not change the recorded traffic or loss. Use it for wiring/smoke tests before online control.")
    else:
        train_env = gym.make("upf-rl/SliceQoS-v0")
        eval_env = gym.make("upf-rl/SliceQoS-v0")

    train_env.unwrapped._log_csv_path = os.path.join(args.log_csv_dir, "train.csv")  # type: ignore[attr-defined]

    obs_dim = int(np.prod(train_env.observation_space.shape))  # type: ignore[arg-type]
    n_actions = int(train_env.action_space.n)                   # type: ignore[arg-type]
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg_dqn)
    log.info("obs_dim=%d  n_actions=%d  device=%s", obs_dim, n_actions, agent.device)
    if writer is not None:
        log.info("TensorBoard: %s", args.tb_dir)

    obs, _ = train_env.reset(seed=args.seed)
    ep_return = 0.0
    ep_len = 0
    episode = 0
    t0 = time.monotonic()

    for step in range(1, args.total_steps + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        agent.observe(obs, action, reward, next_obs, terminated)
        train_metrics = agent.train_step()
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if step % max(args.tb_log_every, 1) == 0:
            _tb_log_step(writer, step, reward, action, info, train_metrics)

        if done:
            episode += 1
            if writer is not None:
                writer.add_scalar("episode/return", ep_return, episode)
                writer.add_scalar("episode/length", ep_len, episode)
            log.info(
                "ep=%4d  step=%7d  ep_len=%4d  ep_return=%+8.2f  eps=%.3f",
                episode, step, ep_len, ep_return, agent.epsilon,
            )
            ep_return = 0.0
            ep_len = 0
            obs, _ = train_env.reset()

        if step % 1000 == 0 and train_metrics:
            elapsed = time.monotonic() - t0
            sps = step / max(elapsed, 1e-6)
            if writer is not None:
                writer.add_scalar("system/steps_per_second", sps, step)
            log.info(
                "step=%7d  loss=%.4f  q_mean=%+.3f  td=%.3f  replay=%d  sps=%.1f",
                step,
                train_metrics["loss"],
                train_metrics["q_mean"],
                train_metrics["td_abs_mean"],
                train_metrics["replay_size"],
                sps,
            )

        if step % args.eval_every == 0:
            log.info("=== eval @ step %d ===", step)
            stats = evaluate(eval_env, agent, args.eval_episodes)
            for k, v in stats.items():
                log.info("  %-25s %+.4f", k, v)
                if writer is not None:
                    writer.add_scalar(k, v, step)

        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{step:07d}.pt")
            agent.save(ckpt_path)
            log.info("saved %s", ckpt_path)

    final_path = os.path.join(args.save_dir, "final.pt")
    agent.save(final_path)
    log.info("final ckpt: %s", final_path)

    train_env.close()
    eval_env.close()
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
