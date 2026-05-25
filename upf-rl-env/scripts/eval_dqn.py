"""Evaluate a saved DQN agent on SliceQoS-v0.

    python scripts/eval_dqn.py --ckpt checkpoints/dqn_raw/final.pt --episodes 5
"""
from __future__ import annotations

import argparse
import logging
import os

import gymnasium as gym
import numpy as np
import torch

import upf_rl_env  # registers env
from upf_rl_env.config import DQNConfig
from upf_rl_agents.dqn import DQNAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed-base", type=int, default=10_000)
    p.add_argument("--log-csv-dir", type=str, default="runs/eval")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("eval")

    os.makedirs(args.log_csv_dir, exist_ok=True)

    env = gym.make("upf-rl/SliceQoS-v0")
    env.unwrapped._log_csv_path = os.path.join(args.log_csv_dir, "eval.csv")  # type: ignore[attr-defined]

    obs_dim = int(np.prod(env.observation_space.shape))  # type: ignore[arg-type]
    n_actions = int(env.action_space.n)                   # type: ignore[arg-type]

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False. Use --device cpu only for a CPU run.")

    cfg = DQNConfig(device=device)
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg)
    agent.load(args.ckpt)
    log.info("loaded %s on device=%s", args.ckpt, agent.device)

    returns, gold_losses, sla_viols, utils = [], [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed_base + ep)
        done = False
        ep_r = 0.0
        gl, sl, ut = [], [], []
        while not done:
            a = agent.act(obs, eval_mode=True)
            obs, r, terminated, truncated, info = env.step(a)
            ep_r += r
            m = info["metrics"]
            gl.append(m["loss_gold"])
            sl.append(1.0 if m["loss_gold"] > 0.01 else 0.0)
            ut.append(m["n6_utilization"])
            done = terminated or truncated
        returns.append(ep_r)
        gold_losses.append(float(np.mean(gl)))
        sla_viols.append(float(np.mean(sl)))
        utils.append(float(np.mean(ut)))
        log.info("ep %d  return=%+.2f  gold_loss=%.4f  sla_viol=%.3f  util=%.3f",
                 ep, ep_r, gold_losses[-1], sla_viols[-1], utils[-1])

    log.info("=== summary over %d eps ===", args.episodes)
    log.info("return        : %+.3f ± %.3f", float(np.mean(returns)), float(np.std(returns)))
    log.info("gold loss     : %.4f ± %.4f", float(np.mean(gold_losses)), float(np.std(gold_losses)))
    log.info("sla viol rate : %.4f ± %.4f", float(np.mean(sla_viols)), float(np.std(sla_viols)))
    log.info("n6 util       : %.4f ± %.4f", float(np.mean(utils)), float(np.std(utils)))

    env.close()


if __name__ == "__main__":
    main()
