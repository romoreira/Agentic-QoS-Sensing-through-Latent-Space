"""Train a DQN agent on SliceQoS-v0.

    python scripts/train_dqn.py --total-steps 200000 --save-dir checkpoints/run_raw
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import gymnasium as gym
import numpy as np

import upf_rl_env  # registers env
from upf_rl_env.config import DQNConfig, EnvConfig
from upf_rl_agents.dqn import DQNAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=DQNConfig.total_steps)
    p.add_argument("--eval-every", type=int, default=DQNConfig.eval_every)
    p.add_argument("--eval-episodes", type=int, default=DQNConfig.eval_episodes)
    p.add_argument("--save-every", type=int, default=DQNConfig.save_every)
    p.add_argument("--save-dir", type=str, default="checkpoints/dqn_raw")
    p.add_argument("--log-csv-dir", type=str, default="runs/dqn_raw")
    p.add_argument("--seed", type=int, default=DQNConfig.seed)
    p.add_argument("--device", type=str, default=DQNConfig.device)
    p.add_argument("--lr", type=float, default=DQNConfig.lr)
    p.add_argument("--batch-size", type=int, default=DQNConfig.batch_size)
    p.add_argument("--gamma", type=float, default=DQNConfig.gamma)
    return p.parse_args()


def evaluate(env: gym.Env, agent: DQNAgent, episodes: int) -> dict:
    rewards, gold_loss_means, sla_viol_rates, utils = [], [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=10_000 + ep)  # fixed eval seeds
        done = False
        ep_r = 0.0
        ep_gloss, ep_sla, ep_util = [], [], []
        while not done:
            a = agent.act(obs, eval_mode=True)
            obs, r, terminated, truncated, info = env.step(a)
            ep_r += r
            m = info["metrics"]
            ep_gloss.append(m["loss_gold"])
            ep_sla.append(1.0 if m["loss_gold"] > 0.01 else 0.0)
            ep_util.append(m["n6_utilization"])
            done = terminated or truncated
        rewards.append(ep_r)
        gold_loss_means.append(float(np.mean(ep_gloss)))
        sla_viol_rates.append(float(np.mean(ep_sla)))
        utils.append(float(np.mean(ep_util)))
    return {
        "eval/return_mean": float(np.mean(rewards)),
        "eval/return_std": float(np.std(rewards)),
        "eval/gold_loss_mean": float(np.mean(gold_loss_means)),
        "eval/sla_viol_rate": float(np.mean(sla_viol_rates)),
        "eval/n6_util_mean": float(np.mean(utils)),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("train")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_csv_dir, exist_ok=True)

    cfg_env = EnvConfig.default()
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
        device=args.device,
    )

    # Training env (with CSV logging)
    train_env = gym.make("upf-rl/SliceQoS-v0")
    train_env.unwrapped._log_csv_path = os.path.join(args.log_csv_dir, "train.csv")  # type: ignore[attr-defined]

    # Separate eval env (no CSV — to keep things fast). Override if you want
    # eval traces on disk too.
    eval_env = gym.make("upf-rl/SliceQoS-v0")

    obs_dim = int(np.prod(train_env.observation_space.shape))  # type: ignore[arg-type]
    n_actions = int(train_env.action_space.n)                   # type: ignore[arg-type]
    log.info("obs_dim=%d  n_actions=%d  device=%s", obs_dim, n_actions, cfg_dqn.device)

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, cfg=cfg_dqn)

    obs, _ = train_env.reset(seed=args.seed)
    ep_return = 0.0
    ep_len = 0
    episode = 0
    t0 = time.monotonic()

    for step in range(1, args.total_steps + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        agent.observe(obs, action, reward, next_obs, terminated)  # done flag = terminated only (truncated isn't a real terminal)
        train_metrics = agent.train_step()
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if done:
            episode += 1
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

        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{step:07d}.pt")
            agent.save(ckpt_path)
            log.info("saved %s", ckpt_path)

    # Final save
    final_path = os.path.join(args.save_dir, "final.pt")
    agent.save(final_path)
    log.info("final ckpt: %s", final_path)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
