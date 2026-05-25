"""DQN agent — Double + Dueling + Prioritized Replay.

Public API:
    agent = DQNAgent(obs_dim, n_actions, config)
    action = agent.act(obs)                       # epsilon-greedy
    agent.observe(obs, action, reward, next_obs, done)
    metrics = agent.train_step()                  # returns dict or {} if not ready
    agent.save(path); agent.load(path)
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from upf_rl_env.config import DQNConfig
from upf_rl_agents.networks import build_q_network
from upf_rl_agents.replay import PrioritizedReplayBuffer


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: Optional[DQNConfig] = None,
    ) -> None:
        self.cfg = cfg or DQNConfig.default()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = _select_device(self.cfg.device)

        # Seeding
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

        # Networks
        self.online = build_q_network(
            obs_dim, n_actions, dueling=self.cfg.dueling, hidden_sizes=self.cfg.hidden_sizes,
        ).to(self.device)
        self.target = build_q_network(
            obs_dim, n_actions, dueling=self.cfg.dueling, hidden_sizes=self.cfg.hidden_sizes,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = Adam(self.online.parameters(), lr=self.cfg.lr)

        # Replay
        self.replay = PrioritizedReplayBuffer(
            capacity=self.cfg.replay_capacity,
            obs_dim=obs_dim,
            alpha=self.cfg.per_alpha,
            eps=self.cfg.per_eps,
        )

        # Counters
        self._env_steps = 0
        self._train_steps = 0

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        c = self.cfg
        if self._env_steps >= c.eps_decay_steps:
            return c.eps_end
        frac = self._env_steps / max(c.eps_decay_steps, 1)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def _beta(self) -> float:
        c = self.cfg
        if self._train_steps >= c.per_beta_anneal_steps:
            return c.per_beta_end
        frac = self._train_steps / max(c.per_beta_anneal_steps, 1)
        return c.per_beta_start + frac * (c.per_beta_end - c.per_beta_start)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
            q = self.online(x)
            return int(q.argmax(dim=-1).item())

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(obs, action, reward, next_obs, done)
        self._env_steps += 1

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self) -> dict:
        c = self.cfg
        if len(self.replay) < max(c.replay_min_size, c.batch_size):
            return {}
        if self._env_steps % c.train_every != 0:
            return {}

        beta = self._beta()
        batch = self.replay.sample(c.batch_size, beta=beta)

        obs = torch.from_numpy(batch.obs).to(self.device)
        next_obs = torch.from_numpy(batch.next_obs).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)
        weights = torch.from_numpy(batch.weights).to(self.device)

        # Current Q(s, a)
        q_all = self.online(obs)                           # (B, A)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Double DQN target
        with torch.no_grad():
            next_q_online = self.online(next_obs)          # (B, A)
            next_actions = next_q_online.argmax(dim=-1)    # (B,)
            next_q_target = self.target(next_obs)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + c.gamma * (1.0 - dones) * next_q

        td_error = target - q_sa
        loss = (weights * F.smooth_l1_loss(q_sa, target, reduction="none")).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if c.grad_clip_norm is not None and c.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip_norm)
        self.optimizer.step()

        # Update priorities
        self.replay.update_priorities(batch.indices, td_error.detach().cpu().numpy())

        self._train_steps += 1

        # Target net update
        if c.use_soft_update:
            with torch.no_grad():
                for tp, op in zip(self.target.parameters(), self.online.parameters()):
                    tp.data.mul_(1.0 - c.soft_tau).add_(c.soft_tau * op.data)
        else:
            if self._train_steps % c.target_update_every == 0:
                self.target.load_state_dict(self.online.state_dict())

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "td_abs_mean": float(td_error.abs().mean().item()),
            "epsilon": self.epsilon,
            "beta": beta,
            "replay_size": len(self.replay),
            "train_step": self._train_steps,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "env_steps": self._env_steps,
                "train_steps": self._train_steps,
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._env_steps = int(ckpt.get("env_steps", 0))
        self._train_steps = int(ckpt.get("train_steps", 0))
