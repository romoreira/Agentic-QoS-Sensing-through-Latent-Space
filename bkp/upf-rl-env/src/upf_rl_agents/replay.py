"""Prioritized Experience Replay (PER) — proportional variant with sum tree.

Reference: Schaul et al. 2016, "Prioritized Experience Replay".

  - priority p_i = (|TD_i| + eps) ** alpha
  - sampling probability P(i) = p_i / sum_k p_k
  - IS weight w_i = (1 / (N * P(i))) ** beta, normalized by max w
  - beta annealed from beta_start -> 1.0 over training

The sum tree gives O(log N) updates and sampling. We also track the max
priority ever seen so that new samples are added with that priority
(common heuristic — fresh experience is "high priority" by default).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray         # (B, obs_dim)
    actions: np.ndarray     # (B,) int64
    rewards: np.ndarray     # (B,) float32
    next_obs: np.ndarray    # (B, obs_dim)
    dones: np.ndarray       # (B,) float32  (1 if terminal)
    indices: np.ndarray     # (B,) int64    (tree indices for priority update)
    weights: np.ndarray     # (B,) float32  (IS weights)


class _SumTree:
    """Fixed-capacity binary sum tree over priorities."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Tree of size 2*capacity - 1 (leaves at [capacity-1 .. 2*capacity-2])
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, leaf_idx: int, priority: float) -> None:
        tree_idx = leaf_idx + self.capacity - 1
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate up
        idx = tree_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get(self, s: float) -> tuple[int, float]:
        """Return (leaf_idx, priority) such that prefix sum at leaf >= s."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx, float(self.tree[idx])


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.eps = eps

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.tree = _SumTree(capacity)
        self._write = 0
        self._size = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._write
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)

        # New samples get max priority (prioritize unseen)
        self.tree.update(i, self._max_priority ** self.alpha)

        self._write = (self._write + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> Batch:
        if self._size < batch_size:
            raise RuntimeError(f"Not enough samples ({self._size} < {batch_size})")

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        total = self.tree.total()
        if total <= 0.0:
            # Degenerate — fall back to uniform
            indices = np.random.randint(0, self._size, size=batch_size, dtype=np.int64)
            priorities = np.ones(batch_size, dtype=np.float64)
            total = float(self._size)
        else:
            segment = total / batch_size
            for k in range(batch_size):
                s = float(np.random.uniform(k * segment, (k + 1) * segment))
                leaf, p = self.tree.get(s)
                # Defensive — leaf might be at an unwritten slot if rounding hits edges
                if leaf >= self._size:
                    leaf = np.random.randint(0, self._size)
                    p = max(self.tree.tree[leaf + self.capacity - 1], self.eps)
                indices[k] = leaf
                priorities[k] = max(p, self.eps)

        probs = priorities / total
        weights = (self._size * probs) ** (-beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)

        return Batch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        td = np.abs(td_errors) + self.eps
        for idx, err in zip(indices.tolist(), td.tolist()):
            p_alpha = err ** self.alpha
            self.tree.update(int(idx), float(p_alpha))
            if err > self._max_priority:
                self._max_priority = float(err)
