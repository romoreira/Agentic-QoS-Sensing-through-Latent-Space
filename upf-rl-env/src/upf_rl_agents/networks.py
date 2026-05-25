"""Network architectures for DQN."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DuelingMLP(nn.Module):
    """Dueling Q-network with an MLP feature trunk.

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(last, 1)
        self.advantage_head = nn.Linear(last, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.trunk(x)
        v = self.value_head(f)
        a = self.advantage_head(f)
        return v + (a - a.mean(dim=-1, keepdim=True))


class PlainMLP(nn.Module):
    """Non-dueling MLP Q-network (for ablations)."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_q_network(obs_dim: int, n_actions: int, dueling: bool, hidden_sizes: Sequence[int]) -> nn.Module:
    if dueling:
        return DuelingMLP(obs_dim, n_actions, hidden_sizes)
    return PlainMLP(obs_dim, n_actions, hidden_sizes)
