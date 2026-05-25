"""upf_rl_env — Gymnasium environment for slice-aware MBR control on BESS-UPF."""
from __future__ import annotations

from gymnasium.envs.registration import register

from upf_rl_env.config import DQNConfig, EnvConfig
from upf_rl_env.env import SliceQoSEnv

__all__ = ["SliceQoSEnv", "EnvConfig", "DQNConfig"]

register(
    id="upf-rl/SliceQoS-v0",
    entry_point="upf_rl_env.env:SliceQoSEnv",
    max_episode_steps=None,  # env handles truncation internally
)
