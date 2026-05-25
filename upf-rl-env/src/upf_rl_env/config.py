"""Default configuration for the SliceQoS env and DQN agent.

All knobs in one place so sweeps are easy. Values used throughout the
package come from EnvConfig.default() unless overridden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# Slice TEIDs as deployed in the BESS-UPF
SLICE_TEIDS: Tuple[int, int, int] = (1, 11, 21)  # gold, silver, bronze
SLICE_NAMES: Tuple[str, str, str] = ("gold", "silver", "bronze")
NOISE_SLICE_TEID: int = 31                # optional fourth stream on the same uplink
N_SLICES: int = 3

# Nominal MBR per slice in kbps (gold, silver, bronze). The action grid
# scales these by a per-slice percentage. Pick values that, summed, exceed
# nominal N6 capacity — otherwise the agent has nothing to do.
NOMINAL_MBR_KBPS: Tuple[int, int, int] = (200_000, 100_000, 50_000)  # 350 Mbps total
FLOOR_MBR_KBPS: Tuple[int, int, int] = (60_000, 30_000, 15_000)      # minimum MBRs

# N6 link
N6_NOMINAL_CAPACITY_MBPS: float = 300.0  # mean capacity
N6_CAPACITY_DRIFT_STD: float = 0.02      # random walk std per second, fraction
N6_CAPACITY_MIN_FRAC: float = 0.7
N6_CAPACITY_MAX_FRAC: float = 1.1

# Hidden background traffic on N6 (mbps) — INVISIBLE to the agent
BG_MEAN_MBPS: float = 40.0
BG_STD_MBPS: float = 20.0
BG_AR1_RHO: float = 0.85


@dataclass
class EnvConfig:
    # Time
    window_dt_s: float = 0.2            # control / observation window
    frame_stack_k: int = 5              # past frames in observation
    max_episode_steps: int = 3000       # 600 s @ 200ms
    realtime: bool = False              # wall-clock pacing (False = fast sim)

    # Spaces
    n_actions: int = 15
    features_per_window: int = 17       # see features.py
    obs_dim: int = 85                   # K * features_per_window

    # Traffic generator
    base_freqs_hz: Tuple[float, ...] = (0.05, 0.13, 0.27)
    base_amplitudes: Tuple[float, ...] = (0.20, 0.15, 0.10)
    ar1_rho: float = 0.80
    ar1_noise_std: float = 0.05
    burst_rate_per_s: float = 0.05      # Poisson rate of bursts per slice
    burst_lognormal_mu: float = 0.0
    burst_lognormal_sigma: float = 0.6
    burst_decay_s: float = 1.5

    # Regime switching
    regime_dwell_s: Tuple[float, float] = (8.0, 25.0)  # uniform dwell time
    n_regimes: int = 4

    # OOD shift (used by eval to perturb beyond train support)
    ood_freq_scale: float = 1.0
    ood_amplitude_scale: float = 1.0

    # Reward weights
    w_gold_loss: float = 10.0
    w_silver_loss: float = 3.0
    w_bronze_loss: float = 0.5
    w_utilization: float = 1.0
    w_action_change: float = 0.2
    w_sla_violation: float = 50.0
    sla_gold_loss_threshold: float = 0.01

    # Feature normalization (fixed for simplicity)
    norm_mbps: float = 250.0            # divide mbps features by this
    norm_kbps: float = 250_000.0        # divide kbps features by this

    # Logging
    log_dir: str = "runs"
    log_csv_name: str = "metrics.csv"

    @classmethod
    def default(cls) -> "EnvConfig":
        return cls()


@dataclass
class DQNConfig:
    # Networks
    hidden_sizes: Tuple[int, int] = (256, 256)
    dueling: bool = True

    # Optim
    lr: float = 1e-4
    batch_size: int = 64
    gamma: float = 0.95
    grad_clip_norm: float = 10.0

    # Target net
    target_update_every: int = 1000     # hard update steps
    use_soft_update: bool = False
    soft_tau: float = 0.005

    # Replay
    replay_capacity: int = 200_000
    replay_min_size: int = 1_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 100_000
    per_eps: float = 1e-6

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    # Training loop
    train_every: int = 4
    total_steps: int = 200_000
    eval_every: int = 10_000
    eval_episodes: int = 3

    # Save
    save_dir: str = "checkpoints"
    save_every: int = 20_000

    # Misc
    seed: int = 42
    device: str = "auto"                # "cpu" | "cuda" | "auto"

    @classmethod
    def default(cls) -> "DQNConfig":
        return cls()
