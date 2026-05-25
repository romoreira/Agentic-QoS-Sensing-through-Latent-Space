"""Slice traffic dynamics used by the mock metric source.

The real TRex profile is intentionally simple and fixed, one physical uplink
with multiple TEIDs. While TRex/BESS readings are still mocked, this module is
the source of the learning challenge for the DQN.

Slice model
-----------
The generator emits offered rates for four slices. The first three are visible
to the agent, Gold, Silver, Bronze. The fourth is a Noise slice. It consumes
shared N6 capacity in the mock simulator, but it is masked out of the
observation.

Dynamics
--------
Per slice, offered_mbps(t) = mean_i(t) + amp_i(t) * (sinus + AR1 + bursts)

  mean_i(t)   : base mean modulated by the current regime
  amp_i(t)    : base std modulated by the current regime
  sinus       : sum of three sinusoids with per-slice phase shifts
  AR1         : correlated short-term noise
  bursts      : Poisson-triggered lognormal events with exponential decay

A hidden regime switches every 8 to 25 seconds. It is not part of the DQN
observation, but it is logged for audit and future latent-space probing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
# Defaults for the mock learning signal. Keep them here so training and
# future latent-space probes use the same traffic process.
# ----------------------------------------------------------------------

# Base means (Mbps) per slice. Order: gold, silver, bronze, noise.
DEFAULT_BASE_MEANS_MBPS = (130.0, 70.0, 35.0, 30.0)
DEFAULT_BASE_STDS_MBPS = (25.0, 15.0, 10.0, 20.0)  # noise is high-variance

# Per-slice phase offsets (radians) — different slices peak at different times.
DEFAULT_PHASES = (0.0, 2.094, 4.188, 1.047)  # 0, 2π/3, 4π/3, π/3

# Three carrier frequencies (Hz) summed for the sinusoidal trend.
DEFAULT_FREQS_HZ = (0.05, 0.13, 0.27)
DEFAULT_AMPLITUDES = (0.20, 0.15, 0.10)

# AR(1)
DEFAULT_AR1_RHO = 0.80
DEFAULT_AR1_NOISE_STD = 0.05

# Poisson bursts
DEFAULT_BURST_RATE_PER_S = 0.05
DEFAULT_BURST_LN_MU = 0.0
DEFAULT_BURST_LN_SIGMA = 0.6
DEFAULT_BURST_DECAY_S = 1.5

# Regime switching
DEFAULT_REGIME_DWELL_S = (8.0, 25.0)
DEFAULT_N_REGIMES = 4

# Per-regime modulators applied to (mean, amplitude). Shape (n_regimes, 4 slices).
# Row 0 = nominal; rows 1-3 emphasize different slices. Noise slice has its
# own column so it can spike independently of the visible slices.
DEFAULT_REGIME_MEAN_MODS = (
    (1.00, 1.00, 1.00, 1.00),   # 0 nominal
    (1.30, 0.85, 0.85, 1.40),   # 1 gold-heavy + noise-heavy
    (0.85, 1.30, 0.95, 0.80),   # 2 silver-heavy, noise quiet
    (1.15, 1.15, 1.15, 1.60),   # 3 all-hot + noise adversarial
)
DEFAULT_REGIME_AMP_MODS = (
    (1.00, 1.00, 1.00, 1.00),
    (1.20, 0.90, 0.90, 1.50),
    (0.90, 1.20, 1.00, 0.70),
    (1.30, 1.30, 1.30, 1.70),
)

# Minimum / maximum offered rate per slice (Mbps) — applied as a clip.
DEFAULT_MIN_OFFERED_MBPS = 0.0
DEFAULT_MAX_OFFERED_MBPS = 400.0


@dataclass
class TrafficDynamicsConfig:
    n_slices: int = 4
    n_visible: int = 3                    # first n_visible are exposed; rest noise
    window_dt_s: float = 0.2

    base_means_mbps: Sequence[float] = field(default_factory=lambda: DEFAULT_BASE_MEANS_MBPS)
    base_stds_mbps: Sequence[float] = field(default_factory=lambda: DEFAULT_BASE_STDS_MBPS)
    phases: Sequence[float] = field(default_factory=lambda: DEFAULT_PHASES)

    freqs_hz: Sequence[float] = field(default_factory=lambda: DEFAULT_FREQS_HZ)
    amplitudes: Sequence[float] = field(default_factory=lambda: DEFAULT_AMPLITUDES)

    ar1_rho: float = DEFAULT_AR1_RHO
    ar1_noise_std: float = DEFAULT_AR1_NOISE_STD

    burst_rate_per_s: float = DEFAULT_BURST_RATE_PER_S
    burst_ln_mu: float = DEFAULT_BURST_LN_MU
    burst_ln_sigma: float = DEFAULT_BURST_LN_SIGMA
    burst_decay_s: float = DEFAULT_BURST_DECAY_S

    regime_dwell_s: tuple = DEFAULT_REGIME_DWELL_S
    n_regimes: int = DEFAULT_N_REGIMES
    regime_mean_mods: Sequence[Sequence[float]] = field(default_factory=lambda: DEFAULT_REGIME_MEAN_MODS)
    regime_amp_mods: Sequence[Sequence[float]] = field(default_factory=lambda: DEFAULT_REGIME_AMP_MODS)

    min_offered_mbps: float = DEFAULT_MIN_OFFERED_MBPS
    max_offered_mbps: float = DEFAULT_MAX_OFFERED_MBPS

    # OOD shifts (multipliers on freq / amplitude scales — keep at 1.0 for IID)
    ood_freq_scale: float = 1.0
    ood_amplitude_scale: float = 1.0

    def validate(self) -> None:
        if self.n_visible > self.n_slices:
            raise ValueError("n_visible cannot exceed n_slices")
        for attr in ("base_means_mbps", "base_stds_mbps", "phases"):
            if len(getattr(self, attr)) < self.n_slices:
                raise ValueError(f"{attr} has fewer entries than n_slices={self.n_slices}")


@dataclass
class TrafficSample:
    """One window of generator output."""
    offered_mbps: np.ndarray            # shape (n_slices,) — all slices
    visible_mask: np.ndarray            # shape (n_slices,) bool — True for slices the agent sees
    regime: int
    timestamp_s: float

    @property
    def visible_offered_mbps(self) -> np.ndarray:
        return self.offered_mbps[self.visible_mask]

    @property
    def noise_offered_mbps(self) -> np.ndarray:
        return self.offered_mbps[~self.visible_mask]


class SliceTrafficGenerator:
    """Stateful per-slice offered-load generator."""

    def __init__(self, cfg: Optional[TrafficDynamicsConfig] = None) -> None:
        self.cfg = cfg or TrafficDynamicsConfig()
        self.cfg.validate()

        n = self.cfg.n_slices
        self._base_means = np.asarray(self.cfg.base_means_mbps[:n], dtype=np.float32)
        self._base_stds = np.asarray(self.cfg.base_stds_mbps[:n], dtype=np.float32)
        self._phases = np.asarray(self.cfg.phases[:n], dtype=np.float32)
        self._regime_mean_mods = np.asarray(self.cfg.regime_mean_mods, dtype=np.float32)[:, :n]
        self._regime_amp_mods = np.asarray(self.cfg.regime_amp_mods, dtype=np.float32)[:, :n]

        self.rng = np.random.default_rng()

        # Mutable state
        self._t: float = 0.0
        self._regime: int = 0
        self._regime_until_t: float = 0.0
        self._ar_state = np.zeros(n, dtype=np.float32)
        self._burst_amplitude = np.zeros(n, dtype=np.float32)

        # Visible mask
        self._visible_mask = np.zeros(n, dtype=bool)
        self._visible_mask[: self.cfg.n_visible] = True

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> TrafficSample:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0.0
        self._regime = int(self.rng.integers(0, self.cfg.n_regimes))
        self._regime_until_t = self._sample_regime_dwell()
        self._ar_state = (self.rng.normal(0.0, 1.0, size=self.cfg.n_slices).astype(np.float32) * 0.1)
        self._burst_amplitude = np.zeros(self.cfg.n_slices, dtype=np.float32)
        return self.step()

    def step(self) -> TrafficSample:
        self._maybe_switch_regime()
        offered = self._sample_offered_mbps()

        sample = TrafficSample(
            offered_mbps=offered.astype(np.float32),
            visible_mask=self._visible_mask.copy(),
            regime=int(self._regime),
            timestamp_s=float(self._t),
        )
        self._t += self.cfg.window_dt_s
        return sample

    @property
    def t(self) -> float:
        return self._t

    @property
    def regime(self) -> int:
        return int(self._regime)

    @property
    def visible_mask(self) -> np.ndarray:
        return self._visible_mask.copy()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_regime_dwell(self) -> float:
        lo, hi = self.cfg.regime_dwell_s
        return self._t + float(self.rng.uniform(lo, hi))

    def _maybe_switch_regime(self) -> None:
        if self._t >= self._regime_until_t:
            choices = [r for r in range(self.cfg.n_regimes) if r != self._regime]
            self._regime = int(self.rng.choice(choices))
            self._regime_until_t = self._sample_regime_dwell()

    def _sample_offered_mbps(self) -> np.ndarray:
        cfg = self.cfg
        mean_mod = self._regime_mean_mods[self._regime]
        amp_mod = self._regime_amp_mods[self._regime]

        means = self._base_means * mean_mod * cfg.ood_amplitude_scale
        amp_scale = self._base_stds * amp_mod

        # Sinusoidal trend — sum of carriers; per-slice phase shift on each
        sinus = np.zeros(cfg.n_slices, dtype=np.float32)
        for f, a in zip(cfg.freqs_hz, cfg.amplitudes):
            f_eff = f * cfg.ood_freq_scale
            sinus += a * np.sin(2.0 * np.pi * f_eff * self._t + self._phases)
        sinus_mbps = sinus * amp_scale

        # AR(1) noise
        eps = self.rng.normal(0.0, cfg.ar1_noise_std, size=cfg.n_slices).astype(np.float32)
        self._ar_state = cfg.ar1_rho * self._ar_state + eps
        ar_mbps = self._ar_state * amp_scale

        # Poisson bursts with exponential decay
        decay = float(np.exp(-cfg.window_dt_s / max(cfg.burst_decay_s, 1e-3)))
        self._burst_amplitude *= decay
        p_burst = 1.0 - np.exp(-cfg.burst_rate_per_s * cfg.window_dt_s)
        for i in range(cfg.n_slices):
            if self.rng.random() < p_burst:
                mag = float(self.rng.lognormal(cfg.burst_ln_mu, cfg.burst_ln_sigma))
                self._burst_amplitude[i] += mag * means[i] * 0.5

        offered = means + sinus_mbps + ar_mbps + self._burst_amplitude
        return np.clip(offered, cfg.min_offered_mbps, cfg.max_offered_mbps).astype(np.float32)
