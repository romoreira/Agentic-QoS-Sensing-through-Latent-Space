#!/usr/bin/env python3
"""Pure-Python traffic model for four slice streams on one TRex uplink.

Order: gold, silver, bronze, noise.
The first three slices are the future DQN-controlled slices. The fourth one is
background pressure used to make the shared uplink/N6 capacity nontrivial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence

SLICE_NAMES = ("gold", "silver", "bronze", "noise")
SLICE_TEIDS = (1, 11, 21, 31)
SLICE_UE_IPS = ("10.250.0.1", "10.250.0.2", "10.250.0.3", "10.250.0.4")
SLICE_SRC_PORTS = (10001, 10011, 10021, 10031)
SLICE_PG_IDS = (1, 11, 21, 31)


@dataclass
class TrafficModelConfig:
    dt_s: float = 0.5
    base_means_mbps: Sequence[float] = (130.0, 70.0, 35.0, 30.0)
    base_stds_mbps: Sequence[float] = (25.0, 15.0, 10.0, 20.0)
    phases_rad: Sequence[float] = (0.0, 2.094, 4.188, 1.047)
    freqs_hz: Sequence[float] = (0.05, 0.13, 0.27)
    amplitudes: Sequence[float] = (0.20, 0.15, 0.10)
    ar1_rho: float = 0.80
    ar1_noise_std: float = 0.05
    burst_rate_per_s: float = 0.05
    burst_ln_mu: float = 0.0
    burst_ln_sigma: float = 0.6
    burst_decay_s: float = 1.5
    regime_dwell_s: Sequence[float] = (8.0, 25.0)
    max_offered_mbps: float = 400.0
    min_offered_mbps: float = 0.0
    regime_mean_mods: Sequence[Sequence[float]] = field(default_factory=lambda: (
        (1.00, 1.00, 1.00, 1.00),
        (1.30, 0.85, 0.85, 1.40),
        (0.85, 1.30, 0.95, 0.80),
        (1.15, 1.15, 1.15, 1.60),
    ))
    regime_amp_mods: Sequence[Sequence[float]] = field(default_factory=lambda: (
        (1.00, 1.00, 1.00, 1.00),
        (1.20, 0.90, 0.90, 1.50),
        (0.90, 1.20, 1.00, 0.70),
        (1.30, 1.30, 1.30, 1.70),
    ))


@dataclass
class TrafficSample:
    step: int
    t_s: float
    regime: int
    mbps: Dict[str, float]

    def as_csv_row(self) -> Dict[str, float | int]:
        row: Dict[str, float | int] = {"step": self.step, "t_s": self.t_s, "regime": self.regime}
        for name in SLICE_NAMES:
            row[f"{name}_mbps"] = self.mbps[name]
        row["visible_total_mbps"] = self.mbps["gold"] + self.mbps["silver"] + self.mbps["bronze"]
        row["total_mbps"] = row["visible_total_mbps"] + self.mbps["noise"]
        return row


class FourSliceTrafficModel:
    def __init__(self, cfg: Optional[TrafficModelConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = cfg or TrafficModelConfig()
        self.rng = random.Random(seed)
        self.step_idx = 0
        self.t_s = 0.0
        self.regime = self.rng.randrange(len(self.cfg.regime_mean_mods))
        self.regime_until_s = self._sample_regime_until()
        self.ar_state = [self.rng.gauss(0.0, 0.1) for _ in SLICE_NAMES]
        self.burst_amp = [0.0 for _ in SLICE_NAMES]

    def _sample_regime_until(self) -> float:
        lo, hi = self.cfg.regime_dwell_s
        return self.t_s + self.rng.uniform(float(lo), float(hi))

    def _maybe_switch_regime(self) -> None:
        if self.t_s < self.regime_until_s:
            return
        choices = [i for i in range(len(self.cfg.regime_mean_mods)) if i != self.regime]
        self.regime = self.rng.choice(choices)
        self.regime_until_s = self._sample_regime_until()

    def sample(self) -> TrafficSample:
        self._maybe_switch_regime()
        cfg = self.cfg
        mean_mod = cfg.regime_mean_mods[self.regime]
        amp_mod = cfg.regime_amp_mods[self.regime]
        decay = math.exp(-cfg.dt_s / max(cfg.burst_decay_s, 1e-6))
        p_burst = 1.0 - math.exp(-cfg.burst_rate_per_s * cfg.dt_s)

        mbps: Dict[str, float] = {}
        for i, name in enumerate(SLICE_NAMES):
            mean = cfg.base_means_mbps[i] * mean_mod[i]
            amp_scale = cfg.base_stds_mbps[i] * amp_mod[i]

            sinus = 0.0
            for freq, amp in zip(cfg.freqs_hz, cfg.amplitudes):
                sinus += amp * math.sin(2.0 * math.pi * freq * self.t_s + cfg.phases_rad[i])
            sinus_mbps = sinus * amp_scale

            self.ar_state[i] = cfg.ar1_rho * self.ar_state[i] + self.rng.gauss(0.0, cfg.ar1_noise_std)
            ar_mbps = self.ar_state[i] * amp_scale

            self.burst_amp[i] *= decay
            if self.rng.random() < p_burst:
                self.burst_amp[i] += self.rng.lognormvariate(cfg.burst_ln_mu, cfg.burst_ln_sigma) * mean * 0.5

            offered = mean + sinus_mbps + ar_mbps + self.burst_amp[i]
            offered = max(cfg.min_offered_mbps, min(cfg.max_offered_mbps, offered))
            mbps[name] = float(offered)

        sample = TrafficSample(step=self.step_idx, t_s=self.t_s, regime=self.regime, mbps=mbps)
        self.step_idx += 1
        self.t_s += cfg.dt_s
        return sample

    def iter_duration(self, duration_s: float) -> Iterable[TrafficSample]:
        n = int(math.ceil(duration_s / self.cfg.dt_s))
        for _ in range(n):
            yield self.sample()
