"""Mock metric source — a self-contained simulator producing the same
WindowMetrics that a real TRex+BESS testbed would. Used for training and
for fast iteration on the env / agent before plugging real hardware.

Design notes
------------
The simulator combines three layers of dynamics so that a temporal latent
representation has something nontrivial to encode:

  1. Per-slice offered load: a sum of sinusoids + AR(1) noise + Poisson
     bursts with lognormally-distributed magnitude and exponential decay.
  2. A regime variable (4 regimes by default) that switches every 8-25 s
     and modulates means and amplitudes per slice. The regime is hidden;
     the agent only sees its effects on telemetry.
  3. Hidden background traffic on N6 (AR(1)) and slowly-drifting N6
     capacity. Both are completely invisible to the agent; together they
     mean that an agent inferring "spare N6 headroom" must use telemetry
     correlations across time — exactly the signal a latent encoder
     should capture.

Enforcement is modeled as a two-stage pipeline per window:
  Stage A (MBR): rx_after_mbr = min(offered, mbr). Excess counts as
                  qer_red and as loss attributable to the policer.
  Stage B (N6):  if sum(rx_after_mbr) + bg > capacity, drop priority-
                 aware (gold first, silver next, bronze last). Adds to
                 loss but not to qer_red.

This is a deliberately simple physical model; replace later with a more
faithful M/M/1 per-slice or a BESS-faithful replay simulator if needed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from upf_rl_env.config import (
    BG_AR1_RHO,
    BG_MEAN_MBPS,
    BG_STD_MBPS,
    EnvConfig,
    N6_CAPACITY_DRIFT_STD,
    N6_CAPACITY_MAX_FRAC,
    N6_CAPACITY_MIN_FRAC,
    N6_NOMINAL_CAPACITY_MBPS,
    NOMINAL_MBR_KBPS,
    N_SLICES,
)
from upf_rl_env.sources.base import MetricSource, WindowMetrics


def _kbps_to_mbps(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 1000.0


class MockMetricSource(MetricSource):
    def __init__(self, cfg: Optional[EnvConfig] = None) -> None:
        self.cfg = cfg or EnvConfig.default()
        self.rng = np.random.default_rng()

        # Per-slice offered-load shape: nominal target in Mbps.
        # Means picked so that without MBR, sum oversubscribes N6 sometimes.
        # gold ~ 130, silver ~ 70, bronze ~ 35 → 235 mean, peaks well above N6 cap.
        self._base_means_mbps = np.array([130.0, 70.0, 35.0], dtype=np.float32)
        self._base_stds_mbps = np.array([25.0, 15.0, 10.0], dtype=np.float32)

        # Regime modulators applied multiplicatively to means and amplitudes.
        # 4 regimes: nominal, gold-heavy, silver-heavy, all-hot.
        self._regime_mean_mods = np.array(
            [
                [1.00, 1.00, 1.00],   # 0 nominal
                [1.30, 0.85, 0.85],   # 1 gold-heavy
                [0.85, 1.30, 0.95],   # 2 silver-heavy
                [1.15, 1.15, 1.15],   # 3 all-hot
            ],
            dtype=np.float32,
        )
        self._regime_amp_mods = np.array(
            [
                [1.00, 1.00, 1.00],
                [1.20, 0.90, 0.90],
                [0.90, 1.20, 1.00],
                [1.30, 1.30, 1.30],
            ],
            dtype=np.float32,
        )

        # Mutable state
        self._t: float = 0.0
        self._step: int = 0
        self._regime: int = 0
        self._regime_until_t: float = 0.0
        self._ar_state = np.zeros(N_SLICES, dtype=np.float32)
        self._burst_amplitude = np.zeros(N_SLICES, dtype=np.float32)
        self._bg_state: float = BG_MEAN_MBPS
        self._n6_capacity_mbps: float = N6_NOMINAL_CAPACITY_MBPS
        self._last_mbr_kbps = np.array(NOMINAL_MBR_KBPS, dtype=np.int64)

    # ---- API ----------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> WindowMetrics:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0.0
        self._step = 0
        self._regime = int(self.rng.integers(0, self.cfg.n_regimes))
        self._regime_until_t = self._sample_regime_dwell()
        self._ar_state = self.rng.normal(0.0, 1.0, size=N_SLICES).astype(np.float32) * 0.1
        self._burst_amplitude = np.zeros(N_SLICES, dtype=np.float32)
        self._bg_state = float(self.rng.normal(BG_MEAN_MBPS, BG_STD_MBPS * 0.5))
        self._n6_capacity_mbps = N6_NOMINAL_CAPACITY_MBPS
        self._last_mbr_kbps = np.array(NOMINAL_MBR_KBPS, dtype=np.int64)
        return self.tick(self._last_mbr_kbps)

    def tick(self, mbr_kbps: np.ndarray) -> WindowMetrics:
        self._maybe_switch_regime()

        offered = self._sample_offered_mbps()
        bg = self._update_background()
        cap = self._update_capacity()

        # Stage A: MBR enforcement (in Mbps for the math)
        mbr_mbps = _kbps_to_mbps(mbr_kbps)
        after_mbr = np.minimum(offered, mbr_mbps)
        mbr_drops_mbps = np.maximum(0.0, offered - mbr_mbps)

        # Stage B: N6 sharing with priority Gold > Silver > Bronze
        remaining = max(0.0, cap - bg)
        rx = np.zeros(N_SLICES, dtype=np.float32)
        for i in range(N_SLICES):
            rx[i] = min(after_mbr[i], remaining)
            remaining = max(0.0, remaining - rx[i])

        # Loss rate per slice (vs. originally offered traffic)
        loss_rate = np.where(offered > 1e-6, (offered - rx) / np.maximum(offered, 1e-6), 0.0)
        loss_rate = np.clip(loss_rate, 0.0, 1.0).astype(np.float32)

        # qer_red as a packet-count proxy: drops scaled to "packets" assuming
        # an average packet of 1000 bytes = 8 kb. Window length window_dt_s.
        avg_pkt_kb = 8.0
        qer_red = (mbr_drops_mbps * 1000.0 * self.cfg.window_dt_s / avg_pkt_kb).astype(np.float32)

        n6_tx_mbps = float(np.sum(rx) + bg)
        n6_util = float(n6_tx_mbps / max(cap, 1e-6))

        metrics = WindowMetrics(
            offered_mbps=offered.astype(np.float32),
            rx_mbps=rx.astype(np.float32),
            loss_rate=loss_rate,
            qer_red=qer_red,
            n6_tx_mbps=n6_tx_mbps,
            n6_utilization=n6_util,
            mbr_kbps=mbr_kbps.astype(np.int64),
            timestamp_s=self._t,
        )

        self._t += self.cfg.window_dt_s
        self._step += 1
        self._last_mbr_kbps = mbr_kbps.astype(np.int64)
        return metrics

    # ---- Internals ----------------------------------------------------

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

        means = self._base_means_mbps * mean_mod * cfg.ood_amplitude_scale

        # Sinusoidal trend per slice — each slice phase-shifted off the freq table
        sinus = np.zeros(N_SLICES, dtype=np.float32)
        for k, (f, a) in enumerate(zip(cfg.base_freqs_hz, cfg.base_amplitudes)):
            f_eff = f * cfg.ood_freq_scale
            phases = np.array([0.0, 2.094, 4.188], dtype=np.float32)  # 0, 2π/3, 4π/3
            sinus += a * np.sin(2.0 * np.pi * f_eff * self._t + phases)
        amp_scale = self._base_stds_mbps * amp_mod
        sinus_mbps = sinus * amp_scale

        # AR(1) noise
        eps = self.rng.normal(0.0, cfg.ar1_noise_std, size=N_SLICES).astype(np.float32)
        self._ar_state = cfg.ar1_rho * self._ar_state + eps
        ar_mbps = self._ar_state * amp_scale

        # Poisson bursts (per-slice independent) — magnitude lognormal, decay exp
        decay = np.exp(-cfg.window_dt_s / max(cfg.burst_decay_s, 1e-3)).astype(np.float32)
        self._burst_amplitude = self._burst_amplitude * decay
        p_burst = 1.0 - np.exp(-cfg.burst_rate_per_s * cfg.window_dt_s)
        for i in range(N_SLICES):
            if self.rng.random() < p_burst:
                mag = float(self.rng.lognormal(cfg.burst_lognormal_mu, cfg.burst_lognormal_sigma))
                # Burst magnitude scaled to half the base mean — sizeable but not absurd
                self._burst_amplitude[i] += mag * means[i] * 0.5

        offered = means + sinus_mbps + ar_mbps + self._burst_amplitude
        offered = np.maximum(offered, 0.0).astype(np.float32)
        return offered

    def _update_background(self) -> float:
        eps = self.rng.normal(0.0, BG_STD_MBPS * np.sqrt(1.0 - BG_AR1_RHO**2))
        self._bg_state = BG_AR1_RHO * (self._bg_state - BG_MEAN_MBPS) + BG_MEAN_MBPS + float(eps)
        self._bg_state = float(np.clip(self._bg_state, 0.0, BG_MEAN_MBPS * 3.0))
        return self._bg_state

    def _update_capacity(self) -> float:
        # Random walk with reflecting bounds
        drift = self.rng.normal(0.0, N6_CAPACITY_DRIFT_STD * N6_NOMINAL_CAPACITY_MBPS * np.sqrt(self.cfg.window_dt_s))
        self._n6_capacity_mbps += float(drift)
        lo = N6_NOMINAL_CAPACITY_MBPS * N6_CAPACITY_MIN_FRAC
        hi = N6_NOMINAL_CAPACITY_MBPS * N6_CAPACITY_MAX_FRAC
        self._n6_capacity_mbps = float(np.clip(self._n6_capacity_mbps, lo, hi))
        return self._n6_capacity_mbps

    @property
    def regime(self) -> int:
        """Current (hidden) regime index — exposed for diagnostics only."""
        return self._regime

    @property
    def hidden_state(self) -> dict:
        """Hidden ground-truth state, for evaluation / latent probing."""
        return {
            "regime": self._regime,
            "bg_mbps": self._bg_state,
            "n6_capacity_mbps": self._n6_capacity_mbps,
        }
