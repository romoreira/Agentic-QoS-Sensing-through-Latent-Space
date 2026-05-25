"""Mock metric source — synthetic simulator producing the same
WindowMetrics that a real TRex+BESS testbed would.

Design
------
Traffic dynamics are delegated to `SliceTrafficGenerator` (in
upf_rl_traffic.dynamics), which is the same generator the TRex
conductor drives. Both sim and real testbed thus produce statistically
equivalent traffic — that is the bridge between training and deployment.

The generator emits offered rates for 4 slices:
  slice 0 = gold     (visible)
  slice 1 = silver   (visible)
  slice 2 = bronze   (visible)
  slice 3 = noise    (background pressure, NOT in observation)

The noise slice traverses the UPF and consumes N6 capacity but its
per-slice metrics are NOT exposed to the agent. It only manifests
through `n6_tx_mbps` / `n6_utilization` — the agent sees aggregate
load higher than the sum of the three visible slices.

Enforcement (per window, in this order)
---------------------------------------
  Stage A — MBR:
      after_mbr_i = min(offered_i, mbr_i)
      qer_red_i counts excess for the 3 visible slices (the noise slice
      may have its own MBR or run unenforced — see HiddenSliceMode below).
  Stage B — N6:
      All four after-MBR streams compete for N6 capacity. If the sum
      exceeds capacity, drops happen in priority order:
            gold > silver > bronze > noise
      The noise slice is treated as lowest priority by default (its
      role is to inject pressure when capacity is tight).

N6 capacity
-----------
Slow random walk around `N6_NOMINAL_CAPACITY_MBPS` (config.py).
Not directly observable.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from upf_rl_env.config import (
    EnvConfig,
    N6_CAPACITY_DRIFT_STD,
    N6_CAPACITY_MAX_FRAC,
    N6_CAPACITY_MIN_FRAC,
    N6_NOMINAL_CAPACITY_MBPS,
    NOMINAL_MBR_KBPS,
    N_SLICES,
)
from upf_rl_env.sources.base import MetricSource, WindowMetrics
from upf_rl_traffic.dynamics import SliceTrafficGenerator, TrafficDynamicsConfig


# Noise slice, aligned with optional TEID 31 in the TRex profile. Here we only need its MBR.
HIDDEN_SLICE_MBR_KBPS = 50_000  # 50 Mbps, fixed background slice (not under agent control)


class HiddenSliceMode(str, Enum):
    """How the noise slice is treated by the UPF.

    - ENFORCED: subject to its own (fixed) MBR. Realistic — operators
      usually still bound non-managed slices.
    - UNENFORCED: passes through whatever it offers, capped only by N6.
      Maximally adversarial.
    """

    ENFORCED = "enforced"
    UNENFORCED = "unenforced"


def _kbps_to_mbps(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 1000.0


class MockMetricSource(MetricSource):
    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        hidden_mode: HiddenSliceMode = HiddenSliceMode.ENFORCED,
        hidden_slice_mbr_kbps: int = HIDDEN_SLICE_MBR_KBPS,
    ) -> None:
        self.cfg = cfg or EnvConfig.default()
        self.hidden_mode = hidden_mode
        self.hidden_slice_mbr_kbps = hidden_slice_mbr_kbps
        self.rng = np.random.default_rng()

        # Traffic generator — 4 slices, first 3 visible
        self.dyn_cfg = TrafficDynamicsConfig(
            n_slices=4,
            n_visible=3,
            window_dt_s=self.cfg.window_dt_s,
            ood_freq_scale=getattr(self.cfg, "ood_freq_scale", 1.0),
            ood_amplitude_scale=getattr(self.cfg, "ood_amplitude_scale", 1.0),
        )
        self.generator = SliceTrafficGenerator(self.dyn_cfg)

        # N6 capacity drift state
        self._n6_capacity_mbps: float = N6_NOMINAL_CAPACITY_MBPS

        # Bookkeeping
        self._last_mbr_kbps = np.array(NOMINAL_MBR_KBPS, dtype=np.int64)
        self._last_offered_all: np.ndarray = np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> WindowMetrics:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.generator.reset(seed=seed)
        self._n6_capacity_mbps = N6_NOMINAL_CAPACITY_MBPS
        self._last_mbr_kbps = np.array(NOMINAL_MBR_KBPS, dtype=np.int64)
        return self.tick(self._last_mbr_kbps)

    def tick(self, mbr_kbps: np.ndarray) -> WindowMetrics:
        sample = self.generator.step()
        offered_all = sample.offered_mbps                  # (4,)
        visible_mask = sample.visible_mask                  # (4,) bool
        self._last_offered_all = offered_all.copy()

        cap = self._update_capacity()

        # MBR vector for ALL 4 slices — visible MBRs are agent-controlled,
        # noise slice has a fixed policy.
        full_mbr_mbps = np.empty(4, dtype=np.float32)
        full_mbr_mbps[:3] = _kbps_to_mbps(mbr_kbps)
        if self.hidden_mode == HiddenSliceMode.ENFORCED:
            full_mbr_mbps[3] = self.hidden_slice_mbr_kbps / 1000.0
        else:
            full_mbr_mbps[3] = np.inf  # no policer

        # Stage A: MBR enforcement on all slices
        after_mbr = np.minimum(offered_all, full_mbr_mbps)
        mbr_drops = np.maximum(0.0, offered_all - full_mbr_mbps)
        mbr_drops = np.where(np.isfinite(mbr_drops), mbr_drops, 0.0)

        # Stage B: N6 sharing with strict priority (gold > silver > bronze > noise)
        priority_order = [0, 1, 2, 3]  # by index
        remaining = cap
        rx_all = np.zeros(4, dtype=np.float32)
        for i in priority_order:
            rx_all[i] = min(float(after_mbr[i]), max(0.0, remaining))
            remaining = max(0.0, remaining - rx_all[i])

        # Per-slice loss vs. offered (entire pipeline)
        loss_all = np.where(
            offered_all > 1e-6,
            (offered_all - rx_all) / np.maximum(offered_all, 1e-6),
            0.0,
        )
        loss_all = np.clip(loss_all, 0.0, 1.0).astype(np.float32)

        # qer_red proxy (packet count) — only for visible slices, since that
        # is what the BESS exporter will give the env in production.
        avg_pkt_kb = 8.0  # 1000 bytes ~ 8 kb
        qer_red_visible = (
            mbr_drops[:3] * 1000.0 * self.cfg.window_dt_s / avg_pkt_kb
        ).astype(np.float32)

        # Aggregate N6 metrics — INCLUDE noise slice
        n6_tx_mbps = float(rx_all.sum())
        n6_util = float(n6_tx_mbps / max(cap, 1e-6))

        # Mask visible-only for the observation; noise slice does not appear
        offered_vis = offered_all[visible_mask].astype(np.float32)
        rx_vis = rx_all[visible_mask].astype(np.float32)
        loss_vis = loss_all[visible_mask].astype(np.float32)

        metrics = WindowMetrics(
            offered_mbps=offered_vis,
            rx_mbps=rx_vis,
            loss_rate=loss_vis,
            qer_red=qer_red_visible,
            n6_tx_mbps=n6_tx_mbps,
            n6_utilization=n6_util,
            mbr_kbps=mbr_kbps.astype(np.int64),
            timestamp_s=sample.timestamp_s,
        )
        self._last_mbr_kbps = mbr_kbps.astype(np.int64)
        return metrics

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_capacity(self) -> float:
        drift = self.rng.normal(
            0.0,
            N6_CAPACITY_DRIFT_STD * N6_NOMINAL_CAPACITY_MBPS * np.sqrt(self.cfg.window_dt_s),
        )
        self._n6_capacity_mbps += float(drift)
        lo = N6_NOMINAL_CAPACITY_MBPS * N6_CAPACITY_MIN_FRAC
        hi = N6_NOMINAL_CAPACITY_MBPS * N6_CAPACITY_MAX_FRAC
        self._n6_capacity_mbps = float(np.clip(self._n6_capacity_mbps, lo, hi))
        return self._n6_capacity_mbps

    # ------------------------------------------------------------------
    # Diagnostics — exposed for evaluation / latent probing
    # ------------------------------------------------------------------

    @property
    def regime(self) -> int:
        return self.generator.regime

    @property
    def hidden_state(self) -> dict:
        """Ground-truth hidden state. Use only for diagnostics and probing
        — the agent must not see this.
        """
        return {
            "regime": self.generator.regime,
            "n6_capacity_mbps": self._n6_capacity_mbps,
            # The noise slice's offered load from the *last* tick
            "hidden_offered_mbps": float(self._last_offered_all[-1]),
        }
