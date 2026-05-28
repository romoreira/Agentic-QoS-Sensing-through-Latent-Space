#!/usr/bin/env python3
"""Enriched four-slice traffic model with richer *latent* structure.

Motivation
----------
The original ``sine_gold_safe_overload`` model has exactly two hidden factors:
a binary noise on/off process and a fixed sine phase. That is enough to show a
DQN beats fixed/reactive control, but it is *too easy* to justify a learned
latent-space state estimator: a 1-bit hidden state plus a known phase can be
tracked by almost anything.

This model deliberately enriches the latent state so that a predictive
embedding (UNUM-style) has something real to learn that a 1- or 2-step delta
cannot capture:

  1. **3-state hidden noise** (quiet -> ramp -> burst) driven by a small Markov
     chain with dwell times. The agent never sees the noise offered load; it
     only sees its *consequence* through the shared-uplink proxy (pressure,
     shortage). Inferring the current noise state from that history is exactly
     the kind of latent inference the embedder is meant to do.

  2. **Slowly drifting uplink capacity** (bounded random walk). This is a
     path-property latent analogous to wireless capacity changes in UNUM. It is
     invisible to the agent's features but is threaded into the proxy, so it
     directly changes ``available_for_gold`` and ``gold_shortage``.

  3. **Long-period demand envelope on Gold** (period ~600 s). This rewards a
     *longer* observation context: a short window cannot tell whether Gold is
     on a slow rise or a slow fall, but a long window can.

The sample interface is a strict superset of ``traffic_model.TrafficSample``:
``step``, ``t_s``, ``regime`` and ``mbps`` are all present, so the existing
scripts keep working. Two extra fields are added:

  * ``capacity_mbps`` -- the *time-varying* uplink capacity for this step. Pass
    it to ``shared_uplink_proxy(offered, mbr, capacity_mbps)`` instead of the
    fixed CLI value when running on the enriched environment.
  * ``hidden_state`` -- a dict logged for audit only (noise state/level,
    capacity). NEVER feed this to the agent or the embedder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, Iterable, Optional, Sequence

# Reuse the canonical slice identity from the base model so TEIDs / PG-IDs match.
from traffic_model import SLICE_NAMES  # ("gold", "silver", "bronze", "noise")


# Hidden noise states
NOISE_QUIET = 0
NOISE_RAMP = 1
NOISE_BURST = 2
NOISE_STATE_NAMES = ("quiet", "ramp", "burst")


@dataclass
class EnrichedSample:
    """Superset of TrafficSample with a per-step capacity and hidden state."""
    step: int
    t_s: float
    regime: int                       # = noise state index (0 quiet, 1 ramp, 2 burst)
    mbps: Dict[str, float]            # gold, silver, bronze, noise
    capacity_mbps: float              # time-varying uplink capacity for this step
    hidden_state: Dict[str, float]    # audit only, NEVER an agent/embedder feature


@dataclass
class EnrichedLatentConfig:
    dt_s: float = 1.0

    # Controlled slices (kept close to sine_gold_safe_overload for comparability)
    controlled_means_mbps: Sequence[float] = (150.0, 210.0, 140.0)
    controlled_amps_mbps: Sequence[float] = (50.0, 90.0, 60.0)
    controlled_periods_s: Sequence[float] = (300.0, 420.0, 260.0)
    controlled_phases_rad: Sequence[float] = (0.0, 2.094, 4.188)
    controlled_jitter_mbps: float = 1.0

    # (1) Long-period demand envelope on Gold (latent with long memory)
    gold_envelope_period_s: float = 600.0
    gold_envelope_depth: float = 0.25     # +/- 25% slow modulation of Gold mean
    gold_envelope_phase_rad: float = 0.7

    # (2) 3-state hidden noise Markov chain
    noise_level_mbps: Sequence[float] = (30.0, 130.0, 300.0)  # quiet, ramp, burst
    noise_jitter_mbps: float = 4.0
    # Dwell time (s) ranges per state -> structure the embedder can learn
    noise_dwell_s: Sequence[Sequence[float]] = field(default_factory=lambda: (
        (90.0, 180.0),   # quiet
        (15.0, 35.0),    # ramp
        (20.0, 55.0),    # burst
    ))
    # Transition targets when a dwell expires (rows: from-state)
    # quiet -> ramp; ramp -> {burst mostly, sometimes back to quiet};
    # burst -> {ramp (cooldown) mostly, sometimes quiet}
    noise_transitions: Sequence[Sequence[float]] = field(default_factory=lambda: (
        (0.00, 1.00, 0.00),   # from quiet
        (0.20, 0.00, 0.80),   # from ramp
        (0.25, 0.65, 0.10),   # from burst
    ))

    # (3) Slowly drifting uplink capacity (path-property latent)
    capacity_base_mbps: float = 200.0
    capacity_ar1_rho: float = 0.97        # high -> slow, autocorrelated drift
    capacity_step_std_frac: float = 0.015 # per-step innovation as frac of base
    capacity_min_frac: float = 0.80
    capacity_max_frac: float = 1.10
    # Burst slightly depresses effective capacity (contention), making the
    # burst+low-capacity coincidence the hardest moment to protect Gold.
    capacity_burst_penalty_frac: float = 0.06

    max_offered_mbps: float = 400.0
    min_offered_mbps: float = 0.0


class EnrichedLatentTrafficModel:
    """Enriched generator. Same iteration interface as the base models."""

    def __init__(self, cfg: Optional[EnrichedLatentConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = cfg or EnrichedLatentConfig()
        self.rng = random.Random(seed)
        self.step_idx = 0
        self.t_s = 0.0

        # Hidden noise state machine
        self.noise_state = NOISE_QUIET
        self.noise_until_s = self._sample_noise_until(self.noise_state)

        # Hidden capacity drift state (multiplicative factor around base)
        self.cap_factor = 1.0

    # ---- hidden noise ----
    def _sample_noise_until(self, state: int) -> float:
        lo, hi = self.cfg.noise_dwell_s[state]
        return self.t_s + self.rng.uniform(float(lo), float(hi))

    def _maybe_switch_noise(self) -> None:
        if self.t_s < self.noise_until_s:
            return
        probs = self.cfg.noise_transitions[self.noise_state]
        r = self.rng.random()
        acc = 0.0
        nxt = self.noise_state
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                nxt = i
                break
        self.noise_state = nxt
        self.noise_until_s = self._sample_noise_until(self.noise_state)

    # ---- hidden capacity ----
    def _step_capacity(self) -> float:
        cfg = self.cfg
        innov = self.rng.gauss(0.0, cfg.capacity_step_std_frac)
        # AR(1) random walk on the multiplicative factor, mean-reverting to 1.0
        self.cap_factor = 1.0 + cfg.capacity_ar1_rho * (self.cap_factor - 1.0) + innov
        self.cap_factor = max(cfg.capacity_min_frac, min(cfg.capacity_max_frac, self.cap_factor))
        cap = cfg.capacity_base_mbps * self.cap_factor
        if self.noise_state == NOISE_BURST:
            cap *= (1.0 - cfg.capacity_burst_penalty_frac)
        return float(cap)

    def sample(self) -> EnrichedSample:
        self._maybe_switch_noise()
        cfg = self.cfg
        mbps: Dict[str, float] = {}

        # Long-period envelope multiplier for Gold (slow latent)
        env = 1.0 + cfg.gold_envelope_depth * math.sin(
            2.0 * math.pi * self.t_s / max(cfg.gold_envelope_period_s, 1e-9)
            + cfg.gold_envelope_phase_rad
        )

        for i, name in enumerate(SLICE_NAMES[:3]):
            mean = cfg.controlled_means_mbps[i]
            if name == "gold":
                mean *= env
            amp = cfg.controlled_amps_mbps[i]
            period = max(float(cfg.controlled_periods_s[i]), 1e-9)
            phase = cfg.controlled_phases_rad[i]
            value = mean + amp * math.sin(2.0 * math.pi * self.t_s / period + phase)
            if cfg.controlled_jitter_mbps > 0:
                value += self.rng.gauss(0.0, cfg.controlled_jitter_mbps)
            mbps[name] = float(max(cfg.min_offered_mbps, min(cfg.max_offered_mbps, value)))

        # Hidden noise offered load from the current Markov state
        noise_level = cfg.noise_level_mbps[self.noise_state]
        noise = noise_level + self.rng.gauss(0.0, cfg.noise_jitter_mbps)
        mbps["noise"] = float(max(cfg.min_offered_mbps, min(cfg.max_offered_mbps, noise)))

        capacity = self._step_capacity()

        hidden_state = {
            "noise_state": int(self.noise_state),
            "noise_level_mbps": float(noise_level),
            "capacity_mbps": float(capacity),
            "gold_envelope": float(env),
        }

        sample = EnrichedSample(
            step=self.step_idx,
            t_s=self.t_s,
            regime=int(self.noise_state),
            mbps=mbps,
            capacity_mbps=capacity,
            hidden_state=hidden_state,
        )
        self.step_idx += 1
        self.t_s += cfg.dt_s
        return sample

    def iter_duration(self, duration_s: float) -> Iterable[EnrichedSample]:
        n = int(math.ceil(duration_s / self.cfg.dt_s))
        for _ in range(n):
            yield self.sample()


def make_enriched_traffic_model(kind: str, dt_s: float, seed: Optional[int] = None):
    """Factory mirroring traffic_model.make_traffic_model.

    kind='latent_rich'  -> full enriched config (default)
    kind='latent_mild'  -> gentler capacity drift / noise for sanity runs
    """
    kind = (kind or "latent_rich").strip().lower().replace("-", "_")
    if kind in ("latent_rich", "rich", "enriched"):
        return EnrichedLatentTrafficModel(EnrichedLatentConfig(dt_s=dt_s), seed=seed)
    if kind in ("latent_mild", "mild"):
        return EnrichedLatentTrafficModel(
            EnrichedLatentConfig(
                dt_s=dt_s,
                noise_level_mbps=(30.0, 90.0, 200.0),
                capacity_step_std_frac=0.008,
                capacity_burst_penalty_frac=0.03,
                gold_envelope_depth=0.15,
            ),
            seed=seed,
        )
    raise ValueError(f"unknown enriched traffic model: {kind}")


if __name__ == "__main__":
    # Quick self-audit of ranges (no TRex, no torch).
    m = make_enriched_traffic_model("latent_rich", dt_s=1.0, seed=7)
    import statistics as st
    cols = {k: [] for k in ("gold", "silver", "bronze", "noise")}
    caps = []
    states = {0: 0, 1: 0, 2: 0}
    for s in m.iter_duration(1800):
        for k in cols:
            cols[k].append(s.mbps[k])
        caps.append(s.capacity_mbps)
        states[s.regime] += 1
    for k, v in cols.items():
        print(f"{k:7s} min={min(v):6.1f} max={max(v):6.1f} mean={st.mean(v):6.1f}")
    print(f"capacity min={min(caps):6.1f} max={max(caps):6.1f} mean={st.mean(caps):6.1f}")
    print(f"noise state dwell counts (quiet/ramp/burst) = {states}")
