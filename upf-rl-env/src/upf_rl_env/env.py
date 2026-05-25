"""SliceQoSEnv — a Gymnasium environment for slice-aware MBR control.

Observation
-----------
Concatenation of the last K windows of 17 features each (default 5x17=85).
Flat Box space, normalized to ~[-1, 1] for stability under DQN.

Action
------
Discrete(15). See action/space.py for the per-slice MBR percentages each
index maps to.

Reward
------
See reward.py.

Episode boundary
----------------
Truncates after `max_episode_steps` (default 3000 = 600 s at 200 ms).
Never `terminated`. Reset re-seeds the traffic generator.

Plugging real sources
---------------------
The env takes the MetricSource and ActionApplier as constructor args.
Default is Mock + Mock. To run against a real testbed:

    env = SliceQoSEnv(
        metric_source=BessMetricSource(slice_teids=(1, 11, 21)),
        action_applier=RealActionApplier(slice_teids=(1, 11, 21),
                                          nominal_mbr_kbps=(200_000, 100_000, 50_000)),
    )
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from upf_rl_env.action.applier import ActionApplier, MockActionApplier
from upf_rl_env.action.space import N_ACTIONS, action_description
from upf_rl_env.config import EnvConfig, NOMINAL_MBR_KBPS, SLICE_TEIDS
from upf_rl_env.features import FEATURES_PER_WINDOW, metrics_to_features
from upf_rl_env.monitor.logger import METRICS_FIELDNAMES, MetricsLogger
from upf_rl_env.reward import compute_reward
from upf_rl_env.sources.base import MetricSource
from upf_rl_env.sources.mock import MockMetricSource

logger = logging.getLogger(__name__)


class SliceQoSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        metric_source: Optional[MetricSource] = None,
        action_applier: Optional[ActionApplier] = None,
        log_csv_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig.default()

        # Spaces
        self.action_space = spaces.Discrete(N_ACTIONS)
        obs_dim = self.cfg.frame_stack_k * FEATURES_PER_WINDOW
        # Generous bounds — normalized features should sit in ~[-1.5, 1.5]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Components (lazy-replaceable)
        self.metric_source: MetricSource = metric_source or MockMetricSource(self.cfg)
        self.action_applier: ActionApplier = action_applier or MockActionApplier(
            slice_teids=SLICE_TEIDS, nominal_mbr_kbps=NOMINAL_MBR_KBPS
        )

        # Logging
        self._log_csv_path = log_csv_path
        self._mlogger: Optional[MetricsLogger] = None

        # Episode state
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self.cfg.frame_stack_k)
        self._prev_action: Optional[int] = None
        self._step_in_episode: int = 0
        self._episode_idx: int = 0
        self._last_wall_t: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.metric_source.reset(seed=seed)
        else:
            self.metric_source.reset()

        # Reset enforcement to nominal
        mbr_kbps = self.action_applier.reset()
        self._prev_action = None
        self._step_in_episode = 0
        self._episode_idx += 1

        # Open / rotate logger if configured
        self._maybe_open_logger()

        # Bootstrap frame buffer with K-1 ticks at nominal, plus 1 final
        self._frame_buffer.clear()
        for _ in range(self.cfg.frame_stack_k):
            m = self.metric_source.tick(mbr_kbps)
            self._frame_buffer.append(metrics_to_features(m, self.cfg))

        obs = self._build_obs()
        info = {"mbr_kbps": mbr_kbps.tolist(), "episode": self._episode_idx}
        self._last_wall_t = time.monotonic()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if not (0 <= action < N_ACTIONS):
            raise ValueError(f"Invalid action {action} — must be in [0, {N_ACTIONS})")

        # Apply the action (enforces new MBR)
        mbr_kbps = self.action_applier.apply(action)

        # Optional real-time pacing
        if self.cfg.realtime:
            elapsed = time.monotonic() - self._last_wall_t
            sleep = self.cfg.window_dt_s - elapsed
            if sleep > 0:
                time.sleep(sleep)
            self._last_wall_t = time.monotonic()

        # Read next window
        m = self.metric_source.tick(mbr_kbps)

        # Update frame buffer
        self._frame_buffer.append(metrics_to_features(m, self.cfg))

        # Reward
        reward, components = compute_reward(m, action, self._prev_action, self.cfg)

        # Bookkeeping
        self._step_in_episode += 1
        truncated = self._step_in_episode >= self.cfg.max_episode_steps
        terminated = False

        # Build obs + info
        obs = self._build_obs()
        info: dict[str, Any] = {
            "mbr_kbps": mbr_kbps.tolist(),
            "metrics": m.as_dict(),
            "reward_components": components,
            "action_desc": action_description(action),
        }
        # Expose hidden state if the source has any (mock only — useful for eval / probing)
        if hasattr(self.metric_source, "hidden_state"):
            info["hidden_state"] = self.metric_source.hidden_state  # type: ignore[attr-defined]

        # Log row
        if self._mlogger is not None:
            hidden = info.get("hidden_state", {})
            row = {
                "step": self._step_in_episode,
                "action": action,
                "reward": reward,
                "regime": hidden.get("regime", ""),
                "noise_offered_mbps": hidden.get("hidden_offered_mbps", ""),
                "n6_capacity_mbps": hidden.get("n6_capacity_mbps", ""),
            }
            row.update(m.as_dict())
            self._mlogger.log(row)

        self._prev_action = action
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._mlogger is not None:
            self._mlogger.close()
            self._mlogger = None
        self.metric_source.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        # frame_buffer holds frame_stack_k arrays of shape (FEATURES_PER_WINDOW,)
        return np.concatenate(list(self._frame_buffer), axis=0).astype(np.float32)

    def _maybe_open_logger(self) -> None:
        if self._log_csv_path is None:
            return
        # Close previous logger to flush its file
        if self._mlogger is not None:
            self._mlogger.close()
        base, ext = os.path.splitext(self._log_csv_path)
        # One file per episode so traces stay self-contained
        path = f"{base}_ep{self._episode_idx:04d}{ext or '.csv'}"
        self._mlogger = MetricsLogger(path=path, fieldnames=METRICS_FIELDNAMES)
        logger.info("Metrics CSV: %s", path)
