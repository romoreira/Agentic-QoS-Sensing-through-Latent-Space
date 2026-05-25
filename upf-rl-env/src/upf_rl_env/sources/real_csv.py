"""Replay telemetry collected from the real TRex + BESS-UPF testbed.

This source is intended for a safe first integration step between real
telemetry and the Gym/DQN stack. It replays a CSV produced by
``trex_4slice_noise_toolkit/scripts/real_telemetry_probe.py`` and converts
one row per control window into ``WindowMetrics``.

Important limitation
--------------------
This is offline replay. The DQN action does not change the already-recorded
traffic or loss in the CSV. The action only changes the MBR fields passed to
``tick()``. Use this to validate feature shape, normalization, reward,
TensorBoard logging, and GPU training plumbing before closing the real loop.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from upf_rl_env.config import EnvConfig, NOMINAL_MBR_KBPS
from upf_rl_env.sources.base import MetricSource, WindowMetrics


_VISIBLE = ("gold", "silver", "bronze")


def _as_float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class RealCsvMetricSource(MetricSource):
    """MetricSource backed by a real telemetry CSV.

    Parameters
    ----------
    csv_path:
        Path to the CSV produced by ``real_telemetry_probe.py``.
    cfg:
        Environment configuration. Only time/normalization settings are used.
    loop:
        If True, wrap around at EOF. Useful for long DQN smoke tests.
        If False, keep returning the last row after EOF.
    use_action_mbr:
        If True, the MBR fields in the observation come from the action passed
        by the environment. If False, use the MBR columns recorded in the CSV.
        For integration with DQN, True is usually preferable because the obs
        reflects the chosen action even though the replayed traffic is fixed.
    """

    def __init__(
        self,
        csv_path: str | Path,
        cfg: Optional[EnvConfig] = None,
        *,
        loop: bool = True,
        use_action_mbr: bool = True,
    ) -> None:
        self.cfg = cfg or EnvConfig.default()
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.loop = bool(loop)
        self.use_action_mbr = bool(use_action_mbr)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Telemetry CSV not found: {self.csv_path}")

        with self.csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"Telemetry CSV has no rows: {self.csv_path}")

        self._idx = 0
        self._last_mbr_kbps = np.array(NOMINAL_MBR_KBPS, dtype=np.int64)
        self.hidden_state = {
            "source": "real_csv",
            "csv_path": str(self.csv_path),
            "rows": len(self.rows),
        }

        self._validate_minimal_columns()

    def _validate_minimal_columns(self) -> None:
        required = [
            "offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps",
            "rx_gold_mbps", "rx_silver_mbps", "rx_bronze_mbps",
            "loss_gold", "loss_silver", "loss_bronze",
        ]
        missing = [c for c in required if c not in self.rows[0]]
        if missing:
            raise ValueError(
                f"Telemetry CSV is missing required columns: {missing}. "
                f"Path: {self.csv_path}"
            )

    def reset(self, seed: Optional[int] = None) -> WindowMetrics:
        # Deterministic replay from the beginning. The seed argument is accepted
        # for Gym compatibility but does not randomize the CSV order.
        self._idx = 0
        self._last_mbr_kbps = np.array(NOMINAL_MBR_KBPS, dtype=np.int64)
        return self.tick(self._last_mbr_kbps)

    def tick(self, mbr_kbps: np.ndarray) -> WindowMetrics:
        row = self.rows[self._idx]
        if self._idx + 1 < len(self.rows):
            self._idx += 1
        elif self.loop:
            self._idx = 0
        # else stay on the final row

        offered = np.array([_as_float(row, f"offered_{s}_mbps") for s in _VISIBLE], dtype=np.float32)
        rx = np.array([_as_float(row, f"rx_{s}_mbps") for s in _VISIBLE], dtype=np.float32)
        loss = np.array([_as_float(row, f"loss_{s}") for s in _VISIBLE], dtype=np.float32)
        loss = np.clip(loss, 0.0, 1.0)

        qer_red = np.array([
            _as_float(row, f"qer_red_{s}", default=float(loss[i]))
            for i, s in enumerate(_VISIBLE)
        ], dtype=np.float32)
        qer_red = np.clip(qer_red, 0.0, None)

        n6_tx_mbps = _as_float(row, "n6_tx_mbps", default=_as_float(row, "n6_total_mbps"))
        n6_utilization = _as_float(row, "n6_utilization")

        if self.use_action_mbr:
            mbr = np.asarray(mbr_kbps, dtype=np.int64)
        else:
            mbr = np.array([
                int(_as_float(row, "mbr_gold_kbps", NOMINAL_MBR_KBPS[0])),
                int(_as_float(row, "mbr_silver_kbps", NOMINAL_MBR_KBPS[1])),
                int(_as_float(row, "mbr_bronze_kbps", NOMINAL_MBR_KBPS[2])),
            ], dtype=np.int64)
        self._last_mbr_kbps = mbr.copy()

        ts = _as_float(row, "t_s", default=_as_float(row, "ts", default=float(self._idx)))

        self.hidden_state = {
            "source": "real_csv",
            "row_index": self._idx,
            "rows": len(self.rows),
            "pfcp_sessions": _as_float(row, "pfcp_sessions", default=-1.0),
            "noise_offered_mbps": _as_float(row, "noise_offered_mbps"),
            "noise_rx_mbps": _as_float(row, "noise_rx_mbps"),
            "noise_loss": _as_float(row, "noise_loss"),
            "trex_total_tx_pkts": _as_float(row, "trex_total_tx_pkts"),
            "trex_total_rx_pkts": _as_float(row, "trex_total_rx_pkts"),
        }

        return WindowMetrics(
            offered_mbps=offered,
            rx_mbps=rx,
            loss_rate=loss,
            qer_red=qer_red,
            n6_tx_mbps=float(n6_tx_mbps),
            n6_utilization=float(n6_utilization),
            mbr_kbps=mbr,
            timestamp_s=float(ts),
        )
