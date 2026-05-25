"""Metric source interface.

A MetricSource is anything that, given the currently enforced MBR vector,
returns the next-window telemetry observation. The env never knows whether
it is talking to a simulator, a TRex/BESS testbed, or a replay file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WindowMetrics:
    """All telemetry collected over a single control window.

    Per-slice arrays are length 3, ordered (gold, silver, bronze).
    Units:
      *_mbps      : megabits per second over the window
      loss_rate   : in [0, 1], computed as (offered - rx) / offered
      qer_red_*   : packets dropped by the per-slice QER (proxy: red-marked count)
      n6_*        : aggregate north-bound interface metrics
      mbr_kbps    : current MBR in effect during the window
    """

    offered_mbps: np.ndarray          # (3,)
    rx_mbps: np.ndarray               # (3,)
    loss_rate: np.ndarray             # (3,)
    qer_red: np.ndarray               # (3,)
    n6_tx_mbps: float
    n6_utilization: float
    mbr_kbps: np.ndarray              # (3,)
    timestamp_s: float

    def as_dict(self) -> dict:
        return {
            "ts": self.timestamp_s,
            "offered_gold_mbps": float(self.offered_mbps[0]),
            "offered_silver_mbps": float(self.offered_mbps[1]),
            "offered_bronze_mbps": float(self.offered_mbps[2]),
            "rx_gold_mbps": float(self.rx_mbps[0]),
            "rx_silver_mbps": float(self.rx_mbps[1]),
            "rx_bronze_mbps": float(self.rx_mbps[2]),
            "loss_gold": float(self.loss_rate[0]),
            "loss_silver": float(self.loss_rate[1]),
            "loss_bronze": float(self.loss_rate[2]),
            "qer_red_gold": float(self.qer_red[0]),
            "qer_red_silver": float(self.qer_red[1]),
            "qer_red_bronze": float(self.qer_red[2]),
            "n6_tx_mbps": float(self.n6_tx_mbps),
            "n6_utilization": float(self.n6_utilization),
            "mbr_gold_kbps": int(self.mbr_kbps[0]),
            "mbr_silver_kbps": int(self.mbr_kbps[1]),
            "mbr_bronze_kbps": int(self.mbr_kbps[2]),
        }


class MetricSource(ABC):
    """Abstract source of telemetry windows."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> WindowMetrics:
        """Reset internal state. Returns a bootstrap window."""

    @abstractmethod
    def tick(self, mbr_kbps: np.ndarray) -> WindowMetrics:
        """Advance one window with the given enforced MBR vector."""

    def close(self) -> None:
        """Release resources (sockets, files). Default no-op."""
        return None
