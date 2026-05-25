"""TRex metric source — stub.

Wire this to your TRex deployment when leaving the mock. Typical entry
points:
  - TRex STL Python client on TCP/4501 (sync) and 4500 (async/zmq)
    `from trex.stl.api import STLClient`
  - TRex ASTF (advanced stateful) if you are simulating real apps.
  - REST: TRex exposes a small HTTP API on 4500-4501 depending on build.

Required outputs per window:
  - offered_mbps per slice (from streams / port stats — the *sent* rate)
  - rx_mbps per slice (from receive ports — what made it through)
  - timestamps aligned to the BESS window so the env can collate.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from upf_rl_env.sources.base import MetricSource, WindowMetrics


class TrexMetricSource(MetricSource):
    """Stub. Replace `tick` body with real TRex client interaction."""

    def __init__(
        self,
        trex_host: str = "127.0.0.1",
        trex_sync_port: int = 4501,
        slice_to_tx_ports: Optional[dict] = None,
        slice_to_rx_ports: Optional[dict] = None,
    ) -> None:
        self.trex_host = trex_host
        self.trex_sync_port = trex_sync_port
        # Map slice index (0=gold, 1=silver, 2=bronze) -> TRex port id
        self.slice_to_tx_ports = slice_to_tx_ports or {0: 0, 1: 1, 2: 2}
        self.slice_to_rx_ports = slice_to_rx_ports or {0: 3, 1: 4, 2: 5}
        self._client = None

    def _connect(self):
        # TODO: import STLClient and connect
        # from trex.stl.api import STLClient
        # self._client = STLClient(server=self.trex_host, sync_port=self.trex_sync_port)
        # self._client.connect()
        # self._client.reset(ports=list(self.slice_to_tx_ports.values()) + list(self.slice_to_rx_ports.values()))
        raise NotImplementedError("Wire TrexMetricSource._connect to STLClient")

    def reset(self, seed: Optional[int] = None) -> WindowMetrics:
        # TODO: optionally re-seed traffic profiles. Most setups keep TRex
        # running flat and only reset internal counters at episode boundaries.
        raise NotImplementedError(
            "TrexMetricSource is a stub. Use MockMetricSource until wired."
        )

    def tick(self, mbr_kbps: np.ndarray) -> WindowMetrics:
        # TODO: read port stats, return WindowMetrics with offered/rx per slice.
        # Outline:
        #   stats = self._client.get_stats(ports=...)
        #   offered = np.array([stats[p]["tx_bps"] / 1e6 for p in tx_ports])
        #   rx      = np.array([stats[p]["rx_bps"] / 1e6 for p in rx_ports])
        #   loss    = compute from offered - rx
        #   ts      = stats["timestamp"]
        raise NotImplementedError(
            "TrexMetricSource.tick — implement TRex stats polling."
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
