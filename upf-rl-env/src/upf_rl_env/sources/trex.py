"""TRex metric source stub for the single-uplink topology.

The intended real wiring is one TRex TX port carrying multiple GTP-U streams.
Per-slice accounting should come from TRex flow latency stats or BESS/UPF
counters keyed by TEID/PG ID, not from separate TRex ports.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from upf_rl_env.config import SLICE_TEIDS
from upf_rl_env.sources.base import MetricSource, WindowMetrics


class TrexMetricSource(MetricSource):
    """Stub. Replace `tick` with real TRex/BESS polling when wired."""

    def __init__(
        self,
        trex_host: str = "127.0.0.1",
        trex_sync_port: int = 4501,
        tx_port: int = 0,
        slice_teids: Sequence[int] = SLICE_TEIDS,
        slice_pg_ids: Sequence[int] = (1, 11, 21),
    ) -> None:
        self.trex_host = trex_host
        self.trex_sync_port = trex_sync_port
        self.tx_port = tx_port
        self.slice_teids = tuple(slice_teids)
        self.slice_pg_ids = tuple(slice_pg_ids)
        self._client = None

    def _connect(self):
        # TODO: import STLClient and connect.
        # from trex_stl_lib.api import STLClient
        # self._client = STLClient(server=self.trex_host, sync_port=self.trex_sync_port)
        # self._client.connect()
        # self._client.clear_stats(ports=[self.tx_port])
        raise NotImplementedError("Wire TrexMetricSource._connect to STLClient")

    def reset(self, seed: Optional[int] = None) -> WindowMetrics:
        raise NotImplementedError(
            "TrexMetricSource is a stub. Use MockMetricSource until TRex/BESS counters are wired."
        )

    def tick(self, mbr_kbps: np.ndarray) -> WindowMetrics:
        # Expected outline for the real implementation:
        #   1. Read per-PG or per-TEID TX/offered counters for TEIDs 1, 11, 21.
        #   2. Read post-UPF RX or QER counters for the same TEIDs.
        #   3. Read aggregate N6 utilization.
        #   4. Return WindowMetrics with arrays ordered as Gold, Silver, Bronze.
        # Do not assume one TRex port per slice.
        raise NotImplementedError("TrexMetricSource.tick is not implemented yet")

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
