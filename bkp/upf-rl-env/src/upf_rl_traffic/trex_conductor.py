"""Small TRex runner for the single-uplink profile.

This module intentionally does not implement port-per-slice rate modulation.
All slice streams are loaded on one TRex TX port and run continuously, matching
the working exp2 profile style. The current project still mocks TRex readings
inside the RL environment, so the relevant DQN noise comes from the shared mock
traffic dynamics, not from runtime TRex rate updates.
"""
from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Optional

from upf_rl_traffic.trex_profile import UPLINK_PORT, build_streams

logger = logging.getLogger(__name__)


def _import_trex_client():
    try:
        from trex_stl_lib.api import STLClient  # type: ignore[import-not-found]

        return STLClient
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import trex_stl_lib.api.STLClient. Make sure your PYTHONPATH "
            "includes the TRex automation directory, for example:\n"
            "  export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH"
        ) from exc


@dataclass
class ConductorConfig:
    trex_host: str = "127.0.0.1"
    trex_sync_port: int = 4501
    tx_port: int = UPLINK_PORT
    duration_s: Optional[float] = None
    include_noise: bool = True
    pps_scale: float = 1.0


class TrexConductor:
    """Run the single-uplink stream set continuously on one TRex port."""

    def __init__(self, cfg: Optional[ConductorConfig] = None) -> None:
        self.cfg = cfg or ConductorConfig()
        self._client = None
        self._stop = threading.Event()

    def connect(self) -> None:
        STLClient = _import_trex_client()
        self._client = STLClient(server=self.cfg.trex_host, sync_port=self.cfg.trex_sync_port)
        self._client.connect()
        logger.info("Connected to TRex at %s:%d", self.cfg.trex_host, self.cfg.trex_sync_port)

    def setup_streams(self) -> None:
        assert self._client is not None
        streams = build_streams(include_noise=self.cfg.include_noise, pps_scale=self.cfg.pps_scale)
        self._client.acquire(ports=[self.cfg.tx_port], force=True)
        self._client.reset(ports=[self.cfg.tx_port])
        self._client.add_streams(streams=streams, ports=[self.cfg.tx_port])
        logger.info(
            "Loaded %d streams on single uplink port=%d include_noise=%s pps_scale=%.3f",
            len(streams),
            self.cfg.tx_port,
            self.cfg.include_noise,
            self.cfg.pps_scale,
        )

    def start(self) -> None:
        assert self._client is not None
        self._client.start(ports=[self.cfg.tx_port], duration=-1)
        logger.info("Traffic started on uplink port=%d", self.cfg.tx_port)

    def stop(self) -> None:
        if self._client is None:
            return
        try:
            self._client.stop(ports=[self.cfg.tx_port])
        except Exception:
            logger.exception("client.stop failed")
        try:
            self._client.release(ports=[self.cfg.tx_port])
        except Exception:
            logger.exception("client.release failed")
        try:
            self._client.disconnect()
        except Exception:
            logger.exception("client.disconnect failed")
        self._client = None
        logger.info("Disconnected from TRex")

    def run(self) -> None:
        if self._client is None:
            self.connect()
        self.setup_streams()
        self.start()

        def _sig(_signum, _frame):
            logger.info("Caught signal, stopping TRex traffic")
            self._stop.set()

        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)

        t0 = time.monotonic()
        try:
            while not self._stop.is_set():
                if self.cfg.duration_s is not None and (time.monotonic() - t0) >= self.cfg.duration_s:
                    logger.info("Duration reached, stopping")
                    break
                time.sleep(0.5)
        finally:
            self.stop()
