"""Metrics logger — non-blocking CSV writer in a background thread.

Decouples telemetry observation from disk I/O so that env.step() stays
cheap even at 200 ms windows. Survives independently of the agent — you
can keep it open while sweeping seeds or even running the env with no
agent at all (random actions for sanity checks).
"""
from __future__ import annotations

import csv
import logging
import os
import queue
import threading
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


_SENTINEL = object()


class MetricsLogger:
    """Buffered CSV logger. Thread-safe."""

    def __init__(
        self,
        path: str,
        fieldnames: Iterable[str],
        flush_every: int = 50,
    ) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.flush_every = flush_every

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=10_000)
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="MetricsLogger", daemon=True)
        self._thread.start()

    def log(self, row: dict) -> None:
        """Enqueue a row. Drops oldest if queue is full (should not happen
        with sane flush_every / disk speed)."""
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            logger.warning("MetricsLogger queue full — dropping row")

    def close(self) -> None:
        if self._stop_evt.is_set():
            return
        self._stop_evt.set()
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=5.0)

    # ---- internal -----------------------------------------------------

    def _run(self) -> None:
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            n_since_flush = 0
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    f.flush()
                    return
                try:
                    writer.writerow(item)  # type: ignore[arg-type]
                except Exception:
                    logger.exception("MetricsLogger write failed")
                n_since_flush += 1
                if n_since_flush >= self.flush_every:
                    f.flush()
                    n_since_flush = 0


METRICS_FIELDNAMES = [
    "step",
    "ts",
    "offered_gold_mbps",
    "offered_silver_mbps",
    "offered_bronze_mbps",
    "rx_gold_mbps",
    "rx_silver_mbps",
    "rx_bronze_mbps",
    "loss_gold",
    "loss_silver",
    "loss_bronze",
    "qer_red_gold",
    "qer_red_silver",
    "qer_red_bronze",
    "n6_tx_mbps",
    "n6_utilization",
    "mbr_gold_kbps",
    "mbr_silver_kbps",
    "mbr_bronze_kbps",
    "action",
    "reward",
    "regime",  # ground truth (mock only) — empty when not available
]
