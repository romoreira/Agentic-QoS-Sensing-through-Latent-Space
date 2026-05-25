"""BESS-UPF metric source — stub.

The exact path to telemetry depends on the UPF-BESS deployment you use
(aether-upf-bess, omec-cn upf, free5gc + bess-upf, sd-fabric). Until you
confirm which deployment, leave this stub in place and use MockMetricSource.

Typical telemetry sources:
  (a) `bessctl show port <port_name>`     -> per-port byte/packet counters
  (b) `bessctl show module <module_name>` -> per-module counters (QosMeter,
                                              QosMarker, NatLB, ...)
  (c) pfcpiface gRPC / REST                -> per-session, per-QER stats
  (d) Prometheus exporter built into UPF   -> scrape `/metrics`

Required outputs per window:
  - rx_mbps per slice (post-MBR — what actually transits N6)
  - qer_red per slice (drops by the policer)
  - n6_tx_mbps, n6_utilization (aggregate north-bound)
  - timestamp aligned to TRex window

You will likely combine bessctl/pfcpiface (BESS-side) with TRex (offered
traffic) at the env level. The env supports a "compound source" pattern
trivially: instantiate both, call .tick() on each, and merge.
"""
from __future__ import annotations

import subprocess
from typing import Optional, Sequence

import numpy as np

from upf_rl_env.sources.base import MetricSource, WindowMetrics


class BessMetricSource(MetricSource):
    """Stub. Replace command/RPC calls when wiring the testbed."""

    def __init__(
        self,
        bessctl_path: str = "bessctl",
        slice_teids: Sequence[int] = (1, 11, 21),
        pfcpiface_endpoint: Optional[str] = None,
    ) -> None:
        self.bessctl_path = bessctl_path
        self.slice_teids = tuple(slice_teids)
        self.pfcpiface_endpoint = pfcpiface_endpoint

    def reset(self, seed: Optional[int] = None) -> WindowMetrics:
        raise NotImplementedError(
            "BessMetricSource is a stub. Use MockMetricSource until wired."
        )

    def tick(self, mbr_kbps: np.ndarray) -> WindowMetrics:
        # TODO: implement against your UPF-BESS deployment.
        # Sketch using bessctl (replace module names with yours):
        #
        #   out = subprocess.run(
        #       [self.bessctl_path, "show", "module", "qos_meter_gold"],
        #       capture_output=True, text=True, check=True,
        #   ).stdout
        #   # parse counters: in_packets, out_packets, dropped_packets
        #
        # Or via pfcpiface gRPC for per-session/per-QER counters.
        raise NotImplementedError(
            "BessMetricSource.tick — implement bessctl/pfcpiface parsing."
        )

    @staticmethod
    def discover_qer_for_teid_command(teid: int) -> str:
        """Helper: print a starting point command for discovering which
        QER/module a TEID maps to in your deployment.

        Run this against your live UPF and inspect the output to find the
        right module name for each slice TEID, then plug into tick().
        """
        # The right command varies by deployment; common variants:
        return (
            f"# Try one of (depends on your UPF flavor):\n"
            f"#   bessctl show pipeline | grep -i 'qer\\|teid'\n"
            f"#   bessctl show module pfcpiface | grep -i 'teid={teid}'\n"
            f"#   pfcpctl service session list   # then find session for teid={teid}\n"
            f"#   curl -s http://<pfcpiface>:8080/v1/up/session  | jq '.[] | select(.teid=={teid})'\n"
        )
