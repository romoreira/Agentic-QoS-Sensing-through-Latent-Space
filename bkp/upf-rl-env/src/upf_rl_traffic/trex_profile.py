"""Single-uplink TRex STL profile for slice traffic.

This profile deliberately follows the topology that already worked in the
lab: one physical TX uplink, several GTP-U streams on that same port, and
one TEID per slice. There is no port-per-slice assumption here.

Default stream set
------------------
  - GOLD   TEID 1
  - SILVER TEID 11
  - BRONZE TEID 21
  - NOISE  TEID 31, optional fourth slice used only as background pressure

The DQN controls only GOLD, SILVER, and BRONZE MBRs. The NOISE stream is
there to consume the same uplink/N6 headroom and make the control problem
partial-observable when the mock source is used. TRex telemetry is still a
stub in this project, so the training challenge comes from the mock metric
source, while this profile keeps the packet shape compatible with the
working exp2 profile.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from trex_stl_lib.api import (  # type: ignore[import-not-found]
        Ether,
        IP,
        UDP,
        STLPktBuilder,
        STLStream,
        STLTXCont,
        STLFlowLatencyStats,
    )
    _TREX_AVAILABLE = True
except Exception:
    _TREX_AVAILABLE = False
    logger.debug("trex_stl_lib not importable, profile builders unavailable outside TRex")

try:
    from scapy.contrib.gtp import GTP_U_Header  # type: ignore[import-not-found]
    _GTP_AVAILABLE = True
except Exception:
    _GTP_AVAILABLE = False


# Keep these defaults aligned with the working exp2 profile.
GNB_IP = "192.168.70.2"
UPF_N3_IP = "192.168.70.1"
DN_IP = "192.168.80.2"
PAYLOAD_BYTES = 128
UPLINK_PORT = 0

# Software-mode friendly default. With 128-byte payloads this is in the same
# practical range as the user's working 50 kpps profile.
DEFAULT_PPS_PER_SLICE = 50_000
DEFAULT_NOISE_PPS = 35_000


@dataclass(frozen=True)
class SliceDef:
    name: str
    teid: int
    ue_ip: str
    src_port: int
    pg_id: int
    pps: int
    visible: bool = True


SLICE_DEFS: List[SliceDef] = [
    SliceDef("gold", teid=1, ue_ip="10.250.0.1", src_port=10001, pg_id=1, pps=DEFAULT_PPS_PER_SLICE),
    SliceDef("silver", teid=11, ue_ip="10.250.0.2", src_port=10011, pg_id=11, pps=DEFAULT_PPS_PER_SLICE),
    SliceDef("bronze", teid=21, ue_ip="10.250.0.3", src_port=10021, pg_id=21, pps=DEFAULT_PPS_PER_SLICE),
    SliceDef("noise", teid=31, ue_ip="10.250.0.4", src_port=10031, pg_id=31, pps=DEFAULT_NOISE_PPS, visible=False),
]


def _require_trex() -> None:
    if not _TREX_AVAILABLE:
        raise ImportError(
            "trex_stl_lib is not available. Run this inside the TRex Python "
            "environment, for example with /opt/trex/v3.08/automation on PYTHONPATH."
        )
    if not _GTP_AVAILABLE:
        raise ImportError("scapy.contrib.gtp.GTP_U_Header is not importable in this environment.")


def build_slice_packet(slc: SliceDef):
    """Build the same GTP-U packet shape as the validated exp2 profile."""
    _require_trex()
    return (
        Ether()
        / IP(src=GNB_IP, dst=UPF_N3_IP)
        / UDP(sport=2152, dport=2152, chksum=0)
        / GTP_U_Header(teid=slc.teid)
        / IP(src=slc.ue_ip, dst=DN_IP)
        / UDP(sport=slc.src_port, dport=5555, chksum=0)
        / ("X" * PAYLOAD_BYTES)
    )


def build_streams(
    *,
    include_noise: bool = True,
    pps_scale: float = 1.0,
    slice_defs: Optional[Sequence[SliceDef]] = None,
) -> List["STLStream"]:
    """Return all slice streams for one TRex TX port.

    `pps_scale` lets you quickly soften or harden the overload without changing
    the profile file. All streams remain on the same port.
    """
    _require_trex()
    streams = []
    for slc in list(slice_defs or SLICE_DEFS):
        if (not include_noise) and (not slc.visible):
            continue
        pps = max(1, int(round(slc.pps * pps_scale)))
        streams.append(
            STLStream(
                name=f"single_uplink_{slc.name}_teid_{slc.teid}",
                packet=STLPktBuilder(pkt=build_slice_packet(slc)),
                mode=STLTXCont(pps=pps),
                flow_stats=STLFlowLatencyStats(pg_id=slc.pg_id),
            )
        )
    return streams


class SingleUplinkProfile:
    def get_streams(self, direction: int = 0, **kwargs):
        include_noise = bool(kwargs.get("include_noise", True))
        pps_scale = float(kwargs.get("pps_scale", 1.0))
        return build_streams(include_noise=include_noise, pps_scale=pps_scale)


def register():
    return SingleUplinkProfile()
