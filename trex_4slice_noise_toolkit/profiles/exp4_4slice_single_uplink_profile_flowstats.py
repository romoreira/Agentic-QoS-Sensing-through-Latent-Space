from trex_stl_lib.api import *
from scapy.contrib.gtp import GTP_U_Header

# Same packet shape as the validated exp2 profile, but with a fourth TEID.
GNB_IP = "192.168.70.2"
UPF_N3_IP = "192.168.70.1"
DN_IP = "192.168.80.2"
PAYLOAD_BYTES = 128

# Static fallback rates. For dynamic curves, use scripts/run_trex_dynamic_4slice.py.
DEFAULT_PPS_VISIBLE = 50000
DEFAULT_PPS_NOISE = 35000

SLICES = [
    {"name": "gold",   "teid": 1,  "ue_ip": "10.250.0.1", "src_port": 10001, "pg_id": 1,  "pps": DEFAULT_PPS_VISIBLE},
    {"name": "silver", "teid": 11, "ue_ip": "10.250.0.2", "src_port": 10011, "pg_id": 11, "pps": DEFAULT_PPS_VISIBLE},
    {"name": "bronze", "teid": 21, "ue_ip": "10.250.0.3", "src_port": 10021, "pg_id": 21, "pps": DEFAULT_PPS_VISIBLE},
    {"name": "noise",  "teid": 31, "ue_ip": "10.250.0.4", "src_port": 10031, "pg_id": 31, "pps": DEFAULT_PPS_NOISE},
]


def _to_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ("0", "false", "no", "off")


class STLS1(object):
    def get_streams(self, direction=0, **kwargs):
        include_noise = _to_bool(kwargs.get("include_noise", True), True)
        pps_scale = float(kwargs.get("pps_scale", 1.0))
        streams = []
        for s in SLICES:
            if s["name"] == "noise" and not include_noise:
                continue
            pps = max(1, int(round(s["pps"] * pps_scale)))
            pkt = (
                Ether()
                / IP(src=GNB_IP, dst=UPF_N3_IP)
                / UDP(sport=2152, dport=2152, chksum=0)
                / GTP_U_Header(teid=s["teid"])
                / IP(src=s["ue_ip"], dst=DN_IP)
                / UDP(sport=s["src_port"], dport=5555, chksum=0)
                / ("X" * PAYLOAD_BYTES)
            )
            streams.append(
                STLStream(
                    name="exp4_%s_teid_%s" % (s["name"], s["teid"]),
                    packet=STLPktBuilder(pkt=pkt),
                    mode=STLTXCont(pps=pps),
                    flow_stats=STLFlowStats(pg_id=s["pg_id"]),
                )
            )
        return streams


def register():
    return STLS1()
