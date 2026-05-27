from trex_stl_lib.api import *
from scapy.contrib.gtp import GTP_U_Header

GNB_IP = "192.168.70.2"
UPF_N3_IP = "192.168.70.1"
DN_IP = "192.168.80.2"
PAYLOAD_BYTES = 128
PPS = 5000

SLICES = [
    {"teid": 1,  "ue_ip": "10.250.0.1", "src_port": 10001, "pg_id": 1},
    {"teid": 11, "ue_ip": "10.250.0.2", "src_port": 10011, "pg_id": 11},
    {"teid": 21, "ue_ip": "10.250.0.3", "src_port": 10021, "pg_id": 21},
    {"teid": 31, "ue_ip": "10.250.0.4", "src_port": 10031, "pg_id": 31},
]

class STLS1(object):
    def get_streams(self, direction=0, **kwargs):
        streams = []
        for s in SLICES:
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
                    name="flow_teid_%s" % s["teid"],
                    packet=STLPktBuilder(pkt=pkt),
                    mode=STLTXCont(pps=PPS),
                    flow_stats=STLFlowStats(pg_id=s["pg_id"]),
                )
            )
        return streams

def register():
    return STLS1()
