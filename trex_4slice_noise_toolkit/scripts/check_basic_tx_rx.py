#!/usr/bin/env python3
import time
from trex_stl_lib.api import STLClient
from run_trex_dynamic_4slice import import_trex, build_streams

HOST = "127.0.0.1"
TX_PORT = 0
RX_PORT = 1

target_mbps = {
    "gold": 80.0,
    "silver": 80.0,
    "bronze": 80.0,
    "noise": 40.0,
}

api = import_trex()
c = STLClient(server=HOST)
c.connect()

try:
    # Acquire both ports before reset/service-mode operations.
    # This avoids TRexError: port(s) [1]: must be acquired.
    c.acquire(ports=[TX_PORT, RX_PORT], force=True)
    c.reset(ports=[TX_PORT, RX_PORT])
    c.set_service_mode(ports=[RX_PORT], enabled=True)

    streams, _, _ = build_streams(api, target_mbps, wire_overhead_bytes=20)
    c.add_streams(streams, ports=[TX_PORT])

    print("Starting basic TX/RX test for 10 seconds")
    c.start(ports=[TX_PORT], duration=10)

    for i in range(5):
        time.sleep(2)

        stats = c.get_stats()
        port0 = stats[TX_PORT]
        port1 = stats[RX_PORT]

        pg = c.get_pgid_stats()

        print(f"\nSample {i}")
        print(f"Port 0 TX bps: {port0.get('tx_bps_L1', 0):.2f}")
        print(f"Port 1 RX bps: {port1.get('rx_bps', 0):.2f}")
        print(f"Port 0 opackets: {port0.get('opackets', 0)}")
        print(f"Port 1 ipackets: {port1.get('ipackets', 0)}")

        for pgid in [1, 11, 21, 31]:
            f = pg.get("flow_stats", {}).get(pgid, {})
            tx = f.get("tx_pkts", {}).get("total", 0)
            rx = f.get("rx_pkts", {}).get("total", 0)
            print(f"PG {pgid}: tx_pkts={tx} rx_pkts={rx}")

finally:
    c.disconnect()
