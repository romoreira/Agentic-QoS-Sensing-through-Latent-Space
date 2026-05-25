# Four Slice BESS UPF, TRex, PFCP and Real Telemetry Testbed

This README documents the current validated base for the four slice experiment used before connecting the DQN controller. It records how to bring the testbed back, how to create the four PFCP sessions, how to run TRex traffic, how to verify BESS rules, how to collect the real telemetry features, and what remains before plugging the DQN.

## 1. Goal

The goal is to run three controlled slices and one hidden noise slice over the same physical uplink.

```text
One TRex TX uplink, port 0, N3
One TRex RX port, port 1, N6
Four GTP U streams on the same uplink
Three slices observed and controlled by DQN
One fourth slice used as disturbance/noise
```

There is no multiport per slice design here. The slices are separated by TEID, UE IP, UDP source port and PG ID, but they share the same N3 uplink.

## 2. Current validated state

The current state is validated as follows.

```text
PFCP sessions installed in UPF, 4
BESS pdrLookup rules, 8
BESS farLookup rules, 8
BESS appQERLookup rules, 16
BESS sessionQERLookup rules, 8
TRex PG IDs visible, 1, 11, 21, 31
TRex per PG RX works with STLFlowLatencyStats and TRex software mode
UPF Prometheus reachable from TRex host at http://192.168.90.1:8080/metrics
real_telemetry_probe.py works and produces the DQN observation columns
```

## 3. Topology

```text
TRex host                                      UPF host
──────────────────────────────────             ──────────────────────────────────

Management
10.30.6.23/23                                  10.30.6.199/23

N3, GTP U traffic
TRex enp7s0np0, 192.168.70.2/24                UPF enp7s0np0, 192.168.70.1/24
TRex MAC 10:70:fd:c0:ef:80                     UPF MAC 10:70:fd:c1:59:c4

N6, decapsulated traffic
TRex enp8s0np0, 192.168.80.2/24                UPF enp8s0np0, 192.168.80.1/24
TRex MAC 10:70:fd:c0:ef:81                     UPF MAC 10:70:fd:c1:59:c5

PFCP control
TRex enp9s0, 192.168.90.2/24                   UPF enp9s0, 192.168.90.1/24
```

Keep PFCP on `enp9s0`. Do not use N3 or N6 for PFCP control while TRex and BESS own the datapath.

## 4. Slice mapping

```text
Gold    TEID 1    UE 10.250.0.1    UDP sport 10001    PG ID 1     controlled by DQN
Silver  TEID 11   UE 10.250.0.2    UDP sport 10011    PG ID 11    controlled by DQN
Bronze  TEID 21   UE 10.250.0.3    UDP sport 10021    PG ID 21    controlled by DQN
Noise   TEID 31   UE 10.250.0.4    UDP sport 10031    PG ID 31    hidden disturbance
```

Initial MBR values used in the four session setup.

```text
Gold    app MBR 200000 kbps
Silver  app MBR 100000 kbps
Bronze  app MBR  50000 kbps
Noise   app MBR  50000 kbps

All slices session level MBR 1000000 kbps
```

The DQN should control only Gold, Silver and Bronze. The Noise slice should be logged for audit but not included as an observation feature.

## 5. Start or restore the UPF side

Run on the UPF host.

### 5.1 Start DPDK BESS and pfcpiface

```bash
cd ~/bess-upf/config

sudo docker rm -f pfcpiface bess-routectl bess 2>/dev/null || true

sudo mkdir -p /dev/hugepages
sudo mount -t hugetlbfs nodev /dev/hugepages 2>/dev/null || true
sudo sysctl -w vm.nr_hugepages=1024

sudo docker compose -f docker-compose-dpdk.yml up -d bess
sleep 3
sudo docker logs bess --tail 80
```

Load the UPF pipeline.

```bash
sudo docker exec bess bash -lc 'cd /opt/bess/bessctl && ./bessctl run up4'
```

Start pfcpiface.

```bash
cd ~/bess-upf/config
sudo docker compose -f docker-compose-dpdk.yml up -d pfcpiface
sudo docker logs pfcpiface --tail 40
```

### 5.2 Program N6 route and neighbor

This is required because TRex uses DPDK and the Linux kernel on the TRex host may not answer ARP normally on N6.

```bash
sudo ip neigh replace 192.168.80.2 lladdr 10:70:fd:c0:ef:81 dev enp8s0np0 nud permanent
sudo ip route replace 192.168.80.2/32 via 192.168.80.2 dev enp8s0np0 onlink
```

Start route controller.

```bash
sudo docker rm -f bess-routectl 2>/dev/null || true

sudo docker run --name bess-routectl -td --restart unless-stopped \
  --net host \
  --pid container:bess \
  --entrypoint python3 \
  -v ~/bess-upf/upf/conf/route_control.py:/route_control.py \
  upf-bess:2.4.2-dev-mlx5 \
  /route_control.py -i enp7s0np0 enp8s0np0

sudo docker logs bess-routectl --tail 20
```

Validate route module.

```bash
sudo docker exec bess /opt/bess/bessctl/bessctl show module enp8s0np0Routes
```

Expected path.

```text
gate 0 -> enp8s0np0DstMAC1070FDC0EF81
gate 8191 -> enp8s0np0bad_route
```

Bad route must remain at zero during valid tests.

### 5.3 UPF health checks

```bash
sudo docker ps
sudo docker logs bess --tail 30
sudo docker logs pfcpiface --tail 30
sudo docker logs bess-routectl --tail 30

curl -s http://127.0.0.1:8080/metrics | grep pfcp_sessions
sudo docker exec bess /opt/bess/bessctl/bessctl show port
```

Expected.

```text
bess running
pfcpiface running
bess-routectl running
pfcpiface listening on UDP 8805
UPF metrics reachable on 8080
BESS gRPC reachable on 10514
```

## 6. Start pfcpsim on the TRex host

Run on the TRex host.

```bash
sudo docker rm -f pfcpsim 2>/dev/null || true

sudo docker run --rm -d --network host \
  --name pfcpsim \
  pfcpsim:patched \
  -p 12345 \
  --interface enp9s0
```

Configure PFCP and associate.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 service configure \
  --n3-addr 192.168.70.1 \
  --remote-peer-addr 192.168.90.1

sudo docker exec pfcpsim pfcpctl -s localhost:12345 service associate
```

Important meanings.

```text
--n3-addr is the UPF N3 address, 192.168.70.1
--remote-peer-addr is the UPF PFCP control address, 192.168.90.1
--gnb-addr in session create is the TRex N3 address, 192.168.70.2
```

## 7. Create the four PFCP sessions

If the old three sessions exist, delete them first.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session delete \
  --count 3 --baseID 1 || true
```

If four sessions may already exist, use this.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session delete \
  --count 4 --baseID 1 || true
```

Create the four sessions.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session create \
  --count 4 --baseID 1 \
  --ue-pool 10.250.0.0/24 \
  --gnb-addr 192.168.70.2 \
  --app-filter "udp:any:any:allow:100" \
  --session-mbr-uplink   1000000,1000000,1000000,1000000 \
  --session-mbr-downlink 1000000,1000000,1000000,1000000 \
  --app-mbr-uplink       200000,100000,50000,50000 \
  --app-mbr-downlink     200000,100000,50000,50000
```

Check pfcpsim logs.

```bash
sudo docker logs --tail 100 pfcpsim 2>&1 | egrep -i "Create session|sessions were established|Delete|error|fail|baseID"
```

Expected.

```text
Create session idx=0 baseID=1
Create session idx=1 baseID=11
Create session idx=2 baseID=21
Create session idx=3 baseID=31
4 sessions were established using 1 as baseID
```

## 8. Validate BESS rules on the UPF host

Run on the UPF host.

```bash
curl -s http://127.0.0.1:8080/metrics | grep pfcp_sessions

sudo docker exec bess /opt/bess/bessctl/bessctl show module pdrLookup | grep -E "rules|packets|gate"
sudo docker exec bess /opt/bess/bessctl/bessctl show module farLookup | grep -E "rules|packets|gate"
sudo docker exec bess /opt/bess/bessctl/bessctl show module appQERLookup | grep -E "rules|packets|gate"
sudo docker exec bess /opt/bess/bessctl/bessctl show module sessionQERLookup | grep -E "rules|packets|gate"
```

Expected after four sessions.

```text
pfcp_sessions{node_id="192.168.90.2"} 4
pdrLookup, 8 rules
farLookup, 8 rules
appQERLookup, 16 rules
sessionQERLookup, 8 rules
```

## 9. Start TRex server

For per PG latency and RX correlation, start TRex in software mode.

```bash
cd /opt/trex/v3.08
sudo ./t-rex-64 -i --software --no-ofed-check -c 8
```

Use another terminal for `trex-console` or the Python scripts.

```bash
cd /opt/trex/v3.08
sudo ./trex-console
```

## 10. Manual TRex validation with four PG IDs

Inside `trex-console`.

```text
reset
service --port 1
start -f /home/ubuntu/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit/profiles/exp4_4slice_single_uplink_profile.py -p 0 -d 20 --force
stats
stats -l
```

Expected in `stats`.

```text
Port 0 opackets increases
Port 1 ipackets increases
Port 1 Rx bps > 0
oerrors 0
ierrors 0
```

Expected in `stats -l`.

```text
PG ID 1   TX pkts > 0   RX pkts > 0
PG ID 11  TX pkts > 0   RX pkts > 0
PG ID 21  TX pkts > 0   RX pkts > 0
PG ID 31  TX pkts > 0   RX pkts > 0
```

A validated run showed all four PG IDs with RX packets and Bronze loss caused by QER policing.

## 11. BESS delta check for a clean run

Use this when you want to prove that drops are QER drops and not forwarding failures.

On the UPF host, before the TRex run.

```bash
mkdir -p /tmp/exp4_check

sudo docker exec bess /opt/bess/bessctl/bessctl show module appQERLookup > /tmp/exp4_check/app_before.txt
sudo docker exec bess /opt/bess/bessctl/bessctl show module sessionQERLookup > /tmp/exp4_check/session_before.txt
sudo docker exec bess /opt/bess/bessctl/bessctl show module farLookup > /tmp/exp4_check/far_before.txt
sudo docker exec bess /opt/bess/bessctl/bessctl show module enp8s0np0Routes > /tmp/exp4_check/routes_before.txt
```

Run TRex for 20 seconds.

Then on the UPF host.

```bash
sudo docker exec bess /opt/bess/bessctl/bessctl show module appQERLookup > /tmp/exp4_check/app_after.txt
sudo docker exec bess /opt/bess/bessctl/bessctl show module sessionQERLookup > /tmp/exp4_check/session_after.txt
sudo docker exec bess /opt/bess/bessctl/bessctl show module farLookup > /tmp/exp4_check/far_after.txt
sudo docker exec bess /opt/bess/bessctl/bessctl show module enp8s0np0Routes > /tmp/exp4_check/routes_after.txt
```

Summarize deltas.

```bash
python3 - <<'PY'
import re
from pathlib import Path

base = Path('/tmp/exp4_check')
pairs = [
    ('appQERLookup', 'app_before.txt', 'app_after.txt'),
    ('sessionQERLookup', 'session_before.txt', 'session_after.txt'),
    ('farLookup', 'far_before.txt', 'far_after.txt'),
    ('routes', 'routes_before.txt', 'routes_after.txt'),
]
pat = re.compile(r'^\s*(\d+):.*?packets\s+(\d+)\s+.*?(?:->\s+0:([A-Za-z0-9_]+))?', re.M)
for name, before, after in pairs:
    print()
    print(name)
    btxt = (base / before).read_text()
    atxt = (base / after).read_text()
    b = {}
    for gate, packets, target in pat.findall(btxt):
        b[(gate, target)] = int(packets)
    for gate, packets, target in pat.findall(atxt):
        packets = int(packets)
        key = (gate, target)
        delta = packets - b.get(key, packets)
        label = target or ''
        print(f'gate {gate:>4} delta {delta:>12} packets {label}')
PY
```

Validated clean delta example.

```text
appQERLookup passed to sessionQERLookup, 3,500,267 packets
appQERLookup dropped to appQERMeterRed, 199,733 packets
appQERLookupFail, 0
sessionQERMeterRed, 0
farLookupFail, 0
enp8s0np0bad_route, 0
```

Interpretation.

```text
Traffic entered BESS
PDR lookup worked
GTP U decapsulation worked
QER policing worked
FAR lookup worked
N6 route worked
The observed packet loss was policer loss, not datapath failure
```

## 12. Real telemetry probe

The final script before DQN is `scripts/real_telemetry_probe.py` in the `trex_4slice_noise_toolkit_with_real_probe_v3` toolkit.

It does three things in one synchronized loop.

```text
1. Generates dynamic four slice TRex traffic on one TX port
2. Reads TRex PG ID counters through get_pgid_stats
3. Reads UPF Prometheus metrics from the UPF HTTP endpoint
```

The probe must use.

```text
TRex TX port 0
TRex RX port 1
TRex service mode on RX port 1
STLFlowLatencyStats, not STLFlowStats
TRex server started with --software
UPF metrics URL http://192.168.90.1:8080/metrics
```

Run on the TRex host.

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH

python3 scripts/real_telemetry_probe.py \
  --host 127.0.0.1 \
  --port 0 \
  --rx-port 1 \
  --duration 60 \
  --dt 1.0 \
  --upf-metrics-url http://192.168.90.1:8080/metrics \
  --force \
  --csv runs/real_telemetry_probe_60s.csv \
  --jsonl runs/real_telemetry_probe_60s.jsonl \
  --tb runs/tensorboard/real_telemetry_probe_60s
```

Expected output pattern.

```text
Enabled service mode on RX port=1
Connected to TRex 127.0.0.1:4501, tx-port=0, rx-port=1
Reading UPF metrics from http://192.168.90.1:8080/metrics
step=0000 offered=(...) Mbps rx=(...) Mbps loss=(...) n6=(...) Mbps pfcp=4
```

Validated output included positive RX values, dynamic losses, `n6_total_mbps > 0` and `pfcp=4`.

## 13. TensorBoard

Start TensorBoard on the TRex host.

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit
tensorboard --logdir runs/tensorboard --host 0.0.0.0
```

Main telemetry groups to inspect.

```text
traffic/offered_gold_mbps
traffic/offered_silver_mbps
traffic/offered_bronze_mbps
traffic/rx_gold_mbps
traffic/rx_silver_mbps
traffic/rx_bronze_mbps
traffic/loss_gold
traffic/loss_silver
traffic/loss_bronze
traffic/n6_total_mbps
traffic/n6_utilization
mbr/gold_kbps
mbr/silver_kbps
mbr/bronze_kbps
hidden/noise_offered_mbps
hidden/noise_rx_mbps
sanity/pfcp_sessions
```

## 14. DQN observation vector

The DQN should observe 17 features per time window.

```text
0   offered_gold_mbps
1   offered_silver_mbps
2   offered_bronze_mbps
3   rx_gold_mbps
4   rx_silver_mbps
5   rx_bronze_mbps
6   loss_gold
7   loss_silver
8   loss_bronze
9   qer_red_gold
10  qer_red_silver
11  qer_red_bronze
12  n6_total_mbps
13  n6_utilization
14  mbr_gold_kbps
15  mbr_silver_kbps
16  mbr_bronze_kbps
```

The environment stacks five windows.

```text
5 windows × 17 features = observation shape 85
```

## 15. Feature sources

```text
offered per slice
TRex PG ID tx_bps or target generated by traffic model

rx per slice
TRex PG ID rx_bps from STLFlowLatencyStats

loss per slice
computed from PG ID packet counters, not from Mbps
This matters because GTP U decapsulation changes packet size

qer_red per slice
proxy equal to per slice loss for now

n6_total_mbps
UPF Prometheus upf_bytes_count{dir="tx",iface="Core"} delta

n6_utilization
n6_total_mbps divided by configured N6 capacity

mbr values
local controller state, matching the latest pfcpsim QER modification
```

Prometheus is useful for aggregated UPF metrics, but it does not expose per TEID, per UE or per QER counters. BESS exposes gates such as `appQERMeterRed`, `farLookupFail` and `bad_route`, but the visible counters are aggregated by gate, not by slice.

## 16. Audit only metrics

These should be logged to CSV and TensorBoard, but not included in the DQN observation for the first real integration.

```text
noise_offered_mbps
noise_rx_mbps
noise_loss
qer_red_total
session_qer_red_total
pfcp_sessions
upf_access_rx_mbps
upf_access_drop_delta
pdrLookupFail delta
farLookupFail delta
bad_route delta
TRex total TX packets
TRex total RX packets
```

## 17. Important lessons and pitfalls

### 17.1 Do not use multiport per slice

All four slices share the same uplink. The separation is by TEID, UE IP, UDP port and PG ID.

### 17.2 Use STLFlowLatencyStats for per PG RX after decapsulation

`STLFlowStats` did not recover RX counters after the UPF decapsulation path. It showed TX per PG ID but RX zero. `STLFlowLatencyStats` with TRex software mode is the validated method.

### 17.3 Start TRex with software mode

Use.

```bash
sudo ./t-rex-64 -i --software --no-ofed-check -c 8
```

Without software mode, forwarding may work, but per PG latency and RX correlation may be unreliable.

### 17.4 Enable service mode on RX port

Inside console.

```text
service --port 1
```

The real telemetry probe does this automatically with `--rx-port 1`.

### 17.5 Python API returns PG ID keys as integers

`get_pgid_stats()` may return keys as integers.

```text
1, 11, 21, 31
```

The parser must handle both integer and string keys.

### 17.6 Prefer total counters for PG ID stats

The RX packets are on port 1. The parser must prefer.

```text
rx_pkts.total
rx_bps.total
```

and only then fall back to port specific values.

### 17.7 Do not compute loss from Mbps

TX packets include GTP U encapsulation. RX packets are decapsulated. Bytes and Mbps differ naturally. Packet counters are the correct basis for loss.

### 17.8 Prometheus is not per slice

UPF Prometheus gives useful aggregate metrics.

```text
pfcp_sessions
upf_packets_count
upf_bytes_count
upf_dropped_count
```

It does not give per slice QER RED or TEID counters.

### 17.9 BESS QER RED is aggregate by gate

`appQERLookup gate 3 -> appQERMeterRed` is useful as an aggregate policer sanity counter. For per slice control, use TRex PG ID loss as the practical proxy.

## 18. Traffic model

The dynamic generator uses four time varying streams.

```text
Gold, controlled
Silver, controlled
Bronze, controlled
Noise, hidden disturbance
```

The conceptual traffic model is.

```text
traffic(t) = regime mean + non synchronized sinusoids + AR noise + random bursts
```

The fourth slice is intentionally hidden from the DQN observation. It competes for capacity and creates perturbations that make the control problem nontrivial.

## 19. Validated test scenarios

### Scenario 1, PFCP four session validation

Purpose.

```text
Ensure the UPF has four PFCP sessions and BESS has all rules installed
```

Expected.

```text
pfcp_sessions = 4
pdrLookup = 8 rules
farLookup = 8 rules
appQERLookup = 16 rules
sessionQERLookup = 8 rules
```

### Scenario 2, manual four PG forwarding

Purpose.

```text
Ensure all four PG IDs are transmitted and received end to end
```

Command.

```text
reset
service --port 1
start -f /home/ubuntu/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit/profiles/exp4_4slice_single_uplink_profile.py -p 0 -d 20 --force
stats
stats -l
```

Expected.

```text
Port 1 RX > 0
PG IDs 1, 11, 21, 31 have RX pkts > 0
```

### Scenario 3, policer delta validation

Purpose.

```text
Confirm losses are due to QER policing and not forwarding failure
```

Expected.

```text
appQERMeterRed delta > 0 when a slice exceeds MBR
appQERLookupFail delta = 0
farLookupFail delta = 0
bad_route delta = 0
```

### Scenario 4, real telemetry probe

Purpose.

```text
Produce the real 17 feature observation vector and audit metrics in CSV, JSONL and TensorBoard
```

Expected.

```text
pfcp_sessions = 4
rx_gold_mbps > 0
rx_silver_mbps > 0
rx_bronze_mbps > 0
n6_total_mbps > 0
loss values finite and not NaN
```

### Scenario 5, future DQN integration

Purpose.

```text
Use the real telemetry source as the DQN MetricSource and pfcpsim modify as ActionApplier
```

Expected.

```text
DQN observes 85 dimensional stacked state
DQN controls Gold, Silver and Bronze MBRs
Noise slice remains hidden but logged
TensorBoard shows DQN metrics and traffic metrics together
```

## 20. Next step, plug with DQN

The immediate next step is to convert the validated real telemetry probe into a real environment source.

```text
Create RealTrexUpfMetricSource
Reuse the 17 feature order from this README
Keep Noise out of the DQN observation
Keep Noise in TensorBoard as hidden/noise metrics
Use pfcpsim session modify as RealActionApplier
Control only app level MBR for TEID 1, 11 and 21
Do not recreate sessions during DQN steps
```

The action applier should call PFCP Session Modification Request through pfcpsim.

Example shape for later.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session modify \
  --count 4 --baseID 1 \
  --app-filter "udp:any:any:allow:100" \
  --app-mbr-uplink   GOLD_KBPS,SILVER_KBPS,BRONZE_KBPS,50000 \
  --app-mbr-downlink GOLD_KBPS,SILVER_KBPS,BRONZE_KBPS,50000
```

The fourth value must remain fixed for the noise slice unless a specific experiment changes it.

## 21. Minimum recovery checklist

When returning to this testbed after a reboot, use this order.

```text
1. UPF host, start BESS and pfcpiface
2. UPF host, load up4 pipeline
3. UPF host, install N6 neighbor and route
4. UPF host, start bess-routectl
5. TRex host, start pfcpsim on enp9s0
6. TRex host, configure and associate PFCP
7. TRex host, delete old sessions if necessary
8. TRex host, create four sessions
9. UPF host, validate pfcp_sessions and BESS rule counts
10. TRex host, start t-rex-64 with --software
11. TRex console, validate manual PG IDs
12. TRex host, run real_telemetry_probe.py
13. Open TensorBoard and inspect traffic and sanity metrics
```

## 22. Quick command summary

TRex Python path.

```bash
export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH
```

TRex server.

```bash
cd /opt/trex/v3.08
sudo ./t-rex-64 -i --software --no-ofed-check -c 8
```

pfcpsim.

```bash
sudo docker rm -f pfcpsim 2>/dev/null || true
sudo docker run --rm -d --network host --name pfcpsim pfcpsim:patched -p 12345 --interface enp9s0
sudo docker exec pfcpsim pfcpctl -s localhost:12345 service configure --n3-addr 192.168.70.1 --remote-peer-addr 192.168.90.1
sudo docker exec pfcpsim pfcpctl -s localhost:12345 service associate
```

Create four sessions.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session create \
  --count 4 --baseID 1 \
  --ue-pool 10.250.0.0/24 \
  --gnb-addr 192.168.70.2 \
  --app-filter "udp:any:any:allow:100" \
  --session-mbr-uplink   1000000,1000000,1000000,1000000 \
  --session-mbr-downlink 1000000,1000000,1000000,1000000 \
  --app-mbr-uplink       200000,100000,50000,50000 \
  --app-mbr-downlink     200000,100000,50000,50000
```

Real telemetry probe.

```bash
python3 scripts/real_telemetry_probe.py \
  --host 127.0.0.1 \
  --port 0 \
  --rx-port 1 \
  --duration 60 \
  --dt 1.0 \
  --upf-metrics-url http://192.168.90.1:8080/metrics \
  --force \
  --csv runs/real_telemetry_probe_60s.csv \
  --jsonl runs/real_telemetry_probe_60s.jsonl \
  --tb runs/tensorboard/real_telemetry_probe_60s
```


