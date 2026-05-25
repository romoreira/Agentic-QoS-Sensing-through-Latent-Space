# Traffic design, one uplink and hidden pressure

This package uses one physical TRex uplink and several GTP-U streams over that same uplink. It does not require one port per slice.

## Lab topology assumption

```text
TRex TX port 0
  Gold stream    TEID 1
  Silver stream  TEID 11
  Bronze stream  TEID 21
  Noise stream   TEID 31, optional fourth slice
        |
        v
UPF N3, same uplink, same N6 bottleneck
```

The three controlled slices are Gold, Silver, and Bronze. The fourth stream is only background pressure. It is useful for the learning problem, but it is not an action dimension.

## Why keep the fourth stream

If only the three controlled slices exist and all their offered loads are directly observed, the DQN can often learn a mostly reactive mapping. The Noise stream creates partial observability.

The agent sees aggregate N6 utilization and the visible slice losses, but it does not receive a separate `offered_noise_mbps` feature. During training, the hidden value is exposed only in `info["hidden_state"]`, CSV logs, and TensorBoard audit scalars.

This makes the later latent-space comparison meaningful. A temporal encoder can learn whether the last windows suggest a hidden pressure regime, while the raw DQN has to infer that from the stacked observations.

## Mock dynamics

The mock source uses `upf_rl_traffic.dynamics.SliceTrafficGenerator`.

Each slice offered load is generated from.

```text
offered_i(t) = regime_mean_i
             + sinusoidal_i(t)
             + AR1_i(t)
             + burst_i(t)
```

Main properties.

- Gold, Silver, Bronze, and Noise have different phase offsets
- Three sinusoidal frequencies are summed, so the slices are not synchronized
- AR(1) noise creates short-term correlation
- Poisson bursts create sudden overloads with exponential decay
- Regimes switch every 8 to 25 seconds
- The N6 capacity has a slow random walk

The mock then applies two stages.

1. Per-slice MBR enforcement for controlled slices
2. Shared N6 bottleneck, with priority Gold, Silver, Bronze, then Noise

This is enough to make the agent decide when Bronze and Silver must be tightened to preserve Gold.

## TRex profile

The file `src/upf_rl_traffic/trex_profile.py` follows the working profile style.

| Slice  | TEID | UE IP       | UDP sport | PG ID | Default pps | Visible to DQN |
|--------|-----:|-------------|----------:|------:|------------:|---------------:|
| Gold   |    1 | 10.250.0.1  |     10001 |     1 |      50 000 | yes            |
| Silver |   11 | 10.250.0.2  |     10011 |    11 |      50 000 | yes            |
| Bronze |   21 | 10.250.0.3  |     10021 |    21 |      50 000 | yes            |
| Noise  |   31 | 10.250.0.4  |     10031 |    31 |      35 000 | no             |

Packet shape.

```text
Ether()
/ IP(src="192.168.70.2", dst="192.168.70.1")
/ UDP(sport=2152, dport=2152, chksum=0)
/ GTP_U_Header(teid=<slice_teid>)
/ IP(src=<ue_ip>, dst="192.168.80.2")
/ UDP(sport=<slice_src_port>, dport=5555, chksum=0)
/ payload
```

## Running fixed TRex traffic

```bash
export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH
python scripts/run_trex_traffic.py --port 0 --duration 600
```

Optional variants.

```bash
python scripts/run_trex_traffic.py --port 0 --no-noise
python scripts/run_trex_traffic.py --port 0 --pps-scale 0.75
python scripts/run_trex_traffic.py --port 0 --pps-scale 1.25
```

No per-slice TRex port modulation is performed. This avoids the incorrect multiport assumption and keeps traffic continuous.

## Auditing the learning signal

Use TensorBoard during training.

```bash
tensorboard --logdir runs/tensorboard
```

Important groups.

```text
traffic/offered_gold_mbps
traffic/offered_silver_mbps
traffic/offered_bronze_mbps
traffic/loss_gold
traffic/loss_silver
traffic/loss_bronze
traffic/n6_utilization
mbr/gold_kbps
mbr/silver_kbps
mbr/bronze_kbps
hidden/noise_offered_mbps
hidden/n6_capacity_mbps
dqn/loss
dqn/q_mean
dqn/epsilon
eval/return_mean
eval/sla_viol_rate
```

The hidden scalars are only for audit. They are deliberately not part of the observation vector.
