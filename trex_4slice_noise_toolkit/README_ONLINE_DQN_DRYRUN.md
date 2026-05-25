# Online DQN dry-run, live telemetry, no BESS action

This mode is the next integration step after `real_telemetry_probe.py`.
It uses live TRex and UPF telemetry and asks a trained DQN checkpoint to choose an action at each window, but it does not call `pfcpctl`, does not modify BESS, and does not change PFCP sessions.

## What it does

1. Generates four GTP-U slices on one TRex TX port.
2. Enables service mode on the TRex RX port for PG-ID correlation.
3. Reads live PG-ID TX/RX counters from TRex.
4. Reads live UPF Prometheus counters from `http://192.168.90.1:8080/metrics`.
5. Builds the same 85-dimensional stacked observation used by the DQN.
6. Loads a trained DQN checkpoint.
7. Chooses an action and logs the intended MBR values.
8. Writes CSV, JSONL and TensorBoard.

## Safety guarantee

`online_dqn_dryrun.py` does not import or call a real action applier. It logs intended actions only.

No BESS/PFCP modification is performed. The actual MBR fields in the observation remain fixed unless you pass different `--mbr-*` arguments.

## Expected prerequisites

The RL package must be installed:

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/upf-rl-env
python3 -m pip install -e .
```

A trained checkpoint should exist, for example:

```text
~/Agentic-QoS-Sensing-through-Latent-Space/upf-rl-env/checkpoints/dqn_real_csv/final.pt
```

TRex server must be running in another terminal:

```bash
cd /opt/trex/v3.08
sudo ./t-rex-64 -i --software --no-ofed-check -c 8
```

The UPF must expose Prometheus metrics from the TRex host:

```bash
curl -s --connect-timeout 3 http://192.168.90.1:8080/metrics \
  | grep -E "pfcp_sessions|upf_packets_count|upf_bytes_count|upf_dropped_count"
```

## Run

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH

rm -rf runs/online_dqn_dryrun
rm -rf runs/tensorboard/online_dqn_dryrun

python3 scripts/online_dqn_dryrun.py \
  --host 127.0.0.1 \
  --port 0 \
  --rx-port 1 \
  --duration 600 \
  --dt 1.0 \
  --traffic-model sine_abrupt \
  --upf-metrics-url http://192.168.90.1:8080/metrics \
  --checkpoint ../upf-rl-env/checkpoints/dqn_real_csv/final.pt \
  --device cuda \
  --force \
  --csv runs/online_dqn_dryrun/online_dqn_dryrun.csv \
  --jsonl runs/online_dqn_dryrun/online_dqn_dryrun.jsonl \
  --tb runs/tensorboard/online_dqn_dryrun
```

## TensorBoard

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

tensorboard --logdir runs/tensorboard/online_dqn_dryrun \
  --host 0.0.0.0 \
  --port 6006
```

Important tags:

```text
live_metrics/offered_gold_mbps
live_metrics/rx_gold_mbps
live_metrics/loss_gold
live_metrics/n6_total_mbps
live_metrics/n6_utilization

dryrun/action_id
dryrun/intended_mbr_gold_kbps
dryrun/intended_mbr_silver_kbps
dryrun/intended_mbr_bronze_kbps
dryrun/reward_diagnostic
dryrun/q_max
dryrun/q_mean

audit/noise_offered_mbps
audit/noise_rx_mbps
audit/pfcp_sessions
```

## Interpretation

`live_metrics/*` is real online telemetry.

`dryrun/*` is the DQN decision based on that telemetry.

`actual_mbr/*` is the MBR assumed to be currently enforced in the UPF. Since this is dry-run, these values are not modified by the DQN.

`dryrun/intended_mbr_*` is what the DQN would apply if the real action applier were enabled.

## Success criteria

```text
pfcp_sessions remains 4
live_metrics/rx_* is greater than zero
dryrun/action_id changes when losses or N6 utilization change
dryrun/intended_mbr_* makes sense, Bronze should usually be cut before Silver, and Gold last
No pfcpctl modify appears in any log
BESS counters keep running, but MBR rules are not modified by this script
```
