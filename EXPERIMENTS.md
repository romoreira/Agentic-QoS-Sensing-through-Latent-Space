# EXPERIMENTS

## Purpose

This document records the current experimental state of the four slice UPF, TRex, and DQN testbed. It is intended for a reader who opens the repository with no prior context and needs to understand what was built, why it was built, how it was validated, how to reproduce the tests, and what remains before enabling real BESS control.

The current milestone is an online DQN dry run. The DQN observes real live telemetry from TRex and the UPF, chooses an action, writes the intended MBR decision to CSV, JSONL, and TensorBoard, but does not modify BESS or PFCP state.

## High level goal

The experiment studies whether a DQN can learn slice aware MBR decisions for a mobile data plane with one uplink and multiple logical slices.

Main question

```text
Can an RL agent observe real per slice traffic pressure and select MBR actions that preserve higher priority slices while reacting to hidden background load?
```

Current design

```text
Gold    controlled slice, high priority
Silver  controlled slice, medium priority
Bronze  controlled slice, low priority
Noise   fourth slice, hidden from the DQN observation, used as background pressure
```

The fourth slice is deliberately not given to the DQN as a per slice feature. It appears only indirectly through aggregate N6 utilization, loss pressure, and audit logs. This is useful now for DQN stress testing and later for latent space sensing.

## Current status

Already working

* Four PFCP sessions are created and visible in the UPF
* BESS rules are installed for four TEIDs
* TRex sends four GTP-U streams over one uplink port
* TRex receives decapsulated traffic on the return port
* Per slice RX, TX, loss, and latency stats are available through TRex PG IDs
* UPF aggregate metrics are readable from Prometheus
* A sine based traffic model exists for Gold, Silver, and Bronze
* An abrupt background noise model exists for the Noise slice
* Real online telemetry is written to CSV, JSONL, and TensorBoard
* DQN replay from real CSV works, but it is not causal and should not be used as an acting policy
* DQN trained in the mock causal environment loads correctly
* Online DQN dry run works with live metrics and does not modify BESS
* The mock causal checkpoint produced plausible online dry run actions in the first 120 second test, mostly A00 and A02

Not enabled yet

* No real `pfcpctl modify` is called by the online DQN dry run
* The DQN does not enforce MBRs in BESS yet
* No final guardrail layer has been added
* No full long duration online dry run has been accepted yet
* No latent representation has been plugged in yet

## Repository layout

The repository currently has two main parts.

```text
Agentic-QoS-Sensing-through-Latent-Space/
  trex_4slice_noise_toolkit/
  upf-rl-env/
  bkp/upf-rl-env/
```

### `trex_4slice_noise_toolkit`

This directory contains the real traffic and telemetry tooling.

Expected content

```text
trex_4slice_noise_toolkit/
  profiles/
    exp4_4slice_single_uplink_profile.py

  scripts/
    preview_traffic_model.py
    real_telemetry_probe.py
    online_dqn_dryrun.py
    run_trex_dynamic_4slice.py
    audit_trex_stats.py
    audit_bess_stats.py
    traffic_model.py

  bess/
    check_bess_4slices.sh

  runs/
    live_sine_abrupt/
    hour_sine_abrupt/
    online_dqn_dryrun/
    online_dqn_dryrun_mock_causal/
    online_dqn_dryrun_mock_causal_420s/
    tensorboard/
```

Responsibilities

```text
Generate traffic
Collect real TRex PG ID telemetry
Collect UPF Prometheus metrics
Write TensorBoard data
Run online DQN dry run
Never modify BESS in dry run mode
```

### `upf-rl-env`

This directory contains the Gymnasium environment, the DQN agent, the action space, reward function, metric sources, and training scripts.

Expected content

```text
upf-rl-env/
  src/
    upf_rl_env/
      env.py
      config.py
      features.py
      reward.py
      action/
        space.py
        applier.py
      sources/
        base.py
        mock.py
        real_csv.py
        trex.py
        bess.py
      monitor/
        logger.py

    upf_rl_agents/
      dqn.py
      networks.py
      replay.py

  scripts/
    train_dqn.py
    eval_dqn.py
    sanity_check_real_csv.py

  checkpoints/
    dqn_real_csv/
    dqn_mock_causal/

  runs/
    dqn_real_csv/
    dqn_mock_causal/
    tensorboard/
```

Responsibilities

```text
Define the 85 dimension observation
Define the 15 action space
Compute reward
Train DQN
Save checkpoints
Evaluate DQN
Provide model checkpoints for online dry run
```

## Slice mapping

| Slice | TEID | UE IP | Source port | PG ID | Nominal app MBR | Controlled by DQN |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Gold | 1 | 10.250.0.1 | 10001 | 1 | 200000 kbps | yes |
| Silver | 11 | 10.250.0.2 | 10011 | 11 | 100000 kbps | yes |
| Bronze | 21 | 10.250.0.3 | 10021 | 21 | 50000 kbps | yes |
| Noise | 31 | 10.250.0.4 | 10031 | 31 | 50000 kbps | no |

Network endpoints

```text
GNB IP      192.168.70.2
UPF N3 IP   192.168.70.1
DN IP       192.168.80.2
UPF metrics http://192.168.90.1:8080/metrics
```

The four slices share one uplink. There is no multiport experiment here. These are logical slices over the same path.

## Control plane recovery

### Start or verify `pfcpsim`

The known running container has been

```text
container name  pfcpsim
image           pfcpsim:patched
entrypoint      pfcpsim
command         -p 12345 --interface enp9s0
```

Check it with

```bash
sudo docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | egrep -i "pfcpsim|bess|upf|pfcp" || true
sudo docker inspect pfcpsim --format 'Name={{.Name}} Image={{.Config.Image}} Cmd={{json .Config.Cmd}} Entrypoint={{json .Config.Entrypoint}}'
```

If the container is missing, the equivalent startup pattern is

```bash
sudo docker run -d \
  --name pfcpsim \
  --network host \
  --privileged \
  pfcpsim:patched \
  -p 12345 \
  --interface enp9s0
```

Use the interface that reaches the UPF PFCP side in your deployment.

Verify the CLI

```bash
sudo docker run --rm pfcpsim:patched --help
sudo docker exec pfcpsim sh -lc 'pfcpsim --help || /pfcpsim --help || true'
```

### Create the four PFCP sessions

Use this from the TRex host where `pfcpsim` is running.

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session delete \
  --count 4 --baseID 1 || true

sudo docker exec pfcpsim pfcpctl -s localhost:12345 session delete \
  --count 3 --baseID 1 || true

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

Expected log

```bash
sudo docker logs --tail 100 pfcpsim 2>&1 | egrep -i "Create session|sessions were established|Delete|error|fail|baseID"
```

Expected result

```text
Create session idx=0 baseID=1
Create session idx=1 baseID=11
Create session idx=2 baseID=21
Create session idx=3 baseID=31
4 sessions were established using 1 as baseID
```

### Verify BESS and UPF rules

On the UPF host

```bash
curl -s http://127.0.0.1:8080/metrics | grep pfcp_sessions

sudo docker exec bess /opt/bess/bessctl/bessctl show module pdrLookup | grep -E "rules|packets|gate"
sudo docker exec bess /opt/bess/bessctl/bessctl show module farLookup | grep -E "rules|packets|gate"
sudo docker exec bess /opt/bess/bessctl/bessctl show module appQERLookup | grep -E "rules|packets|gate"
sudo docker exec bess /opt/bess/bessctl/bessctl show module sessionQERLookup | grep -E "rules|packets|gate"
```

Expected state

```text
pfcp_sessions      4
pdrLookup          8 rules
farLookup          8 rules
appQERLookup       16 rules
sessionQERLookup   8 rules
```

Useful sanity check

```bash
sudo docker exec bess /opt/bess/bessctl/bessctl show module appQERLookup | grep -E "packets|appQERMeterRed|appQERLookupFail"
sudo docker exec bess /opt/bess/bessctl/bessctl show module sessionQERLookup | grep -E "packets|sessionQERMeterRed|sessionQERStatusDrop"
sudo docker exec bess /opt/bess/bessctl/bessctl show module farLookup | grep -E "packets|farLookupFail"
sudo docker exec bess /opt/bess/bessctl/bessctl show module enp8s0np0Routes | grep -E "packets|bad_route"
```

Expected failure counters

```text
appQERLookupFail       0
sessionQERStatusDrop   0
farLookupFail          0
enp8s0np0bad_route     0
```

QER RED counters may increase when offered traffic exceeds app level limits. That is not a datapath failure.

## TRex recovery and validation

### Start the TRex server

In one terminal on the TRex host

```bash
cd /opt/trex/v3.08
sudo ./t-rex-64 -i --software --no-ofed-check -c 8
```

The `--software` mode is important in this setup.

In any Python terminal that uses TRex STL libraries

```bash
export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH
```

Verify imports

```bash
python3 - <<'PY'
from trex_stl_lib.api import STLClient
print("TRex STL lib OK")
PY
```

### Manual four slice console validation

Open the console

```bash
cd /opt/trex/v3.08
sudo ./trex-console
```

Run

```text
reset
service --port 1
start -f /home/ubuntu/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit/profiles/exp4_4slice_single_uplink_profile.py -p 0 -d 20 --force
stats
stats -l
```

Expected result

```text
Port 1 RX > 0
PG ID 1  TX > 0 and RX > 0
PG ID 11 TX > 0 and RX > 0
PG ID 21 TX > 0 and RX > 0
PG ID 31 TX > 0 and RX > 0
Errors should be zero except expected policy loss on overloaded slices
```

## Important telemetry lessons learned

### Use `STLFlowLatencyStats` for this setup

A test with `STLFlowStats` showed TX counters but RX counters stayed zero after UPF decapsulation. This makes `STLFlowStats` unsuitable for the per slice RX metrics needed by the DQN.

The working path is

```text
STLFlowLatencyStats
TRex server in software mode
service mode enabled on RX port 1
PG ID stats read with total counters
```

### PG ID keys may be integers

The Python API can return PG ID keys as integers, not strings.

Use robust lookup logic.

```python
def pg_get(d, pg_id):
    return d.get(pg_id) or d.get(str(pg_id)) or {}
```

### Prefer total counters

The RX packets for each PG ID are on port 1. The parser must prefer

```text
rx_pkts.total
rx_bytes.total
rx_bps.total
```

then fall back to

```text
rx_pkts.1
rx_bytes.1
rx_bps.1
```

and only then to port 0.

### TensorBoard tag groups

In the current real telemetry probe, the useful tag groups are

```text
traffic_target/*
dqn_features/*
audit/*
```

Older `trex_pgid/*` tags may not exist in the current probe. That is expected.

## Traffic model

The current recommended model is

```text
sine_abrupt
```

Its intent is

```text
Gold    smooth sine wave
Silver  smooth sine wave with different period
Bronze  smooth sine wave with different period
Noise   abrupt background steps and bursts
```

The point is to separate predictable controlled demand from hidden abrupt background pressure.

Preview without TRex

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

python3 scripts/preview_traffic_model.py \
  --traffic-model sine_abrupt \
  --duration 3600 \
  --dt 1.0 \
  --csv runs/hour_sine_abrupt/traffic_plan_1h.csv \
  --tb runs/tensorboard/traffic_plan_1h_sine_abrupt
```

Open TensorBoard

```bash
tensorboard --logdir runs/tensorboard \
  --host 0.0.0.0 \
  --port 6006
```

Look at

```text
traffic_target/gold_mbps
traffic_target/silver_mbps
traffic_target/bronze_mbps
traffic_target/noise_mbps
```

This is only the plan, not real telemetry.

## Real online telemetry without DQN

This mode generates real traffic and collects real telemetry. It does not use DQN and it does not modify BESS.

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH

rm -rf runs/live_sine_abrupt
rm -rf runs/tensorboard/live_sine_abrupt

python3 scripts/real_telemetry_probe.py \
  --host 127.0.0.1 \
  --port 0 \
  --rx-port 1 \
  --duration 3600 \
  --dt 1.0 \
  --traffic-model sine_abrupt \
  --upf-metrics-url http://192.168.90.1:8080/metrics \
  --force \
  --csv runs/live_sine_abrupt/real_telemetry_live.csv \
  --jsonl runs/live_sine_abrupt/real_telemetry_live.jsonl \
  --tb runs/tensorboard/live_sine_abrupt
```

TensorBoard

```bash
tensorboard --logdir runs/tensorboard/live_sine_abrupt \
  --host 0.0.0.0 \
  --port 6006
```

Check

```text
dqn_features/offered_gold_mbps
dqn_features/rx_gold_mbps
dqn_features/loss_gold
dqn_features/n6_total_mbps
dqn_features/n6_utilization
audit/noise_offered_mbps
audit/noise_rx_mbps
audit/pfcp_sessions
```

Criteria

```text
pfcp_sessions = 4
rx_* > 0
n6_total_mbps > 0
noise has abrupt behavior
loss_* reacts to pressure
upf_access_drop_delta is zero or stable
```

## DQN observation

The DQN observation is

```text
5 stacked windows × 17 features = 85 dimensions
```

Feature order per window

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

The Noise slice is excluded from the DQN observation as a per slice feature. It is logged under audit.

Normalization is centralized in

```text
upf-rl-env/src/upf_rl_env/features.py
```

Current normalization

```python
f[0:3] = m.offered_mbps / cfg.norm_mbps
f[3:6] = m.rx_mbps / cfg.norm_mbps
f[6:9] = m.loss_rate
f[9:12] = np.log1p(m.qer_red) / 10.0
f[12] = m.n6_tx_mbps / cfg.norm_mbps
f[13] = m.n6_utilization
f[14:17] = m.mbr_kbps.astype(np.float32) / cfg.norm_kbps
```

Relevant defaults

```text
norm_mbps = 250.0
norm_kbps = 250000.0
```

A check confirmed online observations are normalized, with values roughly between 0.0 and 0.88 during the dry run.

## Action space

The action space is

```text
Discrete(15)
```

Actions map to per slice MBR percentages. The exact mapping is defined in

```text
upf-rl-env/src/upf_rl_env/action/space.py
```

Important actions observed so far

```text
A00 = Gold 100%, Silver 100%, Bronze 100%
A02 = Gold 100%, Silver 100%, Bronze 70%
A11 = Gold 85%, Silver 30%, Bronze 30%
```

A11 was the bad action selected by the checkpoint trained on real CSV replay. A00 and A02 were selected by the checkpoint trained in the mock causal environment.

## Why real CSV replay was not enough

The real CSV contains real metrics from TRex and the UPF. It is useful for validating feature shape, normalization, TensorBoard logging, and offline inference.

It is not enough to train a control policy, because the action does not change the next row of the CSV.

In replay mode

```text
DQN chooses action
environment advances to next CSV row
loss, RX, and utilization are already fixed
```

Therefore, the action has no causal effect on the future telemetry. This caused a policy collapse in one experiment.

Observed behavior

```text
checkpoint trained on real_csv selected A11 for every online dry run step
A11 was too aggressive for low N6 utilization
```

The reward also includes a large SLA penalty when Gold loss exceeds `0.01`. Real measured loss often sits around `0.066`, so this penalty can dominate if used directly with real telemetry.

Conclusion

```text
real_csv is good for validation
real_csv alone is not good for learning a causal control policy
```

## Causal mock training

The mock environment is currently the correct place to train DQN, because in the mock environment the selected action changes MBR and the new MBR changes the next state.

Run causal mock training

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/upf-rl-env

rm -rf runs/tensorboard/dqn_mock_causal
rm -rf runs/dqn_mock_causal
rm -rf checkpoints/dqn_mock_causal

python3 scripts/train_dqn.py \
  --device cuda \
  --metric-source mock \
  --total-steps 200000 \
  --save-dir checkpoints/dqn_mock_causal \
  --log-csv-dir runs/dqn_mock_causal \
  --tb-dir runs/tensorboard/dqn_mock_causal
```

Expected completion

```text
checkpoints/dqn_mock_causal/final.pt
```

Check TensorBoard

```bash
tensorboard --logdir runs/tensorboard/dqn_mock_causal \
  --host 0.0.0.0 \
  --port 6006
```

Useful tags

```text
env/reward
dqn/loss
dqn/q_mean
dqn/td_abs_mean
dqn/epsilon
action/action_id
mbr/gold_kbps
mbr/silver_kbps
mbr/bronze_kbps
traffic/n6_utilization
reward_components/*
```

## Online DQN dry run

This is the current important milestone.

In this mode

```text
TRex generates real traffic
UPF/BESS processes real traffic
the script reads live TRex and UPF metrics
the DQN observes the normalized live metrics
the DQN chooses an action
the action is logged only
no pfcpctl command is issued
BESS is not modified
```

Run short dry run with causal mock checkpoint

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH

rm -rf runs/online_dqn_dryrun_mock_causal
rm -rf runs/tensorboard/online_dqn_dryrun_mock_causal

python3 scripts/online_dqn_dryrun.py \
  --host 127.0.0.1 \
  --port 0 \
  --rx-port 1 \
  --duration 120 \
  --dt 1.0 \
  --traffic-model sine_abrupt \
  --upf-metrics-url http://192.168.90.1:8080/metrics \
  --checkpoint ../upf-rl-env/checkpoints/dqn_mock_causal/final.pt \
  --device cpu \
  --force \
  --csv runs/online_dqn_dryrun_mock_causal/online_dqn_dryrun.csv \
  --jsonl runs/online_dqn_dryrun_mock_causal/online_dqn_dryrun.jsonl \
  --tb runs/tensorboard/online_dqn_dryrun_mock_causal
```

Expected terminal message

```text
DRY RUN MODE: actions are logged only. No pfcpctl call. No BESS/PFCP modification.
```

Open TensorBoard

```bash
tensorboard --logdir runs/tensorboard/online_dqn_dryrun_mock_causal \
  --host 0.0.0.0 \
  --port 6006
```

Useful tags

```text
live_metrics/loss_gold
live_metrics/loss_silver
live_metrics/loss_bronze
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

Summarize actions

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("runs/online_dqn_dryrun_mock_causal/online_dqn_dryrun.csv")

print("Action counts")
print(df["dryrun_action_id"].value_counts().sort_index())

print()
print(df[[
    "loss_gold",
    "loss_silver",
    "loss_bronze",
    "n6_utilization",
    "noise_offered_mbps",
    "dryrun_action_id",
    "dryrun_action_description",
]].describe(include="all"))
PY
```

The first short test with the mock causal checkpoint produced

```text
A00 84 times
A02 36 times
```

This was a better pattern than the earlier real CSV checkpoint, because the DQN mostly preserved all MBRs and reduced only Bronze in a limited interval.

A follow up summary showed that A02 appeared when Gold offered traffic was higher, with mean Gold offered around 188.8 Mbps for A02 versus 172.4 Mbps for A00.

## Current 420 second dry run

The current long short test is a 420 second online dry run with the causal mock checkpoint.

Why 420 seconds

```text
It is long enough to cover the slowest sine period used in the current sine_abrupt model.
It is still short enough for fast debugging.
It does not act on BESS.
```

Run

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

rm -rf runs/online_dqn_dryrun_mock_causal_420s
rm -rf runs/tensorboard/online_dqn_dryrun_mock_causal_420s

python3 scripts/online_dqn_dryrun.py \
  --host 127.0.0.1 \
  --port 0 \
  --rx-port 1 \
  --duration 420 \
  --dt 1.0 \
  --traffic-model sine_abrupt \
  --upf-metrics-url http://192.168.90.1:8080/metrics \
  --checkpoint ../upf-rl-env/checkpoints/dqn_mock_causal/final.pt \
  --device cpu \
  --force \
  --csv runs/online_dqn_dryrun_mock_causal_420s/online_dqn_dryrun.csv \
  --jsonl runs/online_dqn_dryrun_mock_causal_420s/online_dqn_dryrun.jsonl \
  --tb runs/tensorboard/online_dqn_dryrun_mock_causal_420s
```

Summarize

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("runs/online_dqn_dryrun_mock_causal_420s/online_dqn_dryrun.csv")

print("Action counts")
print(df["dryrun_action_id"].value_counts().sort_index())

print("\nMean metrics by action")
print(df.groupby("dryrun_action_id")[[
    "loss_gold",
    "loss_silver",
    "loss_bronze",
    "n6_utilization",
    "noise_offered_mbps",
    "offered_gold_mbps",
    "offered_silver_mbps",
    "offered_bronze_mbps",
]].mean())
PY
```

Acceptance criteria

```text
BESS still unmodified
pfcp_sessions remains 4
rx_* remains positive
n6_total_mbps remains positive
A00 dominates when there is no pressure
Bronze reducing actions appear only under plausible pressure
Gold should not be reduced unless the situation is extreme
No reappearance of the A11 collapse
```

## TensorBoard hygiene

Do not point TensorBoard to broad parent directories when comparing specific tests. Use one logdir per question.

Real telemetry only

```bash
tensorboard --logdir runs/tensorboard/live_sine_abrupt \
  --host 0.0.0.0 \
  --port 6006
```

Causal mock training only

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/upf-rl-env

tensorboard --logdir runs/tensorboard/dqn_mock_causal \
  --host 0.0.0.0 \
  --port 6006
```

Online dry run only

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

tensorboard --logdir runs/tensorboard/online_dqn_dryrun_mock_causal \
  --host 0.0.0.0 \
  --port 6006
```

420 second online dry run only

```bash
tensorboard --logdir runs/tensorboard/online_dqn_dryrun_mock_causal_420s \
  --host 0.0.0.0 \
  --port 6006
```

Stop old TensorBoard instances if needed

```bash
pkill -f tensorboard || true
```

## Interpretation notes

### Target traffic versus real telemetry

`traffic_target/*` is the intended generation pattern.

`dqn_features/*` is what the agent would observe after TRex, BESS, QER, and UPF processing.

`audit/*` is additional information for sanity checks.

### Offered Mbps versus RX Mbps

Offered traffic is measured before UPF decapsulation, while RX traffic is measured after decapsulation.

Before UPF

```text
Ether / IP / UDP / GTP-U / IP / UDP / payload
```

After UPF

```text
Ether / IP / UDP / payload
```

Therefore, Mbps values before and after decapsulation are not exactly comparable by byte count.

For loss, use packet based loss

```text
loss = max(tx_pkts - rx_pkts, 0) / max(tx_pkts, 1)
```

This is the more stable delivery signal.

### Real loss baseline

The live testbed often reports loss around 0.066 even in moderate traffic windows. This must be calibrated before using a strict SLA threshold such as 0.01 in online reward logic.

Do not interpret `loss > 0.01` as immediate service failure until the measurement baseline is calibrated.

## What was tested

### Four session creation

Result

```text
TEID 1, 11, 21, and 31 were created
pfcp_sessions became 4
```

### BESS rule installation

Result

```text
pdrLookup          8 rules
farLookup          8 rules
appQERLookup       16 rules
sessionQERLookup   8 rules
```

### TRex four PG ID forwarding

Result

```text
PG ID 1, 11, 21, and 31 had TX and RX counters
Port 1 received decapsulated traffic
```

### QER behavior

Result

```text
loss was attributed to appQERMeterRed when caps were exceeded
pdrLookupFail stayed zero
farLookupFail stayed zero
bad_route stayed zero
```

### Real telemetry probe

Result

```text
real_telemetry_probe.py writes valid CSV, JSONL, and TensorBoard
RX by PG ID works
Prometheus UPF metrics work
pfcp_sessions remains 4
```

### DQN real CSV replay

Result

```text
valid for observation and normalization checks
not valid as a causal policy training source
policy collapsed to A11
```

### DQN mock causal training

Result

```text
checkpoint was trained in an environment where actions affect future telemetry
checkpoint loaded in online dry run
policy no longer collapsed to A11 in the short test
```

### Online DQN dry run

Result

```text
live metrics were read online
DQN selected actions online
actions were logged only
BESS was not modified
```

## What remains before real BESS control

### Immediate next step

Complete the 420 second dry run and inspect action distribution.

Accept if

```text
A00 dominates normal states
Bronze reducing actions appear under plausible pressure
Gold is preserved
pfcp_sessions remains 4
no BESS modification occurs
```

### Add guardrails

Before allowing any real action, add a safety layer.

Candidate guardrails

```text
Never reduce Gold if n6_utilization < 0.8 and loss_gold is near baseline
Never apply A11 unless n6_utilization is high or sustained loss is severe
Do not apply actions more often than every 10 seconds
Require action persistence for two or three windows before applying
Keep Noise MBR fixed at 50000 kbps
Log proposed action and safe action separately
```

### Implement real action applier

The real action applier should eventually call `pfcpctl session modify`.

Template

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session modify \
  --count 4 --baseID 1 \
  --app-filter "udp:any:any:allow:100" \
  --session-mbr-uplink   1000000,1000000,1000000,1000000 \
  --session-mbr-downlink 1000000,1000000,1000000,1000000 \
  --app-mbr-uplink       GOLD,SILVER,BRONZE,50000 \
  --app-mbr-downlink     GOLD,SILVER,BRONZE,50000
```

The fourth value is always the Noise MBR and should remain fixed during the first online control experiment.

### Run online control in stages

Recommended sequence

```text
1. online telemetry only
2. online DQN dry run, no BESS modification
3. online DQN dry run with guardrails logged
4. online DQN with guardrails and manual confirmation outside the script
5. online DQN with guardrails and automatic pfcpctl modify
```

### Later latent space experiment

Once the raw DQN path is stable, add a latent encoder.

The latent experiment should use the same traffic, same action space, same reward, and same evaluation windows. Only the observation representation changes.

Potential comparison

```text
Raw DQN        uses 5 × 17 handcrafted features
Latent DQN     uses an encoded representation of the recent telemetry sequence
```

The hidden Noise slice is useful here because the latent encoder can learn to infer hidden pressure from the visible controlled slices and aggregate N6 behavior.

## Quick reproduction checklist

Start the infrastructure

```bash
sudo docker ps
sudo docker logs --tail 100 pfcpsim
curl -s http://192.168.90.1:8080/metrics | grep pfcp_sessions
```

Create four sessions

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

Start TRex

```bash
cd /opt/trex/v3.08
sudo ./t-rex-64 -i --software --no-ofed-check -c 8
```

Run online dry run

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH

python3 scripts/online_dqn_dryrun.py \
  --host 127.0.0.1 \
  --port 0 \
  --rx-port 1 \
  --duration 120 \
  --dt 1.0 \
  --traffic-model sine_abrupt \
  --upf-metrics-url http://192.168.90.1:8080/metrics \
  --checkpoint ../upf-rl-env/checkpoints/dqn_mock_causal/final.pt \
  --device cpu \
  --force \
  --csv runs/online_dqn_dryrun_mock_causal/online_dqn_dryrun.csv \
  --jsonl runs/online_dqn_dryrun_mock_causal/online_dqn_dryrun.jsonl \
  --tb runs/tensorboard/online_dqn_dryrun_mock_causal
```

Confirm no BESS modification

```text
The online dry run must print
DRY RUN MODE: actions are logged only. No pfcpctl call. No BESS/PFCP modification.
```

Summarize

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("runs/online_dqn_dryrun_mock_causal/online_dqn_dryrun.csv")
print(df["dryrun_action_id"].value_counts().sort_index())
print(df.groupby("dryrun_action_id")[["n6_utilization", "offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps"]].mean())
PY
```

## Current decision

Do not enable real BESS control yet.

The online dry run path is working. The next decision should be based on the 420 second run summary. If the policy remains conservative and plausible, the next engineering task is to add guardrails and an explicit safe action layer before implementing real `pfcpctl modify`.

