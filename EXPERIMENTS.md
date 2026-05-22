# README

# Sinusoidal Slice MBR Control Baselines

## Goal

This experiment evaluates slice aware MBR control in a BESS UPF deployment.

The objective is to preserve the Gold slice under overload by reducing the MBR of lower priority slices, mainly Bronze first and Silver second.

This stage does not implement queue priority in BESS. It uses PFCP and QER updates through `pfcpctl session modify`.

Correct name for this stage

```text
Slice aware MBR control
```

Avoid calling this stage queue priority scheduling.

## Experiment idea

Generate three non synchronized sinusoidal traffic demands with TRex.

```text
Gold
protected slice

Silver
intermediate slice

Bronze
best effort slice
```

The aggregate offered load must exceed the effective N6 capacity during part of the experiment. This forces the controller to decide which slice should be degraded.

Example demand model

```text
rate(t) = base_rate + amplitude * sin(2*pi*frequency*t + phase)
```

Example starting scenario

| Slice | Base Mbps | Amplitude Mbps | Phase | Expected behavior |
|---|---:|---:|---:|---|
| Gold | 300 | 100 | 0 | Preserve |
| Silver | 250 | 150 | pi divided by 3 | Degrade after Bronze |
| Bronze | 250 | 200 | 2 pi divided by 3 | Degrade first |

Example bottleneck

```text
N6 effective capacity
700 Mbps

Aggregate peak demand
greater than 900 Mbps
```

## Common rule for all controllers

All controllers should choose an action from the same action space.

Example action space

| Action ID | Gold Mbps | Silver Mbps | Bronze Mbps | Meaning |
|---:|---:|---:|---:|---|
| 0 | 1000 | 1000 | 1000 | No restriction |
| 1 | 1000 | 1000 | 500 | Mild Bronze reduction |
| 2 | 1000 | 1000 | 200 | Strong Bronze reduction |
| 3 | 1000 | 700 | 100 | Bronze strong plus Silver moderate |
| 4 | 800 | 600 | 100 | Emergency shared reduction |

Gold should only be reduced in emergency actions.

## Common runtime command

All runtime controllers should apply MBR changes with PFCP.

Example

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session modify   --count 3 --baseID 1   --ue-pool 10.250.0.0/24   --gnb-addr 192.168.70.2   --app-filter "udp:any:any:allow:100"   --session-mbr-uplink   1000000,1000000,1000000   --session-mbr-downlink 1000000,1000000,1000000   --app-mbr-uplink       1000000,700000,100000   --app-mbr-downlink     1000000,700000,100000
```

## Common telemetry

Collect one telemetry row per control window.

Suggested window size

```text
1 second
```

Suggested duration

```text
300 to 600 seconds per run
```

## Common CSV outputs

Every baseline should write these files.

```text
results/<baseline_name>/telemetry_windows.csv
results/<baseline_name>/actions.csv
results/<baseline_name>/trex_stats.csv
results/<baseline_name>/bess_counters.csv
results/<baseline_name>/summary_metrics.csv
```

## telemetry_windows.csv

Recommended columns

| Column | Meaning |
|---|---|
| timestamp | Window timestamp |
| scenario_id | Scenario identifier |
| window_index | Control window index |
| gold_offered_mbps | TRex offered load for Gold |
| silver_offered_mbps | TRex offered load for Silver |
| bronze_offered_mbps | TRex offered load for Bronze |
| gold_rx_mbps | Delivered Gold throughput |
| silver_rx_mbps | Delivered Silver throughput |
| bronze_rx_mbps | Delivered Bronze throughput |
| gold_loss_rate | Gold packet loss ratio |
| silver_loss_rate | Silver packet loss ratio |
| bronze_loss_rate | Bronze packet loss ratio |
| n3_rx_mbps | UPF N3 received throughput |
| n6_tx_mbps | UPF N6 transmitted throughput |
| n6_utilization | N6 throughput divided by N6 capacity |
| app_qer_red_gold | Gold app QER red packets or red delta |
| app_qer_red_silver | Silver app QER red packets or red delta |
| app_qer_red_bronze | Bronze app QER red packets or red delta |
| session_qer_red_gold | Gold session QER red packets or red delta |
| session_qer_red_silver | Silver session QER red packets or red delta |
| session_qer_red_bronze | Bronze session QER red packets or red delta |
| current_mbr_gold_kbps | Current Gold MBR |
| current_mbr_silver_kbps | Current Silver MBR |
| current_mbr_bronze_kbps | Current Bronze MBR |

## actions.csv

Recommended columns

| Column | Meaning |
|---|---|
| timestamp | Decision timestamp |
| scenario_id | Scenario identifier |
| window_index | Window index |
| controller_name | Baseline name |
| action_id | Selected action |
| target_mbr_gold_kbps | New Gold MBR |
| target_mbr_silver_kbps | New Silver MBR |
| target_mbr_bronze_kbps | New Bronze MBR |
| reason | Short decision explanation |
| decision_latency_ms | Time to compute action |
| pfcp_apply_latency_ms | Time to apply PFCP update |
| apply_status | applied, skipped, or failed |

## trex_stats.csv

Recommended columns

| Column | Meaning |
|---|---|
| timestamp | Collection timestamp |
| stream_id | TRex stream identifier |
| slice | Gold, Silver, or Bronze |
| offered_mbps | Desired sinusoidal offered load |
| tx_mbps | TRex transmitted throughput |
| rx_mbps | TRex received throughput |
| tx_packets | Transmitted packets |
| rx_packets | Received packets |
| lost_packets | Lost packets |
| loss_rate | Packet loss rate |
| latency_avg_us | Average latency if latency streams are enabled |
| latency_p95_us | p95 latency if available |
| latency_p99_us | p99 latency if available |
| jitter_us | Jitter if available |

## bess_counters.csv

Recommended columns

| Column | Meaning |
|---|---|
| timestamp | Collection timestamp |
| pdr_rules | Number of PDR rules |
| far_rules | Number of FAR rules |
| app_qer_rules | Number of app QER rules |
| session_qer_rules | Number of session QER rules |
| pdr_packets | PDR matched packets |
| pdr_fail_packets | PDR lookup failures |
| app_qer_green_packets | App QER green packets |
| app_qer_yellow_packets | App QER yellow packets |
| app_qer_red_packets | App QER red packets |
| session_qer_green_packets | Session QER green packets |
| session_qer_yellow_packets | Session QER yellow packets |
| session_qer_red_packets | Session QER red packets |
| far_packets | FAR matched packets |
| far_fail_packets | FAR lookup failures |
| n3_rx_packets | N3 port RX packets |
| n3_rx_bytes | N3 port RX bytes |
| n6_tx_packets | N6 port TX packets |
| n6_tx_bytes | N6 port TX bytes |

## summary_metrics.csv

Recommended columns

| Metric | Meaning |
|---|---|
| gold_sla_violation_rate | Fraction of windows where Gold violates SLA |
| gold_avg_throughput_mbps | Average delivered Gold throughput |
| gold_p95_loss_rate | p95 Gold loss rate |
| silver_avg_throughput_mbps | Average delivered Silver throughput |
| bronze_avg_throughput_mbps | Average delivered Bronze throughput |
| bronze_degradation_ratio | Bronze delivered divided by Bronze offered |
| n6_avg_utilization | Average N6 utilization |
| n6_p95_utilization | p95 N6 utilization |
| total_goodput_mbps | Total delivered throughput |
| total_loss_rate | Aggregate loss rate |
| action_changes | Number of MBR vector changes |
| avg_decision_latency_ms | Average controller decision latency |
| avg_pfcp_apply_latency_ms | Average PFCP apply latency |
| oracle_gap | Distance from Oracle performance |

# Baseline B0

## Static MBR

### Objective

Keep a fixed MBR vector for the entire run.

This answers

```text
What happens when QER limits are static while the offered demand changes over time
```

### Microtasks

1. Define the fixed MBR vector.

2. Start TRex sinusoidal traffic for the three slices.

3. Do not update MBR during the run.

4. Collect TRex stats every window.

5. Collect BESS counters every window.

6. Generate summary metrics.

### Required output

```text
results/b0_static/telemetry_windows.csv
results/b0_static/actions.csv
results/b0_static/trex_stats.csv
results/b0_static/bess_counters.csv
results/b0_static/summary_metrics.csv
```

### Expected behavior

This baseline is expected to be simple and possibly poor under overload.

It is the lower bound of the experiment.

# Baseline B1

## Reactive threshold

### Objective

Use deterministic if then rules based on current telemetry.

This answers

```text
Can a simple reactive controller protect Gold without AI
```

### Example triggers

| Signal | Trigger |
|---|---|
| n6_utilization | greater than 0.90 |
| app_qer_red_bronze | greater than 0 |
| aggregate_red_packets | greater than 0 |
| gold_loss_rate | greater than 0.01 |

### Example policy

```text
If overload starts
reduce Bronze first

If overload continues for K windows
reduce Silver

If overload disappears for R windows
restore Bronze gradually

Reduce Gold only in emergency overload
```

### Microtasks

1. Implement threshold rules.

2. Read telemetry every control window.

3. Select an action from the common action space.

4. Apply the selected MBR vector with `pfcpctl session modify`.

5. Log the rule that triggered the action.

6. Collect TRex and BESS counters.

7. Generate summary metrics.

### Required output

```text
results/b1_threshold/telemetry_windows.csv
results/b1_threshold/actions.csv
results/b1_threshold/trex_stats.csv
results/b1_threshold/bess_counters.csv
results/b1_threshold/summary_metrics.csv
```

### Expected behavior

This baseline should protect Gold better than Static, but it may react late and oscillate.

# Baseline B2

## Raw KNN

### Objective

Use KNN over raw telemetry features.

This answers

```text
Can a simple memory based AI select useful MBR actions without a latent representation
```

### Input

One row per control window.

Use raw features such as

```text
offered throughput
received throughput
loss rate
QER red packet deltas
N6 utilization
current MBR vector
```

### Target

Use the Oracle action as the label.

```text
oracle_action_id
```

### Microtasks

1. Run several sinusoidal training scenarios.

2. Generate Oracle labels for each window.

3. Build `train_windows.csv`.

4. Choose raw features.

5. Train KNN classifier.

6. Save model as `models/raw_knn.joblib`.

7. Evaluate on held out scenarios.

8. Run online or replay control.

### Required output

```text
datasets/raw_knn/train_windows.csv
datasets/raw_knn/test_windows.csv
models/raw_knn.joblib
results/b2_raw_knn/actions.csv
results/b2_raw_knn/summary_metrics.csv
```

### Expected behavior

Raw KNN may improve over Static.

It may fail when phase or frequency changes, which motivates the latent representation.

# Baseline B3

## Raw ML

### Objective

Use a simple supervised model over raw features.

Possible models

```text
Random Forest
Gradient Boosting
MLP
```

This answers

```text
Does AI help if it only sees raw metrics
```

### Microtasks

1. Reuse the same training dataset from Raw KNN.

2. Train a classifier for `oracle_action_id`.

3. Save the model.

4. Evaluate offline accuracy.

5. Run replay or online evaluation.

6. Log actions and decision latency.

### Required output

```text
datasets/raw_ml/train_windows.csv
datasets/raw_ml/test_windows.csv
models/raw_ml.joblib
results/b3_raw_ml/actions.csv
results/b3_raw_ml/summary_metrics.csv
```

### Expected behavior

Raw ML should be a stronger non latent AI baseline.

It does not prove the value of latent state.

# Baseline B4

## Latent KNN

### Objective

Use KNN in a learned embedding space.

This is the key baseline for proving the value of latent state.

It answers

```text
Does a learned temporal representation improve KNN decisions
```

### Encoder

Recommended first encoder

```text
Predictive temporal encoder
```

Input

```text
last W telemetry windows
```

Training target

```text
future telemetry over the next H windows
```

Embedding

```text
z = encoder(last W windows)
```

KNN is then applied to `z`, not to raw features.

### Microtasks

1. Build sequence dataset from telemetry windows.

2. Train temporal encoder.

3. Save encoder as `models/latent_encoder.pt`.

4. Generate embeddings for all training windows.

5. Train KNN over embeddings.

6. Save model as `models/latent_knn.joblib`.

7. Evaluate on held out sinusoidal scenarios.

8. Compare against Raw KNN.

### Required output

```text
datasets/latent_knn/sequences_train.npz
datasets/latent_knn/sequences_test.npz
datasets/latent_knn/embeddings_train.csv
datasets/latent_knn/embeddings_test.csv
models/latent_encoder.pt
models/latent_knn.joblib
results/b4_latent_knn/actions.csv
results/b4_latent_knn/summary_metrics.csv
```

### Expected behavior

If Latent KNN outperforms Raw KNN, the experiment supports the claim that temporal latent state matters.

# Baseline B5

## Latent neural controller

### Objective

Use a temporal encoder and a policy head to choose the MBR action.

This is the main AI controller.

It answers

```text
Can a latent temporal controller preserve Gold better than static, threshold, and raw AI baselines
```

### Model

Recommended first design

```text
Transformer or LSTM encoder
MLP policy head
```

Input

```text
last W telemetry windows
```

Output

```text
action_id
```

Use action classification first because it is easier to audit than direct continuous MBR regression.

### Microtasks

1. Build sequence dataset.

2. Generate Oracle action labels.

3. Train the latent controller.

4. Validate on unseen phases and frequencies.

5. Save model as `models/latent_controller.pt`.

6. Run online or replay evaluation.

7. Apply actions through PFCP.

8. Log confidence and latency.

### Required output

```text
datasets/latent_controller/train_sequences.npz
datasets/latent_controller/test_sequences.npz
models/latent_controller.pt
results/b5_latent_controller/actions.csv
results/b5_latent_controller/telemetry_windows.csv
results/b5_latent_controller/summary_metrics.csv
```

### Expected behavior

The latent controller should reduce Gold SLA violations and reduce control oscillation compared with threshold and raw AI baselines.

# Baseline B6

## Oracle

### Objective

Compute the ideal offline action using known sinusoidal demand and N6 capacity.

The Oracle is not deployable.

It is an upper bound.

It answers

```text
What action would be selected if the controller knew the true slice demand and the bottleneck capacity
```

### Inputs

```text
Gold offered demand
Silver offered demand
Bronze offered demand
N6 capacity
priority rule
action space
```

### Deterministic rule

Use a public and reproducible rule.

Example

```text
1. Serve Gold up to its demand.
2. Serve Silver with the remaining capacity.
3. Serve Bronze with whatever capacity remains.
4. If aggregate demand is below capacity, do not restrict slices.
5. Reduce Gold only if capacity is below Gold demand.
```

### Microtasks

1. Read TRex offered load CSV.

2. Read N6 capacity.

3. For each window, compute aggregate demand.

4. Apply the deterministic priority rule.

5. Convert the ideal MBR vector into the closest action ID.

6. Save Oracle labels.

7. Use Oracle labels to train AI models.

8. Use Oracle performance as the upper bound.

### Required output

```text
datasets/oracle/oracle_actions.csv
datasets/oracle/oracle_mbr_vectors.csv
results/b6_oracle/summary_metrics.csv
```

### Required columns in oracle_actions.csv

```text
timestamp
scenario_id
window_index
gold_demand_mbps
silver_demand_mbps
bronze_demand_mbps
n6_capacity_mbps
oracle_action_id
oracle_mbr_gold_kbps
oracle_mbr_silver_kbps
oracle_mbr_bronze_kbps
oracle_reason
```

### Expected behavior

The Oracle should be fully auditable.

Anyone should be able to regenerate it from the TRex demand CSV, the N6 capacity, the priority rule, and the action space.

# Dataset generation

## Dataset D0

### Scenario catalog

Purpose

```text
List every sinusoidal scenario used in training and testing
```

Required output

```text
datasets/scenario_catalog.csv
```

Columns

```text
scenario_id
gold_base_mbps
gold_amplitude_mbps
gold_frequency
gold_phase
silver_base_mbps
silver_amplitude_mbps
silver_frequency
silver_phase
bronze_base_mbps
bronze_amplitude_mbps
bronze_frequency
bronze_phase
n6_capacity_mbps
duration_sec
window_sec
noise_level
split
```

## Dataset D1

### Sinusoidal demand log

Purpose

```text
Record the exact offered demand generated by TRex
```

Required output

```text
datasets/sinusoidal_demands.csv
```

Columns

```text
timestamp
scenario_id
gold_offered_mbps
silver_offered_mbps
bronze_offered_mbps
aggregate_offered_mbps
n6_capacity_mbps
overload
```

## Dataset D2

### Labeled control windows

Purpose

```text
Train supervised controllers
```

Required output

```text
datasets/control_windows_labeled.csv
```

Columns

```text
telemetry features
oracle_action_id
oracle MBR vector
train or test split
```

# Final comparison

Run every controller on the same held out scenarios.

Recommended final table columns

| Controller | Gold SLA violation | Gold throughput | Bronze degradation | N6 utilization | Total goodput | Action changes | Decision latency | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | | | | | | | | |
| Threshold | | | | | | | | |
| Raw KNN | | | | | | | | |
| Raw ML | | | | | | | | |
| Latent KNN | | | | | | | | |
| Latent controller | | | | | | | | |
| Oracle | | | | | | | | |

Expected interpretation

```text
Oracle is best.
Latent controller should be closest to Oracle.
Latent KNN should outperform Raw KNN.
Threshold should react but may be late or oscillatory.
Static should fail during overload or waste capacity.
```

# Recommended implementation order

1. Implement sinusoidal scenario generator.

2. Implement TRex sinusoidal traffic runner.

3. Implement BESS and TRex telemetry collector.

4. Implement B0 Static.

5. Implement B1 Threshold.

6. Implement B6 Oracle.

7. Build labeled dataset.

8. Implement B2 Raw KNN.

9. Implement B3 Raw ML.

10. Implement B4 Latent KNN.

11. Implement B5 Latent controller.

12. Run all baselines on held out scenarios.

13. Generate plots and final comparison table.

# Key claim

The experiment should support or reject this claim.

```text
A temporal latent representation of UPF telemetry improves slice aware MBR control under dynamic non synchronized traffic demand, preserving Gold service while shifting degradation toward lower priority slices.
```
