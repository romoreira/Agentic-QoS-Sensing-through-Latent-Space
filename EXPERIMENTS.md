# Slice-Aware MBR Control with Latent State Estimation (BESS UPF)

## Goal

Evaluate whether a **temporal latent representation** of UPF telemetry improves slice-aware MBR control under dynamic, partially-observable traffic demand, compared to static, reactive, and non-latent ML baselines.

The objective is to preserve the **Gold** slice under overload by reducing the MBR of lower-priority slices (Bronze first, Silver second), via PFCP/QER updates.

Correct name for this stage:

```text
Slice-aware MBR control
```

Do not call this stage "queue priority scheduling". No BESS queue priority is implemented; control happens through `pfcpctl session modify`.

## Key claim (testable)

```text
A temporal latent representation of UPF telemetry improves slice-aware MBR
control under dynamic, partially-observable demand — preserving Gold service
while shifting degradation toward lower priority slices — over and above
what static rules, reactive thresholds, and raw-feature ML baselines achieve.
```

Two falsifiability conditions must be checked before any AI training (see Pre-flight gates):

1. The problem must contain a **non-trivial latent factor** that the controller cannot observe directly. Without this, latent representations have nothing extra to encode.
2. The **Threshold (B1) → Oracle (B6) gap** must be large enough on held-out scenarios to leave room for AI improvement. If B1 already reaches ≥ 95% of Oracle reward, the scenario is too easy and must be redesigned.

---

## Pre-flight gates (run before B2–B5)

These cheap checks prevent wasting weeks on a setup that cannot prove the claim.

### Gate G1 — PFCP apply latency

Measure end-to-end latency of `pfcpctl session modify` over 100 isolated calls. Record:

```text
p50_pfcp_apply_latency_ms
p95_pfcp_apply_latency_ms
p99_pfcp_apply_latency_ms
```

Rule:

```text
If p95_pfcp_apply_latency_ms > 0.5 * window_sec_ms
  redesign actuation (e.g. async apply, larger window, batched updates)
  before continuing.
```

A slow actuation loop confounds the comparison — predictive controllers will appear to win *because they tolerate actuation lag*, not because of latent state. We need to separate those effects.

### Gate G2 — Threshold vs Oracle gap on the held-out test set

Run B0 (Static), B1 (Threshold), B6 (Oracle) on the held-out scenarios (defined in Dataset D0). Compute on the test split only:

```text
oracle_reward_test     = mean reward of B6 on test set
threshold_reward_test  = mean reward of B1 on test set
static_reward_test     = mean reward of B0 on test set

threshold_gap = (oracle_reward_test - threshold_reward_test) / oracle_reward_test
static_gap    = (oracle_reward_test - static_reward_test)    / oracle_reward_test
```

Rules:

```text
If threshold_gap < 0.05
  scenarios are too easy; increase latent factor strength (see Latent Factor Model)
  before training any ML baseline.

If static_gap < 0.10
  there is no overload; raise aggregate offered demand or lower N6 capacity.

If threshold_gap > 0.05 and static_gap > 0.10
  proceed to B2–B5.
```

### Gate G3 — Latent factor observability check

The latent factor model below introduces hidden state. Verify that **raw features cannot recover it well** by training a simple regressor (linear or shallow MLP) from raw features to the hidden state value. If R² > 0.8, the "latent" is not actually hidden and must be made stronger (more noise, more memory).

---

## Latent factor model (the core redesign)

A plain sinusoid trio is fully recoverable from one window of measurements; that is why naive baselines tie with AI in such setups. To create real room for latent representations, the environment must contain factors that are **not observable from per-window telemetry alone** and that **affect optimal control**.

We introduce three hidden mechanisms. None of them is exposed in any telemetry CSV; only their *effects* on `n6_tx_mbps`, loss, and delay leak through.

### L1 — Hidden background flow

A fourth, unmanaged UDP flow shares N6 with the three slices. It is **not** assigned a QER and is **not** present in offered-load CSVs.

```text
bg_rate(t) = bg_base + bg_amp * sin(2*pi*bg_freq*t + bg_phase)
           + AR1_noise(t, rho=0.85, sigma=bg_sigma)
```

The controller does not see `bg_rate(t)`. Recommended starting values:

```text
bg_base   = 100 Mbps
bg_amp    = 80 Mbps
bg_freq   uniform on [0.005, 0.05] Hz   (different from any slice freq)
bg_phase  uniform on [0, 2*pi]
bg_sigma  = 25 Mbps
```

### L2 — Time-varying effective N6 capacity

Effective N6 capacity drifts on a slow timescale (emulates cross-traffic on the egress path, link sharing, or competing tenants):

```text
cap(t) = cap_nominal * (1 + cap_drift_amp * sin(2*pi*cap_freq*t + cap_phase))
       + cap_jitter(t)
```

Recommended:

```text
cap_nominal     = 700 Mbps
cap_drift_amp   = 0.15           (i.e. +/- 15 percent slow drift)
cap_freq        uniform on [0.002, 0.02] Hz
cap_jitter      Gaussian, sigma = 20 Mbps, refreshed every 1 s
```

The controller never receives `cap(t)`. It only observes `n6_tx_mbps`, which is the *outcome* of `min(aggregate_demand(t) + bg_rate(t), cap(t))` minus drops.

### L3 — Correlated demand noise

Each slice's offered load is the sinusoidal base plus AR(1) noise:

```text
offered_slice(t) = base + amp * sin(2*pi*freq*t + phase) + AR1(rho=0.7, sigma=slice_sigma)
slice_sigma in {Gold: 20 Mbps, Silver: 30 Mbps, Bronze: 40 Mbps}
```

This breaks the property "current telemetry uniquely determines current demand", which is what makes raw-feature controllers win when absent.

### Why this matters

Under L1–L3, the **Oracle uses ground-truth `bg_rate(t)` and `cap(t)`** when computing the ideal action. The deployable controllers do not. A controller that only looks at the current window cannot distinguish "Bronze is consuming capacity" from "the background flow is consuming capacity" — but a controller with a temporal embedding can infer the regime from history. This is the analogue of UNUM's argument for congestion control.

---

## Experiment idea

Three non-synchronized sinusoidal slice demands plus the hidden factors L1–L3 are generated by TRex. The aggregate observable demand exceeds effective N6 capacity during overload windows.

Per-slice demand:

```text
rate_slice(t) = base + amp * sin(2*pi*freq*t + phase) + AR1_noise(t)
```

Baseline scenario (training distribution centers; jittered per scenario — see D0):

| Slice  | Base Mbps | Amplitude Mbps | Phase           | Expected behavior        |
|--------|----------:|---------------:|-----------------|--------------------------|
| Gold   | 300       | 100            | 0               | Preserve                 |
| Silver | 250       | 150            | pi / 3          | Degrade after Bronze     |
| Bronze | 250       | 200            | 2 * pi / 3      | Degrade first            |

```text
Visible aggregate peak (slices only):  ~900 Mbps
Hidden background peak:                ~180 Mbps
Aggregate true peak load:              ~1080 Mbps
Nominal N6 capacity:                   700 Mbps
Effective capacity range under L2:     ~580 to ~820 Mbps
```

---

## Action space

Discrete, but denser than a 5-action toy space, to avoid artificially flattening differences between controllers. 15 actions covering the priority simplex.

| Action ID | Gold Mbps | Silver Mbps | Bronze Mbps | Tier                  |
|----------:|----------:|------------:|------------:|-----------------------|
| 0         | 1000      | 1000        | 1000        | No restriction        |
| 1         | 1000      | 1000        | 700         | Bronze light          |
| 2         | 1000      | 1000        | 400         | Bronze moderate       |
| 3         | 1000      | 1000        | 200         | Bronze strong         |
| 4         | 1000      | 1000        | 80          | Bronze hard           |
| 5         | 1000      | 800         | 400         | Silver light + Bronze |
| 6         | 1000      | 800         | 150         | Silver light + Bronze hard |
| 7         | 1000      | 600         | 200         | Silver moderate       |
| 8         | 1000      | 600         | 80          | Silver moderate + Bronze hard |
| 9         | 1000      | 400         | 100         | Silver strong         |
| 10        | 1000      | 300         | 50          | Silver hard           |
| 11        | 900       | 500         | 100         | Gold light emergency  |
| 12        | 800       | 500         | 100         | Gold moderate emergency |
| 13        | 800       | 400         | 80          | Gold + Silver emergency |
| 14        | 700       | 300         | 50          | Full emergency        |

Gold is reduced only in actions 11–14 (emergency tier).

---

## Common runtime command

All runtime controllers apply MBR changes with PFCP. Example for action 7 (Gold 1000 / Silver 600 / Bronze 200, Mbps):

```bash
sudo docker exec pfcpsim pfcpctl -s localhost:12345 session modify \
  --count 3 --baseID 1 \
  --ue-pool 10.250.0.0/24 \
  --gnb-addr 192.168.70.2 \
  --app-filter "udp:any:any:allow:100" \
  --session-mbr-uplink   1000000,1000000,1000000 \
  --session-mbr-downlink 1000000,1000000,1000000 \
  --app-mbr-uplink       1000000,600000,200000 \
  --app-mbr-downlink     1000000,600000,200000
```

---

## Common telemetry

One telemetry row per control window.

```text
window_sec   = 0.200   (200 ms — fine enough to see dynamics, coarse enough for PFCP)
duration_sec = 600     (per run; longer for held-out runs)
```

Rationale: the original 1 s window kills temporal signal — sinusoidal periods of 20–200 s yield only a handful of samples per period. 200 ms gives ~3000 samples per run and ≥100 samples per period, which is what an encoder needs to learn dynamics.

If Gate G1 forces actuation slower than 200 ms, decouple:

- Telemetry collection: 200 ms (used for embedding input).
- Decision cadence: 1 s (downsampled, but informed by the high-rate embedding).

---

## Common CSV outputs

Every baseline writes:

```text
results/<baseline_name>/telemetry_windows.csv
results/<baseline_name>/actions.csv
results/<baseline_name>/trex_stats.csv
results/<baseline_name>/bess_counters.csv
results/<baseline_name>/summary_metrics.csv
```

Plus, **only for Oracle**, a ground-truth side file with hidden state used to compute the optimal action:

```text
results/b6_oracle/hidden_state.csv
```

This file must **not** be readable by any other baseline's controller code (enforce via separate directory and a simple lint check in the experiment runner).

### telemetry_windows.csv

| Column | Meaning |
|---|---|
| timestamp | Window timestamp |
| scenario_id | Scenario identifier |
| window_index | Control window index |
| gold_offered_mbps | TRex offered load for Gold (visible) |
| silver_offered_mbps | TRex offered load for Silver (visible) |
| bronze_offered_mbps | TRex offered load for Bronze (visible) |
| gold_rx_mbps | Delivered Gold throughput |
| silver_rx_mbps | Delivered Silver throughput |
| bronze_rx_mbps | Delivered Bronze throughput |
| gold_loss_rate | Gold packet loss ratio |
| silver_loss_rate | Silver packet loss ratio |
| bronze_loss_rate | Bronze packet loss ratio |
| n3_rx_mbps | UPF N3 received throughput |
| n6_tx_mbps | UPF N6 transmitted throughput |
| n6_utilization | n6_tx_mbps / cap_nominal (NOT effective cap; controller does not see L2) |
| app_qer_red_gold | Gold app QER red packets delta |
| app_qer_red_silver | Silver app QER red packets delta |
| app_qer_red_bronze | Bronze app QER red packets delta |
| session_qer_red_gold | Gold session QER red packets delta |
| session_qer_red_silver | Silver session QER red packets delta |
| session_qer_red_bronze | Bronze session QER red packets delta |
| current_mbr_gold_kbps | Current Gold MBR |
| current_mbr_silver_kbps | Current Silver MBR |
| current_mbr_bronze_kbps | Current Bronze MBR |
| rtt_avg_ms | Average RTT this window (if latency streams enabled) |
| residual_capacity_estimate_mbps | n6_tx_mbps - aggregate_offered_slices (proxy that leaks L1; do not feed as label) |

### actions.csv

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
| guardrail_triggered | bool, only for B5 |
| fallback_action_id | Action chosen by fallback, only if guardrail triggered |

### trex_stats.csv

| Column | Meaning |
|---|---|
| timestamp | Collection timestamp |
| stream_id | TRex stream identifier |
| slice | Gold, Silver, Bronze, or background |
| offered_mbps | Desired offered load (visible to controller for slices, hidden for background) |
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

### bess_counters.csv

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

### summary_metrics.csv

| Metric | Meaning |
|---|---|
| gold_sla_violation_rate | Fraction of windows where Gold violates SLA |
| gold_avg_throughput_mbps | Average delivered Gold throughput |
| gold_p95_loss_rate | p95 Gold loss rate |
| silver_avg_throughput_mbps | Average delivered Silver throughput |
| bronze_avg_throughput_mbps | Average delivered Bronze throughput |
| bronze_degradation_ratio | Bronze delivered / Bronze offered |
| n6_avg_utilization | Average N6 utilization (vs nominal cap) |
| n6_p95_utilization | p95 N6 utilization |
| total_goodput_mbps | Total delivered throughput |
| total_loss_rate | Aggregate loss rate |
| action_changes | Number of MBR vector changes |
| avg_decision_latency_ms | Average controller decision latency |
| avg_pfcp_apply_latency_ms | Average PFCP apply latency |
| oracle_gap | Distance from Oracle reward (test split only) |
| reward | Composite reward (see Reward function) |

### Reward function (composite, used to rank controllers)

```text
reward(window) =
      w_gold     * gold_rx_mbps
    + w_silver   * silver_rx_mbps
    + w_bronze   * bronze_rx_mbps
    - p_gold     * max(0, gold_offered_mbps - gold_rx_mbps)
    - p_silver   * max(0, silver_offered_mbps - silver_rx_mbps)
    - p_bronze   * max(0, bronze_offered_mbps - bronze_rx_mbps)
    - p_loss     * (gold_loss_rate + 0.5 * silver_loss_rate + 0.1 * bronze_loss_rate)
    - p_change   * action_changed_this_window
```

Default weights:

```text
w_gold = 3.0    w_silver = 1.5   w_bronze = 0.5
p_gold = 10.0   p_silver = 2.0   p_bronze = 0.2
p_loss = 50.0   p_change = 0.5
```

Gold penalty dominates — this encodes the SLA priority. `p_change` discourages oscillation.

---

# Baselines

## B0 — Static MBR

### Objective

Keep a fixed MBR vector for the entire run.

Answers:

```text
What happens when QER limits are static while offered demand and latent state change.
```

### Microtasks

1. Define the fixed MBR vector (use action 0).
2. Start TRex traffic (slices + hidden background).
3. Do not update MBR during the run.
4. Collect TRex and BESS telemetry every window.
5. Generate summary metrics.

### Required output

```text
results/b0_static/...
```

### Expected behavior

Lower bound. Will fail Gold SLA during overload.

---

## B1 — Reactive threshold

### Objective

Deterministic if-then rules over current telemetry.

Answers:

```text
Can a hand-tuned reactive controller protect Gold without modeling latent state.
```

### Example triggers

| Signal | Trigger |
|---|---|
| n6_utilization | > 0.90 |
| app_qer_red_bronze | > 0 |
| aggregate_red_packets | > 0 |
| gold_loss_rate | > 0.01 |

### Example policy

```text
If overload starts          -> reduce Bronze first
If overload persists K wins -> reduce Silver
If overload clears R wins   -> restore Bronze gradually, then Silver
Reduce Gold only in emergency (gold_loss_rate > threshold for K_e windows)
```

Tune K, R, K_e on the **training split only**. Report on the test split.

### Microtasks

1. Implement threshold rules.
2. Read telemetry every control window.
3. Select action from common action space.
4. Apply MBR vector via `pfcpctl session modify`.
5. Log the rule that triggered the action.
6. Generate summary metrics.

### Required output

```text
results/b1_threshold/...
```

### Expected behavior

Better than Static. May react late (no anticipation) and oscillate. **Used in Gate G2.**

---

## B2 — Raw KNN

### Objective

KNN over raw telemetry features.

Answers:

```text
Can a memory-based AI select useful MBR actions without a latent representation.
```

### Input features (per window)

```text
gold_offered_mbps, silver_offered_mbps, bronze_offered_mbps
gold_rx_mbps, silver_rx_mbps, bronze_rx_mbps
gold_loss_rate, silver_loss_rate, bronze_loss_rate
n6_tx_mbps, n6_utilization
app_qer_red_gold, app_qer_red_silver, app_qer_red_bronze
current_mbr_gold_kbps, current_mbr_silver_kbps, current_mbr_bronze_kbps
```

### Target

```text
oracle_action_id
```

### Microtasks

1. Reuse training scenarios from D0 (train split only).
2. Generate Oracle labels for each window.
3. Build `datasets/raw_knn/train_windows.csv`.
4. Train KNN classifier (sweep k in {1, 3, 5, 9, 15}; scale features with StandardScaler).
5. Save model.
6. Evaluate on the held-out test split (D0 test split — including OOD scenarios).
7. Run online or replay control.

### Required output

```text
datasets/raw_knn/train_windows.csv
datasets/raw_knn/test_windows.csv
models/raw_knn.joblib
results/b2_raw_knn/...
```

### Expected behavior

Improves over Static and possibly Threshold on in-distribution scenarios. Should **degrade on OOD test split** (new phases/frequencies) — that degradation is the evidence motivating B4.

---

## B3 — Raw ML

### Objective

Supervised classifier over raw features.

Models to sweep:

```text
Random Forest      (n_estimators in {100, 300})
Gradient Boosting  (n_estimators in {100, 300}, max_depth in {3, 6})
MLP                (hidden in {(64,), (128, 64), (256, 128)})
```

Select best on the validation split.

### Microtasks

1. Reuse Raw KNN dataset.
2. Train classifier for `oracle_action_id`.
3. Save model.
4. Evaluate offline accuracy on test split (including OOD).
5. Run replay or online evaluation.
6. Log actions and decision latency.

### Required output

```text
datasets/raw_ml/train_windows.csv
datasets/raw_ml/test_windows.csv
models/raw_ml.joblib
results/b3_raw_ml/...
```

### Expected behavior

Stronger non-latent AI baseline. Should also degrade on OOD test split. **B3 is the controller B4/B5 must beat to claim latent value.**

---

## B4 — Latent KNN

### Objective

KNN in a learned embedding space.

Answers:

```text
Does a learned temporal representation improve KNN decisions over raw features.
```

### Encoder

Predictive temporal encoder.

```text
Input:    last W = 32 telemetry windows  (~6.4 s of context at 200 ms)
Target:   future telemetry over the next H = 8 windows
Loss:     bucketized classification, one head per feature (UNUM-style)
Bucketization: Quantile-50 per feature, fit on training split only
Architecture: 4-layer Transformer encoder, embedding dim 32, 4 heads
              OR 2-layer LSTM with hidden 64 (fallback for compute)
Embedding: z = encoder(last W windows), dim 32
```

KNN is applied to z, not to raw features.

### Microtasks

1. Build sequence dataset from telemetry windows (training split only).
2. Fit bucket boundaries on training split; freeze.
3. Train temporal encoder via windowed next-token prediction.
4. Save encoder as `models/latent_encoder.pt`.
5. Generate embeddings for all training and test windows.
6. Train KNN over training embeddings (sweep k).
7. Evaluate on held-out test split (including OOD).
8. Compare against B2.

### Required output

```text
datasets/latent_knn/sequences_train.npz
datasets/latent_knn/sequences_test.npz
datasets/latent_knn/embeddings_train.csv
datasets/latent_knn/embeddings_test.csv
models/latent_encoder.pt
models/latent_knn.joblib
results/b4_latent_knn/...
```

### Expected behavior

If B4 > B2 on OOD test split → temporal latent state matters for this control problem.

If B4 ≈ B2 → the latent factor L1–L3 is too weak, or W is too short, or the encoder undertrained. Retrain G3 first.

---

## B5 — Latent neural controller

### Objective

Temporal encoder + policy head choosing the MBR action end-to-end.

Answers:

```text
Can a latent temporal controller preserve Gold better than static, threshold,
and raw AI baselines, especially on OOD scenarios.
```

### Model

```text
Encoder:    same architecture as B4 (Transformer 4-layer or LSTM 2-layer)
            Initialize from B4 encoder weights, then fine-tune end-to-end.
Policy:     MLP head, 2 layers, hidden 128, outputs softmax over 15 actions.
Input:      last W = 32 telemetry windows + current MBR vector
Output:     action_id (classification)
Training:   supervised on oracle_action_id (training split)
            Cross-entropy loss; AdamW; lr 1e-4; early stop on val reward.
```

Use action classification first (auditable). Regression onto continuous MBR is a follow-up if classification saturates.

### Guardrail (mandatory)

Run a decoder head in parallel that predicts future telemetry from z. Track:

```text
pred_error_w = avg bucket index distance between predicted and observed
               over the last G windows (G = 10)
```

Rule:

```text
If pred_error_w > tau for more than G windows
  guardrail_triggered = true
  fall back to B1 (Threshold) for the next F windows (F = 20)
  log fallback_action_id
```

Tune tau on the training split to give a false-trigger rate < 5% on in-distribution data.

Justification: UNUM reports +6 pp reward from guardrails because latent models fail under distribution shift. Same risk here, especially on OOD test scenarios. **B5 without a guardrail is not a defensible system.**

### Microtasks

1. Build sequence dataset.
2. Generate Oracle labels.
3. Pre-train encoder via B4's objective; freeze briefly, then fine-tune.
4. Train policy head + fine-tune encoder.
5. Train decoder head for guardrail predictions.
6. Validate on unseen phases and frequencies (OOD set).
7. Save models.
8. Run online or replay evaluation with guardrail enabled.
9. Apply actions through PFCP.
10. Log confidence, decision latency, and guardrail events.

### Required output

```text
datasets/latent_controller/train_sequences.npz
datasets/latent_controller/test_sequences.npz
models/latent_controller.pt
models/latent_decoder.pt
results/b5_latent_controller/...
```

### Expected behavior

B5 reduces Gold SLA violations and reduces action_changes vs B1/B2/B3. Improvement is **largest on OOD test split** — this is the signature of latent value.

---

## B6 — Oracle

### Objective

Compute the ideal offline action using **ground-truth slice demand, hidden background, and effective N6 capacity**. Not deployable. Upper bound.

### Inputs (uses hidden state — controller-side baselines cannot)

```text
gold_offered_mbps(t)        (visible)
silver_offered_mbps(t)      (visible)
bronze_offered_mbps(t)      (visible)
bg_rate(t)                  (hidden — from L1 generator)
cap(t)                      (hidden — from L2 generator)
priority rule
action space
```

### Deterministic rule

```text
Available for slices = max(0, cap(t) - bg_rate(t))

1. Serve Gold up to gold_offered_mbps, but at most Available.
2. Remaining = Available - served_gold
3. Serve Silver up to silver_offered_mbps, but at most Remaining.
4. Remaining = Remaining - served_silver
5. Serve Bronze up to bronze_offered_mbps, but at most Remaining.
6. Map ideal (served_gold, served_silver, served_bronze) to the closest
   action in the discrete action space (L1 distance in MBR-vector space).
7. Record oracle_action_id and oracle_reason.
```

### Microtasks

1. Read slice offered-load CSV.
2. Read hidden bg_rate(t) and cap(t) trajectories from the scenario generator.
3. For each window, compute Available and ideal allocation.
4. Snap to closest action.
5. Save oracle labels and ideal-MBR trajectory.
6. Use labels for B2/B3/B4/B5 training (training split only).
7. Use Oracle test-split reward as the upper bound.

### Required output

```text
datasets/oracle/oracle_actions.csv
datasets/oracle/oracle_mbr_vectors.csv
results/b6_oracle/summary_metrics.csv
results/b6_oracle/hidden_state.csv
```

### Required columns in oracle_actions.csv

```text
timestamp
scenario_id
window_index
gold_demand_mbps
silver_demand_mbps
bronze_demand_mbps
bg_rate_mbps                (hidden)
effective_capacity_mbps     (hidden)
oracle_action_id
oracle_mbr_gold_kbps
oracle_mbr_silver_kbps
oracle_mbr_bronze_kbps
oracle_reason
```

### Expected behavior

Fully auditable. Anyone can regenerate it from generator seeds + scenario catalog.

---

# Dataset generation

## D0 — Scenario catalog (with explicit OOD holdout)

Purpose: list every sinusoidal scenario used in training, validation, and testing.

Required output:

```text
datasets/scenario_catalog.csv
```

### Sampling distributions

```text
Training distribution (used to draw 60 scenarios):
  gold_base       in [250, 350] Mbps
  gold_amplitude  in [80, 120]  Mbps
  gold_freq       in [0.01, 0.05] Hz
  gold_phase      in [0, 2*pi]
  (silver, bronze: analogous, see below)
  bg_freq         in [0.005, 0.05] Hz
  cap_freq        in [0.002, 0.02] Hz
  noise_level     in {low, medium}

Validation distribution (10 scenarios):
  same ranges, different random seeds.

Test-IND distribution (10 scenarios, in-distribution test):
  same ranges, different random seeds.

Test-OOD distribution (10 scenarios, out-of-distribution holdout):
  gold_freq       in [0.06, 0.10] Hz   (OUTSIDE training range)
  silver_freq     in [0.06, 0.10] Hz   (OUTSIDE training range)
  bronze_freq     in [0.06, 0.10] Hz   (OUTSIDE training range)
  noise_level     = high                (OUTSIDE training range)
  bg_amp          = 1.5 * training_max (OUTSIDE training range)
```

The OOD split is the **only honest test** for the key claim. Report results separately on Test-IND and Test-OOD.

### Columns

```text
scenario_id
gold_base_mbps, gold_amplitude_mbps, gold_frequency, gold_phase
silver_base_mbps, silver_amplitude_mbps, silver_frequency, silver_phase
bronze_base_mbps, bronze_amplitude_mbps, bronze_frequency, bronze_phase
bg_base_mbps, bg_amplitude_mbps, bg_frequency, bg_phase, bg_sigma
cap_nominal_mbps, cap_drift_amp, cap_frequency, cap_phase, cap_jitter_sigma
n6_capacity_mbps    (= cap_nominal, controller-visible nominal only)
duration_sec
window_sec
noise_level
split               (train | val | test_ind | test_ood)
seed                (RNG seed; full reproducibility)
```

## D1 — Sinusoidal demand log

Purpose: record exact offered demand (slices visible; hidden factors in a separate file).

Required output:

```text
datasets/sinusoidal_demands.csv         (visible to all controllers)
datasets/hidden_demands.csv             (Oracle-only; gitignore for controllers)
```

`sinusoidal_demands.csv` columns:

```text
timestamp
scenario_id
gold_offered_mbps
silver_offered_mbps
bronze_offered_mbps
aggregate_offered_mbps_slices
n6_capacity_mbps                (nominal, not effective)
overload_visible                (aggregate_offered_mbps_slices > n6_capacity_mbps)
```

`hidden_demands.csv` columns:

```text
timestamp
scenario_id
bg_rate_mbps
effective_capacity_mbps
true_overload                   (aggregate + bg > effective_capacity)
```

## D2 — Labeled control windows

Purpose: train supervised controllers.

Required output:

```text
datasets/control_windows_labeled.csv
```

Columns:

```text
all telemetry_windows.csv columns
oracle_action_id
oracle_mbr_gold_kbps, oracle_mbr_silver_kbps, oracle_mbr_bronze_kbps
split   (train | val | test_ind | test_ood)
```

Strict rule: when training any model, **filter to split == 'train'**. The runner must assert this.

---

# Final comparison

Run every controller on **both** Test-IND and Test-OOD. Two tables — same columns; the OOD one is the headline result.

| Controller | Gold SLA viol. | Gold thrpt | Bronze degrad. | N6 util. | Total goodput | Action changes | Decision lat. | Reward | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 Static            |  |  |  |  |  |  |  |  |  |
| B1 Threshold         |  |  |  |  |  |  |  |  |  |
| B2 Raw KNN           |  |  |  |  |  |  |  |  |  |
| B3 Raw ML            |  |  |  |  |  |  |  |  |  |
| B4 Latent KNN        |  |  |  |  |  |  |  |  |  |
| B5 Latent controller |  |  |  |  |  |  |  |  |  |
| B5 + Guardrail       |  |  |  |  |  |  |  |  |  |
| B6 Oracle            |  |  |  |  |  |  |  |  |  |

Repeat each controller × scenario combination at least **5 times** with different RNG seeds for the AR(1) noise. Report mean ± 95% CI.

### Interpretation rules (decide before running)

```text
Claim supported if ALL of the following hold on Test-OOD:

  reward(B5+Guardrail) > reward(B3) + 1 CI  (latent beats best raw ML)
  reward(B4)           > reward(B2) + 1 CI  (latent embedding beats raw KNN)
  gold_sla_violation_rate(B5+Guardrail) < gold_sla_violation_rate(B1)
  action_changes(B5+Guardrail) < action_changes(B1)  (less oscillation)
  oracle_gap(B5+Guardrail) < oracle_gap(B3)

Claim partially supported if:
  reward(B5+Guardrail) > reward(B3) on Test-OOD but not on Test-IND
  (latent value shows up under distribution shift only — still a valid result)

Claim rejected if:
  reward(B5+Guardrail) <= reward(B3) on both splits
  In this case, report the negative result and analyze why.
```

A negative result is publishable. A negative result hidden by a weak experimental setup is not.

---

# Recommended implementation order

1. Implement sinusoidal scenario generator **with L1, L2, L3 hidden factors**.
2. Implement TRex traffic runner (slices + hidden background flow).
3. Implement BESS and TRex telemetry collector (200 ms window).
4. **Run Gate G1 (PFCP latency).** Stop and fix if it fails.
5. Implement B0 Static.
6. Implement B1 Threshold.
7. Implement B6 Oracle.
8. Generate D0 catalog (train / val / test_ind / test_ood).
9. Build D1 (visible + hidden demand logs) and D2 (labeled windows).
10. **Run Gate G2 (B0 / B1 / B6 on test set).** Stop and redesign L1–L3 if it fails.
11. **Run Gate G3 (latent observability).** Stop and strengthen latent if R² > 0.8.
12. Implement B2 Raw KNN.
13. Implement B3 Raw ML.
14. Implement B4 Latent KNN.
15. Implement B5 Latent controller + Guardrail.
16. Run all baselines on Test-IND and Test-OOD (≥ 5 seeds each).
17. Generate plots and final comparison tables.
18. Apply the Interpretation rules. Write up the result honestly, whichever direction it goes.

---

# Key claim (restated for clarity)

```text
A temporal latent representation of UPF telemetry improves slice-aware
MBR control under dynamic, partially-observable demand, preserving Gold
service while shifting degradation toward lower priority slices, and
this improvement is largest on out-of-distribution scenarios — which
is the regime where latent state estimation should help most.
```
