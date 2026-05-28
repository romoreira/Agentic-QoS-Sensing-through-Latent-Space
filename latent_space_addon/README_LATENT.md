# Latent-space (UNUM-style) extension for the Gold-slice protection experiment

This add-on lets you compare four control conditions **on the same
environment, proxy, reward, action space and PFCP mechanism**, isolating the
one variable that matters: the **state representation**.

| Condition           | Folder              | Learns | State representation                         |
|---------------------|---------------------|:------:|----------------------------------------------|
| Fixed MBR           | `baseline_fixed/`   |   no   | none (static)                                |
| Reactive threshold  | `baseline_reactive/`|   no   | previous-step proxy scalars                  |
| **DQN pure**        | `dqn_pure/`         |  yes   | 11-dim instantaneous (offered + 1-step delta + MBR + pressure/shortage) |
| **DQN latent**      | `dqn_latent/`       |  yes   | frozen UNUM embedding (+ small anchors)      |

The latent representation is produced by a small **encoder-decoder Transformer**
pretrained offline with **windowed next-token prediction** over a discretized
(quantile-bucketized) stream of *observable* signals. The encoder output is the
embedding the DQN consumes; the decoder's future-bucket predictions drive an
optional **guardrail** that falls back to the reactive policy when predictions
diverge. This mirrors UNUM (Chen et al., NSDI'26): state estimation as a
separately-trained, predictive, first-class module.

## Why an enriched environment

`sine_gold_safe_overload` has only two hidden factors (binary noise on/off + a
fixed sine phase) — too easy to justify a learned latent estimator. The new
`traffic_model_enriched.py` adds latent structure with real temporal memory, so
a predictive embedding has something a 1-step delta cannot capture:

1. **3-state hidden noise** (quiet → ramp → burst) via a Markov chain with
   dwell times. Never a feature; only its proxy *consequence* (pressure,
   shortage) is observable.
2. **Slowly drifting uplink capacity** (bounded random walk, AR(1)). A
   path-property latent analogous to UNUM's wireless capacity changes. Invisible
   to the agent but threaded into the proxy, so it directly moves
   `available_for_gold` / `gold_shortage`.
3. **Long-period Gold demand envelope** (~600 s). Rewards a *longer* observation
   context (short windows can't tell a slow rise from a slow fall).

All four conditions see the identical enriched stream and the identical
time-varying capacity, so any difference is attributable to the representation.

## Run layout

```
runs/${EXP}/
  embedder/
    embedder.pt            # pretrained ONCE, shared by every latent seed (separation principle)
    pretrain_log.csv       # per-epoch train/val loss + within-1 bucket accuracy
  baseline_fixed/seed_007/    {metrics.csv, metrics.jsonl, episodes.csv, tensorboard/}
  baseline_reactive/seed_007/ ...
  dqn_pure/seed_007/          ...
  dqn_latent/seed_007/        ...
  (… seed_013, seed_021)
```

`metrics.csv` is a **superset** of your current schema (same columns plus
`policy, obs_mode, env, capacity_mbps, guardrail_active, embedder_divergence`),
so your existing analysis/plotting keeps working.

## How to run

Off-testbed smoke test (proxy only):

```bash
NO_TREX=1 EXP=smoke DURATION=240 PRETRAIN_DURATION=600 \
PRETRAIN_SEEDS="0 1 2" PRETRAIN_EPOCHS=6 SEEDS="7" DEVICE=cpu EPS_DECAY=160 \
./run_all_experiments.sh
```

Full run on the testbed (defaults: 1800 s, seeds 7/13/21, enriched env):

```bash
./run_all_experiments.sh
```

Useful overrides (env vars): `EXP`, `ENV_KIND` (enriched|original),
`ENRICHED_KIND` (latent_rich|latent_mild), `DURATION`, `DT`, `EPISODE_STEPS`,
`CAP`, `SEEDS`, `EPS_START/EPS_END/EPS_DECAY`, `GUARDRAIL` (0|1), `DEVICE`,
`FORCE_PRETRAIN=1`, and all `PRETRAIN_*`.

To reproduce your **original** experiment (no enrichment), set
`ENV_KIND=original TRAFFIC_MODEL=sine_gold_safe_overload`. Pure DQN there is
identical to `online_dqn_proxy_train.py` (the harness imports and reuses its
exact `make_obs`).

## Pretraining (the embedder)

`pretrain_embedder.py` runs offline (no TRex). It rolls out several
environments × seeds × collection policies (random / reactive / fixed),
quantile-bucketizes the observable tokens, and trains the Transformer to predict
the next `H` tokens from the past `W`. Pretraining seeds are **disjoint** from
the evaluation seeds (7/13/21) to avoid train/test leakage.

The embedder is trained **once per experiment** and loaded **frozen** by all
latent DQN seeds — it is a state estimator, not part of the controller.

## What to measure (suggested, you make the plots)

1. **Embedder quality first**, before any control claim: from
   `embedder/pretrain_log.csv`, the within-1 bucket accuracy; and, if you want,
   the embedder's ability to anticipate noise-state transitions / Gold peaks.
2. **Control**: per-episode `gold_shortage_sum` and `reward_mean` for the four
   folders; the headline is `dqn_latent` ≤ `dqn_pure` ≤ `baseline_reactive` ≤
   `baseline_fixed` for shortage, especially in later episodes and during burst
   + low-capacity coincidences.
3. **Guardrail**: `guardrail_active` / `embedder_divergence` columns in
   `dqn_latent` — how often the fallback fired and whether it cut tail
   degradations.

## Honest caveats

- The latent factors here are low-dimensional (≈ a 3-state HMM + slow capacity
  drift + a slow phase). A Transformer is sufficient but not strictly necessary;
  the *architectural* point (a separately-trained predictive state estimator
  feeding control) is what distinguishes this from a plain DQN. To stress the
  Transformer specifically, enrich further (more states, real traces).
- In pure proxy mode the shortage is a deterministic function of offered + MBR +
  capacity; the embedding helps only because the noise and the future are not in
  the agent's features. Make that explicit when you report.
- Keep the wall intact: the hidden noise and the latent capacity must never
  become embedder/agent features. The token set in `unum_embedder.OBS_FEATURES`
  enforces this.
