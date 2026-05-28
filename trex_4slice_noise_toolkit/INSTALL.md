# INSTALL — latent-space add-on

This add-on extends your existing `trex_4slice_noise_toolkit` with a
UNUM-style latent-space state representation and a single harness that runs
**baselines + pure DQN + latent DQN** under one code path.

It does NOT modify any of your existing files. It only adds four scripts and
one runner that import your current proxy / reward / PFCP machinery.

## 1. Copy the files into your toolkit

From the repo root `~/Agentic-QoS-Sensing-through-Latent-Space`:

```bash
cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit

# the four new python modules go NEXT TO your existing scripts
cp /path/to/addon/scripts/traffic_model_enriched.py  scripts/
cp /path/to/addon/scripts/unum_embedder.py           scripts/
cp /path/to/addon/scripts/pretrain_embedder.py       scripts/
cp /path/to/addon/scripts/run_proxy_experiment.py    scripts/

# the orchestrator + docs go at the toolkit root (next to setup_pfcpsim.sh)
cp /path/to/addon/run_all_experiments.sh  .
cp /path/to/addon/README_LATENT.md        .
chmod +x run_all_experiments.sh
```

The new scripts must live in the SAME `scripts/` directory as
`online_dqn_proxy_train.py`, `online_proxy_baseline.py`,
`run_trex_dynamic_4slice.py` and `traffic_model.py`, because they import from
them.

## 2. Dependencies

```bash
python3 -m pip install torch numpy
# (tensorboard optional — logging falls back to a no-op if absent)
```

## 3. Smoke test OFF the testbed (no TRex, no pfcpctl)

This validates the whole pipeline (pretraining + all four conditions) on a
laptop, using the proxy only:

```bash
NO_TREX=1 EXP=smoke DURATION=240 PRETRAIN_DURATION=600 \
PRETRAIN_SEEDS="0 1 2" PRETRAIN_EPOCHS=6 SEEDS="7" DEVICE=cpu EPS_DECAY=160 \
./run_all_experiments.sh
```

## 4. Real run ON the testbed

Make sure pfcpsim is up (your `setup_pfcpsim.sh`) and the TRex API path is set
(the runner exports a default; override with `TREX_PYTHONPATH=...` if needed),
then:

```bash
./run_all_experiments.sh
```

See `README_LATENT.md` for the full layout, knobs, and the scientific framing.
