# upf-rl-env

Gymnasium environment + DQN agent for slice-aware MBR control on a BESS-UPF.

The env wraps a metric source (mock simulator by default; TRex/BESS stubs for
real wiring) and an action applier (mock by default; real pfcpctl stub
provided). Designed for the **RL-raw vs RL-latent** comparison: same
algorithm, same hyperparameters, same action space — only the observation
representation changes.

## Quickstart

```bash
pip install -e .

# 1. Sanity check — random agent, prints stats, writes CSV
python scripts/sanity_check.py --steps 500 --verbose

# 2. Train DQN (Double + Dueling + PER) on the mock env
python scripts/train_dqn.py --total-steps 200000 --save-dir checkpoints/dqn_raw

# 3. Evaluate a checkpoint
python scripts/eval_dqn.py --ckpt checkpoints/dqn_raw/final.pt --episodes 5
```

Python 3.10+ recommended. CPU works; CUDA used automatically if available.

## Env ID

Registered as `upf-rl/SliceQoS-v0`. Use:

```python
import gymnasium as gym
import upf_rl_env  # noqa: F401  (registers the env)

env = gym.make("upf-rl/SliceQoS-v0")
```

## Observation, Action, Reward

- **Observation**: `Box(85,)` — 5 stacked windows × 17 features each, normalized.
  Feature order documented in `src/upf_rl_env/features.py`.
- **Action**: `Discrete(15)`. Per-slice MBR percentages from `action/space.py`.
  Action 0 = full MBR; higher indices cut bronze first, then silver, then gold.
- **Reward**: composite — weighted negative loss per slice + utilization bonus
  − action-change penalty − SLA-violation penalty. See `reward.py`.

## Slice mapping

| Slice  | TEID | Nominal MBR (kbps) |
|--------|-----:|-------------------:|
| Gold   |    1 |            200 000 |
| Silver |   11 |            100 000 |
| Bronze |   21 |             50 000 |

Change in `src/upf_rl_env/config.py` (`SLICE_TEIDS`, `NOMINAL_MBR_KBPS`).

## Layers

```
src/upf_rl_env/
  env.py              SliceQoSEnv — Gymnasium API
  features.py         metrics_to_features (17 dims/window)
  reward.py           compute_reward
  config.py           EnvConfig, DQNConfig, slice constants
  action/
    space.py          15-action grid (gold_pct, silver_pct, bronze_pct)
    applier.py        MockActionApplier (no-op) + RealActionApplier (pfcpctl stub)
  sources/
    base.py           MetricSource ABC + WindowMetrics dataclass
    mock.py           MockMetricSource — full simulator (default)
    trex.py           TrexMetricSource — stub
    bess.py           BessMetricSource — stub (bessctl / pfcpiface)
  monitor/
    logger.py         MetricsLogger — async CSV writer

src/upf_rl_agents/
  dqn.py              DQNAgent (Double + Dueling + PER)
  networks.py         DuelingMLP, PlainMLP
  replay.py           PrioritizedReplayBuffer (sum-tree)

scripts/
  sanity_check.py     Random agent → smoke test
  train_dqn.py        Train DQN
  eval_dqn.py         Eval a checkpoint
```

## What the mock simulates

Mock metric source is the most opinionated piece — it determines whether
RL has anything nontrivial to learn. The dynamics combine:

- **Per-slice offered load**: sum of sinusoids (3 freqs) + AR(1) noise +
  Poisson bursts with lognormal magnitude and exponential decay.
- **Regime variable** (4 regimes, switches every 8–25 s) that modulates
  per-slice means and amplitudes. *Hidden from the agent.*
- **Hidden N6 background traffic** (AR(1) around 40 Mbps).
- **N6 capacity** as a slow random walk (~300 Mbps nominal).
- **Enforcement**: stage A is MBR (drops counted as `qer_red`); stage B
  is N6 sharing with strict Gold > Silver > Bronze priority on the
  remaining headroom after background.

An agent that ignores temporal context can do well only in the all-quiet
regime. The hidden background + capacity drift is the signal a latent
encoder should pick up later.

## Plugging real TRex / BESS

The env constructor takes a `MetricSource` and an `ActionApplier`. To go
from mock to real:

```python
from upf_rl_env import SliceQoSEnv, EnvConfig
from upf_rl_env.action.applier import RealActionApplier
from upf_rl_env.sources.bess import BessMetricSource
from upf_rl_env.config import NOMINAL_MBR_KBPS, SLICE_TEIDS

env = SliceQoSEnv(
    config=EnvConfig(realtime=True),  # pace to 200 ms wall-clock
    metric_source=BessMetricSource(slice_teids=SLICE_TEIDS),
    action_applier=RealActionApplier(
        slice_teids=SLICE_TEIDS,
        nominal_mbr_kbps=NOMINAL_MBR_KBPS,
    ),
)
```

The stubs raise `NotImplementedError` until you fill in the TODOs — the
exact pfcpctl command, the bessctl module names, and the QER discovery
logic depend on which UPF-BESS flavor you deployed. See comments in
`sources/trex.py`, `sources/bess.py`, `action/applier.py`.

You will likely want a *compound* source that reads offered load from
TRex and post-enforcement metrics from BESS, joined by timestamp. Easy
to write as a small wrapper over `MetricSource` once both stubs work.

## Notes / known limitations

- Mock is fast (no wall-clock pacing). Real testbed should set
  `EnvConfig.realtime=True` so `step()` waits for the window to elapse.
- `truncated` flags do not produce a real terminal — `agent.observe()`
  passes `terminated` only (correct bootstrapping).
- PER uses a sum tree; `replay_capacity=200_000` × `obs_dim=85` keeps
  the buffer under ~70 MB of RAM.
- Logger writes one CSV per episode (suffix `_epNNNN.csv`).
