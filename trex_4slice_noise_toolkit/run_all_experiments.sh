#!/usr/bin/env bash
# =============================================================================
# Run the full comparison: baselines + vanilla DQN + latent-space DQN.
#
# Conditions (all on the SAME environment, proxy, reward, action space):
#   baseline_fixed     : static MBR                        (no learning)
#   baseline_reactive  : rule-based threshold              (no learning)
#   dqn_vanilla        : DQN, 11-dim instantaneous state   (--obs-mode raw)
#   dqn_latent_space   : DQN over a frozen UNUM embedding  (--obs-mode latent)
#
# Run layout (per experiment, per seed):
#   runs/${EXP}/
#     embedder/embedder.pt                <- pretrained ONCE, shared by all latent seeds
#     embedder/pretrain_log.csv
#     baseline_fixed/seed_XXX/{metrics.csv,metrics.jsonl,episodes.csv,tensorboard/}
#     baseline_reactive/seed_XXX/...
#     dqn_vanilla/seed_XXX/...
#     dqn_latent_space/seed_XXX/...
#
# OFF-TESTBED SMOKE TEST:
#   NO_TREX=1 DURATION=180 PRETRAIN_DURATION=600 SEEDS="7" ./run_all_experiments.sh
# ON-TESTBED FULL RUN:
#   ./run_all_experiments.sh
# =============================================================================
set -euo pipefail

# ---- TRex Python API (ignored when NO_TREX=1) -------------------------------
export PYTHONPATH="${TREX_PYTHONPATH:-/opt/trex/v3.08/automation/trex_control_plane/interactive}:${PYTHONPATH:-}"

# ---- knobs (override via env vars) ------------------------------------------
EXP="${EXP:-dqn_proxy_enriched_1800s}"
ENV_KIND="${ENV_KIND:-enriched}"            # enriched | original
ENRICHED_KIND="${ENRICHED_KIND:-latent_rich}"
TRAFFIC_MODEL="${TRAFFIC_MODEL:-sine_gold_safe_overload}"  # only used if ENV_KIND=original

DURATION="${DURATION:-1800}"
DT="${DT:-1}"
EPISODE_STEPS="${EPISODE_STEPS:-60}"
CAP="${CAP:-200}"                            # fixed cap (original) / fallback (enriched)

EPS_START="${EPS_START:-0.35}"
EPS_END="${EPS_END:-0.03}"
EPS_DECAY="${EPS_DECAY:-1200}"

SEEDS="${SEEDS:-7 13 21}"                    # evaluation seeds
DEVICE="${DEVICE:-auto}"

# Pretraining (embedder). Seeds DISJOINT from eval seeds to avoid leakage.
PRETRAIN_SEEDS="${PRETRAIN_SEEDS:-0 1 2 3 4 5 8 9 10 11 14 15}"
PRETRAIN_MODELS="${PRETRAIN_MODELS:-latent_rich sine_gold_safe_overload sine_overload}"
PRETRAIN_POLICIES="${PRETRAIN_POLICIES:-random reactive fixed}"
PRETRAIN_DURATION="${PRETRAIN_DURATION:-1800}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
FORCE_PRETRAIN="${FORCE_PRETRAIN:-0}"

GUARDRAIL="${GUARDRAIL:-1}"                   # 1 = enable guardrail on latent DQN

# ---- TRex toggle ------------------------------------------------------------
TREX_FLAGS=""
if [[ "${NO_TREX:-0}" == "1" ]]; then
  TREX_FLAGS="--no-trex --no-apply"
  echo "[run_all] NO_TREX=1 -> proxy-only simulation (no TRex, no pfcpctl)"
fi

SCRIPTS="$(cd "$(dirname "$0")" && pwd)/scripts"
[[ -d "$SCRIPTS" ]] || SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
ROOT="runs/${EXP}"
EMB_DIR="${ROOT}/embedder"
EMB="${EMB_DIR}/embedder.pt"

mkdir -p "${EMB_DIR}"

echo "=============================================================="
echo " EXP=${EXP}  ENV=${ENV_KIND}(${ENRICHED_KIND})  DURATION=${DURATION}s  SEEDS=[${SEEDS}]"
echo " scripts: ${SCRIPTS}"
echo "=============================================================="

# ---- env-dependent flags ----------------------------------------------------
if [[ "${ENV_KIND}" == "enriched" ]]; then
  ENV_FLAGS="--env enriched --enriched-kind ${ENRICHED_KIND}"
else
  ENV_FLAGS="--env original --traffic-model ${TRAFFIC_MODEL}"
fi

# =============================================================================
# Step 0: pretrain the embedder ONCE (shared by every latent seed)
# =============================================================================
if [[ "${FORCE_PRETRAIN}" == "1" || ! -f "${EMB}" ]]; then
  echo "[pretrain] building embedder -> ${EMB}"
  python3 -u "${SCRIPTS}/pretrain_embedder.py" \
    --out "${EMB}" \
    --log-csv "${EMB_DIR}/pretrain_log.csv" \
    --traffic-models ${PRETRAIN_MODELS} \
    --collect-policies ${PRETRAIN_POLICIES} \
    --seeds ${PRETRAIN_SEEDS} \
    --duration "${PRETRAIN_DURATION}" --dt "${DT}" \
    --uplink-capacity-mbps "${CAP}" \
    --epochs "${PRETRAIN_EPOCHS}" \
    --device "${DEVICE}"
else
  echo "[pretrain] reusing existing embedder ${EMB} (set FORCE_PRETRAIN=1 to rebuild)"
fi

# =============================================================================
# Step 1: run all conditions for each seed
# =============================================================================
run_one () {
  local subdir="$1"; shift
  local out="${ROOT}/${subdir}/seed_${SEED_PAD}"
  mkdir -p "${out}"
  python3 -u "${SCRIPTS}/run_proxy_experiment.py" \
    ${ENV_FLAGS} ${TREX_FLAGS} \
    --duration "${DURATION}" --dt "${DT}" --episode-steps "${EPISODE_STEPS}" \
    --uplink-capacity-mbps "${CAP}" \
    --seed "${SEED}" --device "${DEVICE}" --force \
    --csv "${out}/metrics.csv" \
    --jsonl "${out}/metrics.jsonl" \
    --episodes-csv "${out}/episodes.csv" \
    --tb "${out}/tensorboard" \
    "$@"
}

for SEED in ${SEEDS}; do
  SEED_PAD=$(printf "%03d" "${SEED}")
  echo "-------------------------- seed ${SEED_PAD} --------------------------"

  echo "[${SEED_PAD}] baseline_fixed"
  run_one "baseline_fixed"    --policy fixed_mbr

  echo "[${SEED_PAD}] baseline_reactive"
  run_one "baseline_reactive" --policy reactive_threshold

  echo "[${SEED_PAD}] dqn_vanilla (raw)"
  run_one "dqn_vanilla"       --policy dqn --obs-mode raw \
                              --epsilon-start "${EPS_START}" --epsilon-end "${EPS_END}" \
                              --epsilon-decay-steps "${EPS_DECAY}"

  echo "[${SEED_PAD}] dqn_latent_space (latent)"
  GR_FLAG=""
  [[ "${GUARDRAIL}" == "1" ]] && GR_FLAG="--guardrail"
  run_one "dqn_latent_space"  --policy dqn --obs-mode latent --embedder "${EMB}" ${GR_FLAG} \
                              --epsilon-start "${EPS_START}" --epsilon-end "${EPS_END}" \
                              --epsilon-decay-steps "${EPS_DECAY}"
done

echo "=============================================================="
echo " DONE. Runs under: ${ROOT}/"
echo "   baseline_fixed/  baseline_reactive/  dqn_vanilla/  dqn_latent_space/"
echo " Embedder: ${EMB}"
echo "=============================================================="
