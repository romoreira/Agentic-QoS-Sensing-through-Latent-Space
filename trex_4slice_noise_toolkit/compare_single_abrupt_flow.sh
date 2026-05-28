#!/bin/bash
set -e

cd ~/Agentic-QoS-Sensing-through-Latent-Space/trex_4slice_noise_toolkit
export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH

EXP=dqn_proxy_gold_100_200_1800s
ROOT=runs/${EXP}
mkdir -p ${ROOT}/embedder

# (a) Pré-treinar o embedder UMA vez. Sem TRex, roda na própria máquina.
#     Seeds de pré-treino são disjuntos dos de avaliação (7/13/21) para não vazar.
python3 scripts/pretrain_embedder.py \
  --out ${ROOT}/embedder/embedder.pt \
  --log-csv ${ROOT}/embedder/pretrain_log.csv \
  --traffic-models sine_gold_safe_overload sine_overload latent_rich \
  --collect-policies random reactive fixed \
  --seeds 0 1 2 3 4 5 8 9 10 11 14 15 \
  --duration 1800 --dt 1 \
  --uplink-capacity-mbps 200 \
  --epochs 20 --device auto

# (b) Rodar SÓ o latent, nos seeds de avaliação, no MESMO env do seu DQN vanilla
for SEED in 7 13 21; do
  SEED_PAD=$(printf "%03d" "$SEED")
  mkdir -p ${ROOT}/dqn_latent_space/seed_${SEED_PAD}
  python3 scripts/run_proxy_experiment.py \
    --policy dqn --obs-mode latent \
    --env original --traffic-model sine_gold_safe_overload \
    --embedder ${ROOT}/embedder/embedder.pt \
    --guardrail \
    --duration 1800 --dt 1 --episode-steps 60 \
    --uplink-capacity-mbps 200 \
    --epsilon-start 0.35 --epsilon-end 0.03 --epsilon-decay-steps 1200 \
    --seed "$SEED" --force \
    --csv ${ROOT}/dqn_latent_space/seed_${SEED_PAD}/metrics.csv \
    --jsonl ${ROOT}/dqn_latent_space/seed_${SEED_PAD}/metrics.jsonl \
    --episodes-csv ${ROOT}/dqn_latent_space/seed_${SEED_PAD}/episodes.csv \
    --tb ${ROOT}/dqn_latent_space/seed_${SEED_PAD}/tensorboard
done
