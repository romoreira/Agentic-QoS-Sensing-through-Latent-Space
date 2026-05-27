#!/usr/bin/env python3
"""
analyze_action_features.py
==========================

Pergunta que este script responde:
    "Quando o DQN bruto decide reduzir Gold (A12/A13/A14), o que ele esta
     vendo no estado de entrada que e diferente das janelas em que ele
     escolhe A00?"

Por que isso importa:
    Se ha features com diferenca grande e interpretavel entre os dois
    grupos (ex.: n6_utilization, offered_silver_mbps, noise_offered_mbps),
    o agente aprendeu algo. Se nao ha diferenca sistematica nenhuma, o
    agente esta praticamente sorteando, e o checkpoint precisa ser
    investigado antes de qualquer atuacao real.

Como usar:
    python3 analyze_action_features.py \
        --csv runs/online_dqn_dryrun_bronze12_hier_420s/online_dqn_dryrun.csv

Saida:
    - Tabela de features ordenada por Cohen's d (effect size).
    - Classificacao automatica: NEGLIGIVEL / PEQUENO / MEDIO / GRANDE.
    - Diagnostico final em texto plano com sugestao de proxima acao.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Features que vale a pena auditar. Sao as colunas do CSV que sao
# matematicamente derivadas em metrics_to_features() ou diretamente
# correlacionadas a entrada do DQN.
FEATURES_TO_AUDIT = [
    "offered_gold_mbps",
    "offered_silver_mbps",
    "offered_bronze_mbps",
    "noise_offered_mbps",
    "rx_gold_mbps",
    "rx_silver_mbps",
    "rx_bronze_mbps",
    "loss_gold",
    "loss_silver",
    "loss_bronze",
    "n6_total_mbps",
    "n6_utilization",
]

# raw_action == 0 e o grupo de referencia (A00 = nao intervir).
# raw_action em {12, 13, 14} sao as acoes que reduzem Gold.
RED_GOLD_ACTIONS = {12, 13, 14}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d para duas amostras independentes (pooled std)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mean_diff = a.mean() - b.mean()
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    n_a, n_b = len(a), len(b)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else float("nan")
    if not pooled_std or math.isnan(pooled_std) or pooled_std == 0.0:
        return float("nan")
    return mean_diff / pooled_std


def magnitude_label(d: float) -> str:
    if math.isnan(d):
        return "INDEFINIDO"
    ad = abs(d)
    if ad < 0.2:
        return "NEGLIGIVEL"
    if ad < 0.5:
        return "PEQUENO"
    if ad < 0.8:
        return "MEDIO"
    return "GRANDE"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audita features que o DQN ve quando decide reduzir Gold.")
    ap.add_argument("--csv", required=True, type=Path, help="CSV produzido por online_dqn_dryrun.py")
    ap.add_argument(
        "--include-noise-loss",
        action="store_true",
        help="Inclui noise_loss e noise_rx_mbps no audit (sao colunas opcionais).",
    )
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[ERRO] CSV nao encontrado: {args.csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    needed = {"dryrun_raw_action_id"} | set(FEATURES_TO_AUDIT)
    missing = needed - set(df.columns)
    if missing:
        print(f"[ERRO] CSV nao tem as colunas: {sorted(missing)}", file=sys.stderr)
        return 2

    if args.include_noise_loss and "noise_loss" in df.columns:
        FEATURES_TO_AUDIT.append("noise_loss")
    if args.include_noise_loss and "noise_rx_mbps" in df.columns:
        FEATURES_TO_AUDIT.append("noise_rx_mbps")

    mask_a00 = df["dryrun_raw_action_id"] == 0
    mask_red = df["dryrun_raw_action_id"].isin(RED_GOLD_ACTIONS)

    n_a00 = int(mask_a00.sum())
    n_red = int(mask_red.sum())
    n_total = len(df)
    n_other = n_total - n_a00 - n_red

    print("=" * 78)
    print("ANALISE: o que o DQN bruto ve quando pede reducao de Gold?")
    print("=" * 78)
    print(f"CSV: {args.csv}")
    print(f"Linhas totais: {n_total}")
    print(f"  raw_action == A00 (nao intervir)      : {n_a00:4d}")
    print(f"  raw_action em A12/A13/A14 (reduz Gold): {n_red:4d}")
    print(f"  raw_action em outras                  : {n_other:4d}")
    print()

    if n_red < 5:
        print("[AVISO] Poucas amostras de reducao de Gold. Effect sizes serao instaveis.")
        print()

    rows = []
    for feat in FEATURES_TO_AUDIT:
        a00_vals = df.loc[mask_a00, feat].dropna().to_numpy()
        red_vals = df.loc[mask_red, feat].dropna().to_numpy()
        if len(a00_vals) == 0 or len(red_vals) == 0:
            continue
        d = cohens_d(red_vals, a00_vals)  # positivo = maior em RED do que em A00
        rows.append({
            "feature": feat,
            "mean_A00": float(a00_vals.mean()),
            "std_A00": float(a00_vals.std(ddof=1)) if len(a00_vals) > 1 else float("nan"),
            "mean_RED": float(red_vals.mean()),
            "std_RED": float(red_vals.std(ddof=1)) if len(red_vals) > 1 else float("nan"),
            "delta": float(red_vals.mean() - a00_vals.mean()),
            "cohens_d": d,
            "magnitude": magnitude_label(d),
        })

    out = pd.DataFrame(rows)
    out["abs_d"] = out["cohens_d"].abs()
    out = out.sort_values("abs_d", ascending=False).drop(columns=["abs_d"])

    print("TABELA: features ordenadas por |Cohen's d| (efeito maior em cima)")
    print("-" * 78)
    hdr = f"{'feature':28s} {'mean_A00':>10s} {'mean_RED':>10s} {'delta':>10s} {'d':>7s}  magnitude"
    print(hdr)
    print("-" * 78)
    for _, r in out.iterrows():
        d_str = f"{r['cohens_d']:+.2f}" if not math.isnan(r["cohens_d"]) else "  nan"
        print(
            f"{r['feature']:28s} "
            f"{r['mean_A00']:10.3f} "
            f"{r['mean_RED']:10.3f} "
            f"{r['delta']:+10.3f} "
            f"{d_str:>7s}  "
            f"{r['magnitude']}"
        )
    print()

    # Diagnostico automatico.
    grandes = out[out["magnitude"] == "GRANDE"]
    medios = out[out["magnitude"] == "MEDIO"]
    pequenos = out[out["magnitude"] == "PEQUENO"]
    negligs = out[out["magnitude"] == "NEGLIGIVEL"]

    # Deteccao de "sinal invertido": o agente pede para reduzir Gold quando
    # a demanda/pressao esta MAIS BAIXA, nao mais alta. Isso e o oposto da
    # semantica de protecao.
    def get_d(feat: str) -> float:
        row = out[out["feature"] == feat]
        if row.empty:
            return float("nan")
        return float(row["cohens_d"].iloc[0])

    d_offered_gold = get_d("offered_gold_mbps")
    d_n6_util = get_d("n6_utilization")
    d_offered_silver = get_d("offered_silver_mbps")
    inverted_signals = []
    if not math.isnan(d_offered_gold) and d_offered_gold < -0.5:
        inverted_signals.append(("offered_gold_mbps", d_offered_gold))
    if not math.isnan(d_n6_util) and d_n6_util < -0.5:
        inverted_signals.append(("n6_utilization", d_n6_util))
    if not math.isnan(d_offered_silver) and d_offered_silver < -0.5:
        inverted_signals.append(("offered_silver_mbps", d_offered_silver))

    print("=" * 78)
    print("DIAGNOSTICO AUTOMATICO")
    print("=" * 78)

    if inverted_signals:
        print("[ALERTA CRITICO] Sinal INVERTIDO detectado.")
        print("    O agente pede para reduzir Gold quando estas features estao MAIS BAIXAS:")
        for feat, d in inverted_signals:
            print(f"      - {feat:24s}  d={d:+.2f}")
        print()
        print("    Interpretacao: o checkpoint NAO esta protegendo Gold de pressao.")
        print("    Ele esta cortando MBR em janelas de BAIXA demanda, possivelmente")
        print("    porque a reward function de treino premia 'economizar MBR alocado'")
        print("    quando o trafego e pequeno. Isso e o oposto da tese do experimento.")
        print()
        print("    Esta politica e SEGURA na pratica (o guardrail bloqueia a reducao),")
        print("    mas NAO demonstra protecao hierarquica sob estresse.")
        print()
        print("    Recomendacao: antes de gerar mais traces, decidir entre")
        print("      (a) re-treinar com reward function explicita de protecao de Gold,")
        print("      (b) trocar o checkpoint por um treinado para preservar Gold.")
        print("    Antes disso, rode compute_offered_pressure.py: se o cenario atual")
        print("    nunca chega a pressao alta, qualquer checkpoint vai patinar aqui.")
        print()
        print(f"Resumo: GRANDE={len(grandes)}  MEDIO={len(medios)}  PEQUENO={len(pequenos)}  NEGLIGIVEL={len(negligs)}")
        print()
        return 0

    if len(grandes) >= 1:
        nomes = ", ".join(grandes["feature"].tolist())
        print("[OK] O agente parece estar reagindo a sinais com effect size GRANDE.")
        print(f"     Features dominantes: {nomes}")
        print("     -> O checkpoint aprendeu algo interpretavel neste cenario.")
        print("     -> A duvida agora vira: o sinal que ele persegue corresponde")
        print("        a risco REAL para Gold? Rode compute_offered_pressure.py")
        print("        para checar se essas janelas tem pressao agregada alta.")
    elif len(medios) >= 1:
        nomes = ", ".join(medios["feature"].tolist())
        print("[ATENCAO] Sinais com effect size MEDIO apenas.")
        print(f"     Features candidatas: {nomes}")
        print("     -> Pode ser um padrao real, pode ser correlacao espuria.")
        print("     -> Recomendo: gerar tracado com estimulo programado e ver")
        print("        se essas mesmas features aumentam nas fases de pressao.")
    elif len(pequenos) >= 1:
        print("[ALERTA] Apenas effect sizes PEQUENOS.")
        print("     -> O agente reage a combinacoes sutis. Pode ser:")
        print("        (a) o checkpoint nao aprendeu uma politica robusta,")
        print("        (b) o cenario atual nao gera estados informativos.")
        print("     -> Antes de retrainar, gere estimulo programado.")
    else:
        print("[CRITICO] Nenhuma feature distingue A00 de A12/A13/A14.")
        print("     -> O agente esta praticamente sorteando neste regime.")
        print("     -> Hipoteses:")
        print("        1. dqn_mock_causal foi treinado em estados muito diferentes")
        print("           dos que aparecem aqui (provavel).")
        print("        2. As features estao saturando (ex.: n6_utilization sempre baixo).")
        print("     -> Rode compute_offered_pressure.py para checar a hipotese 2.")
        print("        Se ela se confirmar, o caminho e mexer no trafego ANTES de retrainar.")
    print()
    print(f"Resumo: GRANDE={len(grandes)}  MEDIO={len(medios)}  PEQUENO={len(pequenos)}  NEGLIGIVEL={len(negligs)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

