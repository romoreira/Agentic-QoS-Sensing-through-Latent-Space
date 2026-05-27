#!/usr/bin/env python3
"""
compute_offered_pressure.py
===========================

Pergunta que este script responde:
    "O cenario atual gera, em algum momento, pressao agregada suficiente
     para colocar Gold em risco REAL? Ou e um cenario subcritico em que
     nada nunca chega perto de saturar o N6?"

Por que isso importa:
    Sem janelas de alta pressao, nao existe a situacao em que o agente
    PRECISA escolher entre slices. Sem essa situacao, a tese de protecao
    hierarquica de Gold nao tem como ser demonstrada, qualquer que seja
    o checkpoint. Esta e a checagem mais barata e definitiva.

Como usar:
    python3 compute_offered_pressure.py \
        --csv runs/online_dqn_dryrun_bronze12_hier_420s/online_dqn_dryrun.csv \
        --capacity 1000

    O default de capacity (1000 Mbps) bate com o default do
    online_dqn_dryrun.py. Ajuste se seu enlace fisico for menor.

Saida:
    - Estatisticas de offered total.
    - Histograma textual por bucket de utilizacao.
    - Top janelas de maior pressao.
    - Diagnostico: SUBCRITICO / MODERADO / CRITICO.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Mede pressao agregada e classifica subcriticidade do cenario.")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--capacity", type=float, default=1000.0,
                    help="Capacidade N6 em Mbps (default 1000, bate com online_dqn_dryrun.py).")
    ap.add_argument("--top-k", type=int, default=10, help="Quantas janelas de pico mostrar.")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[ERRO] CSV nao encontrado: {args.csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)

    # Constroi offered_total. Preferimos as colunas offered_*_mbps quando
    # disponiveis; cai para n6_total_mbps senao.
    if "noise_offered_mbps" in df.columns:
        df["offered_total_mbps"] = (
            df["offered_gold_mbps"]
            + df["offered_silver_mbps"]
            + df["offered_bronze_mbps"]
            + df["noise_offered_mbps"]
        )
        source = "soma de offered_{gold,silver,bronze,noise}_mbps"
    elif "n6_total_mbps" in df.columns:
        df["offered_total_mbps"] = df["n6_total_mbps"]
        source = "n6_total_mbps (sem coluna de noise offered)"
    else:
        print("[ERRO] CSV nao tem colunas suficientes para calcular pressao.", file=sys.stderr)
        return 2

    df["pressure_ratio"] = df["offered_total_mbps"] / max(args.capacity, 1e-9)

    n = len(df)
    print("=" * 78)
    print("ANALISE DE PRESSAO AGREGADA")
    print("=" * 78)
    print(f"CSV: {args.csv}")
    print(f"Janelas: {n}")
    print(f"Capacidade assumida: {args.capacity:.1f} Mbps")
    print(f"Fonte de offered_total: {source}")
    print()
    print("Estatisticas de offered_total_mbps:")
    print(df["offered_total_mbps"].describe().to_string())
    print()
    print(f"Min  pressure_ratio: {df['pressure_ratio'].min():.3f}")
    print(f"Mean pressure_ratio: {df['pressure_ratio'].mean():.3f}")
    print(f"Max  pressure_ratio: {df['pressure_ratio'].max():.3f}")
    print()

    # Histograma textual.
    buckets = [
        ("<30%",       0.00, 0.30),
        ("30-50%",     0.30, 0.50),
        ("50-70%",     0.50, 0.70),
        ("70-85%",     0.70, 0.85),
        ("85-95%",     0.85, 0.95),
        (">=95%",      0.95, float("inf")),
    ]
    print("Distribuicao por bucket de pressao (capacidade = N6):")
    print("-" * 78)
    print(f"{'bucket':>10s}  {'janelas':>8s}  {'%':>6s}  barra")
    for label, lo, hi in buckets:
        mask = (df["pressure_ratio"] >= lo) & (df["pressure_ratio"] < hi)
        c = int(mask.sum())
        pct = 100.0 * c / max(n, 1)
        bar = "#" * int(round(pct / 2.0))
        print(f"{label:>10s}  {c:>8d}  {pct:>5.1f}%  {bar}")
    print()

    # Top picos.
    top = df.nlargest(args.top_k, "pressure_ratio")[
        ["t_s", "offered_gold_mbps", "offered_silver_mbps",
         "offered_bronze_mbps", "noise_offered_mbps", "offered_total_mbps",
         "pressure_ratio", "loss_gold"]
    ]
    print(f"Top {args.top_k} janelas por pressao:")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    # Diagnostico.
    p_max = float(df["pressure_ratio"].max())
    p_mean = float(df["pressure_ratio"].mean())
    p_high = float((df["pressure_ratio"] >= 0.85).mean())

    print("=" * 78)
    print("DIAGNOSTICO")
    print("=" * 78)
    if p_max < 0.50:
        print("[SUBCRITICO] Pressao maxima ficou abaixo de 50% da capacidade.")
        print(f"    pressure_ratio_max  = {p_max:.3f}")
        print(f"    pressure_ratio_mean = {p_mean:.3f}")
        print("    -> O cenario NUNCA gerou risco real para Gold neste run.")
        print("    -> Qualquer decisao do DQN aqui esta sendo tomada num regime")
        print("       em que nao ha trade-off entre slices. Validar 'protecao de Gold'")
        print("       neste cenario e impossivel por construcao.")
        print()
        print("    Acao recomendada: subir capacidade alvo dos slices OU reduzir")
        print("    capacidade N6 efetiva ate que pelo menos uma janela atinja >=0.85.")
        print("    O estimulo programado (traffic_scenario_programmed.py)")
        print("    foi desenhado para isto: ele cria fases que forcam pressao alta")
        print("    sem deixar tudo igualmente caotico.")
    elif p_max < 0.85:
        print("[MODERADO] Pressao maxima ficou entre 50% e 85% da capacidade.")
        print(f"    pressure_ratio_max  = {p_max:.3f}")
        print(f"    pressure_ratio_mean = {p_mean:.3f}")
        print("    -> Ha pressao perceptivel, mas nada que force escolha entre slices.")
        print("    -> Recomendacao: gerar estimulo programado para criar janelas >=0.85.")
    else:
        print("[CRITICO] Pelo menos uma janela atingiu >=85% da capacidade.")
        print(f"    pressure_ratio_max         = {p_max:.3f}")
        print(f"    pressure_ratio_mean        = {p_mean:.3f}")
        print(f"    fracao de janelas >=0.85   = {p_high:.1%}")
        print("    -> O cenario gera estresse real. Se mesmo assim o agente nao reage")
        print("       coerentemente nessas janelas, o problema esta no checkpoint.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

