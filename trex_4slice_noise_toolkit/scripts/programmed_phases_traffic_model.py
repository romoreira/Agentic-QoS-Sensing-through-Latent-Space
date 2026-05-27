#!/usr/bin/env python3
"""
programmed_phases_traffic_model.py
==================================

Traffic model determinista em 4 fases, desenhado para tornar a tese de
protecao hierarquica de Gold FALSIFIAVEL.

Por que existir:
    O modelo `sine_abrupt` atual produz um cenario subcritico: a pressao
    agregada nunca ultrapassa ~47% da capacidade N6 default. Sem nenhuma
    janela de alta pressao, e impossivel demonstrar que o agente protege
    Gold quando ha trade-off entre slices.

    Este modelo NAO substitui sine_abrupt. Ele e um cenario de teste
    com FASES PROGRAMADAS e timestamps fixos, para que voce possa fazer
    afirmacoes do tipo:
        "Em 240-360s a pressao agregada esta em ~90% da capacidade.
         Eu esperava ver o agente saindo de A00 e indo para A04 ou A06."

Fases (default, para capacity=1000 Mbps):
    [0   - 120s) Fase 0 BASELINE       : pressao ~30%. Esperado: A00 dominante.
    [120 - 240s) Fase 1 PRESSAO_MODER. : Bronze rampa 50 -> 230 Mbps. Pressao ~50-65%.
                                          Esperado: A04 aparece, Gold preservado.
    [240 - 360s) Fase 2 PRESSAO_ALTA   : Bronze alto + Silver rampa + Noise sustentado.
                                          Pressao ~85-95%. Esperado: A06 aparece.
    [360 - 480s) Fase 3 RELAXAMENTO    : Volta progressiva. Esperado: A04 some, A00 volta.

Como integrar com o online_dqn_dryrun:
    1. Coloque este arquivo em trex_4slice_noise_toolkit/scripts/
    2. Em traffic_model.py, adicione ao make_traffic_model():
           if kind in ("programmed", "programmed_phases", "phases"):
               from programmed_phases_traffic_model import ProgrammedPhasesTrafficModel, ProgrammedPhasesConfig
               return ProgrammedPhasesTrafficModel(ProgrammedPhasesConfig(dt_s=dt_s), seed=seed)
    3. Rode online_dqn_dryrun.py com --traffic-model programmed --duration 480

Como usar standalone (preview):
    python3 programmed_phases_traffic_model.py --duration 480 --dt 1.0 \
        --capacity 1000 --out preview_programmed.csv
    Gera um CSV com colunas step, t_s, phase, gold_mbps, silver_mbps,
    bronze_mbps, noise_mbps, total_mbps, pressure_ratio.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import csv


# Mantemos compatibilidade com o esquema do traffic_model.py original.
SLICE_NAMES = ("gold", "silver", "bronze", "noise")


@dataclass
class TrafficSample:
    """Mesmo formato usado pelo SineAbruptTrafficModel para drop-in compatibilidade."""
    step: int
    t_s: float
    regime: int  # aqui usado como id da fase: 0,1,2,3
    mbps: Dict[str, float]


@dataclass
class ProgrammedPhasesConfig:
    dt_s: float = 1.0

    # Limites de cada fase em segundos. Total = phase_ends[-1].
    # Mudancas entre fases sao suaves (rampa linear) para nao gerar saltos
    # que seriam interpretados como bursts artificiais.
    phase_ends_s: Sequence[float] = (120.0, 240.0, 360.0, 480.0)

    # Capacidade N6 de referencia em Mbps. As intensidades das fases sao
    # calibradas para que a pressao maxima fique perto de 0.90 dessa
    # capacidade.
    capacity_mbps: float = 1000.0

    # Niveis BASELINE (Fase 0). Compativeis com sine_abrupt para que o
    # agente comece em um regime familiar.
    baseline_gold_mbps: float = 145.0
    baseline_silver_mbps: float = 85.0
    baseline_bronze_mbps: float = 45.0
    baseline_noise_mbps: float = 20.0

    # PRESSAO MODERADA (Fase 1). Bronze sobe, Silver leve subida.
    phase1_gold_mbps: float = 145.0
    phase1_silver_mbps: float = 110.0
    phase1_bronze_peak_mbps: float = 230.0    # rampa de baseline ate aqui
    phase1_noise_mbps: float = 30.0

    # PRESSAO ALTA (Fase 2). Pressao agregada perto de 90% da capacidade.
    phase2_gold_mbps: float = 180.0
    phase2_silver_peak_mbps: float = 220.0    # rampa
    phase2_bronze_mbps: float = 230.0
    phase2_noise_mbps: float = 280.0          # noise sustentado alto

    # Jitter pequeno para nao deixar tudo perfeitamente determinista.
    # Importante: usamos seed para reproducibilidade.
    jitter_mbps: float = 1.5

    # Limites de seguranca.
    min_offered_mbps: float = 0.0
    max_offered_mbps: float = 600.0


def _linear_ramp(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    """Rampa linear de v0 (em t0) para v1 (em t1), saturada nas pontas."""
    if t1 <= t0:
        return v1
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


class ProgrammedPhasesTrafficModel:
    """Traffic model determinista com 4 fases programadas.

    A interface (sample, iter_duration, atributo step_idx/t_s/cfg) e
    intencionalmente compativel com SineAbruptTrafficModel para drop-in.
    """

    def __init__(self, cfg: Optional[ProgrammedPhasesConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = cfg or ProgrammedPhasesConfig()
        self.rng = random.Random(seed if seed is not None else 0xC0FFEE)
        self.step_idx = 0
        self.t_s = 0.0

    # --- helpers de fase ---------------------------------------------------

    def _phase_index(self, t: float) -> int:
        for i, end in enumerate(self.cfg.phase_ends_s):
            if t < end:
                return i
        return len(self.cfg.phase_ends_s) - 1

    def _phase_window(self, idx: int) -> tuple[float, float]:
        start = 0.0 if idx == 0 else self.cfg.phase_ends_s[idx - 1]
        end = self.cfg.phase_ends_s[idx]
        return start, end

    # --- niveis-alvo por fase, sem jitter ----------------------------------

    def _target_mbps(self, t: float) -> Dict[str, float]:
        cfg = self.cfg
        idx = self._phase_index(t)
        p_start, p_end = self._phase_window(idx)

        if idx == 0:
            # Baseline plano.
            return {
                "gold":   cfg.baseline_gold_mbps,
                "silver": cfg.baseline_silver_mbps,
                "bronze": cfg.baseline_bronze_mbps,
                "noise":  cfg.baseline_noise_mbps,
            }

        if idx == 1:
            # Fase 1: Bronze faz rampa linear do baseline ate o pico.
            # Silver e gold sobem para um nivel quase plano (rampa curta).
            gold = _linear_ramp(t, p_start, p_start + 30, cfg.baseline_gold_mbps, cfg.phase1_gold_mbps)
            silver = _linear_ramp(t, p_start, p_start + 30, cfg.baseline_silver_mbps, cfg.phase1_silver_mbps)
            bronze = _linear_ramp(t, p_start, p_end - 10, cfg.baseline_bronze_mbps, cfg.phase1_bronze_peak_mbps)
            noise = _linear_ramp(t, p_start, p_start + 30, cfg.baseline_noise_mbps, cfg.phase1_noise_mbps)
            return {"gold": gold, "silver": silver, "bronze": bronze, "noise": noise}

        if idx == 2:
            # Fase 2: Bronze mantem alto. Silver sobe para o pico. Noise alto.
            # Gold sobe um pouco para que haja co-pressao de Gold + Silver +
            # Bronze + Noise simultaneamente.
            gold = _linear_ramp(t, p_start, p_start + 30, cfg.phase1_gold_mbps, cfg.phase2_gold_mbps)
            silver = _linear_ramp(t, p_start, p_end - 20, cfg.phase1_silver_mbps, cfg.phase2_silver_peak_mbps)
            bronze = cfg.phase2_bronze_mbps
            noise = _linear_ramp(t, p_start, p_start + 20, cfg.phase1_noise_mbps, cfg.phase2_noise_mbps)
            return {"gold": gold, "silver": silver, "bronze": bronze, "noise": noise}

        # Fase 3: relaxamento gradual para o baseline.
        gold = _linear_ramp(t, p_start, p_end - 10, cfg.phase2_gold_mbps, cfg.baseline_gold_mbps)
        silver = _linear_ramp(t, p_start, p_end - 10, cfg.phase2_silver_peak_mbps, cfg.baseline_silver_mbps)
        bronze = _linear_ramp(t, p_start, p_end - 10, cfg.phase2_bronze_mbps, cfg.baseline_bronze_mbps)
        noise = _linear_ramp(t, p_start, p_end - 10, cfg.phase2_noise_mbps, cfg.baseline_noise_mbps)
        return {"gold": gold, "silver": silver, "bronze": bronze, "noise": noise}

    # --- API publica -------------------------------------------------------

    def sample(self) -> TrafficSample:
        cfg = self.cfg
        tgt = self._target_mbps(self.t_s)
        mbps: Dict[str, float] = {}
        for name in SLICE_NAMES:
            v = tgt[name] + (self.rng.gauss(0.0, cfg.jitter_mbps) if cfg.jitter_mbps > 0 else 0.0)
            v = max(cfg.min_offered_mbps, min(cfg.max_offered_mbps, v))
            mbps[name] = float(v)
        s = TrafficSample(
            step=self.step_idx,
            t_s=self.t_s,
            regime=self._phase_index(self.t_s),
            mbps=mbps,
        )
        self.step_idx += 1
        self.t_s += cfg.dt_s
        return s

    def iter_duration(self, duration_s: float) -> Iterable[TrafficSample]:
        n = int(math.ceil(duration_s / self.cfg.dt_s))
        for _ in range(n):
            yield self.sample()


# -----------------------------------------------------------------------------
# CLI de preview
# -----------------------------------------------------------------------------

def _phase_name(idx: int) -> str:
    return {
        0: "BASELINE",
        1: "PRESSAO_MODERADA",
        2: "PRESSAO_ALTA",
        3: "RELAXAMENTO",
    }.get(idx, f"FASE_{idx}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Preview do traffic model programado em 4 fases.")
    ap.add_argument("--duration", type=float, default=480.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--capacity", type=float, default=1000.0,
                    help="Capacidade N6 em Mbps usada apenas para calcular pressure_ratio no preview.")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--out", type=Path, default=Path("preview_programmed.csv"))
    args = ap.parse_args()

    cfg = ProgrammedPhasesConfig(dt_s=args.dt, capacity_mbps=args.capacity)
    model = ProgrammedPhasesTrafficModel(cfg=cfg, seed=args.seed)

    rows = []
    for s in model.iter_duration(args.duration):
        total = sum(s.mbps[k] for k in SLICE_NAMES)
        rows.append({
            "step": s.step,
            "t_s": round(s.t_s, 3),
            "phase": s.regime,
            "phase_name": _phase_name(s.regime),
            "gold_mbps": round(s.mbps["gold"], 3),
            "silver_mbps": round(s.mbps["silver"], 3),
            "bronze_mbps": round(s.mbps["bronze"], 3),
            "noise_mbps": round(s.mbps["noise"], 3),
            "total_mbps": round(total, 3),
            "pressure_ratio": round(total / max(args.capacity, 1e-9), 4),
        })

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] preview escrito em {args.out} ({len(rows)} linhas).")

    # Estatisticas por fase.
    print()
    print("Resumo por fase:")
    print("-" * 78)
    print(f"{'phase':>5s}  {'name':<18s}  {'t_start':>7s}  {'t_end':>7s}  "
          f"{'p_min':>5s}  {'p_mean':>6s}  {'p_max':>5s}")
    by_phase: Dict[int, list] = {}
    for r in rows:
        by_phase.setdefault(r["phase"], []).append(r["pressure_ratio"])
    for ph in sorted(by_phase.keys()):
        ratios = by_phase[ph]
        ts = [r["t_s"] for r in rows if r["phase"] == ph]
        print(
            f"{ph:>5d}  {_phase_name(ph):<18s}  "
            f"{min(ts):>7.1f}  {max(ts):>7.1f}  "
            f"{min(ratios):>5.2f}  {sum(ratios)/len(ratios):>6.2f}  {max(ratios):>5.2f}"
        )
    print()
    print("Confira: pressure_ratio na FASE 2 deve ficar perto de 0.85-0.95 com o")
    print("default de capacity=1000 Mbps. Se nao ficou, ajuste capacidade ou")
    print("os niveis em ProgrammedPhasesConfig para casar com o seu enlace real.")
    print()
    print("Para validar a hierarquia depois de rodar o online_dqn_dryrun com")
    print("este modelo, use: validate_hierarchy_by_phase.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

