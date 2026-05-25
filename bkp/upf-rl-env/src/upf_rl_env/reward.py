"""Reward computation.

R_t = -w_g * loss_gold
      -w_s * loss_silver
      -w_b * loss_bronze
      +w_u * n6_utilization
      -w_a * 1[action != prev_action]
      -w_v * 1[loss_gold > SLA_threshold]

Components are returned alongside the scalar so the eval/logger can show
where the reward is coming from — useful when comparing raw vs latent.
"""
from __future__ import annotations

from typing import Optional, Tuple

from upf_rl_env.config import EnvConfig
from upf_rl_env.sources.base import WindowMetrics


def compute_reward(
    m: WindowMetrics,
    action: int,
    prev_action: Optional[int],
    cfg: EnvConfig,
) -> Tuple[float, dict]:
    g_loss = float(m.loss_rate[0])
    s_loss = float(m.loss_rate[1])
    b_loss = float(m.loss_rate[2])

    r_gold = -cfg.w_gold_loss * g_loss
    r_silver = -cfg.w_silver_loss * s_loss
    r_bronze = -cfg.w_bronze_loss * b_loss
    r_util = cfg.w_utilization * float(m.n6_utilization)
    r_change = -cfg.w_action_change * float(prev_action is not None and action != prev_action)
    r_sla = -cfg.w_sla_violation * float(g_loss > cfg.sla_gold_loss_threshold)

    total = r_gold + r_silver + r_bronze + r_util + r_change + r_sla

    components = {
        "r_gold": r_gold,
        "r_silver": r_silver,
        "r_bronze": r_bronze,
        "r_util": r_util,
        "r_change": r_change,
        "r_sla": r_sla,
    }
    return float(total), components
