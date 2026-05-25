"""Discrete action space for slice MBR control.

15 actions, ordered so that the agent walks down a priority-aware degradation
ladder. Action 0 keeps everything at nominal. Higher indices cut more aggressively,
starting from bronze, then silver, then gold. Adjacent indices differ by a
small step so the Q-net does not need to learn large jumps.

Each action is a 3-tuple (gold_pct, silver_pct, bronze_pct) of the nominal MBR.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


ActionTuple = Tuple[float, float, float]

ACTION_GRID: List[ActionTuple] = [
    # 0: full
    (1.00, 1.00, 1.00),
    # 1-4: progressive bronze cuts
    (1.00, 1.00, 0.90),
    (1.00, 1.00, 0.70),
    (1.00, 1.00, 0.50),
    (1.00, 1.00, 0.30),
    # 5-8: bronze floored + progressive silver cuts
    (1.00, 0.90, 0.30),
    (1.00, 0.70, 0.30),
    (1.00, 0.50, 0.30),
    (1.00, 0.30, 0.30),
    # 9-11: silver/bronze floored + light gold cuts
    (0.95, 0.30, 0.30),
    (0.90, 0.30, 0.30),
    (0.85, 0.30, 0.30),
    # 12-14: emergency — heavier gold cuts
    (0.80, 0.30, 0.30),
    (0.70, 0.30, 0.30),
    (0.60, 0.30, 0.30),
]

N_ACTIONS = len(ACTION_GRID)
assert N_ACTIONS == 15, f"Expected 15 actions, got {N_ACTIONS}"

ACTION_GRID_NP = np.array(ACTION_GRID, dtype=np.float32)  # shape (15, 3)


def action_to_mbr_kbps(action_idx: int, nominal_mbr_kbps: Tuple[int, int, int]) -> np.ndarray:
    """Translate action index to per-slice MBR in kbps."""
    pct = ACTION_GRID_NP[action_idx]
    nominal = np.array(nominal_mbr_kbps, dtype=np.float32)
    return (pct * nominal).astype(np.int64)


def action_description(action_idx: int) -> str:
    g, s, b = ACTION_GRID[action_idx]
    return f"A{action_idx:02d}(G={int(g*100)}%,S={int(s*100)}%,B={int(b*100)}%)"
