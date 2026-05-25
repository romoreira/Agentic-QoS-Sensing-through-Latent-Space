"""Feature vector construction and normalization.

The per-window feature vector is the unit shared by RL-raw and (later)
RL-latent: it is exactly what the encoder will ingest. Keep this code as
the single source of truth.

Order (17 dims):
    [offered_g, offered_s, offered_b,
     rx_g, rx_s, rx_b,
     loss_g, loss_s, loss_b,
     qer_red_g, qer_red_s, qer_red_b,
     n6_tx_mbps, n6_utilization,
     mbr_g_kbps, mbr_s_kbps, mbr_b_kbps]
"""
from __future__ import annotations

import numpy as np

from upf_rl_env.config import EnvConfig
from upf_rl_env.sources.base import WindowMetrics


FEATURES_PER_WINDOW = 17


def metrics_to_features(m: WindowMetrics, cfg: EnvConfig) -> np.ndarray:
    """Build a normalized 17-vector for one window."""
    f = np.empty(FEATURES_PER_WINDOW, dtype=np.float32)
    # mbps features: divide by norm_mbps
    f[0:3] = m.offered_mbps / cfg.norm_mbps
    f[3:6] = m.rx_mbps / cfg.norm_mbps
    # loss already in [0,1]
    f[6:9] = m.loss_rate
    # qer_red: log1p (counts vary wildly), divide by log scale
    f[9:12] = np.log1p(m.qer_red) / 10.0
    f[12] = m.n6_tx_mbps / cfg.norm_mbps
    f[13] = m.n6_utilization  # already a ratio
    f[14:17] = m.mbr_kbps.astype(np.float32) / cfg.norm_kbps
    return f
