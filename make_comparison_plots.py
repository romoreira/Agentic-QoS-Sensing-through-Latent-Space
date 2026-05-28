#!/usr/bin/env python3
"""
make_comparison_plots.py

Generates 10 comparison plots between:
  - Online DQN
  - Baseline fixed MBR
  - Baseline reactive threshold

Reads:
  <runs_root>/seed_*/metrics.csv               (DQN)
  <runs_root>/seed_*/episodes.csv              (DQN)
  <runs_root>/baseline_fixed/seed_*/...        (Fixed)
  <runs_root>/baseline_reactive/seed_*/...     (Reactive)

Writes (PDF only):
  <out_dir>/fig1_headline_comparison.pdf
  <out_dir>/fig2_shortage_cdf.pdf
  <out_dir>/fig3_shortage_per_episode.pdf
  <out_dir>/fig4_reward_per_episode.pdf
  <out_dir>/fig5_dqn_training_loss.pdf
  <out_dir>/fig6_action_distribution.pdf
  <out_dir>/fig7_dynamics_time_series.pdf
  <out_dir>/fig8_shortage_by_burst_regime.pdf
  <out_dir>/fig9_protection_score_cdf.pdf
  <out_dir>/fig10_pareto_tradeoff.pdf

Usage:
  python3 make_comparison_plots.py \
      --runs-root trex_4slice_noise_toolkit/runs/dqn_proxy_gold_100_200 \
      --out-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# ACM-style global rcParams
# ---------------------------------------------------------------------------
mpl.rcParams.update(
    {
        # Typography (user asked for size 16 on labels/ticks, no titles)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.6",
        # Lines and ticks
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        # Grid
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        # Layout and PDF settings
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType in PDFs (ACM-friendly)
        "ps.fonttype": 42,
    }
)

# Colorblind-safe palette (Wong 2011). One color per policy, kept consistent.
POLICY_COLOR = {
    "Fixed MBR":      "#D55E00",   # vermillion
    "Reactive":       "#0072B2",   # blue
    "DQN":   "#009E73",   # bluish green
}
POLICY_ORDER = ["Fixed MBR", "Reactive", "DQN"]

# Marker per policy for scatter-style figures (Fig 10)
POLICY_MARKER = {
    "Fixed MBR":     "X",
    "Reactive":      "s",
    "DQN":  "o",
}

# Map on-disk folder layout -> display label
POLICY_DIRS = {
    "Fixed MBR":     "baseline_fixed",
    "Reactive":      "baseline_reactive",
    "DQN": None,  # DQN seeds live at the runs_root level
}

FIG_W, FIG_H = 7.0, 4.2  # inches: ACM single-column-ish proportions

# Threshold (Mbps) splitting the hidden-burst signal into quiet vs burst.
# README: quiet ~30 Mbps; active burst 220-360 Mbps. 100 Mbps is a clean cut.
BURST_THRESHOLD_MBPS = 100.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def list_seed_dirs(parent: Path) -> List[Path]:
    return sorted(p for p in parent.glob("seed_*") if p.is_dir())


def load_steps(runs_root: Path) -> pd.DataFrame:
    """Load all step-level metrics.csv files, tag with policy + seed."""
    frames: List[pd.DataFrame] = []
    for label in POLICY_ORDER:
        sub = POLICY_DIRS[label]
        parent = runs_root if sub is None else runs_root / sub
        if not parent.exists():
            print(f"[warn] missing directory: {parent}")
            continue
        for seed_dir in list_seed_dirs(parent):
            csv = seed_dir / "metrics.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            df["policy"] = label
            df["seed"] = seed_dir.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No metrics.csv found under {runs_root}")
    return pd.concat(frames, ignore_index=True)


def load_episodes(runs_root: Path) -> pd.DataFrame:
    """Load all episodes.csv files."""
    frames: List[pd.DataFrame] = []
    for label in POLICY_ORDER:
        sub = POLICY_DIRS[label]
        parent = runs_root if sub is None else runs_root / sub
        if not parent.exists():
            continue
        for seed_dir in list_seed_dirs(parent):
            csv = seed_dir / "episodes.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            df["policy"] = label
            df["seed"] = seed_dir.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No episodes.csv found under {runs_root}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def aggregate_over_seeds(
    df: pd.DataFrame, x_col: str, y_col: str, group_col: str = "policy"
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each policy, return x, mean(y), std(y) aggregated across seeds."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, sub in df.groupby(group_col):
        piv = sub.pivot_table(index=x_col, columns="seed", values=y_col, aggfunc="mean")
        piv = piv.sort_index()
        x = piv.index.to_numpy()
        mu = piv.mean(axis=1).to_numpy()
        sd = piv.std(axis=1, ddof=0).fillna(0).to_numpy()
        out[label] = (x, mu, sd)
    return out


def style_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_pdf(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"[ok] wrote {path}")


def compute_protection_score(df: pd.DataFrame) -> pd.Series:
    """gold_protection_score = 1 - shortage / offered_gold, clipped to [0, 1]."""
    offered = df["offered_gold_mbps"].astype(float).replace(0, np.nan)
    score = 1.0 - df["gold_shortage_mbps"] / offered
    return score.clip(lower=0.0, upper=1.0).fillna(1.0)


# ---------------------------------------------------------------------------
# Figure 1 — Headline comparison: mean gold shortage (bars + std error bars)
# ---------------------------------------------------------------------------
def fig1_headline(steps: pd.DataFrame, out_dir: Path) -> None:
    agg = (
        steps.groupby(["policy", "seed"])["gold_shortage_mbps"]
        .mean()
        .reset_index()
    )
    means = agg.groupby("policy")["gold_shortage_mbps"].mean()
    stds  = agg.groupby("policy")["gold_shortage_mbps"].std(ddof=0)

    means = means.reindex(POLICY_ORDER)
    stds  = stds.reindex(POLICY_ORDER)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    xs = np.arange(len(POLICY_ORDER))
    colors = [POLICY_COLOR[p] for p in POLICY_ORDER]

    bars = ax.bar(
        xs, means.values, yerr=stds.values, color=colors,
        edgecolor="black", linewidth=0.8, capsize=6, width=0.6,
        error_kw={"elinewidth": 1.2, "ecolor": "0.2"},
    )

    for rect, m, s in zip(bars, means.values, stds.values):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + (s if not np.isnan(s) else 0) + 2.5,
            f"{m:.1f}", ha="center", va="bottom", fontsize=14,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(POLICY_ORDER)
    style_axes(ax, xlabel="Policy", ylabel="Mean Gold shortage (Mbps)")
    ax.set_ylim(0, max(means.values + stds.fillna(0).values) * 1.18)
    save_pdf(fig, out_dir, "fig1_headline_comparison")


# ---------------------------------------------------------------------------
# Figure 2 — CDF of step-level Gold shortage
# ---------------------------------------------------------------------------
def fig2_shortage_cdf(steps: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for policy in POLICY_ORDER:
        vals = np.sort(steps.loc[steps.policy == policy, "gold_shortage_mbps"].to_numpy())
        if vals.size == 0:
            continue
        cdf = np.arange(1, vals.size + 1) / vals.size
        ax.plot(vals, cdf, label=policy, color=POLICY_COLOR[policy])

    ax.axvline(0, color="0.4", linestyle=":", linewidth=1.0)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.005)
    style_axes(ax, xlabel="Gold shortage (Mbps)", ylabel="Empirical CDF")
    ax.legend(loc="lower right")
    save_pdf(fig, out_dir, "fig2_shortage_cdf")


# ---------------------------------------------------------------------------
# Figure 3 — Per-episode Gold shortage (learning curve)
# ---------------------------------------------------------------------------
def fig3_shortage_per_episode(episodes: pd.DataFrame, out_dir: Path) -> None:
    agg = aggregate_over_seeds(episodes, x_col="episode", y_col="gold_shortage_sum")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for policy in POLICY_ORDER:
        if policy not in agg:
            continue
        x, mu, sd = agg[policy]
        c = POLICY_COLOR[policy]
        ax.plot(x, mu, label=policy, color=c, marker="o")
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18, linewidth=0)

    style_axes(ax, xlabel="Episode", ylabel="Gold shortage sum (Mbps)")
    ax.legend(loc="upper right", framealpha=0.4)
    save_pdf(fig, out_dir, "fig3_shortage_per_episode")


# ---------------------------------------------------------------------------
# Figure 4 — Per-episode mean reward
# ---------------------------------------------------------------------------
def fig4_reward_per_episode(episodes: pd.DataFrame, out_dir: Path) -> None:
    agg = aggregate_over_seeds(episodes, x_col="episode", y_col="reward_mean")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for policy in POLICY_ORDER:
        if policy not in agg:
            continue
        x, mu, sd = agg[policy]
        c = POLICY_COLOR[policy]
        ax.plot(x, mu, label=policy, color=c, marker="s")
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18, linewidth=0)
    style_axes(ax, xlabel="Episode", ylabel="Mean reward per step")
    ax.legend(loc="lower right")
    save_pdf(fig, out_dir, "fig4_reward_per_episode")


# ---------------------------------------------------------------------------
# Figure 5 — DQN training loss across episodes
# ---------------------------------------------------------------------------
def fig5_dqn_training_loss(episodes: pd.DataFrame, out_dir: Path) -> None:
    dqn = episodes[episodes.policy == "DQN"].copy()
    dqn = dqn.dropna(subset=["loss_mean"])
    if dqn.empty:
        print("[warn] no DQN loss data; skipping fig5")
        return
    piv = dqn.pivot_table(index="episode", columns="seed", values="loss_mean").sort_index()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    c = POLICY_COLOR["DQN"]
    for col in piv.columns:
        ax.plot(piv.index, piv[col], color=c, alpha=0.25, linewidth=1.2)
    mu = piv.mean(axis=1).to_numpy()
    sd = piv.std(axis=1, ddof=0).fillna(0).to_numpy()
    ax.plot(piv.index, mu, color=c, marker="D", label="DQN (mean over seeds)")
    ax.fill_between(piv.index, mu - sd, mu + sd, color=c, alpha=0.18, linewidth=0)

    style_axes(ax, xlabel="Episode", ylabel="DQN training loss (TD)")
    ax.legend(loc="upper right")
    save_pdf(fig, out_dir, "fig5_dqn_training_loss")


# ---------------------------------------------------------------------------
# Figure 6 — Action distribution per policy (stacked %)
# ---------------------------------------------------------------------------
def fig6_action_distribution(steps: pd.DataFrame, out_dir: Path) -> None:
    action_order = [
        "nominal",
        "cut_non_gold_light",
        "cut_non_gold_medium",
        "cut_non_gold_strong",
        "reserve_gold_light",
        "reserve_gold_strong",
        "restore_silver",
        "restore_bronze",
    ]
    actions_present = [a for a in action_order if a in steps["action_name"].unique()]

    counts = (
        steps.groupby(["policy", "action_name"]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(index=POLICY_ORDER, columns=actions_present, fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100.0

    cmap = plt.get_cmap("tab10")
    action_colors = {a: cmap(i % 10) for i, a in enumerate(actions_present)}

    fig, ax = plt.subplots(figsize=(FIG_W + 1.6, FIG_H))
    bottom = np.zeros(len(pct.index))
    xs = np.arange(len(pct.index))
    for action in actions_present:
        vals = pct[action].to_numpy()
        ax.bar(
            xs, vals, bottom=bottom, label=action,
            color=action_colors[action], edgecolor="white", linewidth=0.6, width=0.65,
        )
        bottom = bottom + vals

    ax.set_xticks(xs)
    ax.set_xticklabels(pct.index)
    ax.set_ylim(0, 100)
    style_axes(ax, xlabel="Policy", ylabel="Action usage (%)")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, fontsize=12, handlelength=1.3,
    )
    save_pdf(fig, out_dir, "fig6_action_distribution")


# ---------------------------------------------------------------------------
# Figure 7 — Dynamic time series for one representative seed
# ---------------------------------------------------------------------------
def fig7_dynamics_time_series(steps: pd.DataFrame, out_dir: Path,
                              seed: str = "seed_007") -> None:
    sub = steps[steps.seed == seed].copy()
    if sub.empty:
        print(f"[warn] seed '{seed}' not found; skipping fig7")
        return
    sub = sub.sort_values(["policy", "step"])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(FIG_W + 1.2, FIG_H * 1.6), sharex=True
    )

    any_policy = sub["policy"].iloc[0]
    t_offered = sub[sub.policy == any_policy].sort_values("step")
    ax_top.plot(
        t_offered["t_s"], t_offered["offered_gold_mbps"],
        color="0.25", linestyle="--", linewidth=1.3, label="Offered Gold",
    )
    for policy in POLICY_ORDER:
        s = sub[sub.policy == policy].sort_values("step")
        if s.empty:
            continue
        ax_top.plot(
            s["t_s"], s["available_for_gold_mbps"],
            color=POLICY_COLOR[policy], linewidth=1.5, label=f"Avail. for Gold — {policy}",
        )
    style_axes(ax_top, xlabel="", ylabel="Gold capacity (Mbps)")
    ax_top.legend(loc="upper right", ncol=2, fontsize=11)

    for policy in POLICY_ORDER:
        s = sub[sub.policy == policy].sort_values("step")
        if s.empty:
            continue
        ax_bot.plot(
            s["t_s"], s["gold_shortage_mbps"],
            color=POLICY_COLOR[policy], linewidth=1.4, label=policy,
        )
    style_axes(ax_bot, xlabel="Time (s)", ylabel="Gold shortage (Mbps)")
    ax_bot.legend(loc="upper right", ncol=3, fontsize=12)

    save_pdf(fig, out_dir, "fig7_dynamics_time_series")


# ---------------------------------------------------------------------------
# Figure 8 — Gold shortage conditional on hidden-burst regime
# ---------------------------------------------------------------------------
def fig8_shortage_by_burst_regime(steps: pd.DataFrame, out_dir: Path) -> None:
    """
    Splits each step into 'quiet' (hidden_burst < threshold) and 'burst'
    (>= threshold), then plots grouped bars: shortage_mean per (policy, regime),
    error bars = std across seeds.
    """
    df = steps.copy()
    df["regime_bin"] = np.where(
        df["hidden_burst_mbps"] >= BURST_THRESHOLD_MBPS, "Burst", "Quiet"
    )

    # Per-seed mean shortage in each (policy, regime)
    per_seed = (
        df.groupby(["policy", "regime_bin", "seed"])["gold_shortage_mbps"]
        .mean()
        .reset_index()
    )
    # Mean and std across seeds
    means = (
        per_seed.groupby(["policy", "regime_bin"])["gold_shortage_mbps"]
        .mean()
        .unstack("regime_bin")
        .reindex(POLICY_ORDER)
    )
    stds = (
        per_seed.groupby(["policy", "regime_bin"])["gold_shortage_mbps"]
        .std(ddof=0)
        .unstack("regime_bin")
        .reindex(POLICY_ORDER)
        .fillna(0.0)
    )
    # Ensure both regimes are columns even if one is missing
    for col in ("Quiet", "Burst"):
        if col not in means.columns:
            means[col] = np.nan
            stds[col] = 0.0
    means = means[["Quiet", "Burst"]]
    stds = stds[["Quiet", "Burst"]]

    fig, ax = plt.subplots(figsize=(FIG_W + 0.5, FIG_H))
    xs = np.arange(len(POLICY_ORDER))
    width = 0.36

    # Each policy keeps its own hue; the two regimes are encoded by
    # alpha (light = quiet, solid = burst).
    policy_face = [POLICY_COLOR[p] for p in POLICY_ORDER]

    b1 = ax.bar(
        xs - width / 2, means["Quiet"].values, width=width,
        yerr=stds["Quiet"].values, color=policy_face, alpha=0.45,
        edgecolor="black", linewidth=0.8, capsize=5,
        error_kw={"elinewidth": 1.0, "ecolor": "0.2"},
    )
    b2 = ax.bar(
        xs + width / 2, means["Burst"].values, width=width,
        yerr=stds["Burst"].values, color=policy_face, alpha=1.0,
        edgecolor="black", linewidth=0.8, capsize=5,
        error_kw={"elinewidth": 1.0, "ecolor": "0.2"},
    )

    # Numeric labels above each bar
    for group in (b1, b2):
        for rect in group:
            h = rect.get_height()
            if np.isnan(h):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2.0, h + 3.0,
                f"{h:.1f}", ha="center", va="bottom", fontsize=12,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(POLICY_ORDER)
    style_axes(ax, xlabel="Policy", ylabel="Mean Gold shortage (Mbps)")
    all_top = np.nanmax(means.values + stds.values) if means.size else 1.0
    # Extra headroom so the inside legend never collides with the tall bar
    ax.set_ylim(0, all_top * 1.45)

    # Neutral legend chips: hue is reserved for policy identity, so the
    # legend only conveys what changes between regimes (alpha = saturation).
    thr = int(BURST_THRESHOLD_MBPS)
    legend_handles = [
        Patch(facecolor="0.45", alpha=0.45, edgecolor="black", linewidth=0.8,
              label=f"Quiet  (hidden burst < {thr} Mbps)"),
        Patch(facecolor="0.45", alpha=1.0, edgecolor="black", linewidth=0.8,
              label=f"Burst  (hidden burst \u2265 {thr} Mbps)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right", fontsize=13,
        frameon=True, framealpha=0.95, edgecolor="0.6",
        handlelength=1.4, handleheight=1.0, borderpad=0.5,
    )
    save_pdf(fig, out_dir, "fig8_shortage_by_burst_regime")


# ---------------------------------------------------------------------------
# Figure 9 — CDF of Gold Protection Score
# ---------------------------------------------------------------------------
def fig9_protection_score_cdf(steps: pd.DataFrame, out_dir: Path) -> None:
    df = steps.copy()
    df["protection_score"] = compute_protection_score(df)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for policy in POLICY_ORDER:
        vals = np.sort(df.loc[df.policy == policy, "protection_score"].to_numpy())
        if vals.size == 0:
            continue
        cdf = np.arange(1, vals.size + 1) / vals.size
        ax.plot(vals, cdf, label=policy, color=POLICY_COLOR[policy])

    ax.axvline(1.0, color="0.4", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.005)
    style_axes(ax, xlabel="Gold protection score (1 \u2212 shortage / offered)",
               ylabel="Empirical CDF")
    ax.legend(loc="upper left")
    save_pdf(fig, out_dir, "fig9_protection_score_cdf")


# ---------------------------------------------------------------------------
# Figure 10 — Pareto trade-off: Gold protection vs non-Gold throughput
# ---------------------------------------------------------------------------
def fig10_pareto_tradeoff(steps: pd.DataFrame, out_dir: Path) -> None:
    """
    For each (policy, seed):
       x = mean non-Gold throughput delivered after MBR (Silver + Bronze) in Mbps
       y = mean Gold protection score in [0, 1]
    Better controllers are UP-RIGHT.
    """
    df = steps.copy()
    df["non_gold_mbps"] = df["after_mbr_silver_mbps"] + df["after_mbr_bronze_mbps"]
    df["protection_score"] = compute_protection_score(df)

    per = (
        df.groupby(["policy", "seed"])
          .agg(non_gold_mbps=("non_gold_mbps", "mean"),
               protection=("protection_score", "mean"))
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Per-seed points (small) + per-policy centroid (large)
    for policy in POLICY_ORDER:
        sub = per[per.policy == policy]
        if sub.empty:
            continue
        c = POLICY_COLOR[policy]
        m = POLICY_MARKER[policy]
        ax.scatter(
            sub["non_gold_mbps"], sub["protection"],
            color=c, marker=m, s=60, alpha=0.55,
            edgecolors="black", linewidths=0.7,
        )
        cx = sub["non_gold_mbps"].mean()
        cy = sub["protection"].mean()
        ax.scatter(
            [cx], [cy], color=c, marker=m, s=220,
            edgecolors="black", linewidths=1.2, label=policy, zorder=5,
        )

    # Indicate "better" direction with an arrow
    ax.annotate(
        "better", xy=(0.97, 0.97), xycoords="axes fraction",
        xytext=(0.78, 0.78), textcoords="axes fraction",
        fontsize=13, ha="center", va="center",
        arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2),
    )

    style_axes(
        ax,
        xlabel="Mean non-Gold throughput after MBR (Mbps)",
        ylabel="Mean Gold protection score",
    )
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower left")
    save_pdf(fig, out_dir, "fig10_pareto_tradeoff")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("trex_4slice_noise_toolkit/runs/dqn_proxy_gold_100_200"),
        help="Root directory containing seed_*/, baseline_fixed/, baseline_reactive/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Where to save the PDF figures.",
    )
    parser.add_argument(
        "--ts-seed",
        type=str,
        default="seed_007",
        help="Seed used for the time-series figure (default: seed_007).",
    )
    args = parser.parse_args()

    runs_root: Path = args.runs_root.resolve()
    out_dir: Path = args.out_dir.resolve()
    print(f"[info] runs_root = {runs_root}")
    print(f"[info] out_dir   = {out_dir}")

    steps = load_steps(runs_root)
    episodes = load_episodes(runs_root)
    print(f"[info] steps rows    = {len(steps):,}  "
          f"(policies: {steps['policy'].unique().tolist()})")
    print(f"[info] episodes rows = {len(episodes):,}")

    fig1_headline(steps, out_dir)
    fig2_shortage_cdf(steps, out_dir)
    fig3_shortage_per_episode(episodes, out_dir)
    fig4_reward_per_episode(episodes, out_dir)
    fig5_dqn_training_loss(episodes, out_dir)
    fig6_action_distribution(steps, out_dir)
    fig7_dynamics_time_series(steps, out_dir, seed=args.ts_seed)
    fig8_shortage_by_burst_regime(steps, out_dir)
    fig9_protection_score_cdf(steps, out_dir)
    fig10_pareto_tradeoff(steps, out_dir)

    print("[done] 10 figures written to", out_dir)


if __name__ == "__main__":
    main()
