#!/usr/bin/env python3
"""
make_comparison_plots.py

Generates 7 indispensable comparison plots between:
  - Online DQN
  - Baseline fixed MBR
  - Baseline reactive threshold

Reads:
  <runs_root>/seed_*/metrics.csv               (DQN)
  <runs_root>/seed_*/episodes.csv              (DQN)
  <runs_root>/baseline_fixed/seed_*/...        (Fixed)
  <runs_root>/baseline_reactive/seed_*/...     (Reactive)

Writes:
  <out_dir>/fig1_headline_comparison.pdf
  <out_dir>/fig2_shortage_cdf.pdf
  <out_dir>/fig3_shortage_per_episode.pdf
  <out_dir>/fig4_reward_per_episode.pdf
  <out_dir>/fig5_dqn_training_loss.pdf
  <out_dir>/fig6_action_distribution.pdf
  <out_dir>/fig7_dynamics_time_series.pdf

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
    "DQN (online)":   "#009E73",   # bluish green
}
POLICY_ORDER = ["Fixed MBR", "Reactive", "DQN (online)"]

# Map on-disk folder layout -> display label
POLICY_DIRS = {
    "Fixed MBR":     "baseline_fixed",
    "Reactive":      "baseline_reactive",
    "DQN (online)": None,  # DQN seeds live at the runs_root level
}

FIG_W, FIG_H = 7.0, 4.2  # inches: ACM single-column-ish proportions


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


# ---------------------------------------------------------------------------
# Figure 1 — Headline comparison: mean gold shortage (bars + 95% CI)
# ---------------------------------------------------------------------------
def fig1_headline(steps: pd.DataFrame, out_dir: Path) -> None:
    # Per-seed mean shortage, then mean ± std across seeds
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

    # Numeric annotations above each bar
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
    ax.legend(loc="upper right")
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
    dqn = episodes[episodes.policy == "DQN (online)"].copy()
    dqn = dqn.dropna(subset=["loss_mean"])
    if dqn.empty:
        print("[warn] no DQN loss data; skipping fig5")
        return
    piv = dqn.pivot_table(index="episode", columns="seed", values="loss_mean").sort_index()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    c = POLICY_COLOR["DQN (online)"]
    # Faint per-seed lines + bold mean
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
    # Stable, semantic order for stacking
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

    # Distinct, ordered colormap for actions (perceptually decent)
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
    """
    Shows, for the same seed, how each policy tracks offered Gold demand and
    how much instantaneous shortage it generates.
    Top panel  : offered Gold (shared signal) + available_for_gold per policy.
    Bottom panel: gold_shortage_mbps per policy.
    """
    sub = steps[steps.seed == seed].copy()
    if sub.empty:
        print(f"[warn] seed '{seed}' not found; skipping fig7")
        return
    sub = sub.sort_values(["policy", "step"])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(FIG_W + 1.2, FIG_H * 1.6), sharex=True
    )

    # --- Top: offered Gold (single ref line) + available_for_gold per policy ---
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

    # --- Bottom: shortage per policy ---
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

    print("[done] 7 figures written to", out_dir)


if __name__ == "__main__":
    main()

