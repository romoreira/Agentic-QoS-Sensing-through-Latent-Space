#!/usr/bin/env python3
"""
make_comparison_plots.py

Generates comparison plots and summary tables between:
  - Fixed MBR baseline
  - Reactive threshold baseline
  - DQN     (vanilla DQN, 11-dim instantaneous state)
  - DQN-LS  (DQN over a frozen UNUM-style latent embedding)

Reads:
  <runs_root>/baseline_fixed/seed_*/...
  <runs_root>/baseline_reactive/seed_*/...
  <runs_root>/dqn_vanilla/seed_*/...
  <runs_root>/dqn_latent_space/seed_*/...

Writes (PDF figures):
  fig1_headline_comparison.pdf
  fig2_shortage_cdf.pdf
  fig3_shortage_per_episode.pdf
  fig4_reward_per_episode.pdf
  fig5_dqn_training_loss.pdf
  fig6_action_distribution.pdf
  fig7_dynamics_time_series.pdf
  fig8_shortage_by_burst_regime.pdf
  fig9_protection_score_cdf.pdf
  fig10_pareto_tradeoff.pdf
  fig11_guardrail_activity.pdf      [NEW] embedder divergence + guardrail rate (DQN-LS)
  fig12_tail_shortage.pdf            [NEW] p50/p90/p95/p99 of Gold shortage per policy
  fig13_paired_episode_delta.pdf     [NEW] paired episode delta DQN-LS minus DQN

Writes (tables, CSV + Markdown):
  summary_table.csv / summary_table.md          per-policy summary
  pairwise_comparison.csv / pairwise_comparison.md   pairwise effect sizes

Usage:
  python3 make_comparison_plots.py \
      --runs-root trex_4slice_noise_toolkit/runs/dqn_proxy_gold_100_200_1800s \
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
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Colorblind-safe palette (Wong 2011).
POLICY_COLOR = {
    "Fixed MBR": "#D55E00",   # vermillion
    "Reactive":  "#0072B2",   # blue
    "DQN":       "#009E73",   # bluish green
    "DQN-LS":    "#CC79A7",   # reddish purple
}
POLICY_ORDER = ["Fixed MBR", "Reactive", "DQN", "DQN-LS"]

POLICY_MARKER = {
    "Fixed MBR": "X",
    "Reactive":  "s",
    "DQN":       "o",
    "DQN-LS":    "D",
}

# Map on-disk folder layout -> display label
POLICY_DIRS = {
    "Fixed MBR": "baseline_fixed",
    "Reactive":  "baseline_reactive",
    "DQN":       "dqn_vanilla",
    "DQN-LS":    "dqn_latent_space",
}

FIG_W, FIG_H = 7.0, 4.2  # inches

# Threshold (Mbps) splitting hidden-burst into quiet vs burst.
BURST_THRESHOLD_MBPS = 100.0

# Guardrail bucket threshold matching run_proxy_experiment.py default.
GUARDRAIL_BUCKET_THRESH = 5.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def list_seed_dirs(parent: Path) -> List[Path]:
    return sorted(p for p in parent.glob("seed_*") if p.is_dir())


def load_steps(runs_root: Path) -> pd.DataFrame:
    """Load all step-level metrics.csv files, tag with policy label + seed."""
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
# Figure 1 — Headline comparison
# ---------------------------------------------------------------------------
def fig1_headline(steps: pd.DataFrame, out_dir: Path) -> None:
    agg = (
        steps.groupby(["policy", "seed"])["gold_shortage_mbps"]
        .mean()
        .reset_index()
    )
    means = agg.groupby("policy")["gold_shortage_mbps"].mean()
    stds = agg.groupby("policy")["gold_shortage_mbps"].std(ddof=0)

    means = means.reindex(POLICY_ORDER)
    stds = stds.reindex(POLICY_ORDER)

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
        m = POLICY_MARKER.get(policy, "o")
        ax.plot(x, mu, label=policy, color=c, marker=m)
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
        m = POLICY_MARKER.get(policy, "o")
        ax.plot(x, mu, label=policy, color=c, marker=m)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.18, linewidth=0)
    style_axes(ax, xlabel="Episode", ylabel="Mean reward per step")
    ax.legend(loc="lower right", framealpha=0.4)
    save_pdf(fig, out_dir, "fig4_reward_per_episode")


# ---------------------------------------------------------------------------
# Figure 5 — DQN training loss across episodes (both DQN variants)
# ---------------------------------------------------------------------------
def fig5_dqn_training_loss(episodes: pd.DataFrame, out_dir: Path) -> None:
    dqn_policies = [p for p in POLICY_ORDER if p.startswith("DQN")]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    found_any = False
    for policy in dqn_policies:
        dqn = episodes[episodes.policy == policy].copy().dropna(subset=["loss_mean"])
        if dqn.empty:
            continue
        found_any = True
        piv = dqn.pivot_table(index="episode", columns="seed", values="loss_mean").sort_index()
        c = POLICY_COLOR[policy]
        m = POLICY_MARKER[policy]
        for col in piv.columns:
            ax.plot(piv.index, piv[col], color=c, alpha=0.25, linewidth=1.2)
        mu = piv.mean(axis=1).to_numpy()
        sd = piv.std(axis=1, ddof=0).fillna(0).to_numpy()
        ax.plot(piv.index, mu, color=c, marker=m, label=policy)
        ax.fill_between(piv.index, mu - sd, mu + sd, color=c, alpha=0.18, linewidth=0)

    if not found_any:
        print("[warn] no DQN loss data; skipping fig5")
        plt.close(fig)
        return

    style_axes(ax, xlabel="Episode", ylabel="DQN training loss (TD)")
    ax.legend(loc="upper right")
    save_pdf(fig, out_dir, "fig5_dqn_training_loss")


# ---------------------------------------------------------------------------
# Figure 6 — Action distribution per policy (stacked %)
# ---------------------------------------------------------------------------
ACTION_LABELS = {
    "nominal": "Nominal",
    "cut_non_gold_light": "Cut Non-Gold (Light)",
    "cut_non_gold_medium": "Cut Non-Gold (Medium)",
    "cut_non_gold_strong": "Cut Non-Gold (Strong)",
    "reserve_gold_light": "Reserve Gold (Light)",
    "reserve_gold_strong": "Reserve Gold (Strong)",
    "restore_silver": "Restore Silver",
    "restore_bronze": "Restore Bronze",
}


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

    counts = steps.groupby(["policy", "action_name"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=POLICY_ORDER, columns=actions_present, fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100.0

    cmap = plt.get_cmap("tab10")
    action_colors = {a: cmap(i % 10) for i, a in enumerate(actions_present)}

    fig, ax = plt.subplots(figsize=(FIG_W + 1.6, FIG_H))
    bottom = np.zeros(len(pct.index))
    xs = np.arange(len(pct.index))

    for action in actions_present:
        vals = pct[action].to_numpy()
        pretty_label = ACTION_LABELS.get(action, action)
        ax.bar(
            xs, vals, bottom=bottom, label=pretty_label,
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
            color=POLICY_COLOR[policy], linewidth=1.5,
            label=f"Avail. for Gold — {policy}",
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
    ax_bot.legend(loc="upper right", ncol=4, fontsize=12)

    save_pdf(fig, out_dir, "fig7_dynamics_time_series")


# ---------------------------------------------------------------------------
# Figure 8 — Gold shortage conditional on hidden-burst regime
# ---------------------------------------------------------------------------
def fig8_shortage_by_burst_regime(steps: pd.DataFrame, out_dir: Path) -> None:
    df = steps.copy()
    df["regime_bin"] = np.where(
        df["hidden_burst_mbps"] >= BURST_THRESHOLD_MBPS, "Burst", "Quiet"
    )

    per_seed = (
        df.groupby(["policy", "regime_bin", "seed"])["gold_shortage_mbps"]
        .mean()
        .reset_index()
    )
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
    for col in ("Quiet", "Burst"):
        if col not in means.columns:
            means[col] = np.nan
            stds[col] = 0.0
    means = means[["Quiet", "Burst"]]
    stds = stds[["Quiet", "Burst"]]

    fig, ax = plt.subplots(figsize=(FIG_W + 1.5, FIG_H))
    xs = np.arange(len(POLICY_ORDER))
    width = 0.36

    policy_face = [POLICY_COLOR[p] for p in POLICY_ORDER]

    b1 = ax.bar(
        xs - width / 2, means["Quiet"].values, width=width,
        yerr=stds["Quiet"].values, color=policy_face, alpha=0.45,
        edgecolor="black", linewidth=0.8, capsize=5, hatch="",
        error_kw={"elinewidth": 1.0, "ecolor": "0.2"},
    )
    b2 = ax.bar(
        xs + width / 2, means["Burst"].values, width=width,
        yerr=stds["Burst"].values, color=policy_face, alpha=1.0,
        edgecolor="black", linewidth=0.8, capsize=5, hatch="//",
        error_kw={"elinewidth": 1.0, "ecolor": "0.2"},
    )

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
    ax.set_ylim(0, all_top * 1.45)

    legend_handles = [
        Patch(facecolor="0.45", alpha=0.45, edgecolor="black", linewidth=0.8,
              hatch="", label="Quiet"),
        Patch(facecolor="0.45", alpha=1.0, edgecolor="black", linewidth=0.8,
              hatch="//", label="Burst"),
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
    ax.legend(loc="upper right", framealpha=0.4)
    save_pdf(fig, out_dir, "fig9_protection_score_cdf")


# ---------------------------------------------------------------------------
# Figure 10 — Pareto trade-off
# ---------------------------------------------------------------------------
def fig10_pareto_tradeoff(steps: pd.DataFrame, out_dir: Path) -> None:
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
# Figure 11 — Guardrail activity (DQN-LS only)
# ---------------------------------------------------------------------------
def fig11_guardrail_activity(steps: pd.DataFrame, out_dir: Path) -> None:
    """Two panels:
      (left)  CDF of embedder_divergence across all DQN-LS steps, with the
              guardrail threshold marked.
      (right) Per-seed bar of the fraction of steps in which the guardrail
              fired, with the cross-seed mean as a dashed line.
    """
    ls = steps[steps.policy == "DQN-LS"].copy()
    if ls.empty:
        print("[warn] no DQN-LS data; skipping fig11")
        return

    has_div = "embedder_divergence" in ls.columns
    has_gr = "guardrail_active" in ls.columns
    if not (has_div and has_gr):
        print("[warn] DQN-LS CSV lacks guardrail/divergence columns; skipping fig11")
        return

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(FIG_W + 2.0, FIG_H))
    c = POLICY_COLOR["DQN-LS"]

    # Left: CDF of divergence
    div_vals = ls["embedder_divergence"].dropna().to_numpy()
    if div_vals.size > 0:
        v = np.sort(div_vals)
        cdf = np.arange(1, v.size + 1) / v.size
        ax_l.plot(v, cdf, color=c, label="DQN-LS")
        ax_l.axvline(GUARDRAIL_BUCKET_THRESH, color="0.4", linestyle=":",
                     linewidth=1.2,
                     label=f"threshold = {GUARDRAIL_BUCKET_THRESH:g}")
        ax_l.set_ylim(0, 1.005)
        style_axes(ax_l, xlabel="Embedder divergence (mean bucket distance)",
                   ylabel="Empirical CDF")
        ax_l.legend(loc="lower right", fontsize=12)

    # Right: per-seed guardrail-active fraction
    gr_by_seed = ls.groupby("seed")["guardrail_active"].mean() * 100.0
    gr_by_seed = gr_by_seed.sort_index()
    xs = np.arange(len(gr_by_seed))
    ax_r.bar(xs, gr_by_seed.values, color=c, edgecolor="black",
             linewidth=0.8, width=0.6)
    mean_pct = float(gr_by_seed.mean())
    ax_r.axhline(mean_pct, color="0.3", linestyle="--", linewidth=1.0,
                 label=f"mean = {mean_pct:.2f}%")
    for x, v in zip(xs, gr_by_seed.values):
        ax_r.text(x, v + 0.05, f"{v:.2f}%", ha="center", va="bottom", fontsize=12)
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels([s.replace("seed_", "") for s in gr_by_seed.index])
    style_axes(ax_r, xlabel="Seed", ylabel="Guardrail-active steps (%)")
    ax_r.set_ylim(0, max(gr_by_seed.values.max() * 1.5, 1.0))
    ax_r.legend(loc="upper right", fontsize=12)

    save_pdf(fig, out_dir, "fig11_guardrail_activity")


# ---------------------------------------------------------------------------
# Figure 12 — Tail of Gold shortage per policy (p50, p90, p95, p99)
# ---------------------------------------------------------------------------
def fig12_tail_shortage(steps: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bars: one cluster per policy, four bars per cluster
    (p50, p90, p95, p99). Error bars = std across seeds for that quantile."""
    quantiles = [0.50, 0.90, 0.95, 0.99]
    q_labels = ["p50", "p90", "p95", "p99"]

    # Per (policy, seed) quantile, then aggregate across seeds.
    rows = []
    for policy in POLICY_ORDER:
        sub = steps[steps.policy == policy]
        if sub.empty:
            continue
        for seed, ssub in sub.groupby("seed"):
            for q, ql in zip(quantiles, q_labels):
                rows.append({
                    "policy": policy, "seed": seed, "quantile": ql,
                    "value": float(np.quantile(ssub["gold_shortage_mbps"], q)),
                })
    qdf = pd.DataFrame(rows)
    if qdf.empty:
        print("[warn] no data for fig12; skipping")
        return

    means = qdf.groupby(["policy", "quantile"])["value"].mean().unstack("quantile").reindex(POLICY_ORDER)
    stds = qdf.groupby(["policy", "quantile"])["value"].std(ddof=0).unstack("quantile").reindex(POLICY_ORDER).fillna(0.0)
    means = means[q_labels]
    stds = stds[q_labels]

    fig, ax = plt.subplots(figsize=(FIG_W + 1.5, FIG_H))
    xs = np.arange(len(POLICY_ORDER))
    n_q = len(q_labels)
    width = 0.78 / n_q

    # One color per quantile band (lightest = p50, darkest = p99) over a
    # neutral palette so the policy colors are not overloaded.
    quantile_shades = ["#a6cee3", "#6baed6", "#3182bd", "#08519c"]

    for i, (ql, shade) in enumerate(zip(q_labels, quantile_shades)):
        offsets = xs - 0.39 + (i + 0.5) * width
        ax.bar(
            offsets, means[ql].values, width=width,
            yerr=stds[ql].values, color=shade, edgecolor="black", linewidth=0.6,
            capsize=4, label=ql,
            error_kw={"elinewidth": 0.9, "ecolor": "0.2"},
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(POLICY_ORDER)
    style_axes(ax, xlabel="Policy", ylabel="Gold shortage (Mbps)")
    ax.legend(loc="upper right", title="quantile", fontsize=12, title_fontsize=12)
    save_pdf(fig, out_dir, "fig12_tail_shortage")


# ---------------------------------------------------------------------------
# Figure 13 — Paired episode delta: DQN-LS minus DQN
# ---------------------------------------------------------------------------
def fig13_paired_episode_delta(episodes: pd.DataFrame, out_dir: Path) -> None:
    """Per episode, compute the per-seed difference in gold_shortage_sum
    between DQN-LS and DQN (paired by seed). Plot the mean delta with a
    cross-seed std envelope. Negative = DQN-LS protects Gold better than DQN."""
    dqn = episodes[episodes.policy == "DQN"][["episode", "seed", "gold_shortage_sum"]]
    ls = episodes[episodes.policy == "DQN-LS"][["episode", "seed", "gold_shortage_sum"]]
    if dqn.empty or ls.empty:
        print("[warn] missing DQN or DQN-LS episodes; skipping fig13")
        return

    merged = pd.merge(
        ls.rename(columns={"gold_shortage_sum": "ls"}),
        dqn.rename(columns={"gold_shortage_sum": "vanilla"}),
        on=["episode", "seed"], how="inner",
    )
    if merged.empty:
        print("[warn] no matching (seed, episode) for DQN vs DQN-LS; skipping fig13")
        return
    merged["delta"] = merged["ls"] - merged["vanilla"]

    agg = merged.groupby("episode")["delta"].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    c = POLICY_COLOR["DQN-LS"]
    ax.plot(agg["episode"], agg["mean"], color=c, marker="D",
            label="DQN-LS \u2212 DQN")
    ax.fill_between(agg["episode"],
                    agg["mean"] - agg["std"],
                    agg["mean"] + agg["std"],
                    color=c, alpha=0.18, linewidth=0)
    ax.axhline(0, color="0.3", linestyle="--", linewidth=1.0)

    style_axes(ax, xlabel="Episode",
               ylabel="Paired \u0394 Gold shortage sum (Mbps)")
    # annotate sign hint
    ax.text(0.02, 0.04, "negative \u2192 DQN-LS better",
            transform=ax.transAxes, fontsize=12, color="0.25",
            ha="left", va="bottom")
    ax.legend(loc="upper right", framealpha=0.4)
    save_pdf(fig, out_dir, "fig13_paired_episode_delta")


# ---------------------------------------------------------------------------
# Summary table (per-policy aggregate stats)
# ---------------------------------------------------------------------------
def write_summary_table(steps: pd.DataFrame, episodes: pd.DataFrame,
                        out_dir: Path) -> pd.DataFrame:
    rows = []
    for policy in POLICY_ORDER:
        s = steps[steps.policy == policy]
        e = episodes[episodes.policy == policy]
        if s.empty:
            continue
        # per-seed mean shortage, then mean+/-std across seeds
        per_seed_mean = s.groupby("seed")["gold_shortage_mbps"].mean()
        mean_short = float(per_seed_mean.mean())
        std_short = float(per_seed_mean.std(ddof=0))
        # tail percentiles, across-seeds mean of per-seed quantile
        p50 = float(s.groupby("seed")["gold_shortage_mbps"]
                    .quantile(0.50).mean())
        p90 = float(s.groupby("seed")["gold_shortage_mbps"]
                    .quantile(0.90).mean())
        p95 = float(s.groupby("seed")["gold_shortage_mbps"]
                    .quantile(0.95).mean())
        p99 = float(s.groupby("seed")["gold_shortage_mbps"]
                    .quantile(0.99).mean())
        mean_reward = float(e.groupby("seed")["reward_mean"].mean().mean()) if not e.empty else float("nan")
        prot = compute_protection_score(s)
        per_seed_prot = prot.groupby(s["seed"]).mean()
        mean_prot = float(per_seed_prot.mean())
        rows.append({
            "policy": policy,
            "n_seeds": int(s["seed"].nunique()),
            "mean_shortage_mbps": round(mean_short, 2),
            "std_shortage_mbps": round(std_short, 2),
            "p50_shortage_mbps": round(p50, 2),
            "p90_shortage_mbps": round(p90, 2),
            "p95_shortage_mbps": round(p95, 2),
            "p99_shortage_mbps": round(p99, 2),
            "mean_reward_per_step": round(mean_reward, 3),
            "mean_protection_score": round(mean_prot, 4),
        })
    df = pd.DataFrame(rows)

    # % improvement in mean shortage relative to Fixed MBR (baseline)
    if "Fixed MBR" in df["policy"].values:
        base = float(df.loc[df.policy == "Fixed MBR", "mean_shortage_mbps"].iloc[0])
        if base > 0:
            df["pct_improvement_vs_fixed"] = ((base - df["mean_shortage_mbps"]) / base * 100.0).round(2)
        else:
            df["pct_improvement_vs_fixed"] = 0.0
    else:
        df["pct_improvement_vs_fixed"] = float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ok] wrote {csv_path}")

    md_path = out_dir / "summary_table.md"
    _df_to_md(df, md_path, title="Summary Table — per-policy aggregate metrics",
              footnote=("Aggregates: mean ± std are cross-seed; "
                        "tail percentiles are the mean of per-seed quantiles; "
                        "`pct_improvement_vs_fixed` is the relative reduction "
                        "in mean shortage compared to Fixed MBR. "
                        "With n=3 seeds, treat all uncertainties as descriptive."))
    print(f"[ok] wrote {md_path}")
    return df


# ---------------------------------------------------------------------------
# Pairwise comparison table
# ---------------------------------------------------------------------------
def write_pairwise_table(steps: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Paired (per-seed) comparisons between policies on mean shortage."""
    per_seed = (steps.groupby(["policy", "seed"])["gold_shortage_mbps"]
                .mean().reset_index())
    piv = per_seed.pivot(index="seed", columns="policy",
                        values="gold_shortage_mbps")

    # The pairs we care about (treatment, control)
    pairs = [
        ("DQN-LS", "DQN"),
        ("DQN-LS", "Reactive"),
        ("DQN-LS", "Fixed MBR"),
        ("DQN", "Reactive"),
        ("DQN", "Fixed MBR"),
        ("Reactive", "Fixed MBR"),
    ]

    rows = []
    for treat, ctrl in pairs:
        if treat not in piv.columns or ctrl not in piv.columns:
            continue
        diff = piv[treat] - piv[ctrl]   # negative = treat protects Gold more
        diff = diff.dropna()
        if diff.empty:
            continue
        n = int(diff.size)
        mean_d = float(diff.mean())
        std_d = float(diff.std(ddof=0)) if n > 1 else 0.0
        # Cohen's d using paired std (pooled SD denominator); with n=3
        # this is descriptive only.
        cohen_d = (mean_d / std_d) if std_d > 0 else float("nan")
        n_treat_wins = int((diff < 0).sum())  # negative -> treat is better
        n_ties = int((diff == 0).sum())
        rows.append({
            "treatment": treat,
            "control": ctrl,
            "n_seeds": n,
            "mean_delta_mbps": round(mean_d, 2),
            "std_delta_mbps": round(std_d, 2),
            "cohen_d_paired": round(cohen_d, 3) if not np.isnan(cohen_d) else "NA",
            "seeds_treatment_wins": f"{n_treat_wins}/{n}",
            "seeds_tied": n_ties,
            "pct_improvement": round(
                (-mean_d / float(piv[ctrl].mean()) * 100.0) if piv[ctrl].mean() > 0 else 0.0,
                2,
            ),
        })
    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pairwise_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ok] wrote {csv_path}")

    md_path = out_dir / "pairwise_comparison.md"
    _df_to_md(
        df, md_path,
        title="Pairwise Comparison — paired-by-seed effect sizes on mean Gold shortage",
        footnote=(
            "delta = mean_shortage(treatment) - mean_shortage(control). "
            "Negative delta -> treatment protects Gold better. "
            "`seeds_treatment_wins = k/n` is the count of seeds where the "
            "treatment had lower mean shortage. "
            "Cohen's d is paired and *descriptive only* given n=3: "
            "with such a small sample, inferential tests (Wilcoxon paired, "
            "paired t) are underpowered and are deliberately not reported."
        ),
    )
    print(f"[ok] wrote {md_path}")
    return df


def _df_to_md(df: pd.DataFrame, path: Path, title: str, footnote: str) -> None:
    """Write a tidy GitHub-flavored Markdown table without external deps."""
    cols = list(df.columns)
    widths = [max(len(c), df[c].astype(str).map(len).max() if len(df) else len(c)) for c in cols]

    def row(cells):
        return "| " + " | ".join(
            f"{str(c):<{w}}" for c, w in zip(cells, widths)
        ) + " |"

    header = row(cols)
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = "\n".join(row([df.iloc[i, j] for j in range(len(cols))])
                     for i in range(len(df)))

    with open(path, "w") as f:
        f.write(f"### {title}\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        f.write(body + "\n\n")
        f.write(f"_{footnote}_\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("trex_4slice_noise_toolkit/runs/dqn_proxy_gold_100_200_1800s"),
        help="Root with baseline_fixed/, baseline_reactive/, dqn_vanilla/, dqn_latent_space/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Where to save the PDF figures and tables.",
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
          f"(policies: {sorted(steps['policy'].unique().tolist())})")
    print(f"[info] episodes rows = {len(episodes):,}")

    # 10 existing figures
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

    # 3 new figures
    fig11_guardrail_activity(steps, out_dir)
    fig12_tail_shortage(steps, out_dir)
    fig13_paired_episode_delta(episodes, out_dir)

    # 2 tables
    summary_df = write_summary_table(steps, episodes, out_dir)
    pairwise_df = write_pairwise_table(steps, out_dir)

    print("\n[summary table]")
    print(summary_df.to_string(index=False))
    print("\n[pairwise comparison]")
    print(pairwise_df.to_string(index=False))

    print(f"\n[done] wrote 13 figures and 2 tables to {out_dir}")


if __name__ == "__main__":
    main()

