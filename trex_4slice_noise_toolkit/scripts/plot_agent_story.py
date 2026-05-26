#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ACTION_LABELS = {
    0: "A00  keep all nominal",
    2: "A02  reduce Bronze only",
    12: "A12  moderate Gold cut",
    13: "A13  stronger Gold cut",
    14: "A14  strongest Gold cut",
}


def guardrail_segments(flag_series):
    segs = []
    start = None
    vals = flag_series.fillna(0).astype(int).tolist()
    for i, v in enumerate(vals):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(vals) - 1))
    return segs


def add_guardrail_shading(ax, x, segs, label_once=False):
    first = True
    for a, b in segs:
        lbl = "Guardrail active" if (label_once and first) else None
        ax.axvspan(x.iloc[a], x.iloc[b], alpha=0.18, label=lbl)
        first = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    x = df["step"] if "step" in df.columns else pd.Series(range(len(df)))
    segs = guardrail_segments(df["dryrun_guardrail_applied"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1
    ax = axes[0]
    ax.plot(x, df["offered_gold_mbps"], label="Gold offered traffic")
    ax.plot(x, df["offered_silver_mbps"], label="Silver offered traffic")
    ax.plot(x, df["offered_bronze_mbps"], label="Bronze offered traffic")
    ax.plot(x, df["noise_offered_mbps"], label="Noise slice offered traffic")
    add_guardrail_shading(ax, x, segs, label_once=True)
    ax.set_ylabel("Mbps")
    ax.legend(loc="upper right")
    ax.set_title("Traffic perturbation and agent reaction")

    # Panel 2
    ax = axes[1]
    ax.plot(x, df["loss_gold"], label="Gold packet loss")
    ax.plot(x, df["n6_utilization"], label="N6 utilization")
    ax.axhline(0.10, linestyle="--", label="Gold loss threshold 0.10")
    add_guardrail_shading(ax, x, segs)
    ax.set_ylabel("Ratio")
    ax.legend(loc="upper right")

    # Panel 3
    ax = axes[2]
    ax.step(x, df["dryrun_raw_action_id"], where="post", label="Raw DQN action")
    ax.step(x, df["dryrun_action_id"], where="post", label="Safe action after guardrail")
    add_guardrail_shading(ax, x, segs)
    ax.set_ylabel("Action ID")
    ax.set_yticks([0, 2, 12, 13, 14])
    ax.set_yticklabels([
        "A00",
        "A02",
        "A12",
        "A13",
        "A14",
    ])
    ax.legend(loc="upper right")

    # Panel 4
    ax = axes[3]
    ax.step(x, df["dryrun_intended_mbr_gold_kbps"], where="post", label="Gold intended MBR")
    ax.step(x, df["dryrun_intended_mbr_silver_kbps"], where="post", label="Silver intended MBR")
    ax.step(x, df["dryrun_intended_mbr_bronze_kbps"], where="post", label="Bronze intended MBR")
    add_guardrail_shading(ax, x, segs)
    ax.set_ylabel("kbps")
    ax.set_xlabel("Time step")
    ax.legend(loc="upper right")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"Wrote figure to {args.out}")

    # Also print a compact action legend for the terminal
    print("\nAction legend")
    for k in [0, 2, 12, 13, 14]:
        print(f"  {k:>2}  {ACTION_LABELS[k]}")


if __name__ == "__main__":
    main()
