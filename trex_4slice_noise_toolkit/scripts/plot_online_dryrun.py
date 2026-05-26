#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def save_line(df, x, cols, title, ylabel, out):
    plt.figure(figsize=(12, 4))
    for c in cols:
        plt.plot(df[x], df[c], label=c)
    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x = "step"

    save_line(
        df, x,
        ["offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps", "noise_offered_mbps"],
        "Offered traffic per slice",
        "Mbps",
        outdir / "01_offered_traffic.png",
    )

    save_line(
        df, x,
        ["rx_gold_mbps", "rx_silver_mbps", "rx_bronze_mbps", "noise_rx_mbps"],
        "Received traffic per slice",
        "Mbps",
        outdir / "02_rx_traffic.png",
    )

    save_line(
        df, x,
        ["loss_gold", "loss_silver", "loss_bronze"],
        "Loss per controlled slice",
        "packet loss ratio",
        outdir / "03_loss_per_slice.png",
    )

    save_line(
        df, x,
        ["dryrun_raw_action_id", "dryrun_action_id", "dryrun_guardrail_applied"],
        "Raw DQN action, safe action, and guardrail",
        "action id or flag",
        outdir / "04_actions_guardrail.png",
    )

    save_line(
        df, x,
        ["dryrun_intended_mbr_gold_kbps", "dryrun_intended_mbr_silver_kbps", "dryrun_intended_mbr_bronze_kbps"],
        "Intended MBR after safety guardrail",
        "kbps",
        outdir / "05_intended_mbr.png",
    )

    save_line(
        df, x,
        ["n6_utilization", "loss_gold", "dryrun_guardrail_applied"],
        "N6 utilization, Gold loss, and guardrail events",
        "ratio or flag",
        outdir / "06_n6_loss_guardrail.png",
    )

    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
