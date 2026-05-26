#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
import numpy as np

ACTION_DESC = {
    0: "A00(G=100%,S=100%,B=100%)",
    1: "A01",
    2: "A02(G=100%,S=100%,B=70%)",
    3: "A03",
    4: "A04",
    5: "A05",
    6: "A06",
    7: "A07",
    8: "A08",
    9: "A09",
    10: "A10",
    11: "A11(G=85%,S=30%,B=30%)",
    12: "A12(G=80%,S=30%,B=30%)",
    13: "A13(G=70%,S=30%,B=30%)",
    14: "A14(G=60%,S=30%,B=30%)",
}

def fail(msg, failures):
    print(f"[FAIL] {msg}")
    failures.append(msg)

def ok(msg):
    print(f"[ OK ] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def pct(x):
    return 100.0 * float(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--loss-gold-threshold", type=float, default=0.10)
    ap.add_argument("--light-util-threshold", type=float, default=0.70)
    ap.add_argument("--noise-jump-mbps", type=float, default=20.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    failures = []

    print()
    print("== Basic file check ==")
    print(f"rows: {len(df)}")
    if len(df) < 60:
        fail("CSV has fewer than 60 samples", failures)
    else:
        ok("CSV has enough samples for basic validation")

    required = [
        "pfcp_sessions",
        "rx_gold_mbps", "rx_silver_mbps", "rx_bronze_mbps",
        "n6_total_mbps", "n6_utilization",
        "loss_gold", "loss_silver", "loss_bronze",
        "noise_offered_mbps", "noise_rx_mbps",
        "offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps",
        "dryrun_raw_action_id", "dryrun_action_id",
        "dryrun_guardrail_applied",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"missing required columns: {missing}", failures)
        print()
        print("Available columns:")
        print(list(df.columns))
        return 2

    if df[required].isna().any().any():
        fail("NaN detected in required columns", failures)
    else:
        ok("No NaN in required columns")

    print()
    print("== Operational safety ==")
    pfcp_bad = df[df["pfcp_sessions"] != 4]
    if len(pfcp_bad) == 0:
        ok("pfcp_sessions stayed at 4")
    else:
        fail(f"pfcp_sessions not equal to 4 in {len(pfcp_bad)} rows", failures)

    if (df[["rx_gold_mbps", "rx_silver_mbps", "rx_bronze_mbps"]] > 0).all().all():
        ok("RX per controlled slice is positive")
    else:
        fail("Some RX values are zero or negative", failures)

    if (df["n6_total_mbps"] > 0).all():
        ok("n6_total_mbps is positive")
    else:
        fail("n6_total_mbps has zero or negative values", failures)

    if "upf_access_drop_delta" in df.columns:
        max_drop = df["upf_access_drop_delta"].max()
        print(f"max upf_access_drop_delta: {max_drop}")
        if max_drop > 0:
            warn("UPF access drop delta is not always zero, inspect if this is expected")

    print()
    print("== Action distribution ==")
    print("Raw actions")
    print(df["dryrun_raw_action_id"].value_counts().sort_index())
    print()
    print("Safe actions")
    print(df["dryrun_action_id"].value_counts().sort_index())
    print()
    print("Guardrail")
    print(df["dryrun_guardrail_applied"].value_counts().sort_index())

    print()
    print("== Guardrail correctness ==")
    light = (df["n6_utilization"] < args.light_util_threshold) & (df["loss_gold"] < args.loss_gold_threshold)

    bad_light_final = df[light & df["dryrun_action_id"].isin([12, 13, 14])]
    if len(bad_light_final) == 0:
        ok("No final Gold-cutting action under light load")
    else:
        fail(f"Gold-cutting safe actions under light load: {len(bad_light_final)} rows", failures)
        print(bad_light_final[[
            "step", "loss_gold", "n6_utilization",
            "dryrun_raw_action_id", "dryrun_action_id",
            "dryrun_action_description",
        ]].head(20).to_string(index=False))

    bad_a12 = df[(df["dryrun_action_id"] == 12) & (df["loss_gold"] < args.loss_gold_threshold)]
    if len(bad_a12) == 0:
        ok("A12 only passed when loss_gold >= threshold")
    else:
        fail(f"A12 passed with loss_gold below threshold: {len(bad_a12)} rows", failures)
        print(bad_a12[[
            "step", "loss_gold", "n6_utilization",
            "dryrun_raw_action_id", "dryrun_action_id",
        ]].head(20).to_string(index=False))

    guarded = df[df["dryrun_guardrail_applied"] == 1]
    if len(guarded) > 0:
        if guarded["dryrun_raw_action_id"].isin([12, 13, 14]).all() and (guarded["dryrun_action_id"] == 2).all():
            ok("Guardrail converts unsafe Gold-cutting actions to A02")
        else:
            fail("Guardrail applied but conversion pattern is unexpected", failures)
    else:
        warn("Guardrail was never applied in this run")

    print()
    print("== Perturbation quality ==")
    noise = df["noise_offered_mbps"]
    noise_q10 = noise.quantile(0.10)
    noise_q50 = noise.quantile(0.50)
    noise_q90 = noise.quantile(0.90)
    noise_q95 = noise.quantile(0.95)
    noise_jump_count = noise.diff().abs().gt(args.noise_jump_mbps).sum()

    print(f"noise q10={noise_q10:.2f} Mbps, q50={noise_q50:.2f} Mbps, q90={noise_q90:.2f} Mbps, q95={noise_q95:.2f} Mbps")
    print(f"noise abrupt jumps > {args.noise_jump_mbps:.1f} Mbps: {int(noise_jump_count)}")

    if noise_q90 - noise_q10 >= 20:
        ok("Noise has meaningful amplitude variation")
    else:
        warn("Noise amplitude variation is small")

    if noise_jump_count > 0:
        ok("Noise has abrupt jumps")
    else:
        warn("No abrupt noise jumps detected by the current threshold")

    controlled = ["offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps"]
    print()
    print("Controlled slice offered traffic summary")
    print(df[controlled].describe().T[["min", "mean", "max", "std"]])

    corr = df[controlled].corr()
    print()
    print("Controlled slice correlation")
    print(corr)

    offdiag = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()
    if offdiag < 0.98:
        ok("Controlled slices are not perfectly synchronized")
    else:
        warn("Controlled slices are almost perfectly synchronized")

    print()
    print("== Spike and valley analysis ==")
    spike = df["noise_offered_mbps"] >= noise_q90
    valley = df["noise_offered_mbps"] <= noise_q10

    metrics = [
        "loss_gold", "loss_silver", "loss_bronze",
        "n6_utilization", "noise_offered_mbps",
        "offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps",
    ]

    print("Noise valleys")
    print(df[valley][metrics].mean())
    print()
    print("Noise spikes")
    print(df[spike][metrics].mean())

    spike_guard = df[spike]["dryrun_guardrail_applied"].mean() if spike.any() else 0
    valley_guard = df[valley]["dryrun_guardrail_applied"].mean() if valley.any() else 0
    print()
    print(f"guardrail rate in noise valleys: {pct(valley_guard):.2f}%")
    print(f"guardrail rate in noise spikes:  {pct(spike_guard):.2f}%")

    print()
    print("Safe action counts in noise spikes")
    print(df[spike]["dryrun_action_id"].value_counts().sort_index())

    print()
    print("Safe action counts in noise valleys")
    print(df[valley]["dryrun_action_id"].value_counts().sort_index())

    print()
    print("== Action context ==")
    print(df.groupby("dryrun_action_id")[metrics].mean())

    print()
    if failures:
        print("VALIDATION RESULT: FAIL")
        for m in failures:
            print(f" - {m}")
        return 1

    print("VALIDATION RESULT: PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
