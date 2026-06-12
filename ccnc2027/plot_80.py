#!/usr/bin/env python3
"""Plot embedder pretraining curves -> fig14_embedder_training.pdf
   Local: /home/ubuntu/Agentic-QoS-Sensing-through-Latent-Space/ccnc2027/plot_80.py"""
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/home/ubuntu/Agentic-QoS-Sensing-through-Latent-Space"
CSV = f"{BASE}/trex_4slice_noise_toolkit/runs/dqn_proxy_gold_100_200_1800s/embedder/pretrain_log.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fig14_embedder_training.pdf")

df = pd.read_csv(CSV)

fig, ax1 = plt.subplots(figsize=(3.5, 2.4))

ax1.plot(df["epoch"], df["train_loss"], color="tab:blue",
         lw=1.6, label="Train loss")
ax1.plot(df["epoch"], df["val_loss"], color="tab:blue", ls="--",
         lw=1.6, label="Val. loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy loss", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(df["epoch"], df["val_within1_acc"] * 100, color="tab:red",
         lw=1.6, label="Val. within-1 acc.")
ax2.set_ylabel("Within-1-bucket acc. (%)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(45, 90)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7, frameon=False)

ax1.grid(alpha=0.3, lw=0.5)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"saved {OUT}: final val_acc={df['val_within1_acc'].iloc[-1]*100:.1f}%")
