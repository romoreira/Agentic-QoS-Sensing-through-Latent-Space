#!/usr/bin/env python3
"""Optional TensorBoard writer with a no-op fallback."""
from __future__ import annotations

from typing import Optional


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def make_writer(logdir: Optional[str]):
    if not logdir:
        return NullWriter()
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=logdir)
    except Exception as exc:
        print(f"[WARN] TensorBoard disabled because SummaryWriter could not be imported: {exc}")
        return NullWriter()
