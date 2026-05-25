#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/four_slices.env"

OUT_DIR="${1:-runs/bess_check_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

echo "Writing BESS/PFCP audit outputs to $OUT_DIR"

echo "== four-slice plan ==" | tee "$OUT_DIR/00_plan.txt"
cat "$HERE/four_slice_pfcp_plan.json" | tee -a "$OUT_DIR/00_plan.txt"

echo "== bessctl show pipeline ==" | tee "$OUT_DIR/01_bess_pipeline.txt"
if command -v "$BESSCTL" >/dev/null 2>&1; then
  "$BESSCTL" show pipeline 2>&1 | tee -a "$OUT_DIR/01_bess_pipeline.txt" || true
else
  echo "bessctl not found as $BESSCTL" | tee -a "$OUT_DIR/01_bess_pipeline.txt"
fi

echo "== bessctl show ports ==" | tee "$OUT_DIR/02_bess_ports.txt"
if command -v "$BESSCTL" >/dev/null 2>&1; then
  "$BESSCTL" show port "$N3_PORT_NAME" 2>&1 | tee -a "$OUT_DIR/02_bess_ports.txt" || true
  "$BESSCTL" show port "$N6_PORT_NAME" 2>&1 | tee -a "$OUT_DIR/02_bess_ports.txt" || true
fi

echo "== pfcp sessions ==" | tee "$OUT_DIR/03_pfcp_sessions.txt"
if command -v "$PFCPCTL" >/dev/null 2>&1; then
  "$PFCPCTL" service session list 2>&1 | tee -a "$OUT_DIR/03_pfcp_sessions.txt" || true
else
  echo "pfcpctl not found as $PFCPCTL" | tee -a "$OUT_DIR/03_pfcp_sessions.txt"
fi

echo "== grep TEIDs in collected outputs ==" | tee "$OUT_DIR/04_teid_grep.txt"
for teid in "$TEID_GOLD" "$TEID_SILVER" "$TEID_BRONZE" "$TEID_NOISE"; do
  echo "---- TEID $teid ----" | tee -a "$OUT_DIR/04_teid_grep.txt"
  grep -R "${teid}" "$OUT_DIR" 2>/dev/null | tee -a "$OUT_DIR/04_teid_grep.txt" || true
done

echo "Done. If TEID 31 does not appear, BESS/PFCP is not yet receiving the fourth noise slice."
