#!/usr/bin/env bash
# Serial exact protocol; use on an allocated CUDA node. No scheduler submission.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: REPRO_PYTHON=python bash run_ablation.sh ASSET_ROOT NEW_WORK_DIR" >&2
  exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPRO_PYTHON=${REPRO_PYTHON:-python}
# Keep arguments as arrays; paths containing spaces are supported.
BASE=("$REPRO_PYTHON" "$SCRIPT_DIR/reproduce.py" ablation)
COMMON=(--assets "$1" --work-dir "$2")
for phase in init prepare preflight; do "${BASE[@]}" "$phase" "${COMMON[@]}"; done
for i in {0..5}; do "${BASE[@]}" screen --index "$i" "${COMMON[@]}"; done
"${BASE[@]}" select "${COMMON[@]}"
for i in {0..11}; do "${BASE[@]}" train --index "$i" "${COMMON[@]}"; done
"${BASE[@]}" freeze-test "${COMMON[@]}"
for i in {0..13}; do "${BASE[@]}" test --index "$i" "${COMMON[@]}"; done
"${BASE[@]}" collect "${COMMON[@]}"
