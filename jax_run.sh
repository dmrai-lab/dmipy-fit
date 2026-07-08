#!/usr/bin/env bash
# Run any command in the dmipy JAX GPU environment.
# Usage: ./jax_run.sh python dmipy/jax/benchmarks/benchmark_real_data.py --quick
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
NV_BASE="$REPO/.venv/lib/python3.10/site-packages/nvidia"
for d in "$NV_BASE"/*/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export PATH="$REPO/.venv/bin:$PATH"
exec "$@"
