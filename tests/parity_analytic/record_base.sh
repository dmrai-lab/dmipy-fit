#!/usr/bin/env bash
# Records, on this machine, the outputs of the base revision of the repository: builds in a virtual environment what
# that revision pins (its constraints, the dmipy packages its workflow names by commit) and runs its harness `record`,
# which writes the reference to BASE_DIR/tests/parity_analytic/record.
#
# usage: record_base.sh BASE_DIR VENV_DIR
set -euo pipefail

base=$(realpath "$1")
venv=$2
workflow="$base/.github/workflows/tests.yml"

pinned() {
    sed -n "s/^  $1: *\([0-9a-f]\{40\}\).*/\1/p" "$workflow" | head -n 1
}

requirements=()
for name in sim replay; do
    variable="DMIPY_${name^^}_REF"
    ref=$(pinned "$variable")
    [ -n "$ref" ] || { echo "$variable is not set in $workflow" >&2; exit 1; }
    requirements+=("dmipy-$name @ git+https://github.com/dmrai-lab/dmipy-$name@$ref")
done
analytic=$(pinned DMIPY_ANALYTIC_REF)                      # the revisions before dmipy-analytic do not pin it
if [ -n "$analytic" ]; then
    requirements+=("dmipy-analytic[special,jax] @ git+https://github.com/dmrai-lab/dmipy-analytic@$analytic")
fi

python -m venv "$venv"
"$venv/bin/python" -m pip install -q --upgrade pip
"$venv/bin/pip" install -q -c "$base/tests/parity_analytic/constraints.txt" "${requirements[@]}" \
    "jax[cpu]" jaxopt scipy packaging
"$venv/bin/pip" install -q --no-deps -e "$base"
cd "$base"
"$venv/bin/python" tests/parity_analytic/harness.py record
