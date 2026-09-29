#!/usr/bin/env bash
# Check your qubit.py before submitting. Runs the same six stages as the grader,
# with the public tests. The grader adds hidden tests on the same functions.
#
#     ./selfcheck.sh                  # checks starter/qubit.py
#     ./selfcheck.sh AIE25001.py      # checks a specific file
set -uo pipefail

# Resolve a relative path before changing directory, so the file you name is the
# file that gets checked.
if [ $# -ge 1 ]; then
  case "$1" in /*) SRC="$1" ;; *) SRC="$PWD/$1" ;; esac
fi
cd "$(dirname "$0")"

SRC="${SRC:-starter/qubit.py}"
[ -f "$SRC" ] || SRC="qubit.py"
[ -f "$SRC" ] || { echo "no qubit.py found"; exit 1; }
python3 -c "import numpy" 2>/dev/null || { echo "NumPy is not installed: pip install numpy"; exit 1; }

rc=0
for stage in states probabilities gates measure tensor expectation; do
  echo "==> $stage"
  python3 tests/public.py "$SRC" --stage "$stage" || rc=1
  echo
done
exit $rc
