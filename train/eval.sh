#!/bin/sh
# Evaluate the exported network and policy of one rule variant in the engine.
#
#   train/eval.sh NETS_DIR VARIANT [PAIRS]
#   NETS_DIR holds VARIANT.nnue (+ .check / .check.bin from export.py) and
#   optionally VARIANT.pol; the other variants come from the same directory.
#
# 1. builds the engine with the files embedded (@embedFile, -Dnets),
# 2. checks bit-exactness against the exporter's integer reference: must
#    print "0/64 mismatches",
# 3. move-ordering accuracy of the policy (--polcheck) on the check records,
# 4. matches against the classical engine at equal time (100 ms) and at an
#    equal node budget (20k nodes/move), PAIRS game pairs each, with Elo and
#    its 95% interval.
set -e
dir=${1:?usage: eval.sh NETS_DIR VARIANT [PAIRS]}; variant=${2:?variant}; pairs=${3:-400}
case $variant in
  standard) rules="" ;;
  no_captures) rules="--no-captures" ;;
  no_double_three) rules="--no-double-three" ;;
  no_captures_no_double_three) rules="--no-captures --no-double-three" ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac
root=$(cd "$(dirname "$0")/.." && pwd)
net="$dir/$variant.nnue"
rel=$(python3 -c 'import os,sys; print(os.path.relpath(os.path.abspath(sys.argv[1]), sys.argv[2]))' "$dir" "$root")
pre=$(mktemp -d)
(cd "$root" && zig build -Doptimize=fast -Dnets="$rel" --prefix "$pre")
bin="$pre/bin/Gomoku"
mm=$("$bin" --nncheck "$net.check.bin" $rules 2>&1 | paste - "$net.check" | awk '{ if ($2 != $4) b++ } END { print b + 0 }')
echo "exactness: $mm/64 mismatches"
polflag=""
if [ -f "$dir/$variant.pol" ]; then
  "$bin" --polcheck "$net.check.bin" $rules
  polflag="--b-pol"
fi
"$bin" --match $rules $polflag --games "$pairs" --time 100 2>&1 | tail -1
"$bin" --match $rules $polflag --games "$pairs" --match-nodes 20000 2>&1 | tail -1
rm -rf "$pre"
