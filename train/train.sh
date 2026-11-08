#!/bin/sh
# Train the value network and the move-ordering policy of one rule variant
# on one or more dataset directories (GPU, CUDA). Produces checkpoints;
# export them with train/export.py.
#
#   train/train.sh OUT_PREFIX TABLES VARIANT DATA_DIR...
#   VARIANT: standard | no_captures | no_double_three | no_captures_no_double_three
#   e.g. train/train.sh runs/standard zig-out/share/pattern_tables.bin standard data/gen0 data/gen1
#   -> runs/standard.value.pt, runs/standard.policy.pt
#
# Value net: C=16 features, 32x32 int8 head, 24 epochs, quantisation-aware
# training for the last 4 (the exported integer network is the trained model).
set -e
out=${1:?usage: train.sh OUT_PREFIX TABLES VARIANT DATA_DIR...}; tables=${2:?tables}; variant=${3:?variant}; shift 3
here=$(cd "$(dirname "$0")" && pwd)
py=${PYTHON:-python3}
policy_flags=""
case $variant in
  standard|no_captures) ;;
  no_double_three|no_captures_no_double_three) policy_flags="--no-double-three" ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$out")"
"$py" "$here/train_value.py" --data "$@" --tables "$tables" --C 16 --H1 32 --H2 32 \
    --epochs 24 --qat-epochs 4 --out "$out.value.pt"
"$py" "$here/train_policy.py" --data "$@" --tables "$tables" $policy_flags --batch 16384 --epochs 4 --out "$out.policy.pt"
