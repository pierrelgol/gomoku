#!/bin/sh
# Generate a self-play dataset for one rule variant with the current engine
# (networks + policies embedded at build time).
#
#   train/gen_data.sh OUT_DIR VARIANT [GAMES] [NODES] [SEED] [-- extra zig build args]
#   VARIANT: standard | no_captures | no_double_three | no_captures_no_double_three
#
# e.g. on an aarch64 Linux box from a Mac:
#   train/gen_data.sh data/no_captures/gen1 no_captures 400000 30000 4 -- -Dtarget=aarch64-linux-gnu -Dcpu=neoverse_v2
# (then copy zig-out/bin/Gomoku to the box and run the printed command there).
#
# Output: OUT_DIR/part-NNN.bin, one 104-byte record per position (see
# src/tools/datagen.zig). Also installs zig-out/share/pattern_tables.bin,
# which the training scripts need (--tables).
set -e
out=${1:?usage: gen_data.sh OUT_DIR VARIANT [GAMES] [NODES] [SEED] [-- zig args]}; variant=${2:?variant}
games=${3:-100000}; nodes=${4:-30000}; seed=${5:-1}
shift $(( $# < 5 ? $# : 5 ))
[ "$1" = "--" ] && shift
case $variant in
  standard) rules="" ;;
  no_captures) rules="--no-captures" ;;
  no_double_three) rules="--no-double-three" ;;
  no_captures_no_double_three) rules="--no-captures --no-double-three" ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac
root=$(cd "$(dirname "$0")/.." && pwd)
(cd "$root" && zig build -Doptimize=fast "$@")
cmd="$root/zig-out/bin/Gomoku --gendata $rules --games $games --nodes $nodes --seed $seed --out $out"
case "$*" in
  *-Dtarget=*) echo "cross-compiled; run on the target machine:"; echo "  $cmd"; exit 0 ;;
esac
mkdir -p "$out"
exec $cmd
