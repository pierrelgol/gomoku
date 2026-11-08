# Gomoku

A Gomoku AI with a terminal UI, written in Zig 0.17.1 for the 42 subject.

It plays on a 19×19 board with captures, the endgame capture rule and the
double-three ban. The engine is a multi-threaded alpha-beta search guided by a
small int8 neural network trained on self-play. It answers in under 500 ms and
searches 13 to 19 plies deep.

## Quick start

```sh
make          # builds ./Gomoku
./Gomoku      # play (terminal of at least 78×24)
make test     # run the tests
make bench    # time and depth on a fixed set of positions
```

There is nothing to install apart from Zig. libvaxis, the UI library, is
vendored in `vendor/` and wired in with `--fork`, so the build works offline.
Without the Makefile:

```sh
zig build --fork=vendor/libvaxis --fork=vendor/uucode -Doptimize=ReleaseFast
```

## Playing

The menu has three modes: **Player vs AI**, **Hotseat** (two players with an AI
hint) and **AI vs AI**. You can also set the AI's time per move, the opening
rule (Standard, Pro, Long Pro, Swap, Swap2) and turn captures or the
double-three ban on and off.

| Key | Action |
|---|---|
| arrows / mouse | move, Enter or click to play |
| `g` | hint |
| `u` / `r` | undo / redo |
| `s` / `o` | save / load |
| `d` | analysis view |
| `p` | pause (AI vs AI) |
| `q` | menu |
| `?` | help |

The side panel shows the AI timer, search depth, nodes/s, the score with a win
probability, the best line and the captured pairs.

## Rules

- Five or more stones in a row win.
- Flanking exactly two enemy stones (`X O O X`) captures them. Capturing 5
  pairs wins.
- A five that the opponent can break with a capture doesn't win yet: the
  opponent gets one move to break it.
- A move can't create two free-threes at once, unless it captures or completes
  a five.

## How it works

- **Board:** every empty cell keeps its line patterns for both colours, and a
  move updates them incrementally.
- **Search:** iterative-deepening PVS with a transposition table, the usual
  pruning and reductions, a threat-only quiescence search and Lazy SMP on up to
  16 threads.
- **Evaluation:** an int8 network over line windows and cell states. It is
  updated incrementally and embedded in the binary. Each rule set (captures
  on or off × double-three ban on or off) has its own network and policy,
  `assets/<variant>.nnue` / `.pol`; `-Dnets=DIR` embeds another set. The
  policy, a learned linear model, orders the moves.
- **Training:** `train/` holds the self-play data generation (`gen_data.sh`),
  the PyTorch trainers, the export step and the Elo evaluation (`eval.sh`).

Against the same engine with hand-written evaluation, the shipped network is
about **+209 Elo** at equal time (800 games).

## Layout

```
src/engine/   rules, board, search, evaluation network, policy
src/ui/       terminal UI
src/tools/    benchmark, self-play, matches, pattern-table generator
assets/       shipped networks and policies, one per rule set (embedded at build time),
              their exactness check and the benchmark positions
train/        training pipeline (PyTorch, CUDA)
vendor/       libvaxis + uucode
```
