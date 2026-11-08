//! Engine-vs-engine match (`--match`): network evaluation (B) against the
//! classical evaluation (A) at equal time per move. Game pairs share a random
//! opening and swap colours. Runs many single-threaded games in parallel.
//! Also `--nncheck file`: print the network eval of raw records (to compare
//! with the trainer's reference values).

const std = @import("std");
const Engine = @import("../engine/Engine.zig");
const Game = @import("../engine/Game.zig");
const Board = @import("../engine/Board.zig");
const eval = @import("../engine/eval.zig");
const datagen = @import("datagen.zig");

pub const Options = struct {
    threads: usize = 0,
    pairs: usize = 100,
    soft_ms: u32 = 100,
    hard_ms: u32 = 150,
    seed: u64 = 7,
    /// > 0: equal node budget per move instead of equal time (measures the
    /// evaluation itself, independent of its speed)
    nodes: u64 = 0,
    /// engine configurations (defaults: A classical, B network; hand-made ordering)
    a_net: bool = false,
    a_pol: bool = false,
    b_net: bool = true,
    b_pol: bool = false,
    rules: Board.Rules,
};

const Shared = struct {
    next: std.atomic.Value(usize) = .init(0),
    done: std.atomic.Value(usize) = .init(0),
    /// results for engine B: win / draw / loss
    wdl: [3]std.atomic.Value(usize) = .{ .init(0), .init(0), .init(0) },
    /// B wins as black / as white
    b_black_wins: std.atomic.Value(usize) = .init(0),
    b_white_wins: std.atomic.Value(usize) = .init(0),
    plies: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
};

fn worker(sh: *Shared, opts: *const Options, io: std.Io, gpa: std.mem.Allocator) void {
    workerImpl(sh, opts, io, gpa) catch |e| {
        std.debug.print("match worker: {s}\n", .{@errorName(e)});
        sh.failed.store(true, .monotonic);
    };
}

fn workerImpl(sh: *Shared, opts: *const Options, io: std.Io, gpa: std.mem.Allocator) !void {
    const ea = try Engine.create(
        gpa,
        io,
        .{ .thread_count = 1, .hash_megabytes = 32, .uses_network = opts.a_net, .uses_policy = opts.a_pol },
    );
    defer ea.destroy();
    const eb = try Engine.create(
        gpa,
        io,
        .{ .thread_count = 1, .hash_megabytes = 32, .uses_network = opts.b_net, .uses_policy = opts.b_pol },
    );
    defer eb.destroy();
    const game = try gpa.create(Game);
    defer gpa.destroy(game);
    var limits: Engine.Limits = .withTime(opts.soft_ms, opts.hard_ms);

    if (opts.nodes > 0) {
        limits = .withTime(600_000, 600_000);
        limits.min_depth = 1;
        limits.max_nodes = opts.nodes;
    }

    while (true) {
        const p = sh.next.fetchAdd(1, .monotonic);

        if (p >= opts.pairs) {
            break;
        }

        for (0..2) |leg| {
            var prng = std.Random.DefaultPrng.init(opts.seed *% 0x9E3779B97F4A7C15 +% p);
            game.reset(opts.rules, .standard);
            datagen.randomOpening(game, prng.random());
            ea.newGame();
            eb.newGame();
            // leg 0: B plays black; leg 1: B plays white
            const b_color: Board.Color = if (leg == 0) .black else .white;

            while (!game.isOver() and game.ply() < 361) {
                const eng = if (game.sideToMove() == b_color) eb else ea;
                const info = eng.think(&game.board, game.pendingFive(), limits);

                if (info.best_move == Board.no_move or game.play(info.best_move) != .ok) {
                    break;
                }
            }

            _ = sh.plies.fetchAdd(game.ply(), .monotonic);

            if (game.isOver() and game.maybe_winner != null) {
                const w = game.maybe_winner.?;

                if (w == b_color) {
                    _ = sh.wdl[0].fetchAdd(1, .monotonic);
                    _ = (if (w == .black) &sh.b_black_wins else &sh.b_white_wins).fetchAdd(1, .monotonic);
                } else {
                    _ = sh.wdl[2].fetchAdd(1, .monotonic);
                }
            } else {
                _ = sh.wdl[1].fetchAdd(1, .monotonic);
            }

            _ = sh.done.fetchAdd(1, .monotonic);
        }
    }
}

fn elo(score: f64) f64 {
    const s = std.math.clamp(score, 1e-4, 1 - 1e-4);
    return -400.0 * std.math.log10(1.0 / s - 1.0);
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, opts: Options) !void {
    eval.init();

    if (!Engine.isNetworkAvailable(opts.rules.variant())) {
        std.debug.print(
            "match: no network embedded for the {s} rules (-Dnets=dir)\n",
            .{@tagName(opts.rules.variant())},
        );
        return;
    }

    var n = opts.threads;

    if (n == 0) {
        n = std.Thread.getCpuCount() catch 1;
    }

    n = @min(n, 256);
    var sh: Shared = .{};
    const threads = try gpa.alloc(?std.Thread, n);
    defer gpa.free(threads);

    for (threads) |*t| {
        t.* = std.Thread.spawn(.{ .stack_size = 16 << 20 }, worker, .{ &sh, &opts, io, gpa }) catch null;
    }

    const total = 2 * opts.pairs;
    const root = std.Progress.start(io, .{ .root_name = "match net vs classic", .estimated_total_items = total });

    while (sh.done.load(.monotonic) < total and !sh.failed.load(.monotonic)) {
        root.setCompletedItems(sh.done.load(.monotonic));
        io.sleep(.fromMilliseconds(250), .awake) catch {};

        if (sh.next.load(.monotonic) >= opts.pairs + n) {
            break;
        }
    }

    for (threads) |*t| {
        if (t.*) |th| {
            th.join();
        }
    }

    root.end();

    const w: f64 = @floatFromInt(sh.wdl[0].load(.monotonic));
    const d: f64 = @floatFromInt(sh.wdl[1].load(.monotonic));
    const l: f64 = @floatFromInt(sh.wdl[2].load(.monotonic));
    const games = w + d + l;
    const score = (w + d / 2) / @max(games, 1);
    // standard error of the mean score -> 95% Elo interval
    const var_s = (w * (1 - score) * (1 - score) + d * (0.5 - score) * (0.5 - score) + l * score * score) / @max(
        games,
        1,
    );
    const se = @sqrt(var_s / @max(games, 1));

    if (opts.nodes > 0) {
        std.debug.print("[equal nodes: {d} per move] ", .{opts.nodes});
    }

    std.debug.print("B (net {} pol {}) vs A (net {} pol {}): ", .{ opts.b_net, opts.b_pol, opts.a_net, opts.a_pol });
    std.debug.print(
        "{d}/{d} ms per move, {d} games: +{d} ={d} -{d}  score {d:.3}  Elo {d:.0} [{d:.0}, {d:.0}]  (B wins as black {d}, as white {d}; avg {d} plies)\n",
        .{
            opts.soft_ms,           opts.hard_ms,                     games,                            w, d, l, score,
            elo(
                score,
            ),
            elo(
                score - 1.96 * se,
            ),
            elo(score + 1.96 * se), sh.b_black_wins.load(.monotonic), sh.b_white_wins.load(.monotonic),
            sh.plies.load(.monotonic) / @max(
                @as(usize, @intFromFloat(games)),
                1,
            ),
        },
    );
}

/// Print the network evaluation of each record of a datagen file.
pub fn nncheck(gpa: std.mem.Allocator, io: std.Io, path: []const u8, rules: Board.Rules) !void {
    if (!Engine.isNetworkAvailable(rules.variant())) {
        std.debug.print("nncheck: no network embedded\n", .{});
        return;
    }

    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(data);
    const b = try gpa.create(Board);
    defer gpa.destroy(b);
    const n = data.len / @sizeOf(datagen.Record);

    for (0..n) |i| {
        var r: datagen.Record = undefined;
        @memcpy(std.mem.asBytes(&r), data[i * @sizeOf(datagen.Record) ..][0..@sizeOf(datagen.Record)]);
        var cells: [Board.cell_count]u8 = undefined;

        for (&cells, 0..) |*c, k| {
            c.* = (r.cells[k >> 2] >> @intCast(2 * (k & 3))) & 3;
        }

        b.setupPosition(rules, &cells, @fromBackingInt(@intCast(r.side)), r.captured);
        b.setUsesNetwork(true);
        std.debug.print("{d} {d}\n", .{ i, eval.evaluate(b) });
    }
}

/// Move-ordering accuracy on datagen records: rank of the search's best move
/// among the generated moves (no history), hand-made vs learned ordering.
pub fn polcheck(gpa: std.mem.Allocator, io: std.Io, path: []const u8, limit: usize, rules: Board.Rules) !void {
    const movepick = @import("../engine/movepick.zig");
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 31));
    defer gpa.free(data);
    const b = try gpa.create(Board);
    defer gpa.destroy(b);
    const list = try gpa.create(movepick.MoveList);
    defer gpa.destroy(list);
    const n = @min(data.len / @sizeOf(datagen.Record), if (limit == 0) std.math.maxInt(usize) else limit);
    var hit: [2][3]usize = @splat(@splat(0));
    var seen: usize = 0;
    var ce: [2]f64 = .{ 0, 0 };

    for (0..n) |i| {
        var r: datagen.Record = undefined;
        @memcpy(std.mem.asBytes(&r), data[i * @sizeOf(datagen.Record) ..][0..@sizeOf(datagen.Record)]);
        var cells: [Board.cell_count]u8 = undefined;

        for (&cells, 0..) |*c, k| {
            c.* = (r.cells[k >> 2] >> @intCast(2 * (k & 3))) & 3;
        }

        b.setupPosition(rules, &cells, @fromBackingInt(@intCast(r.side)), r.captured);
        const best: Board.Move = @intCast(Board.moveAt(r.best % 19, r.best / 19));

        if (r.best >= 361) {
            continue;
        }

        var counted = false;

        for (0..2) |v| {
            b.setUsesPolicy(v == 1);

            if (v == 1 and !b.uses_policy) {
                return error.NoPolicyEmbedded;
            }

            movepick.generate(b, list, .all, movepick.Threats.of(b), null);
            var bs: ?i32 = null;

            for (0..list.len) |k| {
                if (list.moves[k] == best) {
                    bs = list.scores[k];
                }
            }

            const s0 = bs orelse {
                continue;
            };

            if (!counted) {
                seen += 1;
            }

            counted = true;
            var above: usize = 0;
            var lse: f64 = 0;

            for (0..list.len) |k| {
                if (list.scores[k] > s0 or (list.scores[k] == s0 and list.moves[k] < best)) {
                    above += 1;
                }

                if (v == 1) {
                    lse += @exp(
                        @as(f64, @floatFromInt(list.scores[k] - s0)) / (256.0 * @as(
                            f64,
                            @floatFromInt(policy_mod.tuning.score_multiplier),
                        )),
                    );
                }
            }

            if (v == 1) {
                ce[1] += @log(lse);
            }

            if (above < 1) {
                hit[v][0] += 1;
            }

            if (above < 3) {
                hit[v][1] += 1;
            }

            if (above < 5) {
                hit[v][2] += 1;
            }
        }
    }

    const f: f64 = @floatFromInt(@max(seen, 1));

    for (0..2) |v| {
        std.debug.print("{s}: top1 {d:.3} top3 {d:.3} top5 {d:.3}\n", .{
            if (v == 0) "hand-made" else "learned  ",
            @as(f64, @floatFromInt(hit[v][0])) / f,
            @as(f64, @floatFromInt(hit[v][1])) / f,
            @as(f64, @floatFromInt(hit[v][2])) / f,
        });
    }

    std.debug.print("learned ce {d:.4} over {d} positions\n", .{ ce[1] / f, seen });
}

const policy_mod = @import("../engine/policy.zig");
