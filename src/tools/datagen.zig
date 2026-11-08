//! Self-play training-data generator (`--gendata`).
//!
//! N worker threads each own a single-threaded engine and play games from
//! randomized openings with a fixed node budget per move. Every position
//! before an engine move is written as a fixed 104-byte `Record`
//! (little-endian) to `<out>/part-<thread>.bin`; the game result is filled in
//! when the game ends. The trainer (train/train.py) reads the same layout.

const std = @import("std");
const Engine = @import("../engine/Engine.zig");
const Game = @import("../engine/Game.zig");
const Board = @import("../engine/Board.zig");
const eval = @import("../engine/eval.zig");
const pattern = @import("../engine/pattern.zig");

pub const Record = extern struct {
    /// 2 bits per cell (0 empty, 1 black, 2 white), cell i = y*19+x at bits 2i
    cells: [91]u8,
    side: u8, // 0 black, 1 white (side to move)
    captured: [2]u8, // pairs captured by black / white
    result: i8, // final result from the side to move: 1 win, 0 draw, -1 loss
    flags: u8, // bit0: side to move must answer a breakable five
    score: i16, // search score (side to move)
    best: u16, // best move, y*19+x
    ply: u16,
    _pad: [2]u8 = .{ 0, 0 },
};

comptime {
    std.debug.assert(@sizeOf(Record) == 104);
}

pub const Options = struct {
    threads: usize = 0,
    games: usize = 1000,
    nodes: u64 = 30_000,
    out: []const u8 = "data",
    seed: u64 = 1,
    rules: Board.Rules,
};

const MAX_REC = 400;

const Shared = struct {
    next_game: std.atomic.Value(usize) = .init(0),
    games_done: std.atomic.Value(usize) = .init(0),
    positions: std.atomic.Value(usize) = .init(0),
    wins: [3]std.atomic.Value(usize) = .{ .init(0), .init(0), .init(0) },
    failed: std.atomic.Value(bool) = .init(false),
};

pub fn encode(b: *const Board, rec: *Record) void {
    @memset(&rec.cells, 0);

    for (0..Board.size) |y| {
        for (0..Board.size) |x| {
            const c = b.cellAt(Board.moveAt(x, y));
            const i = y * Board.size + x;
            rec.cells[i >> 2] |= c << @intCast(2 * (i & 3));
        }
    }

    rec.side = @backingInt(b.side_to_move);
    rec.captured = b.captured_pairs;
    rec.ply = b.ply;
}

fn toIndex(m: Board.Move) u16 {
    return @intCast(Board.rowOf(m) * Board.size + Board.columnOf(m));
}

/// Random opening: 1..max_random stones, the first within 3 of the centre,
/// the next ones among the candidate cells (radius 2 / lines 3).
pub fn randomOpening(game: *Game, rnd: std.Random) void {
    const n = rnd.intRangeLessThan(usize, 1, 9);
    var k: usize = 0;
    var tries: usize = 0;

    while (k < n and tries < 200 and !game.isOver()) : (tries += 1) {
        var m: Board.Move = undefined;

        if (game.board.stone_count == 0) {
            m = Board.moveAt(rnd.intRangeLessThan(usize, 6, 13), rnd.intRangeLessThan(usize, 6, 13));
        } else {
            var cands: [Board.padded_cell_count]Board.Move = undefined;
            var nc: usize = 0;
            var it = game.board.candidates();

            while (it.next()) |q| {
                cands[nc] = q;
                nc += 1;
            }

            if (nc == 0) {
                break;
            }

            m = cands[rnd.uintLessThan(usize, nc)];
        }

        if (game.play(m) == .ok) {
            k += 1;
        }
    }
}

fn worker(sh: *Shared, opts: *const Options, tid: usize, io: std.Io, gpa: std.mem.Allocator) void {
    workerImpl(sh, opts, tid, io, gpa) catch |e| {
        std.debug.print("datagen worker {d}: {s}\n", .{ tid, @errorName(e) });
        sh.failed.store(true, .monotonic);
    };
}

fn workerImpl(sh: *Shared, opts: *const Options, tid: usize, io: std.Io, gpa: std.mem.Allocator) !void {
    const eng = try Engine.create(
        gpa,
        io,
        .{ .thread_count = 1, .hash_megabytes = 32, .uses_network = true, .uses_policy = true },
    );
    defer eng.destroy();
    const game = try gpa.create(Game);
    defer gpa.destroy(game);
    const recs = try gpa.alloc(Record, MAX_REC);
    defer gpa.free(recs);

    var name_buf: [512]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "{s}/part-{d:0>3}.bin", .{ opts.out, tid });
    const file = try std.Io.Dir.cwd().createFile(io, name, .{ .truncate = false });
    defer file.close(io);
    const end = try file.length(io);
    var fbuf: [1 << 16]u8 = undefined;
    var fw = file.writer(io, &fbuf);
    try fw.seekTo(end);
    const w = &fw.interface;

    var prng = std.Random.DefaultPrng.init(opts.seed *% 0x9E3779B97F4A7C15 +% tid);
    const rnd = prng.random();
    var limits: Engine.Limits = .withTime(60_000, 60_000);
    limits.min_depth = 1;
    limits.max_nodes = opts.nodes;

    while (true) {
        const g = sh.next_game.fetchAdd(1, .monotonic);

        if (g >= opts.games) {
            break;
        }

        game.reset(opts.rules, .standard);
        eng.newGame();
        randomOpening(game, rnd);

        var n: usize = 0;

        while (!game.isOver() and n < MAX_REC) {
            const pending = game.pendingFive();
            const info = eng.think(&game.board, pending, limits);

            if (info.best_move == Board.no_move) {
                break;
            }

            var mv = info.best_move;

            // early-game diversity: sometimes play another near-best root move
            if (game.board.ply < 16 and !eval.isMate(
                info.score,
            ) and info.top_move_count > 1 and rnd.float(f32) < 0.35) {
                var pick: [8]Board.Move = undefined;
                var np: usize = 0;

                for (info.top_moves[0..info.top_move_count]) |t| {
                    if (t.move != Board.no_move and t.score >= info.score - 40) {
                        pick[np] = t.move;
                        np += 1;
                    }
                }

                if (np > 0) {
                    mv = pick[rnd.uintLessThan(usize, np)];
                }
            }

            const r = &recs[n];
            encode(&game.board, r);
            r.flags = @intFromBool(pending != Board.no_move);
            r.score = @intCast(std.math.clamp(info.score, -32000, 32000));
            r.best = toIndex(info.best_move);
            r.result = 0;
            r._pad = .{ 0, 0 };
            n += 1;

            if (game.play(mv) != .ok and game.play(info.best_move) != .ok) {
                break;
            }
        }

        const winner: ?Board.Color = if (game.isOver()) game.maybe_winner else null;

        for (recs[0..n]) |*r| {
            r.result = if (winner) |wc| (if (@backingInt(wc) == r.side) @as(i8, 1) else -1) else 0;
        }

        try w.writeAll(std.mem.sliceAsBytes(recs[0..n]));
        _ = sh.positions.fetchAdd(n, .monotonic);
        _ = sh.wins[if (winner) |wc| wc.index() else 2].fetchAdd(1, .monotonic);
        _ = sh.games_done.fetchAdd(1, .monotonic);
    }

    try w.flush();
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, opts: Options) !void {
    eval.init();
    try std.Io.Dir.cwd().createDirPath(io, opts.out);
    var n = opts.threads;

    if (n == 0) {
        n = std.Thread.getCpuCount() catch 1;
    }

    n = @min(n, 256);

    var sh: Shared = .{};
    const threads = try gpa.alloc(?std.Thread, n);
    defer gpa.free(threads);

    for (threads, 0..) |*t, i| {
        t.* = std.Thread.spawn(.{ .stack_size = 16 << 20 }, worker, .{ &sh, &opts, i, io, gpa }) catch null;
    }

    const root = std.Progress.start(io, .{ .root_name = "datagen", .estimated_total_items = opts.games });
    const start = std.Io.Clock.awake.now(io);

    while (sh.games_done.load(.monotonic) < opts.games and !sh.failed.load(.monotonic)) {
        root.setCompletedItems(sh.games_done.load(.monotonic));
        io.sleep(.fromMilliseconds(250), .awake) catch {};

        // every worker has exited (all games handed out, or spawn failures)?
        if (sh.next_game.load(.monotonic) >= opts.games + n) {
            break;
        }
    }

    for (threads) |*t| {
        if (t.*) |th| {
            th.join();
        }
    }

    root.end();

    const secs = @as(f64, @floatFromInt(start.untilNow(io, .awake).toNanoseconds())) / 1e9;
    const pos = sh.positions.load(.monotonic);
    std.debug.print(
        "datagen: {d} games, {d} positions in {d:.1}s ({d:.0} pos/s); black {d} / white {d} / draw {d}\n",
        .{
            sh.games_done.load(.monotonic), pos,                         secs,
            @as(
                f64,
                @floatFromInt(pos),
            ) / secs,
            sh.wins[0].load(.monotonic),    sh.wins[1].load(.monotonic), sh.wins[2].load(.monotonic),
        },
    );
}
