//! Headless tools (non-TUI mode). Progress is reported through std.Progress on
//! stderr; results are printed on stdout.
//!   --bench     : engine plays games against itself, every move is timed and
//!                 the depth reached recorded; prints per-move and summary stats.
//!   --selfplay  : plays N games and reports results (regression testing).

const std = @import("std");
const Engine = @import("../engine/Engine.zig");
const Game = @import("../engine/Game.zig");
const Board = @import("../engine/Board.zig");
const eval = @import("../engine/eval.zig");

pub const Options = struct {
    threads: usize = 0,
    soft_ms: u32 = 300,
    hard_ms: u32 = 450,
    games: usize = 2,
    moves: usize = 40,
    trace: bool = false,
    depth: i32 = 10,
    pos: ?usize = null,
};

fn printBoard(b: *const Board) void {
    std.debug.print(
        "   A B C D E F G H J K L M N O P Q R S T   side={s} caps={d}/{d}\n",
        .{ @tagName(b.side_to_move), b.captured_pairs[0], b.captured_pairs[1] },
    );

    for (0..Board.size) |y| {
        std.debug.print("{d:>2} ", .{19 - y});

        for (0..Board.size) |x| {
            const c = b.cellAt(Board.moveAt(x, y));
            const ch: u8 = if (c == 1) 'X' else if (c == 2) 'O' else '.';
            std.debug.print("{c} ", .{ch});
        }

        std.debug.print("\n", .{});
    }
}

fn traceInfo(_: ?*anyopaque, info: *const Engine.SearchInfo) void {
    std.debug.print(
        "    it depth {d}/{d} score {d} nodes {d} t {d}ms\n",
        .{ info.depth, info.selective_depth, info.score, info.nodes, info.time_ms },
    );
}

const openings = [_][]const [2]u8{
    &.{ .{ 9, 9 }, .{ 10, 10 }, .{ 9, 11 } },
    &.{ .{ 9, 9 }, .{ 10, 9 }, .{ 8, 10 } },
    &.{ .{ 9, 9 }, .{ 9, 8 }, .{ 11, 9 } },
    &.{ .{ 9, 9 }, .{ 11, 11 }, .{ 10, 8 } },
};

const Stats = struct {
    moves: usize = 0,
    total_ms: u64 = 0,
    max_ms: u64 = 0,
    min_depth: i32 = 1000,
    sum_depth: i64 = 0,
    nodes: u64 = 0,
    over: usize = 0,
};

fn moveStr(m: Board.Move, buf: []u8) []const u8 {
    if (m == Board.no_move) {
        return "--";
    }

    const col = "ABCDEFGHJKLMNOPQRST"[Board.columnOf(m)];
    return std.fmt.bufPrint(buf, "{c}{d}", .{ col, 19 - Board.rowOf(m) }) catch "??";
}

fn playGame(
    eng: *Engine,
    game: *Game,
    opening: []const [2]u8,
    opts: Options,
    stats: *Stats,
    out: *std.Io.Writer,
    node: std.Progress.Node,
    verbose: bool,
) !void {
    game.reset(.standard, .standard);
    eng.newGame();

    for (opening) |xy| {
        _ = game.play(Board.moveAt(xy[0], xy[1]));
    }

    const limits: Engine.Limits = .withTime(opts.soft_ms, opts.hard_ms);
    var n: usize = 0;

    while (n < opts.moves and !game.isOver()) : (n += 1) {
        const info = eng.think(&game.board, game.pendingFive(), limits);

        if (info.best_move == Board.no_move) {
            break;
        }

        const l = game.play(info.best_move);

        if (l != .ok) {
            try out.print("illegal engine move: {s}\n", .{l.message()});
            break;
        }

        stats.moves += 1;
        stats.total_ms += info.time_ms;
        stats.max_ms = @max(stats.max_ms, info.time_ms);

        if (!eval.isMate(info.score)) {
            stats.min_depth = @min(stats.min_depth, info.depth);
        }

        if (info.time_ms > 500) {
            stats.over += 1;
        }

        stats.sum_depth += info.depth;
        stats.nodes += info.nodes;

        if (verbose) {
            var b1: [8]u8 = undefined;
            try out.print(
                "{d:>3} {s:<4} {s:>4}  depth {d:>2}/{d:<3} score {d:>6}  nodes {d:>10}  nps {d:>9}  {d:>4} ms  caps {d}/{d}\n",
                .{
                    game.ply(),                   if (game.sideToMove() == .white) "B" else "W",
                    moveStr(info.best_move, &b1), info.depth,
                    info.selective_depth,         info.score,
                    info.nodes,                   info.nodes_per_second,
                    info.time_ms,                 game.board.captured_pairs[0],
                    game.board.captured_pairs[1],
                },
            );
            try out.flush();
        }

        node.completeOne();
    }

    if (verbose) {
        try out.print(
            "result: {s} winner={s}\n\n",
            .{ @tagName(game.outcome), if (game.maybe_winner) |w| @tagName(w) else "-" },
        );
        try out.flush();
    }
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, opts: Options) !void {
    var out_buf: [4096]u8 = undefined;
    var out_fw = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_fw.interface;

    const eng = try Engine.create(
        gpa,
        io,
        .{
            .thread_count = opts.threads,
            .hash_megabytes = Engine.Options.default.hash_megabytes,
            .uses_network = true,
            .uses_policy = true,
        },
    );
    defer eng.destroy();

    if (opts.trace) {
        eng.reporter = .{ .maybe_context = null, .maybe_on_info = traceInfo };
    }

    const game = try gpa.create(Game);
    defer gpa.destroy(game);

    const root = std.Progress.start(io, .{ .root_name = "bench", .estimated_total_items = opts.games });
    defer root.end();

    try out.print(
        "bench: {d} threads, soft {d} ms / hard {d} ms, {d} games x {d} moves\n",
        .{ eng.workers.len, opts.soft_ms, opts.hard_ms, opts.games, opts.moves },
    );
    var stats: Stats = .{};

    for (0..opts.games) |g| {
        const node = root.startFmt(opts.moves, "game {d}", .{g + 1});
        try playGame(
            eng,
            game,
            openings[g % openings.len],
            opts,
            &stats,
            out,
            node,
            true,
        );
        node.end();
        root.completeOne();
    }

    const avg = if (stats.moves > 0) stats.total_ms / stats.moves else 0;
    try out.print(
        "summary: {d} moves ({d} > 500ms), avg {d} ms, max {d} ms, min non-mate depth {d}, avg depth {d:.1}, avg nps {d}\n",
        .{
            stats.moves,
            stats.over,
            avg,
            stats.max_ms,
            stats.min_depth,
            if (stats.moves > 0) @as(f64, @floatFromInt(stats.sum_depth)) / @as(f64, @floatFromInt(stats.moves)) else 0,
            if (stats.total_ms > 0) stats.nodes * 1000 / stats.total_ms else 0,
        },
    );
    try out.flush();
}

pub fn selfplay(gpa: std.mem.Allocator, io: std.Io, opts: Options) !void {
    var out_buf: [4096]u8 = undefined;
    var out_fw = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_fw.interface;

    const eng = try Engine.create(
        gpa,
        io,
        .{
            .thread_count = opts.threads,
            .hash_megabytes = Engine.Options.default.hash_megabytes,
            .uses_network = true,
            .uses_policy = true,
        },
    );
    defer eng.destroy();
    const game = try gpa.create(Game);
    defer gpa.destroy(game);

    const root = std.Progress.start(io, .{ .root_name = "selfplay", .estimated_total_items = opts.games });
    defer root.end();
    var wins: [3]usize = .{ 0, 0, 0 };
    var stats: Stats = .{};
    var o2 = opts;
    o2.moves = 361;

    for (0..opts.games) |g| {
        const node = root.startFmt(0, "game {d}", .{g + 1});
        try playGame(
            eng,
            game,
            openings[g % openings.len],
            o2,
            &stats,
            out,
            node,
            false,
        );
        node.end();
        root.completeOne();

        if (game.maybe_winner) |w| {
            wins[w.index()] += 1;
        } else {
            wins[2] += 1;
        }

        try out.print(
            "game {d}: {s} by {s} in {d} plies\n",
            .{ g + 1, if (game.maybe_winner) |w| @tagName(w) else "draw", @tagName(game.outcome), game.ply() },
        );
        try out.flush();
    }

    try out.print(
        "black {d} / white {d} / other {d}; avg {d} ms/move\n",
        .{ wins[0], wins[1], wins[2], if (stats.moves > 0) stats.total_ms / stats.moves else 0 },
    );
    try out.flush();
}

/// Deterministic node-count benchmark: a fixed set of positions (generated by
/// a cheap single-threaded depth-4 self-play) is searched to a fixed depth.
/// Total nodes / time are comparable between engine versions.
pub fn fixedDepth(gpa: std.mem.Allocator, io: std.Io, opts: Options) !void {
    var out_buf: [4096]u8 = undefined;
    var out_fw = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_fw.interface;

    const gen = try Engine.create(
        gpa,
        io,
        .{ .thread_count = 1, .hash_megabytes = 16, .uses_network = true, .uses_policy = true },
    );
    defer gen.destroy();
    const eng = try Engine.create(
        gpa,
        io,
        .{
            .thread_count = opts.threads,
            .hash_megabytes = Engine.Options.default.hash_megabytes,
            .uses_network = true,
            .uses_policy = true,
        },
    );
    defer eng.destroy();
    const game = try gpa.create(Game);
    defer gpa.destroy(game);
    const positions = try gpa.alloc(Board, openings.len * 40);
    defer gpa.free(positions);
    const pending = try gpa.alloc(Board.Move, positions.len);
    defer gpa.free(pending);

    // The position set is frozen in assets/bench_positions.txt (one game prefix per
    // line, moves as "x,y"), generated on first use, so that it does not
    // change when the search changes.
    var np: usize = 0;
    const path = "assets/bench_positions.txt";
    const cwd = std.Io.Dir.cwd();

    if (cwd.readFileAlloc(io, path, gpa, .limited(1 << 20))) |data| {
        defer gpa.free(data);
        var lines = std.mem.tokenizeScalar(u8, data, '\n');

        while (lines.next()) |line| {
            if (np >= positions.len) {
                break;
            }

            game.reset(.standard, .standard);
            var toks = std.mem.tokenizeScalar(u8, line, ' ');

            while (toks.next()) |t| {
                var xy = std.mem.splitScalar(u8, t, ',');
                const x = std.fmt.parseInt(usize, xy.next() orelse "0", 10) catch 0;
                const y = std.fmt.parseInt(usize, xy.next() orelse "0", 10) catch 0;
                _ = game.play(Board.moveAt(x, y));
            }

            positions[np] = game.board;
            pending[np] = game.pendingFive();
            np += 1;
        }
    } else |_| {
        var file_buf: std.Io.Writer.Allocating = .init(gpa);
        defer file_buf.deinit();

        for (openings) |op| {
            game.reset(.standard, .standard);
            gen.newGame();

            for (op) |xy| {
                _ = game.play(Board.moveAt(xy[0], xy[1]));
            }

            var k: usize = 0;

            while (k < 40 and !game.isOver()) : (k += 1) {
                if (k % 2 == 0 and np < positions.len) {
                    positions[np] = game.board;
                    pending[np] = game.pendingFive();
                    np += 1;

                    for (game.board.playedMoves()) |mv| {
                        try file_buf.writer.print("{d},{d} ", .{ Board.columnOf(mv), Board.rowOf(mv) });
                    }

                    try file_buf.writer.writeByte('\n');
                }

                const info = gen.think(&game.board, game.pendingFive(), fixedDepthLimits(4, 100000));

                if (info.best_move == Board.no_move or game.play(info.best_move) != .ok) {
                    break;
                }
            }
        }

        cwd.writeFile(io, .{ .sub_path = path, .data = file_buf.written() }) catch {};
    }

    const root = std.Progress.start(io, .{ .root_name = "fixed-depth bench", .estimated_total_items = np });
    defer root.end();
    var total_nodes: u64 = 0;
    var total_ms: u64 = 0;
    var worst: [5]struct { i: usize, ms: u64 } = @splat(.{ .i = 0, .ms = 0 });

    if (opts.trace) {
        eng.reporter = .{ .maybe_context = null, .maybe_on_info = traceInfo };
    }

    for (positions[0..np], 0..) |*p, i| {
        if (opts.pos) |only| {
            if (only != i) {
                continue;
            }
        }

        if (opts.trace) {
            printBoard(p);
        }

        eng.newGame();
        const info = eng.think(p, pending[i], fixedDepthLimits(opts.depth, opts.hard_ms * 10));
        total_nodes += info.nodes;
        total_ms += info.time_ms;
        // keep the 5 slowest
        var j: usize = 0;

        while (j < worst.len) : (j += 1) {
            if (info.time_ms > worst[j].ms) {
                var k = worst.len - 1;

                while (k > j) : (k -= 1) {
                    worst[k] = worst[k - 1];
                }

                worst[j] = .{ .i = i, .ms = info.time_ms };
                break;
            }
        }

        root.completeOne();
    }

    try out.print(
        "fixed depth {d}: {d} positions, {d} nodes, {d} ms, {d} nps\n",
        .{ opts.depth, np, total_nodes, total_ms, if (total_ms > 0) total_nodes * 1000 / total_ms else 0 },
    );
    try out.print("slowest:", .{});

    for (worst) |w| {
        try out.print(" #{d}={d}ms", .{ w.i, w.ms });
    }

    try out.print("\n", .{});
    try out.flush();
}

fn fixedDepthLimits(depth: i32, hard_ms: u32) Engine.Limits {
    var limits: Engine.Limits = .withTime(100000, hard_ms);
    limits.min_depth = depth;
    limits.max_depth = depth;
    return limits;
}
