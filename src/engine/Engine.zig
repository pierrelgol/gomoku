//! The engine as the rest of the program sees it: it owns the transposition
//! table and the search workers, runs a Lazy SMP search on a position, and
//! reports progress through a callback. It knows nothing about the UI.

const std = @import("std");
const Board = @import("Board.zig");
const eval = @import("eval.zig");
const nnue = @import("nnue.zig");
const TranspositionTable = @import("TranspositionTable.zig");
const search = @import("search.zig");
const movepick = @import("movepick.zig");

const Self = @This();
const Move = Board.Move;
const Worker = search.Worker;

pub const Limits = search.Limits;

allocator: std.mem.Allocator,
io: std.Io,
transposition_table: TranspositionTable,
workers: []*Worker,
shared: search.Shared,
reporter: Reporter,
uses_network: bool,
uses_policy: bool,

pub const Options = struct {
    /// Zero picks one thread per core, up to sixteen.
    thread_count: usize,
    hash_megabytes: usize,
    /// Evaluate with the embedded network when one is available.
    uses_network: bool,
    /// Order moves with the embedded learned policy when one is available.
    uses_policy: bool,

    pub const default: Options = .{
        .thread_count = 0,
        .hash_megabytes = 256,
        .uses_network = true,
        .uses_policy = true,
    };
};

pub const RootScore = struct {
    move: Move,
    score: i32,
};

pub const SearchInfo = struct {
    depth: i32,
    selective_depth: i32,
    nodes: u64,
    nodes_per_second: u64,
    time_ms: u64,
    score: i32,
    best_move: Move,
    principal_variation: [32]Move,
    principal_variation_length: usize,
    top_moves: [8]RootScore,
    top_move_count: usize,
    hashfull_permille: u32,
    thread_count: usize,
    is_final: bool,

    pub const empty: SearchInfo = .{
        .depth = 0,
        .selective_depth = 0,
        .nodes = 0,
        .nodes_per_second = 0,
        .time_ms = 0,
        .score = 0,
        .best_move = Board.no_move,
        .principal_variation = @splat(Board.no_move),
        .principal_variation_length = 0,
        .top_moves = @splat(.{ .move = Board.no_move, .score = 0 }),
        .top_move_count = 0,
        .hashfull_permille = 0,
        .thread_count = 1,
        .is_final = false,
    };

    /// A result decided without searching: a single move and its score.
    fn immediate(move: Move, score: i32, thread_count: usize) SearchInfo {
        var info: SearchInfo = .empty;
        info.best_move = move;
        info.score = score;
        info.principal_variation[0] = move;
        info.principal_variation_length = 1;
        info.thread_count = thread_count;
        info.is_final = true;
        return info;
    }
};

pub const Reporter = struct {
    maybe_context: ?*anyopaque,
    maybe_on_info: ?*const fn (maybe_context: ?*anyopaque, info: *const SearchInfo) void,

    pub const silent: Reporter = .{
        .maybe_context = null,
        .maybe_on_info = null,
    };
};

pub fn isNetworkAvailable(variant: Board.Rules.Variant) bool {
    return nnue.isAvailable(variant);
}

pub fn create(allocator: std.mem.Allocator, io: std.Io, options: Options) !*Self {
    const self = try allocator.create(Self);
    errdefer allocator.destroy(self);

    self.* = .{
        .allocator = allocator,
        .io = io,
        .transposition_table = TranspositionTable.init(allocator, options.hash_megabytes),
        .workers = &.{},
        .shared = undefined,
        .reporter = .silent,
        .uses_network = options.uses_network,
        .uses_policy = options.uses_policy,
    };
    errdefer self.transposition_table.deinit();

    // Each worker carries a board and an undo log, so on a small machine we
    // run with as many as fit rather than failing outright.
    var workers: std.ArrayList(*Worker) = .empty;
    defer workers.deinit(allocator);

    for (0..threadCount(options)) |id| {
        const worker = allocator.create(Worker) catch {
            break;
        };
        worker.init(id);
        workers.append(allocator, worker) catch {
            allocator.destroy(worker);
            break;
        };
    }

    errdefer {
        for (workers.items) |worker| {
            allocator.destroy(worker);
        }
    }

    if (workers.items.len == 0) {
        return error.OutOfMemory;
    }

    self.workers = try allocator.dupe(*Worker, workers.items);
    return self;
}

fn threadCount(options: Options) usize {
    if (options.thread_count != 0) {
        return options.thread_count;
    }

    const cpu_count = std.Thread.getCpuCount() catch 1;
    return std.math.clamp(cpu_count, 1, 16);
}

pub fn destroy(self: *Self) void {
    for (self.workers) |worker| {
        self.allocator.destroy(worker);
    }

    self.allocator.free(self.workers);
    self.transposition_table.deinit();
    self.allocator.destroy(self);
}

pub fn newGame(self: *Self) void {
    self.transposition_table.clear();

    for (self.workers) |worker| {
        worker.reset();
    }
}

/// Asks the running search to stop. Safe to call from any thread.
pub fn stop(self: *Self) void {
    self.shared.should_stop.store(true, .monotonic);
}

fn nodesPerSecond(nodes: u64, time_ms: u64) u64 {
    if (time_ms == 0) {
        return nodes * 1000;
    }

    return nodes * 1000 / time_ms;
}

fn byDescendingScore(_: void, left: search.RootMove, right: search.RootMove) bool {
    return left.score > right.score;
}

fn buildInfo(self: *Self, worker: *const Worker, is_final: bool) SearchInfo {
    var info: SearchInfo = .empty;
    info.depth = worker.completed_depth;
    info.selective_depth = worker.selective_depth;
    info.nodes = self.shared.totalNodes();
    info.time_ms = self.shared.elapsedMs();
    info.nodes_per_second = nodesPerSecond(info.nodes, info.time_ms);
    info.score = worker.best_score;
    info.best_move = worker.best_move;
    info.principal_variation_length = @min(worker.principal_variation_length, info.principal_variation.len);
    @memcpy(
        info.principal_variation[0..info.principal_variation_length],
        worker.principal_variation[0..info.principal_variation_length],
    );

    // A helper thread, or an iteration interrupted early, may not have filled
    // its root list yet; the main thread's last one is the next best thing.
    var root_worker = worker;

    if (worker.root_move_count == 0) {
        root_worker = self.workers[0];
    }

    var sorted_root_moves: [movepick.max_moves]search.RootMove = undefined;
    const root_move_count = root_worker.root_move_count;
    @memcpy(sorted_root_moves[0..root_move_count], root_worker.root_moves[0..root_move_count]);
    std.sort.pdq(search.RootMove, sorted_root_moves[0..root_move_count], {}, byDescendingScore);

    info.top_move_count = @min(root_move_count, info.top_moves.len);

    for (sorted_root_moves[0..info.top_move_count], 0..) |root_move, index| {
        info.top_moves[index] = .{ .move = root_move.move, .score = root_move.score };
    }

    info.hashfull_permille = self.transposition_table.hashfullPermille();
    info.thread_count = self.workers.len;
    info.is_final = is_final;
    return info;
}

fn onIteration(maybe_context: ?*anyopaque, worker: *Worker) void {
    const self: *Self = @ptrCast(@alignCast(maybe_context orelse unreachable));

    if (self.reporter.maybe_on_info) |on_info| {
        const info = self.buildInfo(worker, false);
        on_info(self.reporter.maybe_context, &info);
    }
}

fn runWorker(worker: *Worker) void {
    worker.iterativeDeepening();
}

/// Searches `position` for the side to move and blocks until done.
/// `pending_five` is the breakable five the opponent just made, or no_move.
pub fn think(self: *Self, position: *const Board, pending_five: Move, limits: Limits) SearchInfo {
    self.shared = .{
        .transposition_table = &self.transposition_table,
        .should_stop = .init(false),
        .deepest_completed_depth = .init(0),
        .has_proven_result = .init(false),
        .hard_limit_ms = .init(limits.hard_ms),
        .io = self.io,
        .start_time = std.Io.Clock.awake.now(self.io),
        .limits = limits,
        .workers = self.workers,
        .maybe_on_iteration = onIteration,
        .maybe_context = self,
    };
    self.transposition_table.newSearch();

    if (position.stone_count == 0) {
        return .immediate(Board.moveAt(9, 9), 0, self.workers.len);
    }

    for (self.workers) |worker| {
        self.prepareWorker(worker, position, pending_five);
    }

    // The search returns a mate score at the root without a move to go with
    // it, so wins on the spot are picked here instead.
    if (pending_five == Board.no_move) {
        const main_board = &self.workers[0].board;
        const threats = movepick.Threats.of(main_board);

        if (threats.own_capture_win or threats.own_five) {
            if (findImmediateWin(main_board)) |winning_move| {
                var info: SearchInfo = .immediate(winning_move, eval.mateIn(1), self.workers.len);
                info.depth = 1;
                info.time_ms = self.shared.elapsedMs();
                return info;
            }
        }
    }

    var maybe_threads: [64]?std.Thread = @splat(null);

    for (self.workers[1..], 1..) |worker, index| {
        if (index >= maybe_threads.len) {
            break;
        }

        maybe_threads[index] = std.Thread.spawn(.{ .stack_size = 8 << 20 }, runWorker, .{worker}) catch null;
    }

    self.workers[0].iterativeDeepening();
    self.stop();

    for (maybe_threads) |maybe_thread| {
        if (maybe_thread) |thread| {
            thread.join();
        }
    }

    var info = self.buildInfo(self.pickResultWorker(), true);

    if (info.best_move == Board.no_move) {
        info.best_move = self.fallbackMove(position);
    }

    if (info.principal_variation_length == 0) {
        info.principal_variation[0] = info.best_move;
        info.principal_variation_length = 1;
    }

    if (self.reporter.maybe_on_info) |on_info| {
        on_info(self.reporter.maybe_context, &info);
    }

    return info;
}

fn prepareWorker(self: *Self, worker: *Worker, position: *const Board, pending_five: Move) void {
    worker.shared = &self.shared;
    worker.board = position.*;
    worker.undo_log.reset();
    worker.board.maybe_undo_log = &worker.undo_log;
    worker.board.setUsesNetwork(self.uses_network);
    worker.board.setUsesPolicy(self.uses_policy);
    worker.ageHistory();

    for (&worker.stack) |*entry| {
        entry.pending_five = Board.no_move;
        entry.move = Board.no_move;
        entry.killers = .{ Board.no_move, Board.no_move };
        entry.pv_length = 0;
        entry.static_eval = 0;
    }

    // stack[0] is ply -1: what the opponent did just before the root.
    worker.stack[0].pending_five = pending_five;
    worker.stack[0].move = position.lastMove();
}

/// Mates come first; otherwise the deepest completed search wins, as long as
/// its score is not worse than what we already have.
fn pickResultWorker(self: *Self) *Worker {
    var best_worker = self.workers[0];

    for (self.workers[1..]) |worker| {
        if (worker.best_move == Board.no_move) {
            continue;
        }

        if (worker.best_score >= eval.mate_threshold and worker.best_score > best_worker.best_score) {
            best_worker = worker;
        } else if (worker.completed_depth > best_worker.completed_depth and worker.best_score >= best_worker.best_score and !eval.isMate(
            best_worker.best_score,
        )) {
            best_worker = worker;
        }
    }

    return best_worker;
}

fn findImmediateWin(board: *Board) ?Move {
    const color = board.side_to_move;
    const color_index = color.index();

    var candidates = board.candidates();

    while (candidates.next()) |move| {
        const record = &board.cell_records[move];
        const wins_by_capture = board.rules.captures and record.capture_counts[color_index] > 0 and
            board.captured_pairs[color_index] + 1 >= board.rules.capture_win_pairs;
        const completes_five = record.info[color_index].pattern4 == .a_five;

        if (!wins_by_capture and !completes_five) {
            continue;
        }

        board.makeMove(move);
        const wins = board.captured_pairs[color_index] >= board.rules.capture_win_pairs or
            (completes_five and !board.isFiveBreakable(move));
        board.undoMove();

        if (wins) {
            return move;
        }
    }

    return null;
}

/// Used when a search was stopped before finishing a single iteration: any
/// legal move beats returning nothing.
fn fallbackMove(self: *Self, position: *const Board) Move {
    const min_centre_distance = self.shared.limits.root_min_centre_distance;

    var list: movepick.MoveList = undefined;
    movepick.generate(position, &list, .all, movepick.Threats.of(position), null);

    if (min_centre_distance > 0) {
        search.restrictRootMoves(position, &list, min_centre_distance);
    }

    if (list.len > 0) {
        return list.selectBest(0);
    }

    for (0..Board.size) |y| {
        for (0..Board.size) |x| {
            const move = Board.moveAt(x, y);

            if (position.isLegal(move) and search.centreDistance(move) >= min_centre_distance) {
                return move;
            }
        }
    }

    return Board.no_move;
}
