//! Alpha-beta search in negamax form.
//!
//! The shape is a classic modern engine: principal variation search under
//! iterative deepening with aspiration windows, a transposition table shared
//! by all threads (Lazy SMP), and the usual pruning family (mate distance,
//! reverse futility, null move, late-move pruning and reductions, internal
//! iterative reduction).
//!
//! What is specific to Gomoku is how forcing the game is. Move generation
//! narrows the move set whenever someone has a four or an open three, and the
//! quiescence search is a VCF search (victory by continuous fours): at the
//! horizon the attacker may keep playing fours, since each one has a single
//! forced answer, so tactical sequences get resolved instead of being cut off.
//! The capture rules are handled exactly inside the tree: capture wins, fives
//! that can be broken by a capture, and the double-three ban.

const std = @import("std");
const Board = @import("Board.zig");
const eval = @import("eval.zig");
const TranspositionTable = @import("TranspositionTable.zig");
const policy = @import("policy.zig");
const movepick = @import("movepick.zig");
const pattern = @import("pattern.zig");

const Move = Board.Move;
const Color = Board.Color;
const MoveList = movepick.MoveList;
const Threats = movepick.Threats;

pub const max_ply = 96;
const max_quiescence_depth = 4;

const infinite_value = eval.infinite_value;

/// Late-move reduction by depth and move number: 0.4 + ln(depth) * ln(move) / 1.9.
const late_move_reductions: [64][64]u8 = blk: {
    @setEvalBranchQuota(100_000);

    var reductions: [64][64]u8 = @splat(@splat(0));

    for (1..64) |depth| {
        for (1..64) |move_number| {
            const reduction = 0.4 + @log(@as(f64, @floatFromInt(depth))) * @log(
                @as(f64, @floatFromInt(move_number)),
            ) / 1.9;
            reductions[depth][move_number] = @intFromFloat(@max(reduction, 0));
        }
    }

    break :blk reductions;
};

/// How many quiet moves we try at each low depth before pruning the rest.
const late_move_pruning_counts: [8]u16 = .{
    0,
    5,
    8,
    12,
    17,
    23,
    30,
    38,
};

/// The per-ply state the search touches on every node. Sixteen bytes keep the
/// entries for ply-2 up to ply+1 in one cache line; principal variations are
/// cold and live in their own table.
pub const StackEntry = extern struct {
    static_eval: i32,
    killers: [2]Move,
    move: Move,
    /// Set when the move played at this ply made a five the opponent can still break.
    pending_five: Move,
    pv_length: u8,
    padding: [3]u8,
};

comptime {
    std.debug.assert(@sizeOf(StackEntry) == 16);
}

pub const RootMove = struct {
    move: Move,
    score: i32,
};

pub const Limits = struct {
    soft_ms: u32,
    hard_ms: u32,
    min_depth: i32,
    max_depth: i32,
    /// Zero means unlimited.
    max_nodes: u64,
    /// Pro and Long Pro openings force the root move at least this far from the
    /// centre (Chebyshev distance). Zero means no restriction.
    root_min_centre_distance: u8,
    /// Catch-up time. While the last completed iteration scores below
    /// `catch_up_below`, the soft and hard limits are lifted up to `catch_up_ms`
    /// so the engine keeps looking for a move that keeps the game alive. Zero
    /// disables it. Proven losses are final, so they never get extra time.
    catch_up_ms: u32,
    catch_up_below: i32,

    pub const default: Limits = .{
        .soft_ms = 300,
        .hard_ms = 450,
        .min_depth = 10,
        .max_depth = max_ply - 8,
        .max_nodes = 0,
        .root_min_centre_distance = 0,
        .catch_up_ms = 0,
        .catch_up_below = 0,
    };

    pub fn withTime(soft_ms: u32, hard_ms: u32) Limits {
        var limits: Limits = .default;
        limits.soft_ms = soft_ms;
        limits.hard_ms = hard_ms;
        return limits;
    }

    pub fn isCatchingUp(limits: *const Limits, score: i32, completed_depth: i32) bool {
        return limits.catch_up_ms > limits.hard_ms and completed_depth > 0 and
            score < limits.catch_up_below and score > -eval.mate_threshold;
    }

    pub fn softLimitMs(limits: *const Limits, score: i32, completed_depth: i32) u32 {
        if (limits.isCatchingUp(score, completed_depth)) {
            return limits.catch_up_ms;
        }

        return limits.soft_ms;
    }

    pub fn hardLimitMs(limits: *const Limits, score: i32, completed_depth: i32) u32 {
        if (limits.isCatchingUp(score, completed_depth)) {
            return limits.catch_up_ms;
        }

        return limits.hard_ms;
    }
};

/// Chebyshev distance from the centre of the board.
pub fn centreDistance(move: Move) usize {
    const centre = Board.size / 2;
    const x = Board.columnOf(move);
    const y = Board.rowOf(move);
    return @max(@max(x, centre) - @min(x, centre), @max(y, centre) - @min(y, centre));
}

/// Keeps only the root moves an opening rule allows. The generator only
/// proposes cells near stones, and under Long Pro those can all be too close
/// to the centre, so we also add every legal allowed point within two lines of
/// a stone, or failing that, the whole allowed ring.
pub fn restrictRootMoves(board: *const Board, list: *MoveList, min_distance: usize) void {
    var keep: [movepick.max_moves]bool = undefined;

    for (list.moves[0..list.len], 0..) |move, index| {
        keep[index] = centreDistance(move) >= min_distance;
    }

    list.retain(keep[0..list.len]);

    for ([_]usize{ 2, Board.size }) |reach| {
        rows: for (0..Board.size) |y| {
            for (0..Board.size) |x| {
                if (list.len >= list.moves.len) {
                    break :rows;
                }

                const move = Board.moveAt(x, y);
                const distance = centreDistance(move);

                if (distance < min_distance or !board.isLegal(move)) {
                    continue;
                }

                if (reach == Board.size and distance != min_distance) {
                    continue;
                }

                if (reach < Board.size and !hasStoneWithin(board, x, y, reach)) {
                    continue;
                }

                if (std.mem.indexOfScalar(Move, list.moves[0..list.len], move) != null) {
                    continue;
                }

                list.append(move, 0);
            }
        }

        if (list.len > 0) {
            break;
        }
    }

    list.padScores();
}

fn hasStoneWithin(board: *const Board, x: usize, y: usize, reach: usize) bool {
    for (y -| reach..@min(y + reach, Board.size - 1) + 1) |nearby_y| {
        for (x -| reach..@min(x + reach, Board.size - 1) + 1) |nearby_x| {
            if (!board.isEmpty(Board.moveAt(nearby_x, nearby_y))) {
                return true;
            }
        }
    }

    return false;
}

/// State every search thread reads. The atomics are the only fields written
/// while the search runs.
pub const Shared = struct {
    transposition_table: *TranspositionTable,
    should_stop: std.atomic.Value(bool),
    /// The deepest iteration any worker has completed. Helpers often get ahead
    /// of the main thread, and their result is just as usable.
    deepest_completed_depth: std.atomic.Value(i32),
    /// Some worker completed an iteration with a mate score, which is final.
    has_proven_result: std.atomic.Value(bool),
    /// Republished by worker 0 after each iteration, because catch-up time
    /// depends on its latest score.
    hard_limit_ms: std.atomic.Value(u32),
    io: std.Io,
    start_time: std.Io.Timestamp,
    limits: Limits,
    workers: []*Worker,
    /// Called by the main thread after each completed iteration.
    maybe_on_iteration: ?*const fn (maybe_context: ?*anyopaque, worker: *Worker) void,
    maybe_context: ?*anyopaque,

    pub fn elapsedMs(self: *const Shared) u64 {
        const elapsed_ns = self.start_time.untilNow(self.io, .awake).toNanoseconds();
        return @intCast(@max(@divTrunc(elapsed_ns, std.time.ns_per_ms), 0));
    }

    pub fn totalNodes(self: *const Shared) u64 {
        var total: u64 = 0;

        for (self.workers) |worker| {
            total += worker.shared_node_count.load(.monotonic);
        }

        return total;
    }

    fn stop(self: *Shared) void {
        self.should_stop.store(true, .monotonic);
    }

    fn isStopped(self: *const Shared) bool {
        return self.should_stop.load(.monotonic);
    }
};

/// Each worker skips some depths so Lazy SMP threads spread over different
/// iterations instead of all searching the same one.
const helper_skip_sizes = [_]i32{
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
};
const helper_skip_phases = [_]i32{
    0,
    1,
    0,
    1,
    2,
    3,
    0,
    1,
    2,
    3,
    4,
    5,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
};

pub const Worker = struct {
    id: usize,
    shared: *Shared,
    board: Board,
    /// Indexed by ply + 1, so ply -1 holds what the opponent did before the root.
    stack: [max_ply + 4]StackEntry align(64),
    /// Triangular table: pv_lines[ply + 1] is the principal variation from that ply.
    pv_lines: [max_ply + 4][max_ply]Move,
    history: movepick.History,
    /// Published every 1024 nodes so other threads can sum node counts without contention.
    shared_node_count: std.atomic.Value(u64) align(std.atomic.cache_line),
    node_count: u64,
    selective_depth: i32,
    completed_depth: i32,
    best_move: Move,
    best_score: i32,
    principal_variation: [max_ply]Move,
    principal_variation_length: usize,
    root_moves: [movepick.max_moves]RootMove,
    root_move_count: usize,
    undo_log: Board.UndoLog,

    pub fn init(self: *Worker, id: usize) void {
        self.id = id;
        self.shared_node_count = .init(0);
        self.node_count = 0;
        self.selective_depth = 0;
        self.completed_depth = 0;
        self.best_move = Board.no_move;
        self.best_score = 0;
        self.principal_variation_length = 0;
        self.root_move_count = 0;
        self.reset();
    }

    pub fn reset(self: *Worker) void {
        self.history = @splat(@splat(0));

        for (&self.stack) |*entry| {
            entry.killers = .{ Board.no_move, Board.no_move };
            entry.move = Board.no_move;
            entry.pending_five = Board.no_move;
            entry.pv_length = 0;
        }
    }

    /// Halves the history between moves: what it learned is still a good
    /// guess for the next position, but should not outweigh fresh evidence.
    pub fn ageHistory(self: *Worker) void {
        const HistoryVector = @Vector(16, i16);

        for (&self.history) |*plane| {
            var index: usize = 0;

            while (index + 16 <= plane.len) : (index += 16) {
                const values: HistoryVector = plane[index..][0..16].*;
                plane[index..][0..16].* = values >> @splat(1);
            }

            while (index < plane.len) : (index += 1) {
                plane[index] >>= 1;
            }
        }
    }

    inline fn stackAt(self: *Worker, ply: i32) *StackEntry {
        return &self.stack[@intCast(ply + 1)];
    }

    inline fn pvLineAt(self: *Worker, ply: i32) *[max_ply]Move {
        return &self.pv_lines[@intCast(ply + 1)];
    }

    /// The line at `ply` becomes `move` followed by the child's line.
    inline fn updatePrincipalVariation(self: *Worker, ply: i32, move: Move) void {
        const child_length = @min(self.stackAt(ply + 1).pv_length, max_ply - 2);
        const line = self.pvLineAt(ply);
        const child_line = self.pvLineAt(ply + 1);

        line[0] = move;
        @memcpy(line[1 .. child_length + 1], child_line[0..child_length]);
        self.stackAt(ply).pv_length = child_length + 1;
    }

    /// Every thread watches the clock, not only the main one: a single thread
    /// can be descheduled or stuck in a slow node for tens of milliseconds,
    /// and the deadline must not wait for it.
    inline fn shouldStop(self: *Worker) bool {
        if ((self.node_count & 1023) == 0) {
            self.shared_node_count.store(self.node_count, .monotonic);
            self.checkLimits();
        }

        return self.shared.isStopped();
    }

    fn checkLimits(self: *Worker) void {
        const shared = self.shared;
        const limits = &shared.limits;
        const hard_limit_ms = shared.hard_limit_ms.load(.monotonic);
        const deepest_completed_depth = shared.deepest_completed_depth.load(.monotonic);

        // The hard limit only applies once some thread reached the minimum
        // depth, or once a forced result is proven, since mate scores are
        // final. Before that a safety net still keeps a slow search from
        // hanging the game.
        const has_usable_result = deepest_completed_depth >= limits.min_depth or
            deepest_completed_depth >= limits.max_depth or
            shared.has_proven_result.load(.monotonic);

        if (has_usable_result) {
            if (shared.elapsedMs() >= hard_limit_ms) {
                shared.stop();
            }
        } else if (shared.elapsedMs() >= @max(hard_limit_ms, limits.hard_ms * 3 / 2)) {
            shared.stop();
        }

        // Node limits only run single-threaded (data generation, matches), so
        // one thread summing the counters is enough.
        if (self.id == 0 and limits.max_nodes != 0 and shared.totalNodes() >= limits.max_nodes) {
            shared.stop();
        }
    }

    fn publishCompletedIteration(self: *Worker, depth: i32, score: i32) void {
        const shared = self.shared;
        _ = shared.deepest_completed_depth.fetchMax(depth, .monotonic);

        if (eval.isMate(score)) {
            shared.has_proven_result.store(true, .monotonic);
        }

        if (self.id == 0) {
            shared.hard_limit_ms.store(shared.limits.hardLimitMs(score, depth), .monotonic);
        }
    }

    fn updateHistory(self: *Worker, color: Color, move: Move, bonus: i32) void {
        const entry = &self.history[color.index()][move];
        const current: i32 = entry.*;
        // The gravity term pulls large values back so the table never saturates.
        const updated = current + bonus - @divTrunc(current * @as(i32, @intCast(@abs(bonus))), 16384);
        entry.* = @intCast(std.math.clamp(updated, -16384, 16384));
    }

    fn hasWonByCapture(board: *const Board, color: Color) bool {
        return board.captured_pairs[color.index()] >= board.rules.capture_win_pairs;
    }

    /// Whether the side to move can complete a five the opponent cannot break.
    fn hasUnbreakableFive(self: *Worker) bool {
        const board = &self.board;
        const color = board.side_to_move;

        var skip: usize = 0;

        while (movepick.nthFivePoint(board, color, skip)) |five_point| : (skip += 1) {
            board.makeMove(five_point);
            const wins = hasWonByCapture(board, color) or !board.isFiveBreakable(five_point);
            board.undoMove();

            if (wins) {
                return true;
            }

            // Bounded so a position full of breakable fives stays cheap to check.
            if (skip > 8) {
                break;
            }
        }

        return false;
    }

    /// The opponent just made a breakable five, so our only moves are the
    /// captures that break it, or that win the game outright.
    fn generateFiveBreakers(self: *Worker, list: *MoveList) void {
        const board = &self.board;
        const color = board.side_to_move;

        list.len = 0;
        var candidates = board.candidates();

        while (candidates.next()) |move| {
            if (board.cell_records[move].capture_counts[color.index()] == 0) {
                continue;
            }

            board.makeMove(move);
            // Same judgement as Game.applyMove: the five is broken unless one survives somewhere.
            const breaks_five = hasWonByCapture(board, color) or !board.hasFiveAnywhere(color.opponent().stoneCode());
            board.undoMove();

            if (breaks_five) {
                list.appendWithFlags(
                    move,
                    movepick.orderingScore(board, move, color),
                    movepick.moveFlags(board, move, color),
                );
            }
        }

        list.padScores();
    }

    /// Plays `move` and checks that the opponent no longer has a five-point or
    /// a capture win. With `counts_own_capture_win`, winning by capture
    /// ourselves also counts as a defence.
    fn removesWinningThreat(self: *Worker, move: Move, counts_own_capture_win: bool) bool {
        const board = &self.board;
        const color = board.side_to_move;

        board.makeMove(move);
        const threats_after = Threats.of(board);
        const won_by_capture = counts_own_capture_win and hasWonByCapture(board, color);
        const threat_removed = won_by_capture or (!threats_after.own_five and !threats_after.own_capture_win);
        board.undoMove();

        return threat_removed;
    }

    /// In defend_five mode the generator keeps every capture as a possible
    /// defence; here we drop the ones that do not actually neutralise the
    /// threat, which would otherwise each cost a full subtree.
    fn keepRealFiveDefences(self: *Worker, list: *MoveList, threats: Threats) void {
        const board = &self.board;
        const own = board.side_to_move;
        const opponent = own.opponent();

        var keep: [movepick.max_moves]bool = undefined;

        for (list.moves[0..list.len], 0..) |move, index| {
            const record = &board.cell_records[move];
            const own_pattern4 = record.info[own.index()].pattern4;
            const opponent_pattern4 = record.info[opponent.index()].pattern4;
            const makes_or_blocks_five = own_pattern4 == .a_five or opponent_pattern4 == .a_five;
            const takes_capture_point = record.capture_counts[opponent.index()] > 0;

            if (makes_or_blocks_five) {
                keep[index] = true;
            } else if (takes_capture_point) {
                // Sitting on the opponent's capture point stops a capture win
                // by itself, but against a five it has to prove itself.
                keep[index] = threats.opponent_capture_win or self.removesWinningThreat(move, false);
            } else {
                keep[index] = self.removesWinningThreat(move, true);
            }
        }

        list.retain(keep[0..list.len]);
    }

    /// Whether every set cell of `plane` lies on one of the four lines through
    /// `move`, at most four cells away: the only cells a quiet stone can change.
    fn isWithinStar(plane: *const [Board.bitboard_word_count]u64, move: Move) bool {
        const move_x: i32 = @intCast(Board.columnOf(move));
        const move_y: i32 = @intCast(Board.rowOf(move));

        for (plane, 0..) |word, word_index| {
            var remaining_bits = word;

            while (remaining_bits != 0) : (remaining_bits &= remaining_bits - 1) {
                const cell: Move = @intCast(word_index * 64 + @ctz(remaining_bits));
                const dx = @as(i32, @intCast(Board.columnOf(cell))) - move_x;
                const dy = @as(i32, @intCast(Board.rowOf(cell))) - move_y;
                const is_on_line = dx == 0 or dy == 0 or dx == dy or dx == -dy;

                if (!is_on_line or @max(@abs(dx), @abs(dy)) > 4) {
                    return false;
                }
            }
        }

        return true;
    }

    /// Against an open-four threat we keep our own fours, captures, and the
    /// quiet moves after which no open-four point is left. If no quiet move
    /// manages that, we are probably lost, so we leave the list alone and let
    /// the search see it.
    fn keepRealFourDefences(self: *Worker, list: *MoveList) void {
        const board = &self.board;
        const own = board.side_to_move;
        const opponent = own.opponent();

        var keep: [movepick.max_moves]bool = undefined;
        var quiet_defence_count: usize = 0;

        for (list.moves[0..list.len], 0..) |move, index| {
            const record = &board.cell_records[move];
            const own_pattern4 = @backingInt(record.info[own.index()].pattern4);
            keep[index] = own_pattern4 >= @backingInt(pattern.Pattern4.e_block4) or
                (board.rules.captures and record.capture_counts[own.index()] > 0);

            // A quiet stone cannot touch an open-four point off its own lines,
            // so the cheap star test saves most of the make/undo pairs.
            if (!keep[index] and isWithinStar(board.threatPlane(opponent, .b_flex4), move)) {
                board.makeMove(move);
                keep[index] = !board.hasThreat(opponent, .b_flex4) and !board.hasThreat(opponent, .a_five);
                board.undoMove();

                if (keep[index]) {
                    quiet_defence_count += 1;
                }
            }
        }

        if (quiet_defence_count == 0) {
            return;
        }

        list.retain(keep[0..list.len]);
    }

    /// Checks the move just played by `mover`. Returns a mate score when it
    /// wins on the spot; when it makes a breakable five, records it so the
    /// opponent's node knows it has to break it.
    inline fn immediateResultAfter(self: *Worker, move: Move, mover: Color, completes_five: bool, ply: i32) ?i32 {
        const board = &self.board;

        if (hasWonByCapture(board, mover)) {
            return eval.mateIn(ply + 1);
        }

        // Captures only remove opponent stones, so a five-point before the
        // move is always a five after it.
        if (completes_five) {
            std.debug.assert(board.fiveDirectionThrough(move) != null);

            if (!board.isFiveBreakable(move)) {
                return eval.mateIn(ply + 1);
            }

            self.stackAt(ply).pending_five = move;
        }

        return null;
    }

    fn staticEvalFrom(maybe_entry: ?TranspositionTable.Entry, board: *const Board) i32 {
        if (maybe_entry) |entry| {
            return entry.static_eval;
        }

        return eval.evaluate(board);
    }

    fn isUsableBound(entry: TranspositionTable.Entry, value: i32, alpha: i32, beta: i32) bool {
        return switch (entry.bound) {
            .exact => true,
            .lower => value >= beta,
            .upper => value <= alpha,
            .none => false,
        };
    }

    fn futilityMargin(depth: i32, improving: bool) i32 {
        if (improving) {
            return 80 * depth - 30;
        }

        return 80 * depth;
    }

    fn latePruningMoveLimit(depth: i32, improving: bool) u16 {
        const limit = late_move_pruning_counts[@intCast(@max(depth, 0))];

        if (improving) {
            return limit + 3;
        }

        return limit;
    }

    /// How far (x256 nats) the learned policy ranks `move` below the node's
    /// favourite; zero when the policy is not in use.
    fn policyDrop(board: *const Board, move: Move, color: Color, best_policy_score: i32) i32 {
        if (best_policy_score == std.math.minInt(i32)) {
            return 0;
        }

        return best_policy_score - movepick.learnedPolicyScore(board, move, color);
    }

    fn searchBound(best: i32, beta: i32, comptime is_pv_node: bool, best_move: Move) TranspositionTable.Bound {
        if (best >= beta) {
            return .lower;
        }

        if (is_pv_node and best_move != Board.no_move) {
            return .exact;
        }

        return .upper;
    }

    pub fn search(
        self: *Worker,
        comptime is_pv_node: bool,
        alpha_in: i32,
        beta_in: i32,
        depth_in: i32,
        ply: i32,
        is_cut_node: bool,
    ) i32 {
        var alpha = alpha_in;
        var beta = beta_in;
        var depth = depth_in;
        const board = &self.board;
        const color = board.side_to_move;
        const is_root = ply == 0;

        if (is_pv_node) {
            self.stackAt(ply).pv_length = 0;
        }

        self.node_count += 1;

        if (self.shouldStop()) {
            return 0;
        }

        if (is_pv_node and ply > self.selective_depth) {
            self.selective_depth = ply;
        }

        self.stackAt(ply + 1).killers = .{ Board.no_move, Board.no_move };
        self.stackAt(ply).pending_five = Board.no_move;

        var list: MoveList = undefined;
        const pending_five = self.stackAt(ply - 1).pending_five;
        const threats = Threats.of(board);
        var mode = threats.generationMode(is_root);

        if (pending_five != Board.no_move) {
            self.generateFiveBreakers(&list);

            if (list.len == 0) {
                return eval.matedIn(ply);
            }
        } else if (!is_root) {
            if (ply >= max_ply - 4) {
                return eval.evaluate(board);
            }

            alpha = @max(alpha, eval.matedIn(ply));
            beta = @min(beta, eval.mateIn(ply + 1));

            if (alpha >= beta) {
                return alpha;
            }

            if (threats.own_capture_win) {
                return eval.mateIn(ply + 1);
            }

            if (threats.own_five and self.hasUnbreakableFive()) {
                return eval.mateIn(ply + 1);
            }
        }

        if (depth <= 0 and pending_five == Board.no_move) {
            return self.quiescence(is_pv_node, alpha, beta, ply, 0);
        }

        const key = board.transpositionKey();
        var tt_move: Move = Board.no_move;
        var was_pv_node = is_pv_node;
        const maybe_tt_entry = self.shared.transposition_table.probe(key);

        if (maybe_tt_entry) |entry| {
            tt_move = entry.move;
            was_pv_node = was_pv_node or entry.is_pv;

            const tt_value = TranspositionTable.fromStoredValue(entry.value, ply);

            if (!is_pv_node and pending_five == Board.no_move and entry.depth >= depth and isUsableBound(
                entry,
                tt_value,
                alpha,
                beta,
            )) {
                return tt_value;
            }
        }

        const static_eval = staticEvalFrom(maybe_tt_entry, board);
        self.stackAt(ply).static_eval = static_eval;
        const improving = ply >= 2 and static_eval > self.stackAt(ply - 2).static_eval;
        const is_quiet_position = !threats.mustDefend() and !threats.opponent_open_four and pending_five == Board.no_move;

        if (!is_pv_node and !is_root and is_quiet_position and !eval.isMate(beta)) {
            if (depth <= 6 and static_eval - futilityMargin(depth, improving) >= beta) {
                return static_eval;
            }

            if (depth >= 3 and static_eval >= beta and self.stackAt(
                ply - 1,
            ).move != Board.no_move and !threats.own_four) {
                const reduction: i32 = 3 + @divTrunc(depth, 4);
                board.makeNullMove();
                self.stackAt(ply).move = Board.no_move;
                const null_value = -self.search(
                    false,
                    -beta,
                    -beta + 1,
                    depth - reduction,
                    ply + 1,
                    !is_cut_node,
                );
                board.undoNullMove();

                if (self.shared.isStopped()) {
                    return 0;
                }

                if (null_value >= beta) {
                    // A null-move mate is not a real mate, so do not report it.
                    if (eval.isMate(null_value)) {
                        return beta;
                    }

                    return null_value;
                }
            }
        }

        if (tt_move == Board.no_move and depth >= 4) {
            depth -= 1;
        }

        if (pending_five == Board.no_move) {
            movepick.generate(board, &list, mode, threats, &self.history);

            switch (mode) {
                .defend_five => {
                    self.keepRealFiveDefences(&list, threats);
                },
                .defend_four => {
                    self.keepRealFourDefences(&list);
                },
                else => {},
            }

            if (list.len == 0) {
                if (mode == .defend_five) {
                    return eval.matedIn(ply + 1);
                }

                if (mode == .defend_four) {
                    mode = .all;
                    movepick.generate(board, &list, .all, threats, &self.history);
                }

                // A full board, or no legal move left: a draw.
                if (list.len == 0) {
                    return 0;
                }
            }
        } else {
            mode = .defend_five;
        }

        if (is_root and self.shared.limits.root_min_centre_distance > 0) {
            restrictRootMoves(board, &list, self.shared.limits.root_min_centre_distance);

            if (list.len == 0) {
                return 0;
            }
        }

        const first_killer = self.stackAt(ply).killers[0];
        const second_killer = self.stackAt(ply).killers[1];

        for (list.moves[0..list.len], 0..) |move, index| {
            if (move == tt_move) {
                list.scores[index] = std.math.maxInt(i32) - 1;
            } else if (move == first_killer) {
                list.scores[index] += 6000;
            } else if (move == second_killer) {
                list.scores[index] += 5000;
            }
        }

        if (is_root) {
            self.root_move_count = 0;
        }

        var best: i32 = -infinite_value;
        var best_move: Move = Board.no_move;
        var move_count: usize = 0;
        var quiet_moves: [64]Move = undefined;
        var quiet_move_count: usize = 0;
        const is_forced = mode == .defend_five;

        var best_policy_score: i32 = std.math.minInt(i32);

        if (board.uses_policy and (policy.tuning.max_reduction != 0 or policy.tuning.prune_margin != 0)) {
            for (list.moves[0..list.len]) |move| {
                best_policy_score = @max(best_policy_score, movepick.learnedPolicyScore(board, move, color));
            }
        }

        var move_index: usize = 0;

        while (move_index < list.len) : (move_index += 1) {
            const move = list.selectBest(move_index);
            move_count += 1;
            const flags = list.flags[move_index];

            if (!is_pv_node and !is_root and !is_forced and best > -eval.mate_threshold and !flags.is_threat and depth < late_move_pruning_counts.len) {
                if (move_count > latePruningMoveLimit(depth, improving)) {
                    continue;
                }
            }

            const policy_drop = policyDrop(board, move, color, best_policy_score);

            if (policy.tuning.prune_margin != 0 and !is_pv_node and !is_root and !is_forced and move_count > 1 and !flags.is_threat and depth <= 3 and
                best > -eval.mate_threshold and policy_drop > policy.tuning.prune_margin)
            {
                continue;
            }

            self.shared.transposition_table.prefetch(board.transpositionKeyAfter(move));
            board.makeMove(move);
            self.stackAt(ply).move = move;
            self.stackAt(ply).pending_five = Board.no_move;

            var value: i32 = undefined;

            if (self.immediateResultAfter(move, color, flags.completes_five, ply)) |immediate_value| {
                value = immediate_value;
            } else {
                var new_depth = depth - 1;

                // Extending a single forced reply costs no branching, and it
                // stops long forcing lines from falling off the horizon.
                if (mode == .defend_five and list.len == 1 and ply < 2 * depth_in + 8) {
                    new_depth += 1;
                }

                if (self.stackAt(ply).pending_five != Board.no_move) {
                    new_depth = @max(new_depth, 1);
                }

                if (move_count > 1 and depth >= 3 and !flags.is_threat and !is_forced) {
                    var reduction: i32 = late_move_reductions[@intCast(@min(depth, 63))][@min(move_count, 63)];

                    if (is_pv_node) {
                        reduction -= 1;
                    }

                    if (is_cut_node) {
                        reduction += 1;
                    }

                    if (!improving) {
                        reduction += 1;
                    }

                    if (flags.is_semi_threat) {
                        reduction -= 1;
                    }

                    reduction -= @divTrunc(self.history[color.index()][move], 6000);

                    if (policy.tuning.max_reduction != 0) {
                        reduction += std.math.clamp(
                            @divTrunc(policy_drop - policy.tuning.reduction_offset, policy.tuning.reduction_divisor),
                            -1,
                            policy.tuning.max_reduction,
                        );
                    }

                    const reduced_depth = std.math.clamp(new_depth - reduction, 1, new_depth);
                    value = -self.search(
                        false,
                        -alpha - 1,
                        -alpha,
                        reduced_depth,
                        ply + 1,
                        true,
                    );

                    if (value > alpha and reduced_depth < new_depth) {
                        value = -self.search(
                            false,
                            -alpha - 1,
                            -alpha,
                            new_depth,
                            ply + 1,
                            !is_cut_node,
                        );
                    }
                } else if (!is_pv_node or move_count > 1) {
                    value = -self.search(
                        false,
                        -alpha - 1,
                        -alpha,
                        new_depth,
                        ply + 1,
                        !is_cut_node,
                    );
                }

                if (is_pv_node and (move_count == 1 or (value > alpha and value < beta))) {
                    value = -self.search(
                        true,
                        -beta,
                        -alpha,
                        new_depth,
                        ply + 1,
                        false,
                    );
                }
            }

            board.undoMove();

            if (self.shared.isStopped()) {
                return 0;
            }

            if (is_root and self.root_move_count < self.root_moves.len) {
                self.root_moves[self.root_move_count] = .{ .move = move, .score = value };
                self.root_move_count += 1;
            }

            if (value > best) {
                best = value;

                if (value > alpha) {
                    best_move = move;

                    if (is_pv_node) {
                        self.updatePrincipalVariation(ply, move);
                    }

                    if (value >= beta) {
                        if (!flags.is_threat) {
                            self.rewardQuietCutoff(ply, color, move, depth, quiet_moves[0..quiet_move_count]);
                        }

                        break;
                    }

                    alpha = value;
                }
            }

            if (!flags.is_threat and quiet_move_count < quiet_moves.len) {
                quiet_moves[quiet_move_count] = move;
                quiet_move_count += 1;
            }
        }

        // Late-move pruning can skip every move after the first was pruned too.
        if (move_count == 0) {
            return static_eval;
        }

        // A node answering a breakable five depends on the move before it, which
        // the hash key does not see, so it must not be stored.
        if (pending_five == Board.no_move) {
            const bound = searchBound(best, beta, is_pv_node, best_move);
            self.shared.transposition_table.store(
                key,
                TranspositionTable.toStoredValue(
                    best,
                    ply,
                ),
                static_eval,
                depth,
                bound,
                best_move,
                was_pv_node,
            );
        }

        return best;
    }

    fn rewardQuietCutoff(
        self: *Worker,
        ply: i32,
        color: Color,
        move: Move,
        depth: i32,
        tried_quiet_moves: []const Move,
    ) void {
        const entry = self.stackAt(ply);

        if (entry.killers[0] != move) {
            entry.killers[1] = entry.killers[0];
            entry.killers[0] = move;
        }

        const bonus: i32 = @min(depth * depth * 16, 4000);
        self.updateHistory(color, move, bonus);

        for (tried_quiet_moves) |quiet_move| {
            self.updateHistory(color, quiet_move, -bonus);
        }
    }

    /// Under a forced defence every reply is tried. Otherwise the attacker gets
    /// a handful of fours; capping deeper sequences harder measured no gain.
    fn quiescenceMoveBudget(must_defend: bool, list_length: usize) usize {
        if (must_defend) {
            return list_length;
        }

        return @min(list_length, 8);
    }

    fn quiescenceBound(best: i32, alpha_at_entry: i32, beta: i32) TranspositionTable.Bound {
        if (best >= beta) {
            return .lower;
        }

        if (best > alpha_at_entry) {
            return .exact;
        }

        return .upper;
    }

    /// Nodes that never evaluated statically store zero rather than -infinity.
    fn storableStaticEval(static_eval: i32) i32 {
        if (static_eval == -infinite_value) {
            return 0;
        }

        return static_eval;
    }

    fn quiescence(
        self: *Worker,
        comptime is_pv_node: bool,
        alpha_in: i32,
        beta: i32,
        ply: i32,
        quiescence_depth: i32,
    ) i32 {
        var alpha = alpha_in;
        const board = &self.board;
        const color = board.side_to_move;

        if (is_pv_node) {
            self.stackAt(ply).pv_length = 0;
        }

        self.node_count += 1;

        if (self.shouldStop()) {
            return 0;
        }

        if (ply > self.selective_depth) {
            self.selective_depth = ply;
        }

        self.stackAt(ply).pending_five = Board.no_move;

        var list: MoveList = undefined;
        const pending_five = self.stackAt(ply - 1).pending_five;
        const threats = Threats.of(board);

        if (pending_five != Board.no_move) {
            self.generateFiveBreakers(&list);

            if (list.len == 0) {
                return eval.matedIn(ply);
            }
        } else {
            if (threats.own_capture_win) {
                return eval.mateIn(ply + 1);
            }

            if (threats.own_five and self.hasUnbreakableFive()) {
                return eval.mateIn(ply + 1);
            }

            if (ply >= max_ply - 4) {
                return eval.evaluate(board);
            }
        }

        const must_defend = pending_five != Board.no_move or threats.mustDefend();

        // Quiescence entries are stored at depth zero.
        const key = board.transpositionKey();
        var tt_move: Move = Board.no_move;

        if (pending_five == Board.no_move) {
            if (self.shared.transposition_table.probe(key)) |entry| {
                tt_move = entry.move;
                const tt_value = TranspositionTable.fromStoredValue(entry.value, ply);

                if (!is_pv_node and isUsableBound(entry, tt_value, alpha, beta)) {
                    return tt_value;
                }
            }
        }

        const alpha_at_entry = alpha;

        var best: i32 = -infinite_value;
        var static_eval: i32 = -infinite_value;

        if (!must_defend) {
            static_eval = eval.evaluate(board);

            if (static_eval >= beta or quiescence_depth >= max_quiescence_depth) {
                return static_eval;
            }

            // A run of fours rarely rescues a hopeless score unless it wins
            // outright, so far below alpha we only go on with real winning
            // shapes (an open four or a four-three) on the board.
            const has_strong_attack = board.hasThreat(color, .b_flex4) or board.hasThreat(color, .c_block4_flex3);

            if (static_eval + 1500 < alpha and !has_strong_attack) {
                return static_eval;
            }

            best = static_eval;

            if (static_eval > alpha) {
                alpha = static_eval;
            }

            movepick.generate(board, &list, .continuous_fours, threats, &self.history);
        } else if (pending_five == Board.no_move) {
            movepick.generate(board, &list, .defend_five, threats, &self.history);
            self.keepRealFiveDefences(&list, threats);

            if (list.len == 0) {
                return eval.matedIn(ply + 1);
            }
        }

        if (tt_move != Board.no_move) {
            for (list.moves[0..list.len], 0..) |move, index| {
                if (move == tt_move) {
                    list.scores[index] = std.math.maxInt(i32) - 1;
                }
            }
        }

        var best_move: Move = Board.no_move;
        const move_budget = quiescenceMoveBudget(must_defend, list.len);
        var move_index: usize = 0;

        while (move_index < move_budget) : (move_index += 1) {
            const move = list.selectBest(move_index);
            const own_pattern4 = board.cell_records[move].info[color.index()].pattern4;

            self.shared.transposition_table.prefetch(board.transpositionKeyAfter(move));
            board.makeMove(move);
            self.stackAt(ply).move = move;
            self.stackAt(ply).pending_five = Board.no_move;

            var value: i32 = undefined;

            if (self.immediateResultAfter(move, color, own_pattern4 == .a_five, ply)) |immediate_value| {
                value = immediate_value;
            } else {
                value = -self.quiescence(is_pv_node, -beta, -alpha, ply + 1, quiescence_depth + 1);
            }

            board.undoMove();

            if (self.shared.isStopped()) {
                return 0;
            }

            if (value > best) {
                best = value;

                if (value > alpha) {
                    best_move = move;

                    if (is_pv_node) {
                        self.updatePrincipalVariation(ply, move);
                    }

                    if (value >= beta) {
                        break;
                    }

                    alpha = value;
                }
            }
        }

        if (pending_five == Board.no_move) {
            const bound = quiescenceBound(best, alpha_at_entry, beta);
            self.shared.transposition_table.store(
                key,
                TranspositionTable.toStoredValue(best, ply),
                storableStaticEval(
                    static_eval,
                ),
                0,
                bound,
                best_move,
                false,
            );
        }

        return best;
    }

    fn isDepthSkippedByHelper(self: *const Worker, depth: i32) bool {
        if (self.id == 0) {
            return false;
        }

        const pattern_index = (self.id - 1) % helper_skip_sizes.len;
        return @mod(@divTrunc(depth + helper_skip_phases[pattern_index], helper_skip_sizes[pattern_index]), 2) != 0;
    }

    /// Searches one depth with an aspiration window around the previous score,
    /// widening it on each fail until the score lands inside.
    fn aspirationSearch(self: *Worker, depth: i32, previous_score: i32) i32 {
        var delta: i32 = 25;
        var alpha = previous_score - delta;
        var beta = previous_score + delta;

        while (true) {
            const value = self.search(
                true,
                alpha,
                beta,
                depth,
                0,
                false,
            );

            if (self.shared.isStopped()) {
                return value;
            }

            if (value <= alpha) {
                alpha = @max(value - delta, -infinite_value);
            } else if (value >= beta) {
                beta = @min(value + delta, infinite_value);
            } else {
                return value;
            }

            delta += delta;

            if (delta > 1500) {
                alpha = -infinite_value;
                beta = infinite_value;
            }
        }
    }

    /// Iterative deepening. Every thread runs it (Lazy SMP); only the main
    /// thread reports and decides when to stop.
    pub fn iterativeDeepening(self: *Worker) void {
        const shared = self.shared;
        self.completed_depth = 0;
        self.selective_depth = 0;
        self.node_count = 0;
        self.shared_node_count.store(0, .monotonic);
        self.best_move = Board.no_move;
        self.best_score = -infinite_value;
        self.principal_variation_length = 0;

        var previous_score: i32 = 0;
        var stable_iteration_count: u32 = 0;
        var depth: i32 = 1;

        while (depth <= shared.limits.max_depth) : (depth += 1) {
            if (self.isDepthSkippedByHelper(depth)) {
                continue;
            }

            self.selective_depth = 0;
            var score: i32 = undefined;

            if (depth >= 5 and !eval.isMate(previous_score)) {
                score = self.aspirationSearch(depth, previous_score);
            } else {
                score = self.search(
                    true,
                    -infinite_value,
                    infinite_value,
                    depth,
                    0,
                    false,
                );
            }

            // An interrupted iteration is unreliable; keep the last complete one.
            if (shared.isStopped() and depth > 1 and self.completed_depth > 0) {
                break;
            }

            const root_entry = self.stackAt(0);

            if (root_entry.pv_length > 0) {
                const root_line = self.pvLineAt(0);

                if (root_line[0] == self.best_move) {
                    stable_iteration_count += 1;
                } else {
                    stable_iteration_count = 0;
                }

                self.best_move = root_line[0];
                self.principal_variation_length = root_entry.pv_length;
                @memcpy(self.principal_variation[0..root_entry.pv_length], root_line[0..root_entry.pv_length]);
            }

            self.best_score = score;
            self.completed_depth = depth;
            self.publishCompletedIteration(depth, score);
            previous_score = score;
            self.shared_node_count.store(self.node_count, .monotonic);

            if (self.id == 0) {
                if (shared.maybe_on_iteration) |on_iteration| {
                    on_iteration(shared.maybe_context, self);
                }

                if (self.isDoneAfterIteration(depth, score, stable_iteration_count)) {
                    break;
                }
            }

            if (shared.isStopped()) {
                break;
            }
        }

        self.shared_node_count.store(self.node_count, .monotonic);

        if (self.id == 0) {
            shared.stop();
        }
    }

    fn isDoneAfterIteration(self: *const Worker, depth: i32, score: i32, stable_iteration_count: u32) bool {
        const limits = &self.shared.limits;

        if (depth < limits.min_depth) {
            return false;
        }

        const elapsed_ms = self.shared.elapsedMs();

        if (elapsed_ms >= limits.softLimitMs(score, depth)) {
            return true;
        }

        // A mate score from a full iteration is a proven result and will not change.
        if (eval.isMate(score)) {
            return true;
        }

        // The same best move for six iterations in a row will not change in
        // the time we have left, so stop once well past half the budget.
        const is_catching_up = limits.isCatchingUp(score, depth);
        return !is_catching_up and stable_iteration_count >= 6 and elapsed_ms * 2 >= limits.soft_ms;
    }
};

test "four-defence filter keeps only real defences" {
    const Static = struct {
        var worker: Worker = undefined;
    };
    const worker = &Static.worker;
    worker.board.init(.standard);

    // An open white three on row 9, plus two scattered black stones.
    for ([_][2]usize{ .{ 8, 9 }, .{ 9, 9 }, .{ 10, 9 } }) |coordinates| {
        worker.board.side_to_move = .white;
        worker.board.makeMove(Board.moveAt(coordinates[0], coordinates[1]));
    }

    worker.board.side_to_move = .black;
    worker.board.makeMove(Board.moveAt(3, 3));
    worker.board.side_to_move = .black;
    worker.board.makeMove(Board.moveAt(15, 15));
    worker.board.side_to_move = .black;

    const threats = Threats.of(&worker.board);
    try std.testing.expect(threats.opponent_open_four);

    var list: MoveList = undefined;
    movepick.generate(&worker.board, &list, .defend_four, threats, null);
    const length_before = list.len;
    worker.keepRealFourDefences(&list);

    // Only the two cells right at the ends of the three stop the open four.
    try std.testing.expectEqual(@as(usize, 2), list.len);
    try std.testing.expect(list.len < length_before);
}

test "opening root restriction keeps only allowed points" {
    const Static = struct {
        var board: Board = undefined;
        var list: MoveList = undefined;
    };
    const board = &Static.board;

    for ([_]usize{ 3, 4 }) |min_distance| {
        board.init(.standard);
        board.makeMove(Board.moveAt(9, 9));
        board.makeMove(Board.moveAt(10, 9));
        movepick.generate(board, &Static.list, .all, Threats.of(board), null);
        restrictRootMoves(board, &Static.list, min_distance);

        try std.testing.expect(Static.list.len > 0);

        for (Static.list.moves[0..Static.list.len]) |move| {
            try std.testing.expect(centreDistance(move) >= min_distance);
            try std.testing.expect(board.isLegal(move));
        }
    }
}
