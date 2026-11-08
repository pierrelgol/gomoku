//! Move generation, threat-based narrowing of the move set, and ordering.
//!
//! Candidates come straight from Board's candidate bitboard. When the
//! position is forcing we search far fewer of them: if the opponent threatens
//! to win next move (a five-point, or a capture reaching the capture win), only
//! blocks, captures and our own fives can matter. If the opponent threatens an
//! open four, only moves on their lines, our own fours and captures can. That
//! narrowing is where most of the engine's speed in tactical positions comes
//! from.
//!
//! Moves and scores live in separate dense arrays so picking the best
//! remaining move is a SIMD max over eight scores at a time.

const std = @import("std");
const Board = @import("Board.zig");
const pattern = @import("pattern.zig");
const policy = @import("policy.zig");

const Move = Board.Move;
const Color = Board.Color;
const Pattern4 = pattern.Pattern4;

/// At least the 361 cells of the board, rounded up to whole SIMD chunks.
pub const max_moves = 368;
const simd_lanes = 8;
const ScoreVector = @Vector(simd_lanes, i32);

comptime {
    std.debug.assert(max_moves >= Board.cell_count and max_moves % simd_lanes == 0);
}

/// Computed once at generation so the search can decide reductions and
/// pruning without looking at the board again.
pub const MoveFlags = packed struct(u8) {
    /// Fours, double threes, blocking an open-four point and captures: never reduced or pruned.
    is_threat: bool,
    /// Making a three or blocking a four: reduced less.
    is_semi_threat: bool,
    completes_five: bool,
    padding: u5,

    pub const quiet: MoveFlags = .{
        .is_threat = false,
        .is_semi_threat = false,
        .completes_five = false,
        .padding = 0,
    };
};

pub const MoveList = struct {
    scores: [max_moves]i32 align(64),
    moves: [max_moves]Move align(64),
    flags: [max_moves]MoveFlags align(64),
    len: usize,

    pub const empty: MoveList = .{ .scores = undefined, .moves = undefined, .flags = undefined, .len = 0 };

    pub inline fn append(self: *MoveList, move: Move, score: i32) void {
        self.appendWithFlags(move, score, .quiet);
    }

    pub inline fn appendWithFlags(self: *MoveList, move: Move, score: i32, flags: MoveFlags) void {
        std.debug.assert(self.len < max_moves);

        self.moves[self.len] = move;
        self.scores[self.len] = score;
        self.flags[self.len] = flags;
        self.len += 1;
    }

    /// Keeps the entries whose `keep` flag is set, in order.
    pub fn retain(self: *MoveList, keep: []const bool) void {
        std.debug.assert(keep.len >= self.len);

        var kept_count: usize = 0;

        for (0..self.len) |index| {
            if (keep[index]) {
                self.moves[kept_count] = self.moves[index];
                self.scores[kept_count] = self.scores[index];
                self.flags[kept_count] = self.flags[index];
                kept_count += 1;
            }
        }

        self.len = kept_count;
        self.padScores();
    }

    /// Fills the tail of the last SIMD chunk with the lowest score so
    /// selectBest can read whole vectors without a bounds check.
    pub inline fn padScores(self: *MoveList) void {
        const padded_end = std.mem.alignForward(usize, self.len, simd_lanes);

        for (self.len..padded_end) |index| {
            self.scores[index] = std.math.minInt(i32);
        }
    }

    /// One step of selection sort: swaps the best move of [start, len) into
    /// `start`. Searches usually cut off after a few moves, so sorting the
    /// whole list up front would be wasted work.
    pub fn selectBest(self: *MoveList, start: usize) Move {
        const first_chunk = start / simd_lanes * simd_lanes;
        var best_score: i32 = std.math.minInt(i32);
        var best_index: usize = start;

        var index = start;

        while (index < first_chunk + simd_lanes and index < self.len) : (index += 1) {
            if (self.scores[index] > best_score) {
                best_score = self.scores[index];
                best_index = index;
            }
        }

        var chunk_start = first_chunk + simd_lanes;

        while (chunk_start < self.len) : (chunk_start += simd_lanes) {
            const chunk: ScoreVector = self.scores[chunk_start..][0..simd_lanes].*;
            const chunk_max = @reduce(.Max, chunk);

            if (chunk_max > best_score) {
                best_score = chunk_max;
                const is_max: @Vector(simd_lanes, bool) = chunk == @as(ScoreVector, @splat(chunk_max));
                const max_mask: u8 = @bitCast(is_max);
                best_index = chunk_start + @ctz(max_mask);
            }
        }

        std.mem.swap(i32, &self.scores[start], &self.scores[best_index]);
        std.mem.swap(Move, &self.moves[start], &self.moves[best_index]);
        std.mem.swap(MoveFlags, &self.flags[start], &self.flags[best_index]);
        return self.moves[start];
    }
};

pub const GenerationMode = enum {
    all,
    defend_five,
    defend_four,
    continuous_fours,
};

pub const Threats = struct {
    own_five: bool,
    opponent_five: bool,
    own_capture_win: bool,
    opponent_capture_win: bool,
    opponent_open_four: bool,
    /// The side to move can make a four, or a five.
    own_four: bool,

    pub inline fn of(board: *const Board) Threats {
        const own = board.side_to_move;
        const opponent = own.opponent();
        return .{
            .own_five = board.hasThreat(own, .a_five),
            .opponent_five = board.hasThreat(opponent, .a_five),
            .own_capture_win = canWinByCapture(board, own),
            .opponent_capture_win = canWinByCapture(board, opponent),
            .opponent_open_four = board.hasThreat(opponent, .b_flex4),
            .own_four = board.hasThreatAtLeast(own, .e_block4),
        };
    }

    pub inline fn mustDefend(threats: Threats) bool {
        return threats.opponent_five or threats.opponent_capture_win;
    }

    /// The root always looks at every move against an open-four threat, so
    /// a sharp counterattack is never filtered out of the final choice.
    pub inline fn generationMode(threats: Threats, is_root: bool) GenerationMode {
        if (threats.mustDefend()) {
            return .defend_five;
        }

        if (threats.opponent_open_four and !is_root) {
            return .defend_four;
        }

        return .all;
    }
};

inline fn canWinByCapture(board: *const Board, color: Color) bool {
    const color_index = color.index();
    return board.rules.captures and
        board.captured_pairs[color_index] + 1 >= board.rules.capture_win_pairs and
        board.capture_opportunities[color_index] > 0;
}

/// Hand-tuned ordering values by Pattern4: what a move makes for us, and what
/// it takes away from the opponent.
const attack_scores = [pattern.pattern4_count]i32{
    0,
    200,
    250,
    500,
    600,
    1200,
    2000,
    4000,
    1500,
    3000,
    10000,
    20000,
    100000,
};
const defence_scores = [pattern.pattern4_count]i32{
    0,
    100,
    100,
    250,
    300,
    700,
    1200,
    2500,
    600,
    1200,
    5000,
    8000,
    50000,
};

pub inline fn handcraftedScore(board: *const Board, move: Move, color: Color) i32 {
    const own_index = color.index();
    const opponent_index = color.opponent().index();
    const record = &board.cell_records[move];

    var score: i32 = attack_scores[@backingInt(record.info[own_index].pattern4)] + defence_scores[
        @backingInt(
            record.info[opponent_index].pattern4,
        )
    ];
    score += @as(i32, record.scores[own_index]) + @divTrunc(@as(i32, record.scores[opponent_index]) * 3, 4);

    if (board.rules.captures) {
        const own_captures: i32 = record.capture_counts[own_index];
        const opponent_captures: i32 = record.capture_counts[opponent_index];
        // Each pair we take brings the capture win closer, so captures grow
        // more valuable as the game goes on. Taking the opponent's capture
        // point protects our own pair in the same way.
        score += own_captures * (800 + 400 * @as(i32, board.captured_pairs[own_index]));
        score += opponent_captures * (500 + 300 * @as(i32, board.captured_pairs[opponent_index]));
    }

    return score;
}

/// The learned policy when the board has it enabled, the hand-made score otherwise.
pub inline fn orderingScore(board: *const Board, move: Move, color: Color) i32 {
    if (board.uses_policy) {
        return learnedPolicyScore(board, move, color) * policy.tuning.score_multiplier;
    }

    return handcraftedScore(board, move, color);
}

/// Raw learned policy score (x256 nats). Only meaningful when the board uses the policy.
pub inline fn learnedPolicyScore(board: *const Board, move: Move, color: Color) i32 {
    const record = &board.cell_records[move];
    return policy.score(
        board.policyTables(),
        record.line_infos,
        record.info,
        record.capture_counts,
        color.index(),
        board.captured_pairs,
    );
}

inline fn classifyMove(own_pattern4: u8, opponent_pattern4: u8, own_capture_count: u8) MoveFlags {
    return .{
        .is_threat = own_pattern4 >= @backingInt(
            Pattern4.f_flex3_2x,
        ) or opponent_pattern4 >= @backingInt(Pattern4.b_flex4) or own_capture_count > 0,
        .is_semi_threat = own_pattern4 >= @backingInt(Pattern4.h_flex3) or opponent_pattern4 >= @backingInt(
            Pattern4.e_block4,
        ),
        .completes_five = own_pattern4 == @backingInt(Pattern4.a_five),
        .padding = 0,
    };
}

pub inline fn moveFlags(board: *const Board, move: Move, color: Color) MoveFlags {
    const record = &board.cell_records[move];
    return classifyMove(
        @backingInt(record.info[color.index()].pattern4),
        @backingInt(record.info[color.opponent().index()].pattern4),
        record.capture_counts[color.index()],
    );
}

pub const History = [2][Board.padded_cell_count]i16;

/// Fills `list` with the moves `mode` allows for the side to move, scored for
/// ordering. `maybe_history` adds the calling thread's history bonus.
pub fn generate(
    board: *const Board,
    list: *MoveList,
    mode: GenerationMode,
    threats: Threats,
    maybe_history: ?*const History,
) void {
    switch (mode) {
        inline else => |comptime_mode| {
            generateFor(comptime_mode, board, list, threats, maybe_history);
        },
    }
}

/// Specialised per mode at compile time, so the filter in this loop is a
/// fixed expression rather than a switch evaluated for every candidate.
fn generateFor(
    comptime mode: GenerationMode,
    board: *const Board,
    list: *MoveList,
    threats: Threats,
    maybe_history: ?*const History,
) void {
    const color = board.side_to_move;
    const own_index = color.index();
    const opponent_index = color.opponent().index();
    const captures_enabled = board.rules.captures;
    const five = @backingInt(Pattern4.a_five);
    const block_four = @backingInt(Pattern4.e_block4);

    list.len = 0;
    var candidates = board.candidates();

    while (candidates.next()) |move| {
        const record = &board.cell_records[move];
        const own_pattern4: u8 = @backingInt(record.info[own_index].pattern4);
        const opponent_pattern4: u8 = @backingInt(record.info[opponent_index].pattern4);
        const is_own_capture = captures_enabled and record.capture_counts[own_index] > 0;

        const is_relevant = switch (mode) {
            .all => true,
            .defend_five => opponent_pattern4 == five or own_pattern4 == five or is_own_capture or
                (threats.opponent_capture_win and record.capture_counts[opponent_index] > 0),
            .defend_four => opponent_pattern4 >= block_four or own_pattern4 >= block_four or is_own_capture,
            .continuous_fours => own_pattern4 >= block_four,
        };

        if (!is_relevant or board.isForbidden(move, color)) {
            continue;
        }

        var score = orderingScore(board, move, color);

        if (maybe_history) |history| {
            score += history[own_index][move];
        }

        list.appendWithFlags(
            move,
            score,
            classifyMove(
                own_pattern4,
                opponent_pattern4,
                record.capture_counts[own_index],
            ),
        );
    }

    list.padScores();
}

/// The `skip`-th cell, in board order, where `color` completes a five. It walks
/// the five-point bitboard, so it never scans the board.
pub fn nthFivePoint(board: *const Board, color: Color, skip: usize) ?Move {
    var seen_count: usize = 0;

    for (board.threatPlane(color, .a_five), 0..) |word, word_index| {
        var remaining_bits = word;

        while (remaining_bits != 0) : (remaining_bits &= remaining_bits - 1) {
            if (seen_count >= skip) {
                return @intCast(word_index * 64 + @ctz(remaining_bits));
            }

            seen_count += 1;
        }
    }

    return null;
}

test "selectBest returns moves in decreasing score order" {
    var list: MoveList = .empty;

    for (0..37) |move| {
        list.append(@intCast(move), @intCast((move * 7919) % 101));
    }

    list.padScores();

    var previous_score: i32 = std.math.maxInt(i32);

    for (0..list.len) |index| {
        _ = list.selectBest(index);
        try std.testing.expect(list.scores[index] <= previous_score);
        previous_score = list.scores[index];
    }
}
