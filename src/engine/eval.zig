//! Static evaluation, from the side to move's point of view.
//!
//! When a trained network is loaded we simply ask it. Otherwise we fall back
//! to a hand-tuned sum that Board keeps cheap to read:
//!  - pattern potential: for each empty cell, what each color would make by
//!    playing there, maintained incrementally so reading it costs nothing;
//!  - threats: who can make a five, an open four or a four-three. The side to
//!    move gets much more credit because it gets to execute first;
//!  - captures: captured pairs grow non-linearly since five pairs win, and open
//!    capture chances matter more the closer a side is to that win;
//!  - a small tempo bonus.

const std = @import("std");
const pattern = @import("pattern.zig");
const nnue = @import("nnue.zig");
const policy = @import("policy.zig");
const Board = @import("Board.zig");

pub const mate_value: i32 = 30000;
/// Anything beyond this is a forced win, so mate scores stay distinguishable
/// from every static evaluation.
pub const mate_threshold: i32 = mate_value - 1000;
pub const infinite_value: i32 = 32000;
pub const max_static_eval: i32 = 20000;

pub const Weights = struct {
    /// Value of a directional pattern at an empty cell, indexed by Pattern.
    pattern_values: [pattern.pattern_count]i16,
    /// Bonus for the cell's combined Pattern4.
    pattern4_bonuses: [pattern.pattern4_count]i16,
    /// Value of having captured n pairs.
    captured_pair_values: [6]i16,
    /// Value per capture chance of the side to move, by its captured pairs.
    own_capture_chance_values: [6]i16,
    /// Value per capture chance of the opponent, by its captured pairs.
    opponent_capture_chance_values: [6]i16,
    /// Pattern-sum scaling in 1/64, slightly in favour of the side to move.
    own_pattern_scale: i16,
    opponent_pattern_scale: i16,
    own_open_four: i16,
    own_four_three: i16,
    /// The opponent has two five-points: we can only block one.
    opponent_double_five: i16,
    opponent_open_four: i16,
    opponent_four_three: i16,
    own_double_open_three: i16,
    tempo: i16,

    pub const default: Weights = .{
        .pattern_values = .{
            0,
            1,
            3,
            5,
            12,
            16,
            20,
            18,
            40,
            50,
            160,
            400,
        },
        .pattern4_bonuses = .{
            0,
            0,
            2,
            10,
            12,
            4,
            30,
            90,
            0,
            40,
            160,
            300,
            600,
        },
        .captured_pair_values = .{
            0,
            70,
            160,
            300,
            700,
            3000,
        },
        .own_capture_chance_values = .{
            30,
            35,
            45,
            70,
            600,
            0,
        },
        .opponent_capture_chance_values = .{
            20,
            25,
            35,
            55,
            250,
            0,
        },
        .own_pattern_scale = 72,
        .opponent_pattern_scale = 64,
        .own_open_four = 2500,
        .own_four_three = 1200,
        .opponent_double_five = 3000,
        .opponent_open_four = 600,
        .opponent_four_three = 250,
        .own_double_open_three = 300,
        .tempo = 20,
    };
};

const weights: Weights = .default;

/// One entry per combination of four directional patterns: the combined
/// Pattern4 and the cell score side by side, so Board's update loop does a
/// single 4-byte lookup per cell and color.
pub const CellTableEntry = extern struct {
    info: pattern.CellInfo,
    padding: u8,
    score: i16,
};

pub var cell_table: [1 << 16]CellTableEntry align(64) = undefined;
var is_cell_table_ready = false;

/// Indices holding a nibble that is not a Pattern can never be looked up;
/// they score zero.
fn cellScore(index: usize, info: pattern.CellInfo) i16 {
    var score: i32 = 0;

    for (0..4) |direction| {
        const pattern_index = (index >> @intCast(4 * direction)) & 0xF;

        if (pattern_index >= pattern.pattern_count) {
            return 0;
        }

        score += weights.pattern_values[pattern_index];
    }

    score += weights.pattern4_bonuses[@backingInt(info.pattern4)];
    return @intCast(score);
}

/// Filled at startup rather than at comptime: it is a trivial loop at runtime,
/// while 64K entries at comptime would need a huge eval branch quota.
/// Also parses the embedded networks and policies, so the tools can call it
/// once from the main thread before their workers start.
pub fn init() void {
    nnue.init();
    policy.init();

    if (is_cell_table_ready) {
        return;
    }

    for (&cell_table, 0..) |*entry, index| {
        const info = pattern.combine_table[index];
        entry.* = .{ .info = info, .padding = 0, .score = cellScore(index, info) };
    }

    is_cell_table_ready = true;
}

pub fn evaluate(board: *const Board) i32 {
    const own = board.side_to_move;
    const opponent = own.opponent();
    const own_index = own.index();
    const opponent_index = opponent.index();

    if (board.uses_network) {
        const empty_cell_count = Board.cell_count - board.stone_count;
        return nnue.evaluate(
            board.network(),
            &board.accumulator,
            own_index,
            board.captured_pairs[own_index],
            board.captured_pairs[opponent_index],
            empty_cell_count,
        );
    }

    var value: i32 = @divTrunc(
        board.pattern_score_sums[own_index] * weights.own_pattern_scale - board.pattern_score_sums[opponent_index] * weights.opponent_pattern_scale,
        64,
    );
    value += threatValue(board, own, opponent);

    if (board.rules.captures) {
        const own_pairs = @min(board.captured_pairs[own_index], 5);
        const opponent_pairs = @min(board.captured_pairs[opponent_index], 5);
        value += weights.captured_pair_values[own_pairs] - weights.captured_pair_values[opponent_pairs];
        value += @as(i32, board.capture_opportunities[own_index]) * weights.own_capture_chance_values[own_pairs];
        value -= @as(
            i32,
            board.capture_opportunities[opponent_index],
        ) * weights.opponent_capture_chance_values[opponent_pairs];
    }

    value += weights.tempo;
    return std.math.clamp(value, -max_static_eval, max_static_eval);
}

fn threatValue(board: *const Board, own: Board.Color, opponent: Board.Color) i32 {
    const opponent_five_count = board.threatCount(opponent, .a_five);

    var value: i32 = 0;

    if (opponent_five_count >= 2) {
        value -= weights.opponent_double_five;
    }

    // If the opponent already threatens a five, we must block it, so none of
    // our own threats get to count.
    if (opponent_five_count != 0) {
        return value;
    }

    if (board.threatCount(own, .b_flex4) > 0) {
        value += weights.own_open_four;
    } else if (board.threatCount(own, .c_block4_flex3) > 0) {
        value += weights.own_four_three;
    } else if (board.threatCount(own, .f_flex3_2x) > 0 and !opponentHasAnyFour(board, opponent)) {
        value += weights.own_double_open_three;
    }

    if (board.threatCount(opponent, .b_flex4) > 0) {
        value -= weights.opponent_open_four;
    }

    if (board.threatCount(opponent, .c_block4_flex3) > 0) {
        value -= weights.opponent_four_three;
    }

    return value;
}

fn opponentHasAnyFour(board: *const Board, opponent: Board.Color) bool {
    const four_count = board.threatCount(opponent, .b_flex4) +
        board.threatCount(opponent, .c_block4_flex3) +
        board.threatCount(opponent, .d_block4_plus) +
        board.threatCount(opponent, .e_block4);
    return four_count != 0;
}

pub inline fn mateIn(ply: i32) i32 {
    return mate_value - ply;
}

pub inline fn matedIn(ply: i32) i32 {
    return -mate_value + ply;
}

pub inline fn isMate(value: i32) bool {
    return value >= mate_threshold or value <= -mate_threshold;
}
