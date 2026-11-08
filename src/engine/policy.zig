//! Learned move ordering trained by train/policy.py (file format "GPOL").
//!
//! A candidate's score is a sum of table lookups on state Board already
//! keeps per cell: one entry for the cell's rules-aware state, one per line
//! window, and two for the capture situation of each side. It approximates
//! (x256) the log-probability that the search ends up preferring this move.
//! Being only lookups, it costs about the same as the hand-made ordering it
//! replaces.
//!
//! One policy per rule variant, embedded at build time from
//! `<nets>/<variant>.pol` (`-Dnets=dir`). A variant without a valid file is
//! unavailable and movepick keeps its hand-made ordering.

const std = @import("std");
const pattern = @import("pattern.zig");
const nnue = @import("nnue.zig");
const Variant = @import("Board.zig").Rules.Variant;

const variant_count = std.enums.values(Variant).len;
const blobs = blk: {
    var result: [variant_count][]const u8 = undefined;

    for (std.enums.values(Variant), 0..) |variant, variant_index| {
        result[variant_index] = @embedFile("policy_" ++ @tagName(variant));
    }

    break :blk result;
};
const header_size = 16;

pub const Policy = struct {
    /// Indexed by cell state, then by side to move: the white column is the black
    /// table re-indexed with colors swapped, so the trainer only ships one.
    cell_state_scores: [nnue.cell_state_count][2]i32,
    window_scores: [1 << 16][2]i32,
    /// Indexed by min(capture count here, 3) * 6 + min(pairs already captured, 5).
    own_capture_scores: [24]i32,
    opponent_capture_scores: [24]i32,
};

var policies: [variant_count]Policy = undefined;
var is_available: [variant_count]bool = @splat(false);
var is_initialized = false;

/// Search tuning knobs, set from the command line for experiments.
pub const Tuning = struct {
    /// Weight of the policy score against the history heuristic.
    score_multiplier: i32,
    /// A move `drop` (x256 nats) below the best policy move is reduced by
    /// clamp((drop - reduction_offset) / reduction_divisor, -1, max_reduction).
    reduction_offset: i32,
    reduction_divisor: i32,
    /// Zero turns policy reductions off.
    max_reduction: i32,
    /// At depth <= 3, quiet moves this far (x256 nats) below the best are
    /// pruned. Zero turns it off.
    prune_margin: i32,

    pub const default: Tuning = .{
        .score_multiplier = 16,
        .reduction_offset = 512,
        .reduction_divisor = 512,
        .max_reduction = 2,
        .prune_margin = 0,
    };
};

pub var tuning: Tuning = .default;

const BlobReader = struct {
    blob: []const u8,
    offset: usize,

    fn read(reader: *BlobReader, destination: []u8) bool {
        if (reader.offset + destination.len > reader.blob.len) {
            return false;
        }

        @memcpy(destination, reader.blob[reader.offset..][0..destination.len]);
        reader.offset += destination.len;
        return true;
    }
};

/// Maps a black-perspective cell state to the same state seen by white. It
/// mirrors cell_perm in train.py.
fn whitePerspectiveCellState(black_state: usize) usize {
    const forbidden_bits = black_state % 4;
    const capture_bits = (black_state / 4) % 4;
    const pattern4_pair = black_state / 16;
    const black_pattern4 = pattern4_pair / 13;
    const white_pattern4 = pattern4_pair % 13;

    const swapped_pattern4_pair = white_pattern4 * 13 + black_pattern4;
    const swapped_capture_bits = (capture_bits & 1) * 2 + (capture_bits >> 1);
    const swapped_forbidden_bits = (forbidden_bits & 1) * 2 + (forbidden_bits >> 1);
    return (swapped_pattern4_pair * 4 + swapped_capture_bits) * 4 + swapped_forbidden_bits;
}

/// Parses every embedded policy, once, from the main thread.
pub fn init() void {
    if (is_initialized) {
        return;
    }

    is_initialized = true;

    for (blobs, 0..) |blob, variant_index| {
        is_available[variant_index] = parse(&policies[variant_index], blob);
    }
}

fn parse(policy: *Policy, blob: []const u8) bool {
    if (blob.len < header_size or !std.mem.eql(u8, blob[0..4], "GPOL")) {
        return false;
    }

    // The trainer can also export per-window tables, which this engine does
    // not implement; such a file is treated as if there were no policy.
    const has_window_tables = std.mem.readInt(u32, blob[8..12], .little) != 0;

    if (has_window_tables) {
        return false;
    }

    var reader: BlobReader = .{ .blob = blob, .offset = header_size };
    var black_cell_state_scores: [nnue.cell_state_count]i32 = undefined;
    const destinations = [_][]u8{
        std.mem.sliceAsBytes(&black_cell_state_scores),
        std.mem.sliceAsBytes(&policy.window_scores),
        std.mem.sliceAsBytes(&policy.own_capture_scores),
        std.mem.sliceAsBytes(&policy.opponent_capture_scores),
    };

    for (destinations) |destination| {
        if (!reader.read(destination)) {
            return false;
        }
    }

    if (reader.offset != blob.len) {
        return false;
    }

    for (0..nnue.cell_state_count) |cell_state| {
        policy.cell_state_scores[cell_state] = .{
            black_cell_state_scores[cell_state],
            black_cell_state_scores[whitePerspectiveCellState(cell_state)],
        };
    }

    return true;
}

pub fn isAvailable(variant: Variant) bool {
    init();
    return is_available[@backingInt(variant)];
}

/// Precondition: isAvailable(variant).
pub inline fn forVariant(variant: Variant) *const Policy {
    std.debug.assert(is_available[@backingInt(variant)]);
    return &policies[@backingInt(variant)];
}

/// Policy score of an empty candidate cell for the side `color_index`,
/// straight from the cell's Board record.
pub inline fn score(
    policy: *const Policy,
    line_infos: u64,
    info: [2]pattern.CellInfo,
    capture_counts: [2]u8,
    color_index: usize,
    captured_pairs: [2]u8,
) i32 {
    const opponent_index = 1 - color_index;

    var total: i32 = policy.cell_state_scores[nnue.cellStateIndex(info, capture_counts)][color_index];

    inline for (0..4) |direction| {
        total += policy.window_scores[@as(u16, @truncate(line_infos >> (16 * direction)))][color_index];
    }

    const own_capture_index: usize = @as(
        usize,
        @min(capture_counts[color_index], 3),
    ) * 6 + @min(captured_pairs[color_index], 5);
    const opponent_capture_index: usize = @as(
        usize,
        @min(capture_counts[opponent_index], 3),
    ) * 6 + @min(captured_pairs[opponent_index], 5);
    total += policy.own_capture_scores[own_capture_index] + policy.opponent_capture_scores[opponent_capture_index];
    return total;
}
