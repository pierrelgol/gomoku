//! Evaluation network trained by train/train.py (file format "GNN3").
//!
//! Everything runs in integers and matches the trainer's reference bit for
//! bit, so a position scores the same here as it did during training.
//!
//! The accumulator sums, over every empty cell, one row per line window
//! around it (the eight neighbours on that line, as a 16-bit index) and one
//! row for its rules-aware state: both colors' Pattern4, whether each color
//! captures by playing there, and whether it is a forbidden double-three for
//! each color. Rows store both perspectives side by side, so one row load
//! updates both. Board keeps the accumulator relative to the empty board in
//! wrapping i16; the exporter picks its scale so the true value always fits.
//!
//! The head is a small int8 MLP. Activations are clamped to 0..127 so every
//! layer is an int8 x int8 -> i32 dot product, which maps onto NEON sdot.
//!
//! One network per rule variant (Board.Rules.Variant), each embedded at build
//! time from `<nets>/<variant>.nnue` (`-Dnets=dir`). A variant without a
//! valid file is unavailable and the engine uses the classical evaluation.

const std = @import("std");
const pattern = @import("pattern.zig");
const Variant = @import("Board.zig").Rules.Variant;

const variant_count = std.enums.values(Variant).len;
const blobs = blk: {
    var result: [variant_count][]const u8 = undefined;

    for (std.enums.values(Variant), 0..) |variant, variant_index| {
        result[variant_index] = @embedFile("nnue_" ++ @tagName(variant));
    }

    break :blk result;
};
const header_size = 28;

fn hasHeader(comptime blob: []const u8) bool {
    return blob.len >= header_size and std.mem.eql(u8, blob[0..4], "GNN3");
}

fn readHeaderField(blob: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, blob[offset..][0..4], .little);
}

/// The layer sizes come from the embedded files at compile time, so the
/// vector widths below are known constants and the head fully unrolls. All
/// variants share them; the output scale and input shift are per network.
fn sharedLayerSize(comptime offset: usize, comptime fallback: u32) u32 {
    var maybe_size: ?u32 = null;

    for (blobs) |blob| {
        if (!hasHeader(blob)) {
            continue;
        }

        const size = readHeaderField(blob, offset);

        if (maybe_size) |previous_size| {
            if (previous_size != size) {
                @compileError("the embedded networks must all have the same layer sizes");
            }
        }

        maybe_size = size;
    }

    return maybe_size orelse fallback;
}

pub const accumulator_width: usize = sharedLayerSize(4, 32);
pub const hidden1_width: usize = sharedLayerSize(8, 32);
pub const hidden2_width: usize = sharedLayerSize(12, 32);

/// Both perspectives, own then opponent's.
pub const row_width = 2 * accumulator_width;
pub const cell_state_count = 13 * 13 * 16;

pub const Row = [row_width]i16;
pub const Accumulator = [row_width]i16;
pub const VectorRow = @Vector(row_width, i16);
const WideVectorRow = @Vector(row_width, i32);

pub const Network = struct {
    window_rows: [1 << 16]Row align(64),
    cell_state_rows: [cell_state_count]Row align(64),
    empty_board_sums: [row_width]i32 align(64),
    empty_cell_bias: WideVectorRow,
    input_bias: WideVectorRow,
    hidden1_weights: [hidden1_width][row_width]i8 align(64),
    hidden1_biases: [hidden1_width]i32,
    /// Rows 0..5 by the captured pairs of the side to move, rows 6..11 by the opponent's.
    hidden1_capture_biases: [12][hidden1_width]i32,
    hidden2_weights: [hidden2_width][hidden1_width]i8 align(64),
    hidden2_biases: [hidden2_width]i32,
    output_weights: [hidden2_width]i8 align(64),
    output_bias: i32,
    output_scale: i32,
    input_shift: u5,
};

var networks: [variant_count]Network = undefined;
var is_available: [variant_count]bool = @splat(false);
var is_initialized: bool = false;

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

/// Parses every embedded network. Call it from the main thread before any
/// search thread starts: the tables are written without synchronisation.
pub fn init() void {
    if (is_initialized) {
        return;
    }

    is_initialized = true;

    inline for (blobs, 0..) |blob, variant_index| {
        if (comptime hasHeader(blob)) {
            is_available[variant_index] = parse(&networks[variant_index], blob);
        }
    }
}

fn parse(network: *Network, blob: []const u8) bool {
    var reader: BlobReader = .{ .blob = blob, .offset = header_size };
    var empty_cell_bias_row: [row_width]i32 = undefined;
    var input_bias_row: [row_width]i32 = undefined;
    var output_bias_value: [1]i32 = undefined;
    const destinations = [_][]u8{
        std.mem.sliceAsBytes(&network.window_rows),
        std.mem.sliceAsBytes(&network.cell_state_rows),
        std.mem.sliceAsBytes(&empty_cell_bias_row),
        std.mem.sliceAsBytes(&input_bias_row),
        std.mem.sliceAsBytes(&network.hidden1_weights),
        std.mem.sliceAsBytes(&network.hidden1_biases),
        std.mem.sliceAsBytes(&network.hidden1_capture_biases),
        std.mem.sliceAsBytes(&network.hidden2_weights),
        std.mem.sliceAsBytes(&network.hidden2_biases),
        std.mem.sliceAsBytes(&network.output_weights),
        std.mem.sliceAsBytes(&output_bias_value),
    };

    for (destinations) |destination| {
        if (!reader.read(destination)) {
            return false;
        }
    }

    if (reader.offset != blob.len) {
        return false;
    }

    network.empty_cell_bias = empty_cell_bias_row;
    network.input_bias = input_bias_row;
    network.output_bias = output_bias_value[0];
    network.output_scale = @intCast(readHeaderField(blob, 16));
    network.input_shift = @intCast(readHeaderField(blob, 24));
    computeEmptyBoardSums(network);
    return true;
}

pub fn isAvailable(variant: Variant) bool {
    init();
    return is_available[@backingInt(variant)];
}

/// Precondition: isAvailable(variant).
pub inline fn forVariant(variant: Variant) *const Network {
    std.debug.assert(is_available[@backingInt(variant)]);
    return &networks[@backingInt(variant)];
}

/// Black-perspective index of a cell's rules-aware state. It must stay the
/// same formula as cell_index in train.py.
pub inline fn cellStateIndex(info: [2]pattern.CellInfo, capture_counts: [2]u8) usize {
    var is_forbidden: [2]usize = undefined;
    var captures: [2]usize = undefined;

    inline for (0..2) |color_index| {
        captures[color_index] = @intFromBool(capture_counts[color_index] > 0);
        // Board.isForbidden under the standard rules. Every variant keeps the
        // same features, so without the ban it is just a pattern feature.
        is_forbidden[color_index] = @intFromBool(
            info[color_index].double_three and capture_counts[color_index] == 0 and info[color_index].pattern4 != .a_five,
        );
    }

    const pattern4_pair = @as(usize, @backingInt(info[0].pattern4)) * 13 + @backingInt(info[1].pattern4);
    return (pattern4_pair * 4 + captures[0] * 2 + captures[1]) * 4 + is_forbidden[0] * 2 + is_forbidden[1];
}

fn emptyBoardWindowCode(x: i32, y: i32) u16 {
    if (x < 0 or y < 0 or x >= 19 or y >= 19) {
        return 3;
    }

    return 0;
}

/// Accumulator of the empty board, built the same way Board builds its own,
/// so that Board only has to track the difference.
fn computeEmptyBoardSums(network: *Network) void {
    const directions = [4][2]i32{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 }, .{ -1, 1 } };

    var sums: [row_width]i32 = @splat(0);

    for (0..19) |y| {
        for (0..19) |x| {
            var pattern_nibbles: [2]u16 = .{ 0, 0 };
            var capture_counts: [2]u8 = .{ 0, 0 };

            for (directions, 0..) |direction, direction_index| {
                var window: u16 = 0;

                for ([_]i32{
                    -4,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    3,
                    4,
                }) |distance| {
                    const neighbour_x = @as(i32, @intCast(x)) + distance * direction[0];
                    const neighbour_y = @as(i32, @intCast(y)) + distance * direction[1];
                    window |= emptyBoardWindowCode(neighbour_x, neighbour_y) << pattern.neighbourShift(distance);
                }

                for (0..row_width) |lane| {
                    sums[lane] += network.window_rows[window][lane];
                }

                const line_info = pattern.line_table[window];
                pattern_nibbles[0] |= (line_info & 0xF) << @intCast(4 * direction_index);
                pattern_nibbles[1] |= ((line_info >> 8) & 0xF) << @intCast(4 * direction_index);
                capture_counts[0] += @intCast((line_info >> 4) & 3);
                capture_counts[1] += @intCast((line_info >> 12) & 3);
            }

            const info = [2]pattern.CellInfo{
                pattern.combine_table[pattern_nibbles[0]],
                pattern.combine_table[pattern_nibbles[1]],
            };

            for (0..row_width) |lane| {
                sums[lane] += network.cell_state_rows[cellStateIndex(info, capture_counts)][lane];
            }
        }
    }

    network.empty_board_sums = sums;
}

pub const RowTable = enum { window, cell_state };

inline fn rowOf(network: *const Network, comptime table: RowTable, index: usize) VectorRow {
    return switch (table) {
        .window => network.window_rows[index],
        .cell_state => network.cell_state_rows[index],
    };
}

/// Folds a row swap into a pending delta, applied once per move by applyDelta.
pub inline fn accumulateRowChange(
    network: *const Network,
    delta: *VectorRow,
    comptime table: RowTable,
    old_index: usize,
    new_index: usize,
) void {
    delta.* +%= rowOf(network, table, new_index) -% rowOf(network, table, old_index);
}

pub inline fn addRow(
    network: *const Network,
    accumulator: *align(64) Accumulator,
    comptime table: RowTable,
    index: usize,
) void {
    const current: VectorRow = accumulator.*;
    accumulator.* = current +% rowOf(network, table, index);
}

pub inline fn subtractRow(
    network: *const Network,
    accumulator: *align(64) Accumulator,
    comptime table: RowTable,
    index: usize,
) void {
    const current: VectorRow = accumulator.*;
    accumulator.* = current -% rowOf(network, table, index);
}

pub inline fn applyDelta(accumulator: *align(64) Accumulator, delta: VectorRow) void {
    const current: VectorRow = accumulator.*;
    accumulator.* = current +% delta;
}

pub fn relativeToEmptyBoard(network: *const Network, absolute_sums: *const [row_width]i32) Accumulator {
    var accumulator: Accumulator = undefined;

    for (0..row_width) |lane| {
        accumulator[lane] = @truncate(absolute_sums[lane] - network.empty_board_sums[lane]);
    }

    return accumulator;
}

inline fn dotProduct(comptime length: usize, left: @Vector(length, i8), right: @Vector(length, i8)) i32 {
    const Wide = @Vector(length, i32);
    return @reduce(.Add, @as(Wide, left) * @as(Wide, right));
}

/// Shifted, clipped ReLU into the int8 activation range.
inline fn clippedRelu(comptime length: usize, values: @Vector(length, i32), shift: u5) @Vector(length, i8) {
    const clipped = std.math.clamp(
        values >> @splat(shift),
        @as(@Vector(length, i32), @splat(0)),
        @as(@Vector(length, i32), @splat(127)),
    );
    return @intCast(clipped);
}

/// Lane orders that put the side to move's perspective first.
const perspective_orders = blk: {
    var orders: [2][row_width]i32 = undefined;

    for (0..row_width) |lane| {
        orders[0][lane] = @intCast(lane);
        orders[1][lane] = @intCast((lane + accumulator_width) % row_width);
    }

    break :blk orders;
};

/// Shuffle masks must be comptime, so each side to move gets its own shuffle.
inline fn ownPerspectiveFirst(inputs: @Vector(row_width, i8), side_to_move: usize) @Vector(row_width, i8) {
    if (side_to_move == 0) {
        return @shuffle(i8, inputs, undefined, perspective_orders[0]);
    }

    return @shuffle(i8, inputs, undefined, perspective_orders[1]);
}

/// Network output in engine units for the side to move (0 black, 1 white).
pub fn evaluate(
    network: *const Network,
    accumulator: *const Accumulator,
    side_to_move: usize,
    own_captured_pairs: u8,
    opponent_captured_pairs: u8,
    empty_cell_count: u32,
) i32 {
    const empty_cells: WideVectorRow = @splat(@intCast(empty_cell_count));
    const absolute: WideVectorRow = network.input_bias + (@as(
        WideVectorRow,
        @as(VectorRow, accumulator.*),
    ) << @splat(6)) + network.empty_cell_bias * empty_cells;
    const inputs = ownPerspectiveFirst(clippedRelu(row_width, absolute, network.input_shift), side_to_move);

    const own_capture_row: usize = @min(own_captured_pairs, 5);
    const opponent_capture_row: usize = 6 + @as(usize, @min(opponent_captured_pairs, 5));
    var hidden1_sums: [hidden1_width]i32 = undefined;

    inline for (0..hidden1_width) |neuron| {
        hidden1_sums[neuron] = dotProduct(
            row_width,
            inputs,
            network.hidden1_weights[neuron],
        ) + network.hidden1_biases[neuron] +
            network.hidden1_capture_biases[own_capture_row][neuron] + network.hidden1_capture_biases[opponent_capture_row][neuron];
    }

    const hidden1 = clippedRelu(hidden1_width, hidden1_sums, 6);

    var hidden2_sums: [hidden2_width]i32 = undefined;

    inline for (0..hidden2_width) |neuron| {
        hidden2_sums[neuron] = dotProduct(
            hidden1_width,
            hidden1,
            network.hidden2_weights[neuron],
        ) + network.hidden2_biases[neuron];
    }

    const hidden2 = clippedRelu(hidden2_width, hidden2_sums, 6);

    const output = dotProduct(hidden2_width, hidden2, network.output_weights) + network.output_bias;
    return std.math.clamp((output * network.output_scale) >> 13, -19000, 19000);
}

/// Fills the standard variant's tables with random weights so tests can
/// check the incremental accumulator without a trained network.
pub fn initRandomForTests(seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const network = &networks[@backingInt(Variant.standard)];

    for (&network.window_rows) |*row| {
        for (row) |*weight| {
            weight.* = random.intRangeAtMost(i16, -300, 300);
        }
    }

    for (&network.cell_state_rows) |*row| {
        for (row) |*weight| {
            weight.* = random.intRangeAtMost(i16, -300, 300);
        }
    }

    computeEmptyBoardSums(network);
    is_initialized = true;
    is_available[@backingInt(Variant.standard)] = true;
}
