//! The board, with every piece of evaluation state kept incrementally.
//!
//! Cells live in a 27x27 mailbox: the 19x19 board plus four wall cells on each
//! side. Pattern windows reach four cells away from their centre, and with the
//! padding we never have to bounds-check in the hottest loop of the engine.
//!
//! Alongside the cells, every line of the board (rows, columns and both
//! diagonals) is kept as a 64-bit key with two bits per cell. Pulling the
//! nine-cell window around any cell is then a shift and a mask, and its
//! pattern is a table lookup. Placing or removing a stone only changes the 32
//! cells that share a window with it, so we re-classify those and update the
//! per-color sums by delta instead of rescanning the board.

const std = @import("std");
const pattern = @import("pattern.zig");
const eval = @import("eval.zig");
const policy = @import("policy.zig");
const nnue = @import("nnue.zig");

const Self = @This();

pub const Pattern4 = pattern.Pattern4;
pub const CellInfo = pattern.CellInfo;

pub const size = 19;
pub const padding = 4;
pub const padded_width = size + 2 * padding;
pub const padded_cell_count = padded_width * padded_width;
pub const cell_count = size * size;

pub const Move = u16;
/// Cell 0 is a wall, so it can never be a legal move.
pub const no_move: Move = 0;

pub const empty_cell: u8 = 0;
pub const wall_cell: u8 = 3;

pub const Color = enum(u1) {
    black = 0,
    white = 1,

    pub inline fn opponent(color: Color) Color {
        return @fromBackingInt(@intCast(~@backingInt(color)));
    }

    pub inline fn stoneCode(color: Color) u8 {
        return @as(u8, @backingInt(color)) + 1;
    }

    pub inline fn index(color: Color) usize {
        return @backingInt(color);
    }

    pub fn ofStone(stone_code: u8) Color {
        std.debug.assert(stone_code == 1 or stone_code == 2);

        return @fromBackingInt(@intCast(stone_code - 1));
    }
};

/// Horizontal, vertical, diagonal and anti-diagonal steps in the padded board.
pub const line_directions = [4]i32{ 1, padded_width, padded_width + 1, padded_width - 1 };

/// The eight compass directions, which is what captures look along.
pub const compass_directions = [8]i32{
    1,
    -1,
    padded_width,
    -padded_width,
    padded_width + 1,
    -(padded_width + 1),
    padded_width - 1,
    -(padded_width - 1),
};

pub inline fn moveAt(x: usize, y: usize) Move {
    return @intCast((y + padding) * padded_width + (x + padding));
}

pub inline fn columnOf(move: Move) usize {
    return @as(usize, move % padded_width) - padding;
}

pub inline fn rowOf(move: Move) usize {
    return @as(usize, move / padded_width) - padding;
}

pub inline fn isOnBoard(padded_index: usize) bool {
    const x = padded_index % padded_width;
    const y = padded_index / padded_width;
    return x >= padding and x < padding + size and y >= padding and y < padding + size;
}

inline fn step(move: Move, delta: i32) Move {
    return @intCast(@as(i32, move) + delta);
}

const lines_per_direction = 2 * size - 1;

/// Where a cell sits on one of its lines: which key holds the line and the
/// shift that brings the cell's nine-cell window down to bit 0. Packed in two
/// bytes so the four directions of a cell come in with a single 8-byte load.
const LineSlot = extern struct { line_index: u8, shift: u8 };

const line_slots: [padded_cell_count][4]LineSlot align(64) = blk: {
    @setEvalBranchQuota(100_000);

    var slots: [padded_cell_count][4]LineSlot = @splat(@splat(.{ .line_index = 0, .shift = 0 }));

    for (0..size) |y| {
        for (0..size) |x| {
            const cell = moveAt(x, y);
            const anti_diagonal = x + y;

            slots[cell][0] = .{
                .line_index = 0 * lines_per_direction + y,
                .shift = 2 * x,
            };
            slots[cell][1] = .{
                .line_index = 1 * lines_per_direction + x,
                .shift = 2 * y,
            };
            slots[cell][2] = .{
                .line_index = 2 * lines_per_direction + (x + size - 1 - y),
                .shift = 2 * @as(usize, @min(x, y)),
            };
            slots[cell][3] = .{
                .line_index = 3 * lines_per_direction + anti_diagonal,
                .shift = 2 * (y - (anti_diagonal -| (size - 1))),
            };
        }
    }

    break :blk slots;
};

/// Rows and columns use only the first 19 of their 37 line slots; the unused
/// ones stay all walls.
fn lineCellCount(direction: usize, line: usize) usize {
    if (direction < 2) {
        if (line < size) {
            return size;
        }

        return 0;
    }

    if (line < size) {
        return line + 1;
    }

    return lines_per_direction - line;
}

pub const Zobrist = struct {
    stones: [2][padded_cell_count]u64,
    captured_pairs: [2][8]u64,
    side_to_move: u64,
};

pub const zobrist: Zobrist = blk: {
    @setEvalBranchQuota(1_000_000);

    var generator = std.Random.SplitMix64.init(0x9E3779B97F4A7C15);
    var keys: Zobrist = undefined;

    for (&keys.stones) |*color_keys| {
        for (color_keys) |*key| {
            key.* = generator.next();
        }
    }

    for (&keys.captured_pairs) |*color_keys| {
        for (color_keys) |*key| {
            key.* = generator.next();
        }
    }

    keys.side_to_move = generator.next();

    break :blk keys;
};

/// The cells worth considering as moves: everything within two cells, plus the
/// cells three away along a line, which is where long-range threats start.
const neighbourhood_offsets: [32]i32 = blk: {
    var offsets: [32]i32 = undefined;
    var offset_count = 0;

    for (0..5) |row| {
        for (0..5) |column| {
            const dx = @as(i32, column) - 2;
            const dy = @as(i32, row) - 2;

            if (dx == 0 and dy == 0) {
                continue;
            }

            offsets[offset_count] = dy * padded_width + dx;
            offset_count += 1;
        }
    }

    for (compass_directions) |direction| {
        offsets[offset_count] = 3 * direction;
        offset_count += 1;
    }

    break :blk offsets;
};

pub const Rules = struct {
    captures: bool,
    double_three: bool,
    capture_win_pairs: u8,

    pub const standard: Rules = .{
        .captures = true,
        .double_three = true,
        .capture_win_pairs = 5,
    };

    /// The rule sets play differently enough that each gets its own network
    /// and policy; the names are the asset file names.
    pub const Variant = enum {
        standard,
        no_captures,
        no_double_three,
        no_captures_no_double_three,
    };

    pub fn variant(rules: Rules) Variant {
        if (rules.captures and rules.double_three) {
            return .standard;
        }

        if (rules.double_three) {
            return .no_captures;
        }

        if (rules.captures) {
            return .no_double_three;
        }

        return .no_captures_no_double_three;
    }
};

pub const max_history = 1024;

/// Threat bitboards: one plane per Pattern4 strong enough to matter tactically
/// (f_flex3_2x up to a_five), plus one plane of capture points.
pub const ThreatPlane = enum(u8) {
    f_flex3_2x = @backingInt(Pattern4.f_flex3_2x),
    e_block4 = @backingInt(Pattern4.e_block4),
    d_block4_plus = @backingInt(Pattern4.d_block4_plus),
    c_block4_flex3 = @backingInt(Pattern4.c_block4_flex3),
    b_flex4 = @backingInt(Pattern4.b_flex4),
    a_five = @backingInt(Pattern4.a_five),
    capture_point = 255,
};

const first_threat_pattern4: u8 = @backingInt(Pattern4.f_flex3_2x);
const threat_category_count: usize = pattern.pattern4_count - first_threat_pattern4;
const capture_plane_index: usize = threat_category_count;
const planes_per_color: usize = threat_category_count + 1;

pub const bitboard_word_count = 12;
const threat_word_count: usize = 2 * planes_per_color * bitboard_word_count;

comptime {
    std.debug.assert(bitboard_word_count * 64 >= padded_cell_count);
}

inline fn threatWordIndex(color_index: usize, plane_index: usize, word: usize) usize {
    return (color_index * planes_per_color + plane_index) * bitboard_word_count + word;
}

/// Everything the evaluation needs about one cell, in 16 bytes so four cells
/// share a cache line. The update loop reads and writes all of it for both
/// colors at once, so grouping by cell beats one array per field.
pub const CellRecord = extern struct {
    /// Four directions of 16 bits: black's LineInfo in the low byte, white's in the high byte.
    line_infos: u64,
    info: [2]CellInfo,
    capture_counts: [2]u8,
    scores: [2]i16,
};

comptime {
    std.debug.assert(@sizeOf(CellRecord) == 16);
}

/// A search thread's undo log. makeMove saves everything it overwrites and
/// undoMove copies it back, which is much cheaper than recomputing the
/// patterns around the removed stones.
pub const UndoLog = struct {
    pub const cell_capacity = 1 << 16;
    pub const key_capacity = 1 << 13;
    pub const frame_capacity = 256;
    /// The placed stone plus up to sixteen captured ones, each touching 32 neighbours and itself.
    const max_cells_per_move = 17 * 33;
    const max_keys_per_move = 17 * 4;

    const Frame = struct {
        cell_start: u32,
        key_start: u32,
        bit_start: u32,
        pattern_score_sums: [2]i32,
        capture_opportunities: [2]i16,
        hash: u64,
        candidate_bits: [bitboard_word_count]u64,
        accumulator: nnue.Accumulator align(64),
    };

    cell_indices: [cell_capacity]u16,
    old_cells: [cell_capacity]CellRecord,
    key_indices: [key_capacity]u8,
    old_keys: [key_capacity]u64,
    bit_word_indices: [key_capacity]u8,
    old_bit_words: [key_capacity]u64,
    frames: [frame_capacity]Frame,
    cell_count: u32,
    key_count: u32,
    bit_count: u32,
    frame_count: u32,

    pub fn reset(self: *UndoLog) void {
        self.cell_count = 0;
        self.key_count = 0;
        self.bit_count = 0;
        self.frame_count = 0;
    }

    inline fn hasRoomForMove(self: *const UndoLog) bool {
        return self.frame_count < frame_capacity and
            self.cell_count + max_cells_per_move <= cell_capacity and
            self.key_count + max_keys_per_move <= key_capacity and
            self.bit_count + 4 * max_cells_per_move <= key_capacity;
    }
};

cell_records: [padded_cell_count]CellRecord align(64),
cells: [padded_cell_count]u8 align(64),
neighbour_counts: [padded_cell_count]u8,
line_keys: [4 * lines_per_direction]u64 align(64),
candidate_bits: [bitboard_word_count]u64,

/// Network accumulator for both perspectives, only maintained when uses_network.
accumulator: nnue.Accumulator align(64),
pattern_score_sums: [2]i32,
/// Number of capturing moves available to each color.
capture_opportunities: [2]i16,
threat_bits: [threat_word_count]u64 align(64),

captured_pairs: [2]u8,
side_to_move: Color,
/// Stones and captured pairs only; transpositionKey() folds in the side to move.
hash: u64,
stone_count: u16,
ply: u16,
rules: Rules,

/// Without a log, undoMove falls back to recomputing.
maybe_undo_log: ?*UndoLog,
is_logging: bool,
uses_network: bool,
uses_policy: bool,

move_history: [max_history]Move,
captured_stone_counts: [max_history]u8,
is_move_logged: [max_history]bool,
captured_stones: [max_history]Move,
captured_stone_top: u16,

pub fn init(self: *Self, rules: Rules) void {
    eval.init();

    self.rules = rules;
    self.maybe_undo_log = null;
    self.is_logging = false;
    self.uses_network = false;
    self.uses_policy = false;
    self.accumulator = @splat(0);

    @memset(&self.cells, wall_cell);

    for (0..size) |y| {
        for (0..size) |x| {
            self.cells[moveAt(x, y)] = empty_cell;
        }
    }

    for (0..4) |direction| {
        for (0..lines_per_direction) |line| {
            var line_key: u64 = ~@as(u64, 0);

            for (0..lineCellCount(direction, line)) |position| {
                line_key &= ~(@as(u64, 3) << @intCast(2 * (position + padding)));
            }

            self.line_keys[direction * lines_per_direction + line] = line_key;
        }
    }

    @memset(&self.neighbour_counts, 0);
    @memset(&self.candidate_bits, 0);
    @memset(&self.threat_bits, 0);
    self.pattern_score_sums = .{ 0, 0 };
    self.capture_opportunities = .{ 0, 0 };
    self.captured_pairs = .{ 0, 0 };
    self.side_to_move = .black;
    self.hash = 0;
    self.stone_count = 0;
    self.ply = 0;
    self.captured_stone_top = 0;

    @memset(&self.cell_records, std.mem.zeroes(CellRecord));

    for (0..padded_cell_count) |padded_index| {
        if (self.cells[padded_index] == wall_cell) {
            continue;
        }

        const cell: Move = @intCast(padded_index);
        var line_infos: u64 = 0;

        for (0..4) |direction| {
            line_infos |= @as(u64, pattern.line_table[self.windowIndex(cell, direction)]) << @intCast(16 * direction);
        }

        self.cell_records[cell].line_infos = line_infos;
        self.addEmptyCell(cell);
    }
}

/// Sets up a position without move history: `cells[y * 19 + x]` holds
/// 0 for empty, 1 for black and 2 for white.
pub fn setupPosition(
    self: *Self,
    rules: Rules,
    cells: *const [cell_count]u8,
    side_to_move: Color,
    captured_pairs: [2]u8,
) void {
    self.init(rules);

    for (0..size) |y| {
        for (0..size) |x| {
            const stone_code = cells[y * size + x];

            if (stone_code == 1 or stone_code == 2) {
                const cell = moveAt(x, y);
                self.placeStone(cell, stone_code);
                self.candidate_bits[cell >> 6] &= ~(@as(u64, 1) << @intCast(cell & 63));
            }
        }
    }

    self.side_to_move = side_to_move;

    for (0..2) |color_index| {
        self.captured_pairs[color_index] = captured_pairs[color_index];
        self.hash ^= zobrist.captured_pairs[color_index][0] ^ zobrist.captured_pairs[color_index][
            @min(
                captured_pairs[color_index],
                7,
            )
        ];
    }
}

/// Each rule variant has its own network and policy; a variant without one
/// quietly stays on the classical code.
pub fn setUsesNetwork(self: *Self, enabled: bool) void {
    self.uses_network = enabled and nnue.isAvailable(self.rules.variant());

    if (self.uses_network) {
        self.accumulator = self.accumulatorFromScratch();
    }
}

pub fn setUsesPolicy(self: *Self, enabled: bool) void {
    self.uses_policy = enabled and policy.isAvailable(self.rules.variant());
}

/// Precondition: uses_network.
pub inline fn network(self: *const Self) *const nnue.Network {
    return nnue.forVariant(self.rules.variant());
}

/// Precondition: uses_policy.
pub inline fn policyTables(self: *const Self) *const policy.Policy {
    return policy.forVariant(self.rules.variant());
}

inline fn windowIndex(self: *const Self, cell: Move, direction: usize) u16 {
    const slot = line_slots[cell][direction];
    const window_bits: u32 = @truncate(self.line_keys[slot.line_index] >> @intCast(slot.shift));
    return pattern.windowIndex(window_bits & 0x3FFFF);
}

/// Gathers the low nibble of each 16-bit lane (one color's four patterns)
/// into a 16-bit combine-table index, without a loop.
inline fn gatherPatternNibbles(line_infos: u64) u16 {
    var gathered = line_infos & 0x000F_000F_000F_000F;
    gathered = (gathered | (gathered >> 12)) & 0x0000_00FF_0000_00FF;
    gathered = (gathered | (gathered >> 24)) & 0xFFFF;
    return @intCast(gathered);
}

/// Adds up the 2-bit capture counts sitting at bits 4-5 of each lane.
inline fn sumCaptureCounts(line_infos: u64) u8 {
    const counts = (line_infos >> 4) & 0x0003_0003_0003_0003;
    return @intCast((counts *% 0x0001_0001_0001_0001) >> 48);
}

/// Logging is only switched on by makeMove after it found a log with room.
inline fn undoLog(self: *Self) *UndoLog {
    return self.maybe_undo_log orelse unreachable;
}

inline fn logCell(self: *Self, cell: Move) void {
    if (self.is_logging) {
        const log = self.undoLog();
        log.cell_indices[log.cell_count] = cell;
        log.old_cells[log.cell_count] = self.cell_records[cell];
        log.cell_count += 1;
    }
}

inline fn logLineKey(self: *Self, line_index: u8) void {
    if (self.is_logging) {
        const log = self.undoLog();
        log.key_indices[log.key_count] = line_index;
        log.old_keys[log.key_count] = self.line_keys[line_index];
        log.key_count += 1;
    }
}

inline fn recordFromLineInfos(line_infos: u64) CellRecord {
    var record: CellRecord = .{
        .line_infos = line_infos,
        .info = undefined,
        .capture_counts = undefined,
        .scores = undefined,
    };

    inline for (0..2) |color_index| {
        const color_line_infos = line_infos >> (8 * color_index);
        const entry = eval.cell_table[gatherPatternNibbles(color_line_infos)];
        record.info[color_index] = entry.info;
        record.capture_counts[color_index] = sumCaptureCounts(color_line_infos);
        record.scores[color_index] = entry.score;
    }

    return record;
}

inline fn flipThreatBit(self: *Self, word_index: usize, mask: u64) void {
    if (self.is_logging) {
        const log = self.undoLog();
        log.bit_word_indices[log.bit_count] = @intCast(word_index);
        log.old_bit_words[log.bit_count] = self.threat_bits[word_index];
        log.bit_count += 1;
    }

    self.threat_bits[word_index] ^= mask;
}

/// A cell with no record is occupied, and occupied cells hold no threats.
inline fn pattern4CodeOf(maybe_record: ?CellRecord, color_index: usize) u8 {
    if (maybe_record) |record| {
        return @backingInt(record.info[color_index].pattern4);
    }

    return 0;
}

inline fn isCapturePointOf(maybe_record: ?CellRecord, color_index: usize) bool {
    if (maybe_record) |record| {
        return record.capture_counts[color_index] > 0;
    }

    return false;
}

/// Moves `cell` between threat planes; most updates leave the category
/// unchanged, so we only touch the bitboards when it really moves.
inline fn updateThreatBits(self: *Self, cell: Move, maybe_old_record: ?CellRecord, maybe_new_record: ?CellRecord) void {
    const word = cell >> 6;
    const mask = @as(u64, 1) << @intCast(cell & 63);

    inline for (0..2) |color_index| {
        const old_code = pattern4CodeOf(maybe_old_record, color_index);
        const new_code = pattern4CodeOf(maybe_new_record, color_index);

        if (old_code != new_code) {
            if (old_code >= first_threat_pattern4) {
                self.flipThreatBit(threatWordIndex(color_index, old_code - first_threat_pattern4, word), mask);
            }

            if (new_code >= first_threat_pattern4) {
                self.flipThreatBit(threatWordIndex(color_index, new_code - first_threat_pattern4, word), mask);
            }
        }

        if (isCapturePointOf(maybe_old_record, color_index) != isCapturePointOf(maybe_new_record, color_index)) {
            self.flipThreatBit(threatWordIndex(color_index, capture_plane_index, word), mask);
        }
    }
}

inline fn addToSums(self: *Self, record: CellRecord) void {
    inline for (0..2) |color_index| {
        self.pattern_score_sums[color_index] += record.scores[color_index];
        self.capture_opportunities[color_index] += record.capture_counts[color_index];
    }
}

inline fn subtractFromSums(self: *Self, record: CellRecord) void {
    inline for (0..2) |color_index| {
        self.pattern_score_sums[color_index] -= record.scores[color_index];
        self.capture_opportunities[color_index] -= record.capture_counts[color_index];
    }
}

inline fn addEmptyCell(self: *Self, cell: Move) void {
    const record = recordFromLineInfos(self.cell_records[cell].line_infos);
    self.cell_records[cell] = record;
    self.addToSums(record);
    self.updateThreatBits(cell, null, record);

    if (self.uses_network) {
        self.updateCellAccumulator(cell, .add);
    }
}

inline fn removeEmptyCell(self: *Self, cell: Move) void {
    self.subtractFromSums(self.cell_records[cell]);
    self.updateThreatBits(cell, self.cell_records[cell], null);

    if (self.uses_network) {
        self.updateCellAccumulator(cell, .subtract);
    }
}

inline fn networkCellState(record: CellRecord) usize {
    return nnue.cellStateIndex(record.info, record.capture_counts);
}

const Update = enum {
    add,
    subtract,
};

inline fn updateCellAccumulator(self: *Self, cell: Move, comptime update: Update) void {
    inline for (0..4) |direction| {
        const window = self.windowIndex(cell, direction);

        switch (update) {
            .add => {
                nnue.addRow(self.network(), &self.accumulator, .window, window);
            },
            .subtract => {
                nnue.subtractRow(self.network(), &self.accumulator, .window, window);
            },
        }
    }

    const cell_state = networkCellState(self.cell_records[cell]);

    switch (update) {
        .add => {
            nnue.addRow(self.network(), &self.accumulator, .cell_state, cell_state);
        },
        .subtract => {
            nnue.subtractRow(self.network(), &self.accumulator, .cell_state, cell_state);
        },
    }
}

fn accumulatorFromScratch(self: *const Self) nnue.Accumulator {
    var sums: [nnue.row_width]i32 = @splat(0);

    for (0..padded_cell_count) |padded_index| {
        const cell: Move = @intCast(padded_index);

        if (self.cells[cell] != empty_cell) {
            continue;
        }

        inline for (0..4) |direction| {
            for (&sums, self.network().window_rows[self.windowIndex(cell, direction)]) |*sum, weight| {
                sum.* += weight;
            }
        }

        for (&sums, self.network().cell_state_rows[networkCellState(self.cell_records[cell])]) |*sum, weight| {
            sum.* += weight;
        }
    }

    return nnue.relativeToEmptyBoard(self.network(), &sums);
}

inline fn setCandidateBit(self: *Self, cell: Move, is_candidate: bool) void {
    const word = cell >> 6;
    const bit = @as(u64, 1) << @intCast(cell & 63);

    if (is_candidate) {
        self.candidate_bits[word] |= bit;
    } else {
        self.candidate_bits[word] &= ~bit;
    }
}

const neighbour_distances = [8]i32{
    -4,
    -3,
    -2,
    -1,
    1,
    2,
    3,
    4,
};
const Vector8u64 = @Vector(8, u64);
const Vector8u32 = @Vector(8, u32);

/// Re-classifies the eight neighbours of `cell` on each of its four lines.
/// They all read the same line key, so the eight windows come out of one
/// vector shift and mask. `old_cell_code` is what `cell` held before (0 when a
/// stone was just placed, the stone code when one was removed); the network
/// needs it to rebuild the neighbours' old windows.
fn reclassifyNeighbours(self: *Self, cell: Move, old_cell_code: u16) void {
    // 4 directions x 8 neighbours of inlined code is past the default quota.
    @setEvalBranchQuota(20_000);

    const slots = line_slots[cell];
    const uses_network = self.uses_network;
    var accumulator_delta: nnue.VectorRow = @splat(0);

    inline for (0..4) |direction| {
        const direction_step = line_directions[direction];
        const line_key = self.line_keys[slots[direction].line_index];

        // Near the wall the neighbour's shift goes negative; clamping keeps the
        // shift legal and the window it produces is a wall cell we never use.
        const centre_shift: i32 = slots[direction].shift;
        var shifts: Vector8u64 = undefined;

        inline for (0..8) |neighbour| {
            shifts[neighbour] = @intCast(@max(centre_shift + 2 * neighbour_distances[neighbour], 0));
        }

        const windows: Vector8u32 = @truncate(
            (@as(Vector8u64, @splat(line_key)) >> @as(
                @Vector(8, u6),
                @truncate(shifts),
            )) & @as(Vector8u64, @splat(0x3FFFF)),
        );
        const window_indices: Vector8u32 = (windows & @as(Vector8u32, @splat(0xFF))) | ((windows >> @splat(2)) & @as(
            Vector8u32,
            @splat(0xFF00),
        ));

        inline for (0..8) |neighbour| {
            const neighbour_cell = step(cell, neighbour_distances[neighbour] * direction_step);
            const new_line_info: u64 = pattern.line_table[window_indices[neighbour]];

            if (uses_network and self.cells[neighbour_cell] == empty_cell) {
                const cell_shift: u4 = comptime pattern.neighbourShift(-neighbour_distances[neighbour]);
                const new_window: u16 = @intCast(window_indices[neighbour]);
                const old_window: u16 = (new_window & ~(@as(u16, 3) << cell_shift)) | (old_cell_code << cell_shift);
                nnue.accumulateRowChange(self.network(), &accumulator_delta, .window, old_window, new_window);
            }

            const lane_mask: u64 = 0xFFFF << (16 * direction);
            const old_line_infos = self.cell_records[neighbour_cell].line_infos;
            const new_line_infos = (old_line_infos & ~lane_mask) | (new_line_info << (16 * direction));

            if (new_line_infos != old_line_infos) {
                self.logCell(neighbour_cell);

                if (self.cells[neighbour_cell] == empty_cell) {
                    const new_record = recordFromLineInfos(new_line_infos);

                    if (uses_network) {
                        const old_cell_state = networkCellState(self.cell_records[neighbour_cell]);
                        const new_cell_state = networkCellState(new_record);

                        if (old_cell_state != new_cell_state) {
                            nnue.accumulateRowChange(
                                self.network(),
                                &accumulator_delta,
                                .cell_state,
                                old_cell_state,
                                new_cell_state,
                            );
                        }
                    }

                    self.subtractFromSums(self.cell_records[neighbour_cell]);
                    self.addToSums(new_record);
                    self.updateThreatBits(neighbour_cell, self.cell_records[neighbour_cell], new_record);
                    self.cell_records[neighbour_cell] = new_record;
                } else {
                    self.cell_records[neighbour_cell].line_infos = new_line_infos;
                }
            }
        }
    }

    if (uses_network) {
        nnue.applyDelta(&self.accumulator, accumulator_delta);
    }
}

fn placeStone(self: *Self, cell: Move, stone_code: u8) void {
    std.debug.assert(self.cells[cell] == empty_cell);
    std.debug.assert(stone_code == 1 or stone_code == 2);

    self.removeEmptyCell(cell);
    self.cells[cell] = stone_code;
    self.hash ^= zobrist.stones[stone_code - 1][cell];

    const slots = line_slots[cell];

    inline for (0..4) |direction| {
        self.logLineKey(slots[direction].line_index);
        self.line_keys[slots[direction].line_index] |= @as(u64, stone_code) << @intCast(slots[direction].shift + 8);
    }

    self.reclassifyNeighbours(cell, 0);
    self.stone_count += 1;
    self.setCandidateBit(cell, false);
    self.updateNeighbourCounts(cell, .add);

    // Every empty neighbour is now near a stone, so it becomes a candidate.
    // Done branch-free because whether a neighbour is empty is unpredictable.
    for (neighbourhood_offsets) |offset| {
        const neighbour_cell = step(cell, offset);
        self.candidate_bits[neighbour_cell >> 6] |= @as(
            u64,
            @intFromBool(self.cells[neighbour_cell] == empty_cell),
        ) << @intCast(neighbour_cell & 63);
    }
}

fn removeStone(self: *Self, cell: Move) void {
    const stone_code = self.cells[cell];
    std.debug.assert(stone_code == 1 or stone_code == 2);

    self.cells[cell] = empty_cell;
    self.hash ^= zobrist.stones[stone_code - 1][cell];

    const slots = line_slots[cell];

    inline for (0..4) |direction| {
        self.logLineKey(slots[direction].line_index);
        self.line_keys[slots[direction].line_index] &= ~(@as(u64, 3) << @intCast(slots[direction].shift + 8));
    }

    self.reclassifyNeighbours(cell, stone_code);
    self.logCell(cell);
    self.addEmptyCell(cell);
    self.stone_count -= 1;
    self.updateNeighbourCounts(cell, .subtract);

    for (neighbourhood_offsets) |offset| {
        const neighbour_cell = step(cell, offset);
        self.candidate_bits[neighbour_cell >> 6] &= ~(@as(
            u64,
            @intFromBool(self.neighbour_counts[neighbour_cell] == 0),
        ) << @intCast(neighbour_cell & 63));
    }

    self.setCandidateBit(cell, self.neighbour_counts[cell] > 0);
}

const neighbour_row_increment: u64 = 0x0000_0001_0101_0101;
const neighbour_middle_row_increment: u64 = 0x0000_0001_0100_0101;

inline fn neighbourRowIncrement(row: usize) u64 {
    if (row == 2) {
        return neighbour_middle_row_increment;
    }

    return neighbour_row_increment;
}

/// The 5x5 block around a cell is five runs of five adjacent bytes, so each
/// row is updated with a single 8-byte add. Counts never exceed 32, so no
/// carry can spill into the next byte.
inline fn updateNeighbourCounts(self: *Self, cell: Move, comptime update: Update) void {
    inline for (0..5) |row| {
        const row_start: usize = @intCast(@as(i32, cell) + (@as(i32, row) - 2) * padded_width - 2);
        const row_counts: *align(1) u64 = @ptrCast(&self.neighbour_counts[row_start]);

        switch (update) {
            .add => {
                row_counts.* +%= neighbourRowIncrement(row);
            },
            .subtract => {
                row_counts.* -%= neighbourRowIncrement(row);
            },
        }
    }

    inline for (compass_directions) |direction| {
        const distant_cell = step(cell, 3 * direction);

        switch (update) {
            .add => {
                self.neighbour_counts[distant_cell] += 1;
            },
            .subtract => {
                self.neighbour_counts[distant_cell] -= 1;
            },
        }
    }
}

pub inline fn cellAt(self: *const Self, cell: Move) u8 {
    return self.cells[cell];
}

pub inline fn isEmpty(self: *const Self, cell: Move) bool {
    return self.cells[cell] == empty_cell;
}

pub inline fn captureCountAt(self: *const Self, cell: Move, color: Color) u8 {
    if (!self.rules.captures) {
        return 0;
    }

    return self.cell_records[cell].capture_counts[color.index()];
}

pub inline fn threatPlane(
    self: *const Self,
    color: Color,
    comptime plane: ThreatPlane,
) *const [bitboard_word_count]u64 {
    const plane_index: usize = switch (plane) {
        .capture_point => capture_plane_index,
        else => @backingInt(plane) - first_threat_pattern4,
    };
    return self.threat_bits[threatWordIndex(color.index(), plane_index, 0)..][0..bitboard_word_count];
}

pub inline fn hasThreat(self: *const Self, color: Color, comptime plane: ThreatPlane) bool {
    const words: @Vector(bitboard_word_count, u64) = self.threatPlane(color, plane).*;
    return @reduce(.Or, words) != 0;
}

pub inline fn threatCount(self: *const Self, color: Color, comptime plane: ThreatPlane) u32 {
    const words: @Vector(bitboard_word_count, u64) = self.threatPlane(color, plane).*;
    return @reduce(.Add, @as(@Vector(bitboard_word_count, u32), @popCount(words)));
}

/// Planes are stored in increasing strength, so "this threat or stronger" is
/// one contiguous OR-reduce.
pub inline fn hasThreatAtLeast(self: *const Self, color: Color, comptime plane: ThreatPlane) bool {
    const first_plane = @backingInt(plane) - first_threat_pattern4;
    const word_count = bitboard_word_count * (threat_category_count - first_plane);
    const words: @Vector(word_count, u64) = self.threat_bits[threatWordIndex(
        color.index(),
        first_plane,
        0,
    )..][0..word_count].*;
    return @reduce(.Or, words) != 0;
}

inline fn sideToMoveKey(side_to_move: Color) u64 {
    if (side_to_move == .white) {
        return zobrist.side_to_move;
    }

    return 0;
}

pub inline fn transpositionKey(self: *const Self) u64 {
    return self.hash ^ sideToMoveKey(self.side_to_move);
}

/// Good enough to prefetch the right bucket; captures are ignored because
/// they are rare and a wrong prefetch only costs a cache miss.
pub inline fn transpositionKeyAfter(self: *const Self, move: Move) u64 {
    return self.hash ^ zobrist.stones[self.side_to_move.index()][move] ^ sideToMoveKey(self.side_to_move.opponent());
}

pub fn isLegal(self: *const Self, move: Move) bool {
    if (move >= padded_cell_count or self.cells[move] != empty_cell) {
        return false;
    }

    return !self.isForbidden(move, self.side_to_move);
}

/// A double free-three is forbidden, unless the move completes a five or
/// captures, which the rules always allow.
pub inline fn isForbidden(self: *const Self, move: Move, color: Color) bool {
    if (!self.rules.double_three) {
        return false;
    }

    const record = &self.cell_records[move];
    const info = record.info[color.index()];

    if (!info.double_three or info.pattern4 == .a_five) {
        return false;
    }

    if (self.rules.captures and record.capture_counts[color.index()] > 0) {
        return false;
    }

    return true;
}

/// Plays `move` for the side to move. Legality (double-three) is the
/// caller's job; the cell only has to be empty.
pub fn makeMove(self: *Self, move: Move) void {
    const color = self.side_to_move;
    const stone_code = color.stoneCode();
    const opponent_code = color.opponent().stoneCode();
    const ply = self.ply;

    self.is_logging = false;

    if (self.maybe_undo_log) |log| {
        if (log.hasRoomForMove()) {
            log.frames[log.frame_count] = .{
                .cell_start = log.cell_count,
                .key_start = log.key_count,
                .bit_start = log.bit_count,
                .pattern_score_sums = self.pattern_score_sums,
                .capture_opportunities = self.capture_opportunities,
                .hash = self.hash,
                .candidate_bits = self.candidate_bits,
                .accumulator = undefined,
            };

            if (self.uses_network) {
                log.frames[log.frame_count].accumulator = self.accumulator;
            }

            log.frame_count += 1;
            self.is_logging = true;
        }
    }

    self.is_move_logged[ply] = self.is_logging;
    self.move_history[ply] = move;

    var captured_stone_count: u8 = 0;
    const captures_something = self.rules.captures and self.cell_records[move].capture_counts[color.index()] > 0;
    self.placeStone(move, stone_code);

    if (captures_something) {
        inline for (compass_directions) |direction| {
            const near_stone = step(move, direction);
            const far_stone = step(move, 2 * direction);

            if (self.cells[near_stone] == opponent_code and self.cells[far_stone] == opponent_code and self.cells[
                step(
                    move,
                    3 * direction,
                )
            ] == stone_code) {
                self.removeStone(near_stone);
                self.removeStone(far_stone);
                self.captured_stones[self.captured_stone_top] = near_stone;
                self.captured_stones[self.captured_stone_top + 1] = far_stone;
                self.captured_stone_top += 2;
                captured_stone_count += 2;
            }
        }

        if (captured_stone_count > 0) {
            const color_index = color.index();
            const pairs_before = @min(self.captured_pairs[color_index], 7);
            self.captured_pairs[color_index] += captured_stone_count / 2;
            self.hash ^= zobrist.captured_pairs[color_index][pairs_before] ^ zobrist.captured_pairs[color_index][
                @min(
                    self.captured_pairs[color_index],
                    7,
                )
            ];
        }
    }

    self.captured_stone_counts[ply] = captured_stone_count;
    self.is_logging = false;
    self.side_to_move = color.opponent();
    self.ply += 1;
}

pub fn undoMove(self: *Self) void {
    std.debug.assert(self.ply > 0);

    self.ply -= 1;
    const ply = self.ply;
    self.side_to_move = self.side_to_move.opponent();
    const color = self.side_to_move;
    const move = self.move_history[ply];
    const captured_stone_count = self.captured_stone_counts[ply];
    const opponent_code = color.opponent().stoneCode();

    if (captured_stone_count > 0) {
        self.captured_pairs[color.index()] -= captured_stone_count / 2;
    }

    if (self.is_move_logged[ply]) {
        self.restoreFromUndoLog(move, captured_stone_count, opponent_code);
        return;
    }

    if (captured_stone_count > 0) {
        const color_index = color.index();
        const pairs_before_undo = @min(self.captured_pairs[color_index] + captured_stone_count / 2, 7);
        self.hash ^= zobrist.captured_pairs[color_index][pairs_before_undo] ^ zobrist.captured_pairs[color_index][
            @min(
                self.captured_pairs[color_index],
                7,
            )
        ];

        for (0..captured_stone_count) |_| {
            self.captured_stone_top -= 1;
            self.placeStone(self.captured_stones[self.captured_stone_top], opponent_code);
        }
    }

    self.removeStone(move);
}

/// Copies back, newest first, everything makeMove overwrote. Only the cells
/// and neighbour counts are redone by hand, because they are cheap.
fn restoreFromUndoLog(self: *Self, move: Move, captured_stone_count: u8, opponent_code: u8) void {
    const log = self.undoLog();
    log.frame_count -= 1;
    const frame = &log.frames[log.frame_count];

    var cell_index = log.cell_count;

    while (cell_index > frame.cell_start) {
        cell_index -= 1;
        self.cell_records[log.cell_indices[cell_index]] = log.old_cells[cell_index];
    }

    log.cell_count = frame.cell_start;

    var key_index = log.key_count;

    while (key_index > frame.key_start) {
        key_index -= 1;
        self.line_keys[log.key_indices[key_index]] = log.old_keys[key_index];
    }

    log.key_count = frame.key_start;

    var bit_index = log.bit_count;

    while (bit_index > frame.bit_start) {
        bit_index -= 1;
        self.threat_bits[log.bit_word_indices[bit_index]] = log.old_bit_words[bit_index];
    }

    log.bit_count = frame.bit_start;

    self.pattern_score_sums = frame.pattern_score_sums;
    self.capture_opportunities = frame.capture_opportunities;
    self.hash = frame.hash;
    self.candidate_bits = frame.candidate_bits;

    if (self.uses_network) {
        self.accumulator = frame.accumulator;
    }

    self.cells[move] = empty_cell;
    self.updateNeighbourCounts(move, .subtract);

    for (0..captured_stone_count) |_| {
        self.captured_stone_top -= 1;
        const captured_stone = self.captured_stones[self.captured_stone_top];
        self.cells[captured_stone] = opponent_code;
        self.updateNeighbourCounts(captured_stone, .add);
    }

    self.stone_count = self.stone_count - 1 + captured_stone_count;
}

pub fn makeNullMove(self: *Self) void {
    self.side_to_move = self.side_to_move.opponent();
}

pub fn undoNullMove(self: *Self) void {
    self.side_to_move = self.side_to_move.opponent();
}

pub fn lastMove(self: *const Self) Move {
    if (self.ply == 0) {
        return no_move;
    }

    return self.move_history[self.ply - 1];
}

pub fn playedMoves(self: *const Self) []const Move {
    return self.move_history[0..self.ply];
}

/// The stones removed by the last move, oldest first.
pub fn lastCapturedStones(self: *const Self) []const Move {
    if (self.ply == 0) {
        return &.{};
    }

    const captured_stone_count = self.captured_stone_counts[self.ply - 1];
    std.debug.assert(captured_stone_count <= self.captured_stone_top);

    return self.captured_stones[self.captured_stone_top - captured_stone_count .. self.captured_stone_top];
}

/// Length of the run of `stone_code` through `cell` along `direction`.
pub fn runLength(self: *const Self, cell: Move, direction: usize, stone_code: u8) usize {
    var length: usize = 1;

    var forward = step(cell, line_directions[direction]);

    while (self.cells[forward] == stone_code) : (forward = step(forward, line_directions[direction])) {
        length += 1;
    }

    var backward = step(cell, -line_directions[direction]);

    while (self.cells[backward] == stone_code) : (backward = step(backward, -line_directions[direction])) {
        length += 1;
    }

    return length;
}

/// The direction of a five (or longer) through the stone at `cell`, if any.
pub fn fiveDirectionThrough(self: *const Self, cell: Move) ?usize {
    const stone_code = self.cells[cell];

    if (stone_code != 1 and stone_code != 2) {
        return null;
    }

    for (0..4) |direction| {
        if (self.runLength(cell, direction, stone_code) >= 5) {
            return direction;
        }
    }

    return null;
}

fn isFlankedByOneOpponent(first_flank: u8, second_flank: u8, opponent_code: u8) bool {
    return (first_flank == opponent_code and second_flank == empty_cell) or
        (first_flank == empty_cell and second_flank == opponent_code);
}

/// Whether the opponent can capture the stone at `cell` with a single move.
pub fn isStoneCapturable(self: *const Self, cell: Move) bool {
    const stone_code = self.cells[cell];
    const opponent_code: u8 = 3 - stone_code;

    for (line_directions) |direction_step| {
        if (self.cells[step(cell, direction_step)] == stone_code) {
            const before_pair = self.cells[step(cell, -direction_step)];
            const after_pair = self.cells[step(cell, 2 * direction_step)];

            if (isFlankedByOneOpponent(before_pair, after_pair, opponent_code)) {
                return true;
            }
        }

        if (self.cells[step(cell, -direction_step)] == stone_code) {
            const before_pair = self.cells[step(cell, -2 * direction_step)];
            const after_pair = self.cells[step(cell, direction_step)];

            if (isFlankedByOneOpponent(before_pair, after_pair, opponent_code)) {
                return true;
            }
        }
    }

    return false;
}

/// The endgame capture rule: a five only wins if the opponent cannot break it
/// by capturing one of its stones, or win outright by capturing a last pair.
/// The stone at `cell` must already be on the board.
pub fn isFiveBreakable(self: *const Self, cell: Move) bool {
    if (!self.rules.captures) {
        return false;
    }

    const stone_code = self.cells[cell];
    const opponent = Color.ofStone(stone_code).opponent();

    if (self.capture_opportunities[opponent.index()] == 0) {
        return false;
    }

    if (self.captured_pairs[opponent.index()] + 1 >= self.rules.capture_win_pairs) {
        return true;
    }

    for (0..4) |direction| {
        if (self.runLength(cell, direction, stone_code) < 5) {
            continue;
        }

        var run_cell = cell;

        while (self.cells[run_cell] == stone_code) : (run_cell = step(run_cell, -line_directions[direction])) {
            if (self.isStoneCapturable(run_cell)) {
                return true;
            }
        }

        run_cell = step(cell, line_directions[direction]);

        while (self.cells[run_cell] == stone_code) : (run_cell = step(run_cell, line_directions[direction])) {
            if (self.isStoneCapturable(run_cell)) {
                return true;
            }
        }
    }

    return false;
}

pub fn hasFiveAnywhere(self: *const Self, stone_code: u8) bool {
    for (0..size) |y| {
        for (0..size) |x| {
            const cell = moveAt(x, y);

            if (self.cells[cell] == stone_code and self.fiveDirectionThrough(cell) != null) {
                return true;
            }
        }
    }

    return false;
}

pub const CandidateIterator = struct {
    words: [bitboard_word_count]u64,
    word_index: usize,

    pub inline fn next(iterator: *CandidateIterator) ?Move {
        while (iterator.word_index < bitboard_word_count) {
            const word = iterator.words[iterator.word_index];

            if (word != 0) {
                iterator.words[iterator.word_index] = word & (word - 1);
                return @intCast(iterator.word_index * 64 + @ctz(word));
            }

            iterator.word_index += 1;
        }

        return null;
    }
};

pub inline fn candidates(self: *const Self) CandidateIterator {
    return .{ .words = self.candidate_bits, .word_index = 0 };
}

/// Rebuilds the position from its stones and compares every incremental
/// structure against it. Tests lean on this to catch drift in the deltas.
pub fn matchesRecomputedState(self: *const Self) bool {
    var fresh: Self = undefined;
    fresh.init(self.rules);

    for (0..padded_cell_count) |padded_index| {
        if (self.cells[padded_index] == 1 or self.cells[padded_index] == 2) {
            fresh.placeStone(@intCast(padded_index), self.cells[padded_index]);
        }
    }

    if (!std.mem.eql(u8, &fresh.cells, &self.cells)) {
        return false;
    }

    if (!std.mem.eql(i32, &fresh.pattern_score_sums, &self.pattern_score_sums)) {
        return false;
    }

    if (!std.mem.eql(i16, &fresh.capture_opportunities, &self.capture_opportunities)) {
        return false;
    }

    if (!std.mem.eql(u64, &fresh.threat_bits, &self.threat_bits)) {
        return false;
    }

    if (!std.mem.eql(u64, &fresh.candidate_bits, &self.candidate_bits)) {
        return false;
    }

    if (!std.mem.eql(u64, &fresh.line_keys, &self.line_keys)) {
        return false;
    }

    if (!std.mem.eql(u8, &fresh.neighbour_counts, &self.neighbour_counts)) {
        return false;
    }

    for (0..padded_cell_count) |padded_index| {
        if (self.cells[padded_index] == wall_cell) {
            continue;
        }

        if (fresh.cell_records[padded_index].line_infos != self.cell_records[padded_index].line_infos) {
            return false;
        }

        if (self.cells[padded_index] == empty_cell and !std.meta.eql(
            fresh.cell_records[padded_index],
            self.cell_records[padded_index],
        )) {
            return false;
        }
    }

    var expected_hash = fresh.hash;

    for (0..2) |color_index| {
        expected_hash ^= zobrist.captured_pairs[color_index][0] ^ zobrist.captured_pairs[color_index][
            @min(
                self.captured_pairs[color_index],
                7,
            )
        ];
    }

    if (expected_hash != self.hash) {
        return false;
    }

    if (self.uses_network and !std.mem.eql(i16, &self.accumulator, &self.accumulatorFromScratch())) {
        return false;
    }

    return true;
}

/// Boards are too big for the test runner's stack, so each test borrows a static one.
fn testBoard(rules: Rules) *Self {
    const Static = struct {
        var board: Self = undefined;
    };
    Static.board.init(rules);
    return &Static.board;
}

/// Places stones from a small diagram ('X' black, 'O' white) without
/// captures and without changing the side to move.
fn placeDiagram(board: *Self, origin_x: usize, origin_y: usize, rows: []const []const u8) void {
    for (rows, 0..) |row, dy| {
        for (row, 0..) |character, dx| {
            const cell = moveAt(origin_x + dx, origin_y + dy);

            switch (character) {
                'X' => {
                    board.placeStone(cell, 1);
                },
                'O' => {
                    board.placeStone(cell, 2);
                },
                else => {},
            }
        }
    }
}

fn randomLegalMoveNearCentre(board: *const Self, random: std.Random) Move {
    // Staying near the centre crowds the stones, which is where captures happen.
    for (0..100) |_| {
        const cell = moveAt(4 + random.uintLessThan(usize, 11), 4 + random.uintLessThan(usize, 11));

        if (board.isLegal(cell)) {
            return cell;
        }
    }

    return no_move;
}

test "capture" {
    const board = testBoard(.standard);
    placeDiagram(board, 5, 5, &.{ "O...", ".X..", "..X.", "...." });
    board.side_to_move = .white;

    try std.testing.expect(board.captureCountAt(moveAt(8, 8), .white) == 1);
    board.makeMove(moveAt(8, 8));
    try std.testing.expectEqual(empty_cell, board.cellAt(moveAt(6, 6)));
    try std.testing.expectEqual(empty_cell, board.cellAt(moveAt(7, 7)));
    try std.testing.expectEqual(@as(u8, 1), board.captured_pairs[1]);
    try std.testing.expect(board.matchesRecomputedState());

    board.undoMove();
    try std.testing.expectEqual(@as(u8, 1), board.cellAt(moveAt(6, 6)));
    try std.testing.expectEqual(@as(u8, 0), board.captured_pairs[1]);
    try std.testing.expect(board.matchesRecomputedState());
}

test "moving into a capture is safe" {
    const board = testBoard(.standard);
    placeDiagram(board, 5, 5, &.{"OX.O"});
    board.side_to_move = .black;

    board.makeMove(moveAt(7, 5));
    try std.testing.expectEqual(@as(u8, 1), board.cellAt(moveAt(6, 5)));
    try std.testing.expectEqual(@as(u8, 1), board.cellAt(moveAt(7, 5)));
    try std.testing.expectEqual(@as(u8, 0), board.captured_pairs[1]);
}

test "double three" {
    const board = testBoard(.standard);
    placeDiagram(board, 0, 0, &.{
        "........",
        ".X......",
        "..X.....",
        "........",
        ".....XX.",
        "........",
    });
    board.side_to_move = .black;

    const double_three_point = moveAt(4, 4);
    try std.testing.expect(board.isForbidden(double_three_point, .black));
    try std.testing.expect(!board.isLegal(double_three_point));

    // Blocking one end of the row three makes it a plain three again.
    board.placeStone(moveAt(3, 4), 2);
    try std.testing.expect(!board.isForbidden(double_three_point, .black));
}

test "double three allowed when capturing" {
    const board = testBoard(.standard);
    placeDiagram(board, 0, 0, &.{
        "........",
        ".X......",
        "..X.....",
        "........",
        ".....XX.",
        "....O...",
        "....O...",
        "....X...",
    });
    board.side_to_move = .black;

    try std.testing.expect(board.captureCountAt(moveAt(4, 4), .black) == 1);
    try std.testing.expect(!board.isForbidden(moveAt(4, 4), .black));
}

test "five breakable by capture" {
    const board = testBoard(.standard);
    placeDiagram(board, 2, 2, &.{
        "XXXX.",
        "X....",
        "O....",
    });
    board.side_to_move = .black;

    // The stone at (2,2) pairs with (2,3), flanked by white below and empty above.
    board.makeMove(moveAt(6, 2));
    try std.testing.expect(board.fiveDirectionThrough(moveAt(6, 2)) != null);
    try std.testing.expect(board.isFiveBreakable(moveAt(6, 2)));
}

test "random games keep incremental state consistent" {
    const board = testBoard(.standard);
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    for (0..40) |_| {
        board.init(.standard);

        for (0..150) |move_number| {
            const move = randomLegalMoveNearCentre(board, random);

            if (move == no_move) {
                break;
            }

            board.makeMove(move);

            if (move_number % 7 == 0) {
                try std.testing.expect(board.matchesRecomputedState());
            }
        }

        try std.testing.expect(board.matchesRecomputedState());

        while (board.ply > 0) {
            board.undoMove();
        }

        try std.testing.expect(board.matchesRecomputedState());
        try std.testing.expectEqual(@as(u64, 0), board.hash);
        try std.testing.expectEqual(@as(u16, 0), board.stone_count);
    }
}

test "undo log restores exactly" {
    const Static = struct {
        var board: Self = undefined;
        var undo_log: UndoLog = undefined;
    };
    const board = &Static.board;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();

    for (0..30) |_| {
        board.init(.standard);
        Static.undo_log.reset();
        board.maybe_undo_log = &Static.undo_log;

        for (0..120) |_| {
            const move = randomLegalMoveNearCentre(board, random);

            if (move == no_move) {
                break;
            }

            board.makeMove(move);

            if (random.uintLessThan(u8, 4) == 0) {
                board.undoMove();
                try std.testing.expect(board.matchesRecomputedState());
                board.makeMove(move);
            }
        }

        try std.testing.expect(board.matchesRecomputedState());

        while (board.ply > 0) {
            board.undoMove();

            if (board.ply % 5 == 0) {
                try std.testing.expect(board.matchesRecomputedState());
            }
        }

        try std.testing.expectEqual(@as(u64, 0), board.hash);
        try std.testing.expectEqual(@as(u32, 0), Static.undo_log.frame_count);
    }
}

test "network accumulator stays consistent through logged and recomputed undo" {
    const Static = struct {
        var board: Self = undefined;
        var undo_log: UndoLog = undefined;
    };
    nnue.initRandomForTests(3);
    const board = &Static.board;
    var prng = std.Random.DefaultPrng.init(11);
    const random = prng.random();

    for (0..30) |game_number| {
        board.init(.standard);
        board.setUsesNetwork(true);
        try std.testing.expect(board.uses_network);

        Static.undo_log.reset();

        if (game_number % 2 == 0) {
            board.maybe_undo_log = &Static.undo_log;
        } else {
            board.maybe_undo_log = null;
        }

        for (0..120) |move_number| {
            const move = randomLegalMoveNearCentre(board, random);

            if (move == no_move) {
                break;
            }

            board.makeMove(move);

            if (random.uintLessThan(u8, 4) == 0) {
                board.undoMove();
                try std.testing.expect(board.matchesRecomputedState());
                board.makeMove(move);
            }

            if (move_number % 9 == 0) {
                try std.testing.expect(board.matchesRecomputedState());
            }
        }

        try std.testing.expect(board.matchesRecomputedState());

        while (board.ply > 0) {
            board.undoMove();

            if (board.ply % 5 == 0) {
                try std.testing.expect(board.matchesRecomputedState());
            }
        }

        try std.testing.expect(board.matchesRecomputedState());
    }
}
