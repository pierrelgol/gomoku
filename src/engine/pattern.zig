//! Line-pattern tables.
//!
//! Every empty intersection is classified, for each of the four line directions
//! and for each color, by looking at the nine-cell window centred on it. Cells
//! are encoded on two bits (empty, black, white, wall). The centre is always
//! empty, so we drop it and the eight neighbours give a 16-bit key into a
//! 64K-entry table.
//!
//! Patterns are defined recursively, the way Rapfi does it, because that keeps
//! the definition short and provably consistent: a five-point (F5) completes
//! five through the centre, F4 makes two five-points the opponent cannot both
//! block, B4 makes exactly one, F3/B3 allow a further move that yields F4/B4,
//! and so on down to F1/B1. Every threat has to go through the centre stone, so
//! the pattern says exactly what that stone contributes on its line.

const std = @import("std");

const empty_code: u2 = 0;
const black_code: u2 = 1;
const white_code: u2 = 2;
const wall_code: u2 = 3;

pub const Pattern = enum(u4) {
    dead = 0,
    b1,
    f1,
    b2,
    f2,
    /// A two that can grow into two different free threes.
    f2a,
    /// A two that can grow into three or more different free threes.
    f2b,
    b3,
    f3,
    b4,
    f4,
    f5,
};

pub const pattern_count = 12;

/// What the four directional patterns of one cell add up to for one color.
pub const Pattern4 = enum(u4) {
    none = 0,
    l_flex2,
    k_block3,
    j_flex2_2x,
    i_block3_plus,
    h_flex3,
    g_flex3_plus,
    f_flex3_2x,
    e_block4,
    d_block4_plus,
    c_block4_flex3,
    b_flex4,
    a_five,
};

pub const pattern4_count = 13;

/// One direction, one color, packed in a byte so a 16-bit table entry holds
/// both colors.
pub const LineInfo = packed struct(u8) {
    pattern: Pattern,
    /// Pairs this color captures on the line by playing here (0..2).
    captures: u2,
    /// Playing here sets up a capture (C O O . or . O O C).
    capture_threat: bool,
    /// Playing here leaves a pair of our own stones capturable (O C C . or . C C O).
    vulnerable: bool,
};

pub const CellInfo = packed struct(u8) {
    pattern4: Pattern4,
    /// Two free threes at once: forbidden unless the move captures.
    double_three: bool,
    padding: u3,
};

/// The tables are computed once at build time by src/tools/gen_tables.zig and
/// embedded here, so the engine never pays for the recursive classification.
const blob = @embedFile("pattern_tables");

/// Black's LineInfo in the low byte, white's in the high byte. We benchmarked a
/// compact 121x121 variant indexed through two ordinal tables and it was 3%
/// slower on an Apple M5: the 128 KB table sits in L2 and the extra dependent
/// loads cost more than the misses they save.
pub const line_table: [1 << 16]u16 align(64) = @bitCast(blob[0 .. 2 << 16].*);

/// Indexed by the four directional patterns packed as nibbles.
pub const combine_table: [1 << 16]CellInfo align(64) = @bitCast(@as([1 << 16]u8, blob[2 << 16 ..][0 .. 1 << 16].*));

const window_length = 9;
const window_centre = 4;
const Window = [window_length]u2;

fn keyToWindow(key: u16) Window {
    var window: Window = undefined;
    var remaining_key = key;

    for (0..window_length) |index| {
        if (index == window_centre) {
            window[index] = empty_code;
            continue;
        }

        window[index] = @truncate(remaining_key & 3);
        remaining_key >>= 2;
    }

    return window;
}

fn windowToKey(window: Window) u16 {
    var key: u16 = 0;

    var index: usize = window_length;

    while (index > 0) {
        index -= 1;

        if (index == window_centre) {
            continue;
        }

        key = (key << 2) | window[index];
    }

    return key;
}

/// Drops the centre cell from an 18-bit nine-cell window to get the table key.
pub inline fn windowIndex(window_bits: u32) u16 {
    const cells_before_centre: u32 = window_bits & 0xFF;
    const cells_after_centre: u32 = (window_bits >> 10) & 0xFF;
    return @intCast(cells_before_centre | (cells_after_centre << 8));
}

/// Bit position, inside a table key, of the neighbour `offset` cells away from
/// the centre (offset in -4..-1 or 1..4).
pub fn neighbourShift(offset: i32) u4 {
    std.debug.assert(offset != 0 and offset >= -4 and offset <= 4);

    if (offset < 0) {
        return @intCast(2 * (offset + 4));
    }

    return @intCast(2 * (offset - 1) + 8);
}

/// Length of the run of `code` through the centre, counting the centre itself.
fn runThroughCentre(window: Window, code: u2) usize {
    var run_length: usize = 1;

    var left: usize = window_centre;

    while (left > 0 and window[left - 1] == code) {
        left -= 1;
        run_length += 1;
    }

    var right: usize = window_centre + 1;

    while (right < window_length and window[right] == code) {
        right += 1;
        run_length += 1;
    }

    return run_length;
}

/// Classification memo indexed by window key, from black's point of view;
/// white is classified by swapping the colors first.
const Memo = [1 << 16]u8;
const memo_unset = 0xFF;

fn swapColors(window: Window) Window {
    var swapped = window;

    for (&swapped) |*cell| {
        switch (cell.*) {
            black_code => {
                cell.* = white_code;
            },
            white_code => {
                cell.* = black_code;
            },
            else => {},
        }
    }

    return swapped;
}

fn opponentCode(code: u2) u2 {
    std.debug.assert(code == black_code or code == white_code);

    if (code == black_code) {
        return white_code;
    }

    return black_code;
}

/// Classifies the line for black, with the centre still empty.
fn classify(memo: *Memo, window: Window) Pattern {
    const key = windowToKey(window);

    if (memo[key] != memo_unset) {
        return @fromBackingInt(@intCast(memo[key]));
    }

    const pattern = classifyUncached(memo, window);
    memo[key] = @backingInt(pattern);
    return pattern;
}

fn classifyUncached(memo: *Memo, window: Window) Pattern {
    if (runThroughCentre(window, black_code) >= 5) {
        return .f5;
    }

    var five_point_count: usize = 0;

    for (0..window_length) |index| {
        if (index == window_centre or window[index] != empty_code) {
            continue;
        }

        var extended = window;
        extended[index] = black_code;

        if (runThroughCentre(extended, black_code) >= 5) {
            five_point_count += 1;
        }
    }

    if (five_point_count >= 2) {
        return .f4;
    }

    if (five_point_count == 1) {
        return .b4;
    }

    var flex4_count: usize = 0;
    var block4_count: usize = 0;
    var flex3_count: usize = 0;
    var block3_count: usize = 0;
    var flex2_count: usize = 0;
    var block2_count: usize = 0;

    for (0..window_length) |index| {
        if (index == window_centre or window[index] != empty_code) {
            continue;
        }

        var extended = window;
        extended[index] = black_code;

        switch (classify(memo, extended)) {
            .f5 => {
                unreachable;
            },
            .f4 => {
                flex4_count += 1;
            },
            .b4 => {
                block4_count += 1;
            },
            .f3 => {
                flex3_count += 1;
            },
            .b3 => {
                block3_count += 1;
            },
            .f2, .f2a, .f2b => {
                flex2_count += 1;
            },
            .b2 => {
                block2_count += 1;
            },
            else => {},
        }
    }

    if (flex4_count > 0) {
        return .f3;
    }

    if (block4_count > 0) {
        return .b3;
    }

    if (flex3_count >= 3) {
        return .f2b;
    }

    if (flex3_count == 2) {
        return .f2a;
    }

    if (flex3_count == 1) {
        return .f2;
    }

    if (block3_count > 0) {
        return .b2;
    }

    if (flex2_count > 0) {
        return .f1;
    }

    if (block2_count > 0) {
        return .b1;
    }

    return .dead;
}

const CaptureInfo = struct {
    captures: u2,
    threat: bool,
    vulnerable: bool,
};

fn isFlankedByOneOpponent(first_flank: u2, second_flank: u2, opponent: u2) bool {
    return (first_flank == opponent and second_flank == empty_code) or
        (first_flank == empty_code and second_flank == opponent);
}

fn captureInfo(window: Window, code: u2) CaptureInfo {
    const opponent = opponentCode(code);
    var captures: u2 = 0;
    var threat = false;
    var vulnerable = false;

    const sides = [2][3]usize{ .{ 5, 6, 7 }, .{ 3, 2, 1 } };

    for (sides) |side| {
        if (window[side[0]] == opponent and window[side[1]] == opponent) {
            if (window[side[2]] == code) {
                captures += 1;
            }

            if (window[side[2]] == empty_code) {
                threat = true;
            }
        }
    }

    // After we play, the centre and a neighbour form a pair; it is capturable
    // when one flank holds an opponent stone and the other is empty.
    if (window[5] == code and isFlankedByOneOpponent(window[3], window[6], opponent)) {
        vulnerable = true;
    }

    if (window[3] == code and isFlankedByOneOpponent(window[2], window[5], opponent)) {
        vulnerable = true;
    }

    return .{ .captures = captures, .threat = threat, .vulnerable = vulnerable };
}

fn combine(patterns: [4]Pattern) CellInfo {
    var counts: [pattern_count]u8 = @splat(0);

    for (patterns) |pattern| {
        counts[@backingInt(pattern)] += 1;
    }

    const f5 = counts[@backingInt(Pattern.f5)];
    const f4 = counts[@backingInt(Pattern.f4)];
    const b4 = counts[@backingInt(Pattern.b4)];
    const f3 = counts[@backingInt(Pattern.f3)];
    const b3 = counts[@backingInt(Pattern.b3)];
    const f2 = counts[@backingInt(Pattern.f2)] + counts[@backingInt(Pattern.f2a)] + counts[@backingInt(Pattern.f2b)];

    const pattern4: Pattern4 = blk: {
        if (f5 > 0) {
            break :blk .a_five;
        }

        if (f4 > 0 or b4 >= 2) {
            break :blk .b_flex4;
        }

        if (b4 > 0 and f3 > 0) {
            break :blk .c_block4_flex3;
        }

        if (b4 > 0 and (b3 > 0 or f2 > 0)) {
            break :blk .d_block4_plus;
        }

        if (b4 > 0) {
            break :blk .e_block4;
        }

        if (f3 >= 2) {
            break :blk .f_flex3_2x;
        }

        if (f3 > 0 and (b3 > 0 or f2 > 0)) {
            break :blk .g_flex3_plus;
        }

        if (f3 > 0) {
            break :blk .h_flex3;
        }

        if (b3 >= 2 or (b3 > 0 and f2 > 0)) {
            break :blk .i_block3_plus;
        }

        if (f2 >= 2) {
            break :blk .j_flex2_2x;
        }

        if (b3 > 0) {
            break :blk .k_block3;
        }

        if (f2 > 0) {
            break :blk .l_flex2;
        }

        break :blk .none;
    };

    return .{ .pattern4 = pattern4, .double_three = f3 >= 2, .padding = 0 };
}

/// Walls only appear as a contiguous run touching the window border (the board
/// edge), so only 121 x 121 of the 65536 keys can actually occur.
fn isValidWindow(window: Window) bool {
    var wall_seen = false;
    var index: usize = window_centre;

    while (index > 0) {
        index -= 1;

        if (window[index] == wall_code) {
            wall_seen = true;
        } else if (wall_seen) {
            return false;
        }
    }

    wall_seen = false;

    for (window_centre + 1..window_length) |right_index| {
        if (window[right_index] == wall_code) {
            wall_seen = true;
        } else if (wall_seen) {
            return false;
        }
    }

    return true;
}

fn lineInfo(memo: *Memo, window: Window, code: u2) LineInfo {
    const capture_info = captureInfo(window, code);
    const black_view = switch (code) {
        black_code => window,
        else => swapColors(window),
    };

    return .{
        .pattern = classify(memo, black_view),
        .captures = capture_info.captures,
        .capture_threat = capture_info.threat,
        .vulnerable = capture_info.vulnerable,
    };
}

pub fn buildLineTable() [1 << 16]u16 {
    var memo: Memo = undefined;
    @memset(&memo, memo_unset);

    var table: [1 << 16]u16 = @splat(0);

    for (0..1 << 16) |key| {
        const window = keyToWindow(@intCast(key));

        if (!isValidWindow(window)) {
            continue;
        }

        const black_info: u8 = @bitCast(lineInfo(&memo, window, black_code));
        const white_info: u8 = @bitCast(lineInfo(&memo, window, white_code));
        table[key] = @as(u16, black_info) | (@as(u16, white_info) << 8);
    }

    return table;
}

fn combineTableEntry(index: usize) CellInfo {
    var patterns: [4]Pattern = undefined;

    for (0..4) |direction| {
        const nibble: u4 = @truncate(index >> @intCast(4 * direction));

        if (nibble >= pattern_count) {
            return .{ .pattern4 = .none, .double_three = false, .padding = 0 };
        }

        patterns[direction] = @fromBackingInt(@intCast(nibble));
    }

    return combine(patterns);
}

pub fn buildCombineTable() [1 << 16]CellInfo {
    var table: [1 << 16]CellInfo = undefined;

    for (&table, 0..) |*entry, index| {
        entry.* = combineTableEntry(index);
    }

    return table;
}

/// Parses nine characters: 'X' black, 'O' white, '#' wall, anything else empty.
fn parseWindow(text: []const u8) Window {
    std.debug.assert(text.len == window_length);

    var window: Window = undefined;

    for (text, 0..) |character, index| {
        window[index] = switch (character) {
            'X' => black_code,
            'O' => white_code,
            '#' => wall_code,
            else => empty_code,
        };
    }

    return window;
}

fn blackLineInfoOf(text: []const u8) LineInfo {
    const key = windowToKey(parseWindow(text));
    return @bitCast(@as(u8, @truncate(line_table[key])));
}

test "pattern classification" {
    try std.testing.expectEqual(Pattern.f5, blackLineInfoOf("XXXX_....").pattern);
    try std.testing.expectEqual(Pattern.f5, blackLineInfoOf("..XX_XX..").pattern);
    try std.testing.expectEqual(Pattern.f4, blackLineInfoOf("..XX_X...").pattern);
    try std.testing.expectEqual(Pattern.b4, blackLineInfoOf(".OXX_X...").pattern);
    try std.testing.expectEqual(Pattern.b4, blackLineInfoOf("..X._XX..").pattern);
    try std.testing.expectEqual(Pattern.f3, blackLineInfoOf("...X_X...").pattern);
    try std.testing.expectEqual(Pattern.f3, blackLineInfoOf("..X._X...").pattern);

    // A three squeezed between opponent stones (O.XXX.O) can never become free.
    try std.testing.expectEqual(Pattern.b3, blackLineInfoOf(".O.X_X.O.").pattern);
    try std.testing.expectEqual(Pattern.b3, blackLineInfoOf("#..X_XO..").pattern);

    // Only four cells between the blockers, so no five fits.
    try std.testing.expectEqual(Pattern.dead, blackLineInfoOf("O..._O...").pattern);
}

test "capture info" {
    try std.testing.expectEqual(@as(u2, 1), blackLineInfoOf(".....OOX.").captures);
    try std.testing.expectEqual(@as(u2, 2), blackLineInfoOf(".XOO.OOX.").captures);
    try std.testing.expect(blackLineInfoOf("...O_X...").vulnerable);
}

test "neighbour shift matches the key layout" {
    for ([_]i32{
        -4,
        -3,
        -2,
        -1,
        1,
        2,
        3,
        4,
    }) |offset| {
        var window: Window = @splat(empty_code);
        window[@intCast(window_centre + offset)] = wall_code;
        try std.testing.expectEqual(@as(u16, wall_code) << neighbourShift(offset), windowToKey(window));
    }
}
