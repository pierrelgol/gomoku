//! Drawing primitives on the root vaxis window, in absolute signed screen
//! coordinates (everything clips silently). Also collects the clickable /
//! hoverable regions of the frame for mouse handling.
//! Text must outlive the frame (string literals or the frame arena).

const std = @import("std");
const vaxis = @import("vaxis");
const T = @import("theme.zig");

pub const Style = vaxis.Style;
pub const Color = T.Color;
pub const Alloc = std.mem.Allocator;

pub const Rect = struct {
    x: i32,
    y: i32,
    w: i32,
    h: i32,

    pub fn contains(r: Rect, x: i32, y: i32) bool {
        return x >= r.x and y >= r.y and x < r.x + r.w and y < r.y + r.h;
    }

    pub fn inset(r: Rect, dx: i32, dy: i32) Rect {
        return .{ .x = r.x + dx, .y = r.y + dy, .w = @max(0, r.w - 2 * dx), .h = @max(0, r.h - 2 * dy) };
    }

    pub fn right(r: Rect) i32 {
        return r.x + r.w;
    }

    pub fn bottom(r: Rect) i32 {
        return r.y + r.h;
    }
};

/// What a mouse region does.
pub const Act = union(enum) {
    /// behave as if this key was pressed
    key: u21,
    /// menu row (select on hover, activate on click)
    menu: u8,
    /// highlight a board point while hovered (candidate list, move list…)
    point: u16,
};

pub const Hit = struct { r: Rect, act: Act };

pub const Hits = struct {
    items: [96]Hit = undefined,
    len: usize = 0,

    pub fn add(h: *Hits, r: Rect, act: Act) void {
        if (h.len < h.items.len) {
            h.items[h.len] = .{ .r = r, .act = act };
            h.len += 1;
        }
    }

    /// Last registered region under (x,y) (later = drawn on top).
    pub fn at(h: *const Hits, x: i32, y: i32) ?Act {
        var i = h.len;

        while (i > 0) {
            i -= 1;

            if (h.items[i].r.contains(x, y)) {
                return h.items[i].act;
            }
        }

        return null;
    }
};

pub const Canvas = struct {
    win: vaxis.Window,
    a: Alloc,
    hits: *Hits,
    /// animation clock (ms since program start)
    ms: u64,
    /// drawing is restricted to this rect (cards use it so long text never
    /// spills over their borders or a neighbour)
    clip: Rect = .{ .x = 0, .y = 0, .w = std.math.maxInt(i32), .h = std.math.maxInt(i32) },

    /// Copy of the canvas whose drawing is limited to `r` (intersected with
    /// the current clip).
    pub fn clipped(c: *const Canvas, r: Rect) Canvas {
        var cc = c.*;
        const x0 = @max(c.clip.x, r.x);
        const y0 = @max(c.clip.y, r.y);
        const x1 = @min(c.clip.x +| c.clip.w, r.x +| r.w);
        const y1 = @min(c.clip.y +| c.clip.h, r.y +| r.h);
        cc.clip = .{ .x = x0, .y = y0, .w = @max(0, x1 - x0), .h = @max(0, y1 - y0) };
        return cc;
    }

    fn visible(c: *const Canvas, x: i32, y: i32) bool {
        return x >= 0 and y >= 0 and x < c.win.width and y < c.win.height and
            x >= c.clip.x and y >= c.clip.y and x - c.clip.x < c.clip.w and y - c.clip.y < c.clip.h;
    }

    pub fn width(c: *const Canvas) i32 {
        return c.win.width;
    }

    pub fn height(c: *const Canvas) i32 {
        return c.win.height;
    }

    pub fn print(c: *const Canvas, comptime f: []const u8, args: anytype) []const u8 {
        return std.fmt.allocPrint(c.a, f, args) catch "?";
    }

    pub fn cell(c: *const Canvas, x: i32, y: i32, g: []const u8, st: Style) void {
        if (!c.visible(x, y)) {
            return;
        }

        c.win.writeCell(@intCast(x), @intCast(y), .{ .char = .{ .grapheme = g, .width = 1 }, .style = st });
    }

    /// Background color currently at (x,y) (for blending overlays).
    pub fn bgAt(c: *const Canvas, x: i32, y: i32) Color {
        if (x < 0 or y < 0 or x >= c.win.width or y >= c.win.height) {
            return T.bg;
        }

        const cl = c.win.readCell(@intCast(x), @intCast(y)) orelse {
            return T.bg;
        };
        return if (cl.style.bg == .rgb) cl.style.bg else T.bg;
    }

    /// Blend the background of an existing cell toward `col`.
    pub fn tint(c: *const Canvas, x: i32, y: i32, col: Color, a: f32) void {
        if (a <= 0.005 or !c.visible(x, y)) {
            return;
        }

        var cl = c.win.readCell(@intCast(x), @intCast(y)) orelse {
            return;
        };
        const base = if (cl.style.bg == .rgb) cl.style.bg else T.bg;
        cl.style.bg = T.mix(base, col, a);
        c.win.writeCell(@intCast(x), @intCast(y), cl);
    }

    /// Print single-width text; returns the column after it.
    pub fn put(c: *const Canvas, x: i32, y: i32, s: []const u8, st: Style) i32 {
        var col = x;
        const view = std.unicode.Utf8View.init(s) catch {
            // invalid UTF-8 (never expected): one '?' per non-ASCII byte
            for (s) |b| {
                c.cell(col, y, if (b < 0x80) ascii[b .. b + 1] else "?", st);
                col += 1;
            }

            return col;
        };
        var it = view.iterator();

        while (it.nextCodepointSlice()) |cp| : (col += 1) {
            c.cell(col, y, cp, st);
        }

        return col;
    }

    /// Print text ending at column `end` (exclusive).
    pub fn putRight(c: *const Canvas, end: i32, y: i32, s: []const u8, st: Style) void {
        _ = c.put(end - textWidth(s), y, s, st);
    }

    /// Print text centered in [x, x+w).
    pub fn putCenter(
        c: *const Canvas,
        x: i32,
        w: i32,
        y: i32,
        s: []const u8,
        st: Style,
    ) void {
        _ = c.put(x + @divFloor(w - textWidth(s), 2), y, s, st);
    }

    pub fn fill(c: *const Canvas, r: Rect, st: Style) void {
        var y = r.y;

        while (y < r.y + r.h) : (y += 1) {
            var x = r.x;

            while (x < r.x + r.w) : (x += 1) {
                c.cell(x, y, " ", st);
            }
        }
    }

    pub fn hline(
        c: *const Canvas,
        x: i32,
        y: i32,
        w: i32,
        g: []const u8,
        st: Style,
    ) void {
        var i: i32 = 0;

        while (i < w) : (i += 1) {
            c.cell(x + i, y, g, st);
        }
    }

    /// Card: filled rounded box, title in the top border, optional note on
    /// the right. Returns the inner rect (1 cell padding on the sides).
    pub fn card(c: *const Canvas, r: Rect, title: []const u8, note: []const u8, o: CardOpts) Rect {
        const bgc = if (o.active) T.card_hi else T.card;
        const bc = o.border orelse (if (o.active) T.border_hi else T.border);
        c.fill(r, .{ .bg = bgc });
        const bs: Style = .{ .fg = bc, .bg = bgc };
        c.cell(r.x, r.y, "╭", bs);
        c.cell(r.right() - 1, r.y, "╮", bs);
        c.cell(r.x, r.bottom() - 1, "╰", bs);
        c.cell(r.right() - 1, r.bottom() - 1, "╯", bs);
        c.hline(r.x + 1, r.y, r.w - 2, "─", bs);
        c.hline(r.x + 1, r.bottom() - 1, r.w - 2, "─", bs);
        var y = r.y + 1;

        while (y < r.bottom() - 1) : (y += 1) {
            c.cell(r.x, y, "│", bs);
            c.cell(r.right() - 1, y, "│", bs);
        }

        if (title.len > 0) {
            var x = c.put(r.x + 2, r.y, " ", .{ .bg = bgc });

            if (o.icon.len > 0) {
                x = c.put(x, r.y, o.icon, .{ .fg = o.icon_color orelse o.title_color, .bg = bgc, .bold = true });
                x += 1;
            }

            x = c.put(x, r.y, title, .{ .fg = o.title_color, .bg = bgc, .bold = true });
            _ = c.put(x, r.y, " ", .{ .bg = bgc });
        }

        if (note.len > 0) {
            const nw = textWidth(note);

            if (r.x + 4 + textWidth(title) + 6 + nw < r.right()) {
                _ = c.put(r.right() - 3 - nw, r.y, " ", .{ .bg = bgc });
                c.putRight(r.right() - 2, r.y, note, .{ .fg = o.note_color, .bg = bgc });
                _ = c.put(r.right() - 2, r.y, " ", .{ .bg = bgc });
            }
        }

        return .{ .x = r.x + 2, .y = r.y + 1, .w = @max(0, r.w - 4), .h = @max(0, r.h - 2) };
    }

    /// Horizontal gauge with eighth-block resolution.
    pub fn bar(
        c: *const Canvas,
        x: i32,
        y: i32,
        w: i32,
        frac_in: f32,
        on: Color,
        off: Color,
    ) void {
        const frac = std.math.clamp(frac_in, 0, 1);
        const eighths: i32 = @intFromFloat(@round(frac * @as(f32, @floatFromInt(w * 8))));
        const glyphs = [_][]const u8{
            " ",
            "▏",
            "▎",
            "▍",
            "▌",
            "▋",
            "▊",
            "▉",
            "█",
        };
        var i: i32 = 0;

        while (i < w) : (i += 1) {
            const f = std.math.clamp(eighths - i * 8, 0, 8);

            if (f == 8) {
                c.cell(x + i, y, " ", .{ .bg = on });
            } else {
                c.cell(x + i, y, glyphs[@intCast(f)], .{ .fg = on, .bg = off });
            }
        }
    }

    /// Bar whose color follows a gradient along its length.
    pub fn gradientBar(
        c: *const Canvas,
        x: i32,
        y: i32,
        w: i32,
        frac_in: f32,
        from: Color,
        to: Color,
        off: Color,
    ) void {
        const frac = std.math.clamp(frac_in, 0, 1);
        const eighths: i32 = @intFromFloat(@round(frac * @as(f32, @floatFromInt(w * 8))));
        const glyphs = [_][]const u8{
            " ",
            "▏",
            "▎",
            "▍",
            "▌",
            "▋",
            "▊",
            "▉",
            "█",
        };
        var i: i32 = 0;

        while (i < w) : (i += 1) {
            const col = T.mix(from, to, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(@max(1, w - 1))));
            const f = std.math.clamp(eighths - i * 8, 0, 8);

            if (f == 8) {
                c.cell(x + i, y, " ", .{ .bg = col });
            } else {
                c.cell(x + i, y, glyphs[@intCast(f)], .{ .fg = col, .bg = off });
            }
        }
    }

    /// Vertical mini bars (▁…█), values 0..1.
    pub fn spark(c: *const Canvas, x: i32, y: i32, v: f32, st: Style) void {
        const g = [_][]const u8{
            "▁",
            "▂",
            "▃",
            "▄",
            "▅",
            "▆",
            "▇",
            "█",
        };
        const k: usize = @intFromFloat(std.math.clamp(@round(v * 7), 0, 7));
        c.cell(x, y, g[k], st);
    }

    /// Key cap " k " followed by a label; clickable (sends the key).
    pub fn keycap(
        c: *const Canvas,
        x: i32,
        y: i32,
        key: []const u8,
        label: []const u8,
        cp: u21,
        o: KeyOpts,
    ) i32 {
        const kbg = if (o.hot) T.accent else T.border;
        const kfg = if (o.hot) T.bg else T.text;
        var col = c.put(x, y, " ", .{ .bg = kbg });
        col = c.put(col, y, key, .{ .fg = kfg, .bg = kbg, .bold = true });
        col = c.put(col, y, " ", .{ .bg = kbg });
        col = c.put(col + 1, y, label, .{ .fg = if (o.hot) T.accent else T.dim, .bg = o.bg });
        c.hits.add(.{ .x = x, .y = y, .w = col - x, .h = 1 }, .{ .key = cp });
        return col + 2;
    }

    /// Width taken by keycap(key, label) including the trailing gap.
    pub fn keycapWidth(key: []const u8, label: []const u8) i32 {
        return textWidth(key) + 2 + 1 + textWidth(label) + 2;
    }

    /// Width of button(key, label).
    pub fn buttonWidth(key: []const u8, label: []const u8) i32 {
        return textWidth(key) + 2 + 1 + textWidth(label) + 1;
    }

    /// Button "[k] label" with its hotkey in a darker cap; clickable.
    pub fn button(
        c: *const Canvas,
        x: i32,
        y: i32,
        key: []const u8,
        label: []const u8,
        cp: u21,
        fg: Color,
        bgc: Color,
    ) i32 {
        const cap = T.mix(bgc, T.bg_deep, 0.35);
        var col = c.put(x, y, " ", .{ .bg = cap });
        col = c.put(col, y, key, .{ .fg = fg, .bg = cap, .bold = true });
        col = c.put(col, y, " ", .{ .bg = cap });
        col = c.put(col, y, " ", .{ .bg = bgc });
        col = c.put(col, y, label, .{ .fg = fg, .bg = bgc, .bold = true });
        col = c.put(col, y, " ", .{ .bg = bgc });
        c.hits.add(.{ .x = x, .y = y, .w = col - x, .h = 1 }, .{ .key = cp });
        return col;
    }
};

pub const CardOpts = struct {
    title_color: Color = T.dim,
    icon: []const u8 = "",
    icon_color: ?Color = null,
    note_color: Color = T.faint,
    border: ?Color = null,
    active: bool = false,
};

pub const KeyOpts = struct {
    hot: bool = false,
    bg: Color = T.bg,
};

const ascii: [128]u8 = blk: {
    var t: [128]u8 = undefined;

    for (&t, 0..) |*b, i| {
        b.* = if (i < 0x20 or i == 0x7f) '?' else @intCast(i);
    }

    break :blk t;
};

pub fn textWidth(s: []const u8) i32 {
    const n = std.unicode.utf8CountCodepoints(s) catch s.len;
    return @intCast(n);
}

// ---- big digits (3 rows, 3 columns + 1 spacing) ------------------------------

const big_digits = [10][3][]const u8{
    .{ "╭─╮", "│ │", "╰─╯" },
    .{ " ╶┐", "  │", "  ╵" },
    .{ "╶─╮", "╭─╯", "╰─╴" },
    .{ "╶─╮", " ─┤", "╶─╯" },
    .{ "╷ ╷", "╰─┤", "  ╵" },
    .{ "╭─╴", "╰─╮", "╶─╯" },
    .{ "╭─╴", "├─╮", "╰─╯" },
    .{ "╶─╮", "  │", "  ╵" },
    .{ "╭─╮", "├─┤", "╰─╯" },
    .{ "╭─╮", "╰─┤", "╶─╯" },
};

const big_digits_bold = [10][3][]const u8{
    .{ "█▀█", "█ █", "▀▀▀" },
    .{ "▀█ ", " █ ", "▀▀▀" },
    .{ "▀▀█", "█▀▀", "▀▀▀" },
    .{ "▀▀█", " ▀█", "▀▀▀" },
    .{ "█ █", "▀▀█", "  ▀" },
    .{ "█▀▀", "▀▀█", "▀▀▀" },
    .{ "█▀▀", "█▀█", "▀▀▀" },
    .{ "▀▀█", "  █", "  ▀" },
    .{ "█▀█", "█▀█", "▀▀▀" },
    .{ "█▀█", "▀▀█", "▀▀▀" },
};

/// Digits, '.' and ':' in a 3-row font; returns the column after.
pub fn bigText(
    c: *const Canvas,
    x: i32,
    y: i32,
    s: []const u8,
    st: Style,
    bold: bool,
) i32 {
    var col = x;

    for (s) |ch| {
        switch (ch) {
            '0'...'9' => {
                const g = if (bold) big_digits_bold[ch - '0'] else big_digits[ch - '0'];

                for (g, 0..) |row, r| {
                    var it = std.unicode.Utf8View.initUnchecked(row).iterator();
                    var k: i32 = 0;

                    while (it.nextCodepointSlice()) |cp| : (k += 1) {
                        if (cp[0] != ' ') {
                            c.cell(col + k, y + @as(i32, @intCast(r)), cp, st);
                        }
                    }
                }

                col += 4;
            },
            '.' => {
                c.cell(col, y + 2, if (bold) "▀" else "•", st);
                col += 2;
            },
            ':' => {
                c.cell(col, y, "•", st);
                c.cell(col, y + 2, "•", st);
                col += 2;
            },
            else => {
                col += 2;
            },
        }
    }

    return col;
}

pub fn spinner(ms: u64) []const u8 {
    const frames = [_][]const u8{
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏",
    };
    return frames[@intCast((ms / 80) % frames.len)];
}

// ---- braille plotting ------------------------------------------------------------

const braille_table: [256][3]u8 = blk: {
    var t: [256][3]u8 = undefined;

    for (0..256) |i| {
        const cp: u21 = 0x2800 + @as(u21, @intCast(i));
        t[i] = .{ @intCast(0xE0 | (cp >> 12)), @intCast(0x80 | ((cp >> 6) & 0x3F)), @intCast(0x80 | (cp & 0x3F)) };
    }

    break :blk t;
};

fn dotBit(dx: usize, dy: usize) u8 {
    const bits = [2][4]u8{ .{ 0x01, 0x02, 0x04, 0x40 }, .{ 0x08, 0x10, 0x20, 0x80 } };
    return bits[dx][dy];
}

/// Line chart of `vals` (0..1, NaN = unknown, carried forward) in a w x h
/// cell area using braille dots. Above `mid` uses `hi`, below uses `lo`.
pub fn plot(
    c: *const Canvas,
    r: Rect,
    vals: []const f32,
    mid: f32,
    hi: Color,
    lo: Color,
    bgc: Color,
) void {
    if (r.w <= 0 or r.h <= 0) {
        return;
    }

    const W: usize = @intCast(r.w);
    const H: usize = @intCast(r.h);
    const dots_w = W * 2;
    const dots_h = H * 4;
    const grid = c.a.alloc(u8, W * H) catch {
        return;
    };
    @memset(grid, 0);

    // baseline
    const mid_dot: usize = @intFromFloat(
        std.math.clamp(@round((1 - mid) * @as(f32, @floatFromInt(dots_h - 1))), 0, @as(f32, @floatFromInt(dots_h - 1))),
    );
    const mid_row = mid_dot / 4;
    var x: usize = 0;

    while (x < W) : (x += 1) {
        const yy: i32 = @intCast(mid_row);
        c.cell(r.x + @as(i32, @intCast(x)), r.y + yy, if (mid_dot % 4 < 2) "╌" else "╌", .{ .fg = T.ghost, .bg = bgc });
    }

    if (vals.len == 0) {
        return;
    }

    var prev_y: ?usize = null;
    var dx: usize = 0;

    while (dx < dots_w) : (dx += 1) {
        // sample: map dot column to value index (last column = last value)
        const idx: usize = if (vals.len == 1) 0 else @min(vals.len - 1, dx * (vals.len - 1) / @max(1, dots_w - 1));
        // carry the last known value forward
        var v = vals[idx];

        if (std.math.isNan(v)) {
            var k = idx;

            while (k > 0 and std.math.isNan(vals[k])) {
                k -= 1;
            }

            v = vals[k];
        }

        if (std.math.isNan(v)) {
            prev_y = null;
            continue;
        }

        const fy: f32 = (1 - std.math.clamp(v, 0, 1)) * @as(f32, @floatFromInt(dots_h - 1));
        const y: usize = @intFromFloat(@round(fy));
        const y0 = if (prev_y) |p| @min(p, y) else y;
        const y1 = if (prev_y) |p| @max(p, y) else y;
        var yy = y0;

        while (yy <= y1) : (yy += 1) {
            grid[(yy / 4) * W + dx / 2] |= dotBit(dx % 2, yy % 4);
        }

        prev_y = y;
    }

    for (0..H) |row| {
        for (0..W) |col| {
            const b = grid[row * W + col];

            if (b == 0) {
                continue;
            }

            const above = row * 4 + 2 <= mid_dot;
            c.cell(
                r.x + @as(i32, @intCast(col)),
                r.y + @as(i32, @intCast(row)),
                &braille_table[b],
                .{ .fg = if (above) hi else lo, .bg = bgc },
            );
        }
    }
}

test "braille table" {
    try std.testing.expectEqualStrings("⠀", &braille_table[0]);
    try std.testing.expectEqualStrings("⣿", &braille_table[255]);
}
