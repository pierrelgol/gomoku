//! The goban: grained wood, grid, coordinates, stones, overlays and effects.
//!
//! Two scales (both roughly square on screen):
//!  * large   : 4 columns x 2 rows per intersection; stones are 3-cell pills
//!  * compact : 2 columns x 1 row per intersection; stones are "●"
//!
//! Draw passes: wood + grid -> background tints (heat map, ripples, glows)
//! -> stones / ghosts -> marks (cursor, forbidden points) -> floating labels.

const std = @import("std");
const Board = @import("../engine/Board.zig");
const Game = @import("../engine/Game.zig");
const Engine = @import("../engine/Engine.zig");
const T = @import("theme.zig");
const G = @import("gfx.zig");
const fx_mod = @import("fx.zig");

const Move = Board.Move;
const NONE = Board.no_move;
const N = Board.size;
const Canvas = G.Canvas;
const Rect = G.Rect;
const Color = T.Color;

pub const col_letters = "ABCDEFGHJKLMNOPQRST";

/// "K10" style coordinates (columns A..T without I, rows 1..19 from the bottom).
pub fn moveName(buf: []u8, m: Move) []const u8 {
    if (m == NONE or m >= Board.padded_cell_count or !Board.isOnBoard(m)) {
        return "--";
    }

    return std.fmt.bufPrint(buf, "{c}{d}", .{ col_letters[Board.columnOf(m)], N - Board.rowOf(m) }) catch "?";
}

pub const Geo = struct {
    cw: i32,
    ch: i32,
    /// screen position of intersection (0,0) = A19
    ox: i32,
    oy: i32,
    rect: Rect,

    pub fn large(g: Geo) bool {
        return g.cw >= 4;
    }

    pub fn pt(g: Geo, x: usize, y: usize) [2]i32 {
        return .{ g.ox + g.cw * @as(i32, @intCast(x)), g.oy + g.ch * @as(i32, @intCast(y)) };
    }

    pub fn ptm(g: Geo, m: Move) [2]i32 {
        return g.pt(Board.columnOf(m), Board.rowOf(m));
    }
};

pub fn size(large: bool) struct { w: i32, h: i32 } {
    const cw: i32 = if (large) 4 else 2;
    const ch: i32 = if (large) 2 else 1;
    return .{ .w = cw * (N - 1) + 1 + 8, .h = ch * (N - 1) + 1 + 4 };
}

pub fn geometry(large: bool, x: i32, y: i32) Geo {
    const s = size(large);
    return .{
        .cw = if (large) 4 else 2,
        .ch = if (large) 2 else 1,
        .ox = x + 4,
        .oy = y + 2,
        .rect = .{ .x = x, .y = y, .w = s.w, .h = s.h },
    };
}

/// Nearest intersection to a screen cell, if inside the grid area.
pub fn hit(g: Geo, col: i32, row: i32) ?[2]usize {
    const dx = col - g.ox;
    const dy = row - g.oy;
    const x = @divFloor(dx + @divFloor(g.cw, 2), g.cw);
    const y = @divFloor(dy + @divFloor(g.ch, 2), g.ch);

    if (x < 0 or y < 0 or x >= N or y >= N) {
        return null;
    }

    if (dx < -@divFloor(g.cw, 2) or dy < -@divFloor(g.ch, 2)) {
        return null;
    }

    return .{ @intCast(x), @intCast(y) };
}

pub const View = struct {
    cursor: ?[2]usize = null,
    cursor_legal: bool = true,
    /// stone color previewed under the cursor (null = no preview)
    ghost: ?Board.Color = null,
    /// board point highlighted from the side panel
    hover: Move = NONE,
    suggestion: Move = NONE,
    show_forbidden: bool = false,
    /// analysis overlay: engine's best line + candidate heat map
    pv: []const Move = &.{},
    pv_side: Board.Color = .black,
    heat: []const Engine.RootScore = &.{},
    vulnerable: bool = false,
    thinking: bool = false,
    thinking_color: Color = T.ai,
};

fn isHoshi(x: usize, y: usize) bool {
    return (x == 3 or x == 9 or x == 15) and (y == 3 or y == 9 or y == 15);
}

fn junction(x: usize, y: usize) []const u8 {
    if (y == 0) {
        return if (x == 0) "┌" else if (x == N - 1) "┐" else "┬";
    }

    if (y == N - 1) {
        return if (x == 0) "└" else if (x == N - 1) "┘" else "┴";
    }

    if (x == 0) {
        return "├";
    }

    if (x == N - 1) {
        return "┤";
    }

    if (isHoshi(x, y)) {
        return "╋";
    }

    return "┼";
}

fn hash(x: i32, y: i32) f32 {
    var h: u32 = @bitCast(x *% 374761393 +% y *% 668265263);
    h = (h ^ (h >> 13)) *% 1274126177;
    h ^= h >> 16;
    return @as(f32, @floatFromInt(h & 0xffff)) / 65535.0;
}

/// Wood grain: long, slightly wavy rings plus a little fibre noise.
fn grain(rx: i32, ry: i32) Color {
    const x: f32 = @floatFromInt(rx);
    const y: f32 = @floatFromInt(ry);
    const ring = @sin(y * 0.55 + @sin(x * 0.045) * 2.2 + @sin(x * 0.011 + y * 0.23) * 1.4);
    const t = 0.5 + 0.32 * ring + 0.16 * (hash(rx, ry) - 0.5);
    return T.mix(T.wood_a, T.wood_b, t * 0.75);
}

const nums = [_][]const u8{
    " 1",
    " 2",
    " 3",
    " 4",
    " 5",
    " 6",
    " 7",
    " 8",
    " 9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
};
const letter = [_][]const u8{
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
};
const digits = [_][]const u8{
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
};

/// Mark every stone belonging to a run of 5+ of color `code`.
pub fn markFives(b: *const Board, code: u8, out: *[Board.padded_cell_count]bool) bool {
    @memset(out, false);
    var any = false;
    const dirs = [_][2]i32{ .{ 1, 0 }, .{ 0, 1 }, .{ 1, 1 }, .{ 1, -1 } };

    for (0..N) |y| {
        for (0..N) |x| {
            if (b.cellAt(Board.moveAt(x, y)) != code) {
                continue;
            }

            for (dirs) |d| {
                const px = @as(i32, @intCast(x)) - d[0];
                const py = @as(i32, @intCast(y)) - d[1];

                if (inside(px, py) and b.cellAt(Board.moveAt(@intCast(px), @intCast(py))) == code) {
                    continue;
                }

                var len: usize = 0;
                var cx: i32 = @intCast(x);
                var cy: i32 = @intCast(y);

                while (inside(cx, cy) and b.cellAt(Board.moveAt(@intCast(cx), @intCast(cy))) == code) : (len += 1) {
                    cx += d[0];
                    cy += d[1];
                }

                if (len < 5) {
                    continue;
                }

                any = true;
                cx = @intCast(x);
                cy = @intCast(y);

                for (0..len) |_| {
                    out[Board.moveAt(@intCast(cx), @intCast(cy))] = true;
                    cx += d[0];
                    cy += d[1];
                }
            }
        }
    }

    return any;
}

fn inside(x: i32, y: i32) bool {
    return x >= 0 and y >= 0 and x < N and y < N;
}

fn stoneColor(code: u8) Color {
    return if (code == 1) T.stone_black else T.stone_white;
}

/// Perimeter index of a rim cell (clockwise from the top-left corner).
fn rimIndex(r: Rect, x: i32, y: i32) i32 {
    if (y == r.y) {
        return x - r.x;
    }

    if (x == r.right() - 1) {
        return r.w + (y - r.y);
    }

    if (y == r.bottom() - 1) {
        return r.w + r.h + (r.right() - 1 - x);
    }

    return 2 * r.w + r.h + (r.bottom() - 1 - y);
}

pub fn draw(c: *const Canvas, g: Geo, game: *const Game, v: View, fx: *const fx_mod.Fx) void {
    const b = &game.board;
    const r = g.rect;
    const now = c.ms;
    const big = g.large();
    const grid_w = g.cw * (N - 1);
    const grid_h = g.ch * (N - 1);

    // ---- what glows on the rim
    var win_mask: [Board.padded_cell_count]bool = undefined;
    var five_mask: [Board.padded_cell_count]bool = undefined;
    const won = game.outcome == .five and game.maybe_winner != null and markFives(
        b,
        game.maybe_winner.?.stoneCode(),
        &win_mask,
    );

    if (!won) {
        @memset(&win_mask, false);
    }

    const pf = game.pendingFive();
    const pending = pf != NONE and b.cellAt(pf) != 0 and markFives(b, b.cellAt(pf), &five_mask);

    if (!pending) {
        @memset(&five_mask, false);
    }

    const perim = 2 * (r.w + r.h);
    const head: i32 = @intCast((now / 9) % @as(u64, @intCast(perim)));

    // ---- pass 1: wood, rim, grid and labels
    var sy = r.y;

    while (sy < r.bottom()) : (sy += 1) {
        var sx = r.x;

        while (sx < r.right()) : (sx += 1) {
            const wood = grain(sx - r.x, sy - r.y);
            const top = sy == r.y;
            const bot = sy == r.bottom() - 1;
            const lft = sx == r.x;
            const rgt = sx == r.right() - 1;

            if (top or bot or lft or rgt) {
                // half-cell rim: the outer half is the "glow" channel
                var glow = T.bg;

                if (v.thinking) {
                    const d = @mod(head - rimIndex(r, sx, sy), perim);
                    const tail: i32 = 34;

                    if (d < tail) {
                        glow = T.mix(T.bg, v.thinking_color, 1 - @as(f32, @floatFromInt(d)) / tail);
                    }

                    const d2 = @mod(head + @divFloor(perim, 2) - rimIndex(r, sx, sy), perim);

                    if (d2 < tail) {
                        glow = T.mix(glow, v.thinking_color, 0.6 * (1 - @as(f32, @floatFromInt(d2)) / tail));
                    }
                } else if (won) {
                    glow = T.mix(T.bg, T.win_glow, 0.25 + 0.45 * T.pulse(now, 1400));
                } else if (pending) {
                    glow = T.mix(T.bg, T.bad, 0.2 + 0.5 * T.pulse(now, 700));
                }

                const g_: []const u8 = if (top and lft) "▗" else if (top and rgt) "▖" else if (bot and lft) "▝" else if (bot and rgt) "▘" else if (top) "▄" else if (bot) "▀" else if (lft) "▐" else "▌";
                c.cell(sx, sy, g_, .{ .fg = wood, .bg = glow });
                continue;
            }

            const dx = sx - g.ox;
            const dy = sy - g.oy;
            var glyph: []const u8 = " ";
            var st: G.Style = .{ .fg = T.line, .bg = wood };

            if (dx >= 0 and dy >= 0 and dx <= grid_w and dy <= grid_h) {
                const on_h = @mod(dy, g.ch) == 0;
                const on_v = @mod(dx, g.cw) == 0;

                if (on_h and on_v) {
                    const x: usize = @intCast(@divExact(dx, g.cw));
                    const y: usize = @intCast(@divExact(dy, g.ch));
                    glyph = junction(x, y);
                    st.bold = isHoshi(x, y);
                } else if (on_h) {
                    glyph = "─";
                } else if (on_v) {
                    glyph = "│";
                }
            } else {
                st = .{ .fg = T.label, .bg = wood, .bold = true };
                const cur = v.cursor;

                // column letters
                if ((sy == g.oy - 1 or sy == g.oy + grid_h + 1) and dx >= 0 and dx <= grid_w and @mod(dx, g.cw) == 0) {
                    const x: usize = @intCast(@divExact(dx, g.cw));
                    glyph = letter[x];

                    if (cur != null and cur.?[0] == x) {
                        st.fg = T.mix(T.label, T.last_mark, 0.85);
                    }
                }

                // row numbers
                if (dy >= 0 and dy <= grid_h and @mod(dy, g.ch) == 0) {
                    const y: usize = @intCast(@divExact(dy, g.ch));
                    const t = nums[N - 1 - y];

                    if (cur != null and cur.?[1] == y) {
                        st.fg = T.mix(T.label, T.last_mark, 0.85);
                    }

                    if (sx == r.x + 1) {
                        glyph = t[0..1];
                    } else if (sx == r.x + 2) {
                        glyph = t[1..2];
                    }

                    const rx = g.ox + grid_w + 2;

                    if (t[0] == ' ') {
                        if (sx == rx) {
                            glyph = t[1..2];
                        }
                    } else {
                        if (sx == rx) {
                            glyph = t[0..1];
                        } else if (sx == rx + 1) {
                            glyph = t[1..2];
                        }
                    }

                    if (std.mem.eql(u8, glyph, " ")) {
                        st.fg = T.line;
                    }
                }
            }

            c.cell(sx, sy, glyph, st);
        }
    }

    // ---- pass 2: background tints
    // candidate heat map
    if (v.heat.len > 0) {
        const best = v.heat[0].score;

        for (v.heat, 0..) |h, i| {
            if (h.move == NONE or h.move >= Board.padded_cell_count or !b.isEmpty(h.move)) {
                continue;
            }

            const diff: f32 = @floatFromInt(std.math.clamp(@as(i64, best) - h.score, 0, 800));
            var a: f32 = 0.18 + 0.5 * (1 - diff / 800.0);

            if (i == 0 and v.thinking) {
                a *= 0.7 + 0.3 * T.pulse(now, 600);
            }

            const col = if (i == 0) T.good else T.mix(T.good, T.cyan, @as(f32, @floatFromInt(i)) / 8.0);
            regionTint(c, g, h.move, col, a);
        }
    }

    // hint
    if (v.suggestion != NONE and b.isEmpty(v.suggestion)) {
        regionTint(c, g, v.suggestion, T.good, 0.35 + 0.35 * T.pulse(now, 900));
    }

    // winning line / breakable five glow (sweeping highlight)
    if (won or pending) {
        for (0..N) |y| {
            for (0..N) |x| {
                const m = Board.moveAt(x, y);

                if (!(win_mask[m] or five_mask[m])) {
                    continue;
                }

                const k: f32 = @floatFromInt(2 * x + 3 * y);
                const tt: f32 = @floatFromInt(now % 100000);
                const sweep = @max(0, @cos(k * 0.55 - tt / 140.0));
                const col = if (won) T.win_glow else T.bad;
                regionTint(c, g, m, col, 0.45 + 0.5 * sweep * sweep);
            }
        }
    }

    // placement ripple
    if (fx.place_m != NONE and b.cellAt(fx.place_m) != 0) {
        if (fx_mod.Fx.progress(now, fx.place_t, fx_mod.PLACE_MS)) |p| {
            ripple(
                c,
                g,
                fx.place_m,
                p,
                T.win_glow,
                0.55,
            );
        }
    }

    // captures: red shockwave
    for (fx.caps[0..fx.ncaps]) |cp| {
        if (fx_mod.Fx.progress(now, cp.t, fx_mod.CAPTURE_MS)) |p| {
            if (p < 0.55) {
                ripple(
                    c,
                    g,
                    cp.m,
                    p / 0.55,
                    T.bad,
                    0.5,
                );
            }
        }
    }

    // ---- pass 3: stones, ghosts, marks
    const last = b.lastMove();
    var pv_idx: [Board.padded_cell_count]u8 = @splat(0);

    for (v.pv[0..@min(v.pv.len, digits.len)], 0..) |m, i| {
        if (m != NONE and m < Board.padded_cell_count and b.isEmpty(m) and pv_idx[m] == 0) {
            pv_idx[m] = @intCast(i + 1);
        }
    }

    for (0..N) |y| {
        for (0..N) |x| {
            const m = Board.moveAt(x, y);
            const p = g.pt(x, y);
            const code = b.cellAt(m);

            if (code == 1 or code == 2) {
                var col = stoneColor(code);
                var mark: []const u8 = " ";
                var mark_fg = T.last_mark;

                if (m == last) {
                    mark = "●";
                }

                if (v.vulnerable and m != last and b.isStoneCapturable(m)) {
                    mark = "•";
                    mark_fg = T.bad;
                }

                // pop-in for a freshly placed stone
                var scale: f32 = 1;

                if (m == fx.place_m) {
                    if (fx_mod.Fx.progress(now, fx.place_t, 160)) |pp| {
                        scale = pp;
                    }
                }

                if (win_mask[m]) {
                    col = T.mix(col, T.win_glow, if (code == 1) 0.15 else 0.25);
                }

                stone(
                    c,
                    g,
                    p,
                    col,
                    mark,
                    mark_fg,
                    scale,
                    code == 1,
                );
                continue;
            }

            // dissolving captured stones
            var dissolving = false;

            for (fx.caps[0..fx.ncaps]) |cp| {
                if (cp.m != m) {
                    continue;
                }

                if (fx_mod.Fx.progress(now, cp.t, fx_mod.CAPTURE_MS)) |pp| {
                    dissolving = true;
                    const bgc = c.bgAt(p[0], p[1]);
                    const col = T.mix(stoneColor(cp.code), T.bad, @min(1, pp * 3) * 0.6);
                    const faded = T.mix(col, bgc, T.easeOut(pp));
                    stone(
                        c,
                        g,
                        p,
                        faded,
                        if (pp > 0.3) "×" else " ",
                        T.mix(
                            T.bad,
                            bgc,
                            pp,
                        ),
                        1 - pp * 0.7,
                        cp.code == 1,
                    );
                }
            }

            if (dissolving) {
                continue;
            }

            if (pv_idx[m] != 0) {
                const k = pv_idx[m];
                const mover_black = (k % 2 == 1) == (v.pv_side == .black);
                const bgc = c.bgAt(p[0], p[1]);
                const strength: f32 = if (k == 1) 0.7 else 0.5;
                const col = T.mix(bgc, if (mover_black) T.stone_black else T.stone_white, strength);
                const fg = if (mover_black) T.stone_white else T.stone_black;

                if (!big) {
                    c.cell(
                        p[0],
                        p[1],
                        digits[k - 1],
                        .{
                            .fg = if (mover_black) T.stone_black else T.stone_white,
                            .bg = T.mix(bgc, T.pv, 0.55),
                            .bold = true,
                        },
                    );
                    continue;
                }

                stone(
                    c,
                    g,
                    p,
                    col,
                    digits[k - 1],
                    T.mix(fg, T.pv, 0.25),
                    1,
                    mover_black,
                );
                continue;
            }

            if (m == v.suggestion) {
                c.cell(
                    p[0],
                    p[1],
                    "◆",
                    .{
                        .fg = T.mix(T.rgb(0x1E6B34), T.good, T.pulse(now, 900)),
                        .bg = c.bgAt(p[0], p[1]),
                        .bold = true,
                    },
                );
                continue;
            }

            if (v.show_forbidden and b.isForbidden(m, b.side_to_move)) {
                c.cell(
                    p[0],
                    p[1],
                    "×",
                    .{ .fg = T.mix(T.bad, T.last_mark, 0.4), .bg = c.bgAt(p[0], p[1]), .bold = true },
                );
            }
        }
    }

    // ---- pass 4: cursor, hover, ghost preview
    if (v.hover != NONE and v.hover < Board.padded_cell_count and Board.isOnBoard(v.hover)) {
        const p = g.ptm(v.hover);

        if (b.isEmpty(v.hover)) {
            ghostStone(c, g, p, if (b.side_to_move == .black) T.stone_black else T.stone_white, 0.5);
        }

        frame(c, g, p, T.mix(T.ai, T.text, T.pulse(now, 700)));
    }

    if (v.cursor) |cur| {
        const p = g.pt(cur[0], cur[1]);
        const m = Board.moveAt(cur[0], cur[1]);
        const breathe = T.pulse(now, 1600);
        const col = if (v.cursor_legal) T.mix(
            T.accent,
            T.rgb(0xFFF1D0),
            breathe * 0.6,
        ) else T.mix(T.bad, T.last_mark, breathe);

        if (v.ghost) |gc| {
            if (b.isEmpty(m) and v.cursor_legal) {
                ghostStone(c, g, p, if (gc == .black) T.stone_black else T.stone_white, 0.45 + 0.15 * breathe);
            }
        }

        if (!v.cursor_legal and b.isEmpty(m) and v.ghost != null) {
            c.cell(p[0], p[1], "×", .{ .fg = T.bad, .bg = c.bgAt(p[0], p[1]), .bold = true });
        }

        frame(c, g, p, col);
    }

    // ---- pass 5: floating labels (+2 on captures)
    for (fx.floats[0..fx.nfloats]) |fl| {
        const pp = fx_mod.Fx.progress(now, fl.t, fx_mod.FLOAT_MS) orelse {
            continue;
        };
        const p = g.ptm(fl.m);
        const rise: i32 = @intFromFloat(T.easeOut(pp) * @as(f32, @floatFromInt(2 * g.ch)));
        const y = p[1] - 1 - rise;
        const txt = fl.text[0..fl.len];
        const x = p[0] - @divFloor(G.textWidth(txt), 2);
        var i: i32 = 0;
        var it = std.unicode.Utf8View.initUnchecked(txt).iterator();

        while (it.nextCodepointSlice()) |cp| : (i += 1) {
            const bgc = c.bgAt(x + i, y);
            const a: f32 = if (pp < 0.7) 1 else 1 - (pp - 0.7) / 0.3;
            c.cell(x + i, y, cp, .{ .fg = T.mix(bgc, T.bad, a), .bg = T.mix(bgc, T.bg, 0.35 * a), .bold = true });
        }
    }
}

/// Tint the area of one intersection (3 cells in large mode).
fn regionTint(c: *const Canvas, g: Geo, m: Move, col: Color, a: f32) void {
    const p = g.ptm(m);

    if (g.large()) {
        c.tint(p[0] - 1, p[1], col, a);
        c.tint(p[0], p[1], col, a);
        c.tint(p[0] + 1, p[1], col, a);
        c.tint(p[0] - 1, p[1] - 1, col, a * 0.35);
        c.tint(p[0], p[1] - 1, col, a * 0.45);
        c.tint(p[0] + 1, p[1] - 1, col, a * 0.35);
        c.tint(p[0] - 1, p[1] + 1, col, a * 0.35);
        c.tint(p[0], p[1] + 1, col, a * 0.45);
        c.tint(p[0] + 1, p[1] + 1, col, a * 0.35);
    } else {
        c.tint(p[0], p[1], col, a);
    }
}

/// Expanding ring around an intersection; `p` = progress 0..1.
fn ripple(
    c: *const Canvas,
    g: Geo,
    m: Move,
    p: f32,
    col: Color,
    strength: f32,
) void {
    const ctr = g.ptm(m);
    const radius = T.easeOut(p) * 3.2;
    const width: f32 = 0.6;
    const fade = (1 - p) * strength;
    const reach: i32 = 4;
    var dy: i32 = -reach * g.ch;

    while (dy <= reach * g.ch) : (dy += 1) {
        var dx: i32 = -reach * g.cw;

        while (dx <= reach * g.cw) : (dx += 1) {
            const x = ctr[0] + dx;
            const y = ctr[1] + dy;

            if (!g.rect.inset(1, 1).contains(x, y)) {
                continue;
            }

            const fx_: f32 = @as(f32, @floatFromInt(dx)) / @as(f32, @floatFromInt(g.cw));
            const fy: f32 = @as(f32, @floatFromInt(dy)) / @as(f32, @floatFromInt(g.ch));
            const d = @sqrt(fx_ * fx_ + fy * fy);
            const a = (1 - @abs(d - radius) / width) * fade;

            if (a > 0) {
                c.tint(x, y, col, a);
            }
        }
    }
}

/// Draw a stone. `scale` < 1 gives the pop-in / dissolve frames.
fn stone(
    c: *const Canvas,
    g: Geo,
    p: [2]i32,
    col: Color,
    mark: []const u8,
    mark_fg: Color,
    scale: f32,
    dark: bool,
) void {
    _ = dark;
    const x = p[0];
    const y = p[1];

    if (scale < 0.45) {
        c.cell(x, y, "•", .{ .fg = col, .bg = c.bgAt(x, y), .bold = true });
        return;
    }

    if (!g.large() or scale < 0.8) {
        const glyph = if (g.large() or std.mem.eql(u8, mark, " ")) "●" else mark;
        const fg = if (g.large() or std.mem.eql(u8, mark, " ")) col else mark_fg;
        const bgc = c.bgAt(x, y);

        if (!g.large() and !std.mem.eql(u8, mark, " ")) {
            // compact: the mark is shown as a tinted background under the stone
            c.cell(x, y, "●", .{ .fg = col, .bg = T.mix(bgc, mark_fg, 0.55), .bold = true });
            return;
        }

        c.cell(x, y, glyph, .{ .fg = fg, .bg = bgc, .bold = true });
        return;
    }

    c.cell(x - 1, y, "▐", .{ .fg = col, .bg = c.bgAt(x - 1, y) });
    c.cell(x, y, mark, .{ .fg = mark_fg, .bg = col, .bold = true });
    c.cell(x + 1, y, "▌", .{ .fg = col, .bg = c.bgAt(x + 1, y) });
}

fn ghostStone(c: *const Canvas, g: Geo, p: [2]i32, col: Color, a: f32) void {
    const bgc = c.bgAt(p[0], p[1]);
    stone(
        c,
        g,
        p,
        T.mix(bgc, col, a),
        " ",
        col,
        1,
        false,
    );
}

/// Selection frame: rounded corners around an intersection.
fn frame(c: *const Canvas, g: Geo, p: [2]i32, col: Color) void {
    const x = p[0];
    const y = p[1];

    if (g.large()) {
        c.cell(x - 2, y - 1, "╭", .{ .fg = col, .bg = c.bgAt(x - 2, y - 1), .bold = true });
        c.cell(x + 2, y - 1, "╮", .{ .fg = col, .bg = c.bgAt(x + 2, y - 1), .bold = true });
        c.cell(x - 2, y + 1, "╰", .{ .fg = col, .bg = c.bgAt(x - 2, y + 1), .bold = true });
        c.cell(x + 2, y + 1, "╯", .{ .fg = col, .bg = c.bgAt(x + 2, y + 1), .bold = true });
        c.cell(x - 2, y, "╴", .{ .fg = col, .bg = c.bgAt(x - 2, y) });
        c.cell(x + 2, y, "╶", .{ .fg = col, .bg = c.bgAt(x + 2, y) });
    } else {
        c.cell(x - 1, y, "▕", .{ .fg = col, .bg = c.bgAt(x - 1, y), .bold = true });
        c.cell(x + 1, y, "▏", .{ .fg = col, .bg = c.bgAt(x + 1, y), .bold = true });
    }
}
