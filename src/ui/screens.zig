//! Screens: game (header, board, dashboard, footer, toasts, modals), menu
//! (animated title on a goban strip) and help.

const std = @import("std");
const Board = @import("../engine/Board.zig");
const app_mod = @import("app.zig");
const board_view = @import("board_view.zig");
const sidebar = @import("sidebar.zig");
const fx_mod = @import("fx.zig");
const T = @import("theme.zig");
const G = @import("gfx.zig");

const App = app_mod.App;
const Canvas = G.Canvas;
const Rect = G.Rect;
const Color = T.Color;
const NONE = Board.no_move;

const SIDE_MIN: i32 = 40;
const SIDE_MAX: i32 = 50;
/// narrowest dashboard column that is still legible
const SIDE_NARROW: i32 = 32;

pub fn draw(app: *App, c: *const Canvas) void {
    c.fill(.{ .x = 0, .y = 0, .w = c.width(), .h = c.height() }, .{ .bg = T.bg });
    app.geo = null;

    switch (app.screen) {
        .menu => {
            menu(app, c);
        },
        .game => {
            game(app, c);
        },
        .help => {
            help(app, c);
        },
    }
}

// ---- game ------------------------------------------------------------------------

fn game(app: *App, c: *const Canvas) void {
    const W = c.width();
    const H = c.height();

    const lg = board_view.size(true);
    const sm = board_view.size(false);
    // header and footer rows are dropped when the height is tight (80x24)
    const show_footer = H >= sm.h + 1;
    const show_header = H >= sm.h + 2;
    const top: i32 = if (show_header) 1 else 0;
    const avail_h = H - top - @as(i32, if (show_footer) 1 else 0);

    var large = false;

    if (app.zoom == .auto and lg.h <= avail_h and lg.w + 1 + SIDE_MIN <= W) {
        large = true;
    }

    const bs = board_view.size(large);

    // dashboard columns: beside the board, else below it
    const free = W - bs.w - 1;
    var ncols: i32 = 0;
    var cw: i32 = 0;

    if (free >= 2 * SIDE_MIN + 1 and !large and avail_h < 40) {
        ncols = 2;
        cw = @min(SIDE_MAX, @divFloor(free - 1, 2));
    } else if (free >= SIDE_NARROW) {
        ncols = 1;
        cw = @min(SIDE_MAX, free);
    }

    const below_h = avail_h - bs.h;
    const below = ncols == 0 and below_h >= 5 and W >= bs.w;

    if (bs.w > W or bs.h > avail_h or (ncols == 0 and !below)) {
        tooSmall(c, sm.w + 1 + SIDE_NARROW, sm.h + 1);
        return;
    }

    const total = bs.w + (if (ncols > 0) 1 + ncols * cw + (ncols - 1) else 0);
    const ox = @max(0, @divFloor(W - total, 2));
    const by = top + if (below) 0 else @max(0, @divFloor(avail_h - bs.h, 2));
    const geo = board_view.geometry(large, ox, by);
    app.geo = geo;

    if (show_header) {
        header(app, c, ox, total);
    }

    const show_cursor = app.settings.mode != .ava and !app.game.isOver() and app.cursor_on_board;
    const cur_m = Board.moveAt(app.cursor_x, app.cursor_y);
    const can = app.humanCanPlay();
    const info_ok = app.show_analysis and app.have_info;
    board_view.draw(c, geo, app.game, .{
        .cursor = if (show_cursor) .{ app.cursor_x, app.cursor_y } else null,
        .cursor_legal = !can or app.game.legality(cur_m) == .ok,
        .ghost = if (can) app.game.sideToMove() else null,
        .hover = app.hover_point,
        .suggestion = app.suggestion,
        .show_forbidden = can and app.game.board.rules.double_three,
        .pv = if (info_ok) app.last_info.principal_variation[0..@min(
            app.last_info.principal_variation_length,
            app.last_info.principal_variation.len,
        )] else &.{},
        .pv_side = app.info_side,
        .heat = if (info_ok and (app.thinking.load(
            .acquire,
        ) or app.info_is_suggestion)) app.last_info.top_moves[0..@min(
            app.last_info.top_move_count,
            app.last_info.top_moves.len,
        )] else &.{},
        .vulnerable = app.show_analysis and app.game.board.rules.captures,
        .thinking = app.thinking.load(.acquire),
        .thinking_color = if (app.job_is_suggestion) T.good else T.ai,
    }, &app.fx);

    // board click / hover region (registered first: everything else is on top)
    c.hits.add(geo.rect, .{ .key = 0 });

    if (ncols > 0) {
        const sx = geo.rect.right() + 1;
        var cols: [2]Rect = undefined;

        for (0..@intCast(ncols)) |i| {
            cols[i] = .{ .x = sx + @as(i32, @intCast(i)) * (cw + 1), .y = top, .w = cw, .h = avail_h };
        }

        sidebar.draw(c, app, cols[0..@intCast(ncols)]);
    } else if (below) {
        const col = [_]Rect{.{ .x = geo.rect.x, .y = geo.rect.bottom(), .w = geo.rect.w, .h = below_h }};
        sidebar.draw(c, app, &col);
    }

    // cursor legality hint under the board / in the toast area
    if (show_cursor and can) {
        const l = app.game.legality(cur_m);

        if (l != .ok and l != .occupied) {
            const msg = c.print(" × {s} ", .{l.message()});
            const y = geo.rect.y;
            c.putCenter(
                geo.rect.x,
                geo.rect.w,
                y,
                msg,
                .{ .fg = T.text, .bg = T.mix(T.bg, T.bad, 0.55), .bold = true },
            );
        }
    }

    toast(app, c, geo);

    if (app.game.isOver() and !app.hide_banner) {
        gameOver(app, c, geo);
    }

    if (app.game.pendingChoice() != .none and !app.isAiTurn() and !app.game.isOver()) {
        swapModal(app, c, geo);
    }

    if (show_footer) {
        footer(app, c);
    }
}

fn header(app: *const App, c: *const Canvas, x0: i32, w: i32) void {
    const W = c.width();
    c.fill(.{ .x = 0, .y = 0, .w = W, .h = 1 }, .{ .bg = T.bg_deep });
    const bgc = T.bg_deep;
    var x: i32 = @max(0, x0);
    const wood = T.mix(T.wood_a, T.wood_b, 0.4);
    c.cell(x, 0, "▐", .{ .fg = wood, .bg = bgc });
    c.cell(x + 1, 0, "●", .{ .fg = T.stone_black, .bg = wood, .bold = true });
    c.cell(x + 2, 0, "●", .{ .fg = T.stone_white, .bg = wood, .bold = true });
    c.cell(x + 3, 0, "▌", .{ .fg = wood, .bg = bgc });
    x = c.put(x + 5, 0, "GOMOKU", .{ .fg = T.text, .bg = bgc, .bold = true });
    x += 2;
    const s = &app.settings;
    const b = &app.game.board;
    const chips = [_][]const u8{
        s.mode.name(),
        c.print("{s} opening", .{app.game.opening.name()}),
        if (b.rules.captures) "captures" else "no captures",
        if (b.rules.double_three) "no double-three" else "double-three ok",
        if (s.endlessOn()) c.print(
            "endless · game {d}",
            .{app.series.games + @as(u32, if (app.game.isOver()) 0 else 1)},
        ) else "",
        if (s.catchUpOn()) "catch-up time" else "",
    };

    for (chips) |t| {
        if (t.len == 0) {
            continue;
        }

        if (x + G.textWidth(t) + 3 > x0 + w - 18) {
            break;
        }

        x = c.put(x, 0, c.print(" {s} ", .{t}), .{ .fg = T.dim, .bg = T.card_hi });
        x += 1;
    }

    // right: engine state pill
    const live = app.thinking.load(.acquire);
    const pill = if (live) c.print(
        " {s} {s} ",
        .{
            G.spinner(c.ms),
            if (app.job_is_suggestion) "hint…" else if (app.settings.catchUpOn() and app.thinkElapsedMs() > app.budgetMs()) "catching up…" else "thinking",
        },
    ) else if (app.game.isOver()) " game over " else " ready ";
    const pc: Color = if (live) T.ai else if (app.game.isOver()) T.win_glow else T.good;
    const end = @min(W, x0 + w);
    c.putRight(
        end,
        0,
        pill,
        .{ .fg = T.bg, .bg = T.mix(pc, T.bg_deep, if (live) 0.25 * T.pulse(c.ms, 800) else 0.15), .bold = true },
    );
}

fn footer(app: *const App, c: *const Canvas) void {
    const y = c.height() - 1;
    c.fill(.{ .x = 0, .y = y, .w = c.width(), .h = 1 }, .{ .bg = T.bg });
    const g = app.game;
    // (cap, priority): when the line is too narrow, the lowest priorities go
    const Item = struct { cap: Cap, prio: u8 };
    var items: [16]Item = undefined;
    var n: usize = 0;
    const add = struct {
        fn f(list: *[16]Item, len: *usize, cap: Cap, prio: u8) void {
            if (len.* < list.len) {
                list[len.*] = .{ .cap = cap, .prio = prio };
                len.* += 1;
            }
        }
    }.f;

    if (g.isOver()) {
        add(&items, &n, .{ .key = "n", .label = "New game", .cp = 'n', .hot = true }, 9);
        add(&items, &n, .{ .key = "v", .label = if (app.hide_banner) "Show result" else "View board", .cp = 'v' }, 6);
    } else if (g.pendingChoice() != .none and !app.isAiTurn()) {
        add(&items, &n, .{ .key = "b", .label = "Black", .cp = 'b', .hot = true }, 9);
        add(&items, &n, .{ .key = "w", .label = "White", .cp = 'w', .hot = true }, 9);

        if (g.pendingChoice() == .swap2_first) {
            add(&items, &n, .{ .key = "p", .label = "Place two", .cp = 'p', .hot = true }, 9);
        }
    } else if (app.settings.mode == .ava) {
        add(
            &items,
            &n,
            .{ .key = "p", .label = if (app.paused) "Resume" else "Pause", .cp = 'p', .hot = app.paused },
            9,
        );
    } else {
        add(&items, &n, .{ .key = "↵", .label = "Play", .cp = '\r' }, 5);
        add(&items, &n, .{ .key = "g", .label = "Hint", .cp = 'g', .hot = app.suggestion != NONE }, 7);
    }

    add(&items, &n, .{ .key = "u", .label = "Undo", .cp = 'u' }, 8);
    add(&items, &n, .{ .key = "r", .label = "Redo", .cp = 'r' }, 4);
    add(&items, &n, .{ .key = "d", .label = "Analysis", .cp = 'd', .hot = app.show_analysis }, 2);
    add(&items, &n, .{ .key = "z", .label = "Zoom", .cp = 'z' }, 1);
    add(&items, &n, .{ .key = "s", .label = "Save", .cp = 's' }, 6);
    add(&items, &n, .{ .key = "o", .label = "Load", .cp = 'o' }, 3);

    if (app.settings.mode != .ava and !g.isOver()) {
        add(&items, &n, .{ .key = "n", .label = "New", .cp = 'n' }, 3);
    }

    add(&items, &n, .{ .key = "?", .label = "Help", .cp = '?' }, 7);
    add(&items, &n, .{ .key = "q", .label = "Menu", .cp = 'q' }, 9);

    var keep: [16]bool = @splat(true);
    var width: i32 = 1;

    for (items[0..n]) |it| {
        width += Canvas.keycapWidth(it.cap.key, it.cap.label);
    }

    var p: u8 = 0;

    while (width - 2 > c.width() and p < 10) : (p += 1) {
        for (items[0..n], 0..) |it, i| {
            if (keep[i] and it.prio == p) {
                keep[i] = false;
                width -= Canvas.keycapWidth(it.cap.key, it.cap.label);
            }
        }
    }

    var x: i32 = 1;

    for (items[0..n], 0..) |it, i| {
        if (keep[i]) {
            x = c.keycap(
                x,
                y,
                it.cap.key,
                it.cap.label,
                it.cap.cp,
                .{ .hot = it.cap.hot },
            );
        }
    }
}

fn toast(app: *const App, c: *const Canvas, geo: board_view.Geo) void {
    const f = &app.fx;

    if (f.toast_len == 0) {
        return;
    }

    const age = c.ms -| f.toast_t;
    const col: Color = switch (f.toast_kind) {
        .info => T.text,
        .good => T.good,
        .bad => T.bad,
        .ai => T.ai,
    };
    const icon: []const u8 = switch (f.toast_kind) {
        .info => "•",
        .good => "✓",
        .bad => "×",
        .ai => "◆",
    };
    // floats over the bottom rows of the board, never wider than it
    const area = geo.rect.inset(2, 0);
    var txt = c.print("  {s}  {s}  ", .{ icon, f.toastText() });

    if (G.textWidth(txt) > area.w) {
        txt = c.print(" {s} {s} ", .{ icon, f.toastText() });
    }

    const cc = c.clipped(area);
    const slide: i32 = if (age < 180) 1 else 0;
    const fade: f32 = if (age + 500 > fx_mod.TOAST_MS) @as(f32, @floatFromInt(fx_mod.TOAST_MS -| age)) / 500.0 else 1;
    const y = geo.rect.bottom() - @as(i32, if (geo.large()) 4 else 3) + slide;
    const w = @min(G.textWidth(txt), area.w);
    const x = area.x + @divFloor(area.w - w, 2);
    var i: i32 = 0;
    var it = std.unicode.Utf8View.initUnchecked(txt).iterator();
    const pill_bg = T.mix(T.card_hi, col, 0.18);

    while (it.nextCodepointSlice()) |cp| : (i += 1) {
        if (i >= w) {
            break;
        }

        const under = c.bgAt(x + i, y);
        // an ellipsis when the text is cut
        const g = if (i == w - 1 and G.textWidth(txt) > w) "…" else cp;
        cc.cell(x + i, y, g, .{ .fg = T.mix(under, col, fade), .bg = T.mix(under, pill_bg, fade), .bold = true });
    }

    c.cell(x - 1, y, "▐", .{ .fg = T.mix(c.bgAt(x - 1, y), pill_bg, fade), .bg = c.bgAt(x - 1, y) });
    c.cell(x + w, y, "▌", .{ .fg = T.mix(c.bgAt(x + w, y), pill_bg, fade), .bg = c.bgAt(x + w, y) });
}

fn gameOver(app: *const App, c: *const Canvas, geo: board_view.Geo) void {
    const age = c.ms -| app.fx.over_t;

    if (age < 450) {
        return; // let the winning line sweep first
    }

    const g = app.game;
    const w: i32 = @min(48, geo.rect.w - 2);
    const h: i32 = 9;
    const x = geo.rect.x + @divFloor(geo.rect.w - w, 2);
    // keep the winning stones visible: use the half away from the last move
    const last = g.board.lastMove();
    const upper_half = last != NONE and Board.rowOf(last) < Board.size / 2;
    const y = if (upper_half) geo.rect.bottom() - h - 2 else geo.rect.y + 2;
    const r: Rect = .{ .x = x, .y = y, .w = w, .h = h };
    // drop shadow
    c.fill(.{ .x = x + 1, .y = y + 1, .w = w, .h = h }, .{ .bg = T.bg_deep });
    const in = c.card(
        r,
        "RESULT",
        "",
        .{ .border = T.mix(T.border, T.win_glow, 0.4 + 0.4 * T.pulse(c.ms, 1400)), .title_color = T.win_glow },
    );
    const bgc = T.card;
    const title = if (g.maybe_winner) |wc| c.print(
        "★  {s} WINS  ★",
        .{if (wc == .black) "BLACK" else "WHITE"},
    ) else "DRAW";
    // shimmering title
    const tw = G.textWidth(title);
    const tx = in.x + @divFloor(in.w - tw, 2);
    var i: i32 = 0;
    var it = std.unicode.Utf8View.initUnchecked(title).iterator();
    const sweep: f32 = @floatFromInt((c.ms / 40) % 80);

    while (it.nextCodepointSlice()) |cp| : (i += 1) {
        const d = @abs(@as(f32, @floatFromInt(i)) - (sweep - 20));
        const glint = @max(0, 1 - d / 4);
        c.cell(tx + i, in.y + 1, cp, .{ .fg = T.mix(T.win_glow, T.rgb(0xFFFFFF), glint), .bg = bgc, .bold = true });
    }

    const who = if (g.maybe_winner) |wc| app.ownerName(wc) else "";
    const reason = switch (g.outcome) {
        .five => c.print("{s} aligned five stones", .{who}),
        .capture => c.print("{s} captured 10 stones", .{who}),
        else => "the board is full",
    };
    c.putCenter(in.x, in.w, in.y + 2, reason, .{ .fg = T.dim, .bg = bgc });
    const avg = if (app.think_count > 0) app.total_think_ms / app.think_count else 0;
    c.putCenter(
        in.x,
        in.w,
        in.y + 4,
        c.print("{d} moves · AI avg {d} ms · max {d} ms", .{ g.ply(), avg, app.max_think_ms }),
        .{ .fg = T.faint, .bg = bgc },
    );

    if (app.settings.endlessOn()) {
        const left = app_mod.ENDLESS_PAUSE_MS -| (c.ms -| app.fx.over_t);
        const sr = &app.series;
        c.putCenter(
            in.x,
            in.w,
            in.y + 3,
            c.print(
                "series ● {d} ○ {d} = {d} · next in {d}s",
                .{ sr.wins[0], sr.wins[1], sr.draws, (left + 999) / 1000 },
            ),
            .{ .fg = T.ai, .bg = bgc },
        );
    }

    // buttons
    const by = in.y + 6;
    const full = Canvas.buttonWidth("n", "New game") + 2 + Canvas.buttonWidth("u", "Undo") + 2 + Canvas.buttonWidth(
        "v",
        "View board",
    );
    const narrow = full > in.w;
    const view = if (narrow) "View" else "View board";
    const gap: i32 = if (narrow) 1 else 2;
    const bw = Canvas.buttonWidth(
        "n",
        "New game",
    ) + gap + Canvas.buttonWidth("u", "Undo") + gap + Canvas.buttonWidth("v", view);
    var bx = in.x + @max(0, @divFloor(in.w - bw, 2));
    bx = c.button(
        bx,
        by,
        "n",
        "New game",
        'n',
        T.bg,
        T.win_glow,
    ) + gap;
    bx = c.button(
        bx,
        by,
        "u",
        "Undo",
        'u',
        T.text,
        T.border_hi,
    ) + gap;
    _ = c.button(
        bx,
        by,
        "v",
        view,
        'v',
        T.text,
        T.border_hi,
    );
}

fn swapModal(app: *const App, c: *const Canvas, geo: board_view.Geo) void {
    const g = app.game;
    const ch = g.pendingChoice();
    const w: i32 = @min(52, geo.rect.w - 2);
    const h: i32 = 8;
    const x = geo.rect.x + @divFloor(geo.rect.w - w, 2);
    const y = geo.rect.y + 2;
    c.fill(.{ .x = x + 1, .y = y + 1, .w = w, .h = h }, .{ .bg = T.bg_deep });
    const in = c.card(
        .{ .x = x, .y = y, .w = w, .h = h },
        c.print("{s} OPENING", .{g.opening.name()}),
        "",
        .{ .border = T.accent, .title_color = T.accent },
    );
    const bgc = T.card;
    c.putCenter(
        in.x,
        in.w,
        in.y,
        c.print("{s}, choose your color", .{app.actorName()}),
        .{ .fg = T.text, .bg = bgc, .bold = true },
    );
    c.putCenter(in.x, in.w, in.y + 1, switch (ch) {
        .swap => "three stones down: keep or swap sides",
        .swap2_first => "take a color, or place two more stones",
        else => "two more stones placed: pick a color",
    }, .{ .fg = T.dim, .bg = bgc });
    const by = in.y + 4;
    const two = ch == .swap2_first;
    const full = Canvas.buttonWidth("b", "Black ●") + 2 + Canvas.buttonWidth("w", "White ○") +
        (if (two) 2 + Canvas.buttonWidth("p", "Place two") else 0);
    const narrow = full > in.w;
    const gap: i32 = if (narrow) 1 else 2;
    const lb = if (narrow) "Black" else "Black ●";
    const lw = if (narrow) "White" else "White ○";
    const lp = if (narrow) "Place 2" else "Place two";
    const bw = Canvas.buttonWidth(
        "b",
        lb,
    ) + gap + Canvas.buttonWidth("w", lw) + (if (two) gap + Canvas.buttonWidth("p", lp) else 0);
    var bx = in.x + @max(0, @divFloor(in.w - bw, 2));
    const pulse = T.pulse(c.ms, 1200);
    bx = c.button(
        bx,
        by,
        "b",
        lb,
        'b',
        T.text,
        T.mix(T.stone_black, T.border_hi, 0.5 + 0.3 * pulse),
    ) + gap;
    bx = c.button(
        bx,
        by,
        "w",
        lw,
        'w',
        T.stone_black,
        T.mix(T.stone_white_lo, T.stone_white, pulse),
    ) + gap;

    if (two) {
        _ = c.button(
            bx,
            by,
            "p",
            lp,
            'p',
            T.bg,
            T.accent,
        );
    }
}

fn tooSmall(c: *const Canvas, need_w: i32, need_h: i32) void {
    const W = c.width();
    const H = c.height();
    const wide = W >= 34;
    const y = @max(0, @divFloor(H, 2) - 1);
    c.putCenter(0, W, y, if (wide) "Terminal too small" else "too small", .{ .fg = T.bad, .bg = T.bg, .bold = true });
    c.putCenter(
        0,
        W,
        y + 1,
        c.print("need {d}×{d}, have {d}×{d}", .{ need_w, need_h, W, H }),
        .{ .fg = T.dim, .bg = T.bg },
    );

    if (wide) {
        c.putCenter(0, W, y + 2, c.print("{s} enlarge the window", .{G.spinner(c.ms)}), .{ .fg = T.faint, .bg = T.bg });
    }
}

// ---- menu ------------------------------------------------------------------------------

const font = [_][5][]const u8{
    .{ ".###.", "#....", "#.###", "#...#", ".###." }, // G
    .{ ".###.", "#...#", "#...#", "#...#", ".###." }, // O
    .{ "#...#", "##.##", "#.#.#", "#...#", "#...#" }, // M
    .{ ".###.", "#...#", "#...#", "#...#", ".###." }, // O
    .{ "#...#", "#..#.", "###..", "#..#.", "#...#" }, // K
    .{ "#...#", "#...#", "#...#", "#...#", ".###." }, // U
};

fn hash(i: u32) u32 {
    var h = i *% 2654435761;
    h ^= h >> 15;
    h *%= 2246822519;
    h ^= h >> 13;
    return h;
}

/// Ambient background: a slowly breathing dot grid and drifting stones.
fn ambient(c: *const Canvas) void {
    const t: f32 = @floatFromInt(c.ms % 1_000_000);
    var y: i32 = 0;

    while (y < c.height()) : (y += 2) {
        var x: i32 = 2;

        while (x < c.width()) : (x += 4) {
            const fx_: f32 = @floatFromInt(x);
            const fy: f32 = @floatFromInt(y);
            const wv = 0.5 + 0.5 * @sin(fx_ * 0.08 + fy * 0.21 - t / 900.0) * @cos(fx_ * 0.03 - t / 1700.0);
            // quantized so most frames leave most dots unchanged
            const q = @round(wv * 5) / 5;
            c.cell(x, y, "·", .{ .fg = T.mix(T.bg, T.faint, 0.25 + 0.6 * q), .bg = T.bg });
        }
    }

    // drifting stones
    var i: u32 = 0;

    while (i < 18) : (i += 1) {
        const h = hash(i + 7);
        const speed: f32 = 0.0018 + @as(f32, @floatFromInt(h % 100)) / 100.0 * 0.003;
        const hh: f32 = @floatFromInt(c.height() + 4);
        const py = @mod(@as(f32, @floatFromInt(h % 997)) + t * speed, hh) - 2;
        const px: i32 = @intCast((h >> 8) % @as(u32, @intCast(@max(1, c.width()))));
        const white = (h >> 3) & 1 == 1;
        const a: f32 = 0.18 + @as(f32, @floatFromInt((h >> 5) % 10)) / 60.0;
        c.cell(
            px,
            @intFromFloat(py),
            if (white) "○" else "●",
            .{ .fg = T.mix(T.bg, if (white) T.stone_white else T.dim, a), .bg = T.bg },
        );
    }
}

fn titleStrip(c: *const Canvas, x0: i32, y0: i32, t0: u64) void {
    const pw: i32 = 2; // columns per pixel
    const letter_w: i32 = 5 * pw;
    const gap: i32 = 2;
    const w: i32 = 6 * letter_w + 5 * gap + 4;
    const h: i32 = 7;
    const age = c.ms -| t0;
    // goban strip
    var y: i32 = 0;

    while (y < h) : (y += 1) {
        var x: i32 = 0;

        while (x < w) : (x += 1) {
            const wood = T.mix(
                T.wood_a,
                T.wood_b,
                0.5 + 0.3 * @sin(@as(f32, @floatFromInt(y)) * 1.3 + @as(f32, @floatFromInt(x)) * 0.05),
            );
            const edge_t = y == 0;
            const edge_b = y == h - 1;

            if (edge_t or edge_b) {
                c.cell(x0 + x, y0 + y, if (edge_t) "▄" else "▀", .{ .fg = wood, .bg = T.bg });
                continue;
            }

            const gx = x - 2;
            const on_v = gx >= 0 and gx < w - 4 and @mod(gx, pw) == 0;
            c.cell(x0 + x, y0 + y, if (on_v) "┼" else "─", .{ .fg = T.mix(wood, T.line, 0.45), .bg = wood });
        }
    }

    // stones, revealed left to right, with a periodic glint
    const glint_pos: f32 = @as(f32, @floatFromInt(age % 4200)) / 22.0 - 20;

    for (font, 0..) |glyph, li| {
        const white = li % 2 == 1;
        const lx = x0 + 2 + @as(i32, @intCast(li)) * (letter_w + gap);

        for (glyph, 0..) |row, ry| {
            for (row, 0..) |px, rx| {
                if (px != '#') {
                    continue;
                }

                const sx = lx + @as(i32, @intCast(rx)) * pw;
                const sy = y0 + 1 + @as(i32, @intCast(ry));
                const order: u64 = @intCast(sx - x0);
                const appear = order * 9;

                if (age < appear) {
                    continue;
                }

                const wood = c.bgAt(sx, sy);
                var col = if (white) T.stone_white else T.stone_black;
                const d = @abs(@as(f32, @floatFromInt(sx - x0)) + @as(f32, @floatFromInt(ry)) * 2 - glint_pos);
                const gl = @max(0, 1 - d / 5);
                col = T.mix(col, if (white) T.rgb(0xFFFFFF) else T.stone_black_hi, gl);
                const pop = age - appear < 70;
                c.cell(
                    sx,
                    sy,
                    if (pop) "•" else "●",
                    .{ .fg = col, .bg = T.mix(wood, T.win_glow, gl * 0.35), .bold = true },
                );
            }
        }
    }
}

const MenuRow = app_mod.MenuRow;

/// One line in a row of key caps; drawn centered, dropping caps that do
/// not fit.
const Cap = struct { key: []const u8, label: []const u8, cp: u21, hot: bool = false };

fn capsCentered(c: *const Canvas, y: i32, caps: []const Cap) void {
    var total: i32 = 0;
    var n: usize = 0;

    for (caps) |k| {
        const w = Canvas.keycapWidth(k.key, k.label);

        if (total + w - 2 > c.width()) {
            break;
        }

        total += w;
        n += 1;
    }

    var x: i32 = @max(0, @divFloor(c.width() - (total - 2), 2));
    c.fill(.{ .x = x - 1, .y = y, .w = total, .h = 1 }, .{ .bg = T.bg });

    for (caps[0..n]) |k| {
        x = c.keycap(
            x,
            y,
            k.key,
            k.label,
            k.cp,
            .{ .hot = k.hot },
        );
    }
}

/// Value shown on the right of a settings row.
fn menuValue(app: *const App, c: *const Canvas, r: MenuRow) []const u8 {
    const s = &app.settings;
    return switch (r) {
        .mode => s.mode.name(),
        .human_first => if (s.human_first) "Black · first" else "White · second",
        .opening => s.opening.name(),
        .think => if (app.budgetMs() > app_mod.time_limit_ms) c.print(
            "{d} ms · analysis",
            .{app.budgetMs()},
        ) else c.print("{d} ms", .{app.budgetMs()}),
        .captures => if (s.captures) "on" else "off",
        .double_three => if (s.double_three) "forbidden" else "allowed",
        .endless => if (s.endless) "on" else "off",
        .catch_up => if (s.catch_up) c.print("on · ≤ {d} s", .{app.catchUpMs() / 1000}) else "off",
        else => "",
    };
}

fn menuLabel(r: MenuRow) []const u8 {
    return switch (r) {
        .resume_game => "Continue",
        .mode => "Mode",
        .human_first => "You play",
        .opening => "Opening",
        .think => "AI time / move",
        .captures => "Captures",
        .double_three => "Double-three",
        .endless => "Endless",
        .catch_up => "Catch-up time",
        .start => "New game",
        .load => "Load saved game",
        .quit => "Quit",
    };
}

fn menuActive(app: *const App, r: MenuRow) bool {
    const m = app.settings.mode;
    return switch (r) {
        .human_first => m == .pva,
        .endless => m == .ava,
        .catch_up => m != .hotseat,
        else => true,
    };
}

/// Setting that only takes effect in a new game (shown when a game exists).
fn nextGameOnly(r: MenuRow) bool {
    return r == .opening or r == .captures or r == .double_three;
}

fn menuDesc(app: *const App, c: *const Canvas, r: MenuRow) []const u8 {
    const s = &app.settings;
    const g = app.game;
    const later = app.has_game and nextGameOnly(r);
    const desc: []const u8 = switch (r) {
        .resume_game => if (g.isOver())
            (if (g.maybe_winner) |w| c.print(
                "Back to the finished game ({s} won in {d} moves).",
                .{ app_mod.colorName(w), g.ply() },
            ) else "Back to the finished game (draw).")
        else
            c.print("Back to the game: move {d}, {s} to play.", .{ g.ply() + 1, app_mod.colorName(g.sideToMove()) }),
        .mode => switch (s.mode) {
            .pva => "Play against the engine.",
            .hotseat => "Two players, one keyboard; the engine suggests a move (◆) each turn.",
            .ava => "Watch the engine play both sides. p pauses.",
        },
        .human_first => if (s.mode == .pva) "Black moves first." else "Only used in Player vs AI mode.",
        .opening => switch (s.opening) {
            .standard => "No restriction on the first moves.",
            .pro => "First stone at the centre; Black's second stone 3+ away from it.",
            .long_pro => "First stone at the centre; Black's second stone 4+ away from it.",
            .swap => "After 3 stones, the second player picks a color.",
            .swap2 => "After 3 stones: pick a color, or add 2 stones and let the other pick.",
        },
        .think => if (app.budgetMs() <= 450)
            "Time budget for each AI move, within the subject's 0.5 s limit."
        else
            "Analysis only: beyond the subject's 0.5 s limit (the timer flags it).",
        .captures => "Flank exactly two enemy stones to remove them; 10 captured stones win.",
        .double_three => "A move creating two free-threes at once is forbidden.",
        .endless => if (s.mode == .ava) "Play games back to back; the series score is kept." else "Only for AI vs AI.",
        .catch_up => if (s.mode != .hotseat) "Analysis only: when behind, the AI thinks past the 0.5 s limit." else "Only when the AI plays.",
        .start => if (app.has_game) "Abandon the current game and start a new one with these settings." else "Start a new game with these settings.",
        .load => c.print("Load the game saved in {s} (s saves during a game).", .{app_mod.save_path}),
        .quit => "Leave the program.",
    };
    return if (later) c.print("{s} (next game)", .{desc}) else desc;
}

fn menu(app: *App, c: *const Canvas) void {
    ambient(c);
    const W = c.width();
    const H = c.height();
    var buf: [app_mod.MENU_MAX]MenuRow = undefined;
    const rows = app.menuRows(&buf);
    const sel = @min(app.menu_sel, rows.len - 1);

    // row offsets inside the card: a gap after "Continue" and before the actions
    var offs: [app_mod.MENU_MAX]i32 = undefined;
    var off: i32 = 0;

    for (rows, 0..) |r, i| {
        if (i > 0 and (r == .start or (r == .mode and rows[0] == .resume_game))) {
            off += 1;
        }

        offs[i] = off;
        off += 1;
    }

    const rows_h = off;
    const card_w: i32 = @min(64, W - 2);
    const min_h = rows_h + 2 + 3; // card + description + keys

    if (W < 50 or H < min_h) {
        tooSmall(c, 50, min_h);
        return;
    }

    // extras by priority as the height allows: title, engine line, padding
    var room = H - min_h;
    var title_h: i32 = 0;

    if (W >= 82 and room >= 9 + 2 + 2) {
        title_h = 9;
    } else if (room >= 2) {
        title_h = 2;
    }

    room -= title_h;
    const engine_line = room >= 2;

    if (engine_line) {
        room -= 2;
    }

    const pad: i32 = if (room >= 2) 1 else 0;
    const card_h: i32 = rows_h + 2 + 2 * pad;
    const total_h = title_h + card_h + 3 + @as(i32, if (engine_line) 2 else 0);
    const y0 = @max(0, @divFloor(H - total_h, 2));
    const cx = @divFloor(W - card_w, 2);

    if (title_h == 9) {
        const tw: i32 = 6 * 10 + 5 * 2 + 4;
        titleStrip(c, @divFloor(W - tw, 2), y0, app.menu_t);
        c.putCenter(
            0,
            W,
            y0 + 7,
            "five in a row  ·  pair captures  ·  no double-three  ·  19×19",
            .{ .fg = T.dim, .bg = T.bg },
        );
    } else if (title_h == 2) {
        c.putCenter(0, W, y0, "● ○  G O M O K U  ● ○", .{ .fg = T.accent, .bg = T.bg, .bold = true });
    }

    const cr: Rect = .{ .x = cx, .y = y0 + title_h, .w = card_w, .h = card_h };
    const in = c.card(
        cr,
        if (app.has_game) "MENU" else "NEW GAME",
        if (app.has_game) app.settings.mode.name() else "",
        .{ .title_color = T.accent },
    );
    const base = in.y + pad;
    // sliding highlight (fractional row index -> interpolated offset)
    const disp = std.math.clamp(app.fx.menu_disp, 0, @as(f32, @floatFromInt(rows.len - 1)));
    const lo: usize = @intFromFloat(@floor(disp));
    const hi = @min(lo + 1, rows.len - 1);
    const frac = disp - @floor(disp);
    const hyf = @as(f32, @floatFromInt(offs[lo])) + (@as(f32, @floatFromInt(offs[hi] - offs[lo]))) * frac;
    const hy: i32 = base + @as(i32, @intFromFloat(@round(hyf)));
    c.fill(.{ .x = in.x - 1, .y = hy, .w = in.w + 2, .h = 1 }, .{ .bg = T.card_hi });
    c.cell(in.x - 1, hy, "▌", .{ .fg = T.accent, .bg = T.card_hi });

    for (rows, 0..) |r, i| {
        const y = base + offs[i];
        const is_sel = i == sel;
        const bgc = c.bgAt(in.x + 1, y);
        c.hits.add(.{ .x = in.x - 1, .y = y, .w = in.w + 2, .h = 1 }, .{ .menu = @intCast(i) });

        switch (r) {
            .resume_game, .start => {
                const primary = r == .resume_game or !app.has_game;
                const lbl = if (r == .resume_game)
                    (if (app.game.isOver()) "  ◀  BACK TO GAME  " else "  ▶  CONTINUE  ")
                else if (app.has_game) "  ✚  NEW GAME  " else "  ▶  START GAME  ";
                const hue = if (r == .resume_game) T.good else T.accent;
                const lw = G.textWidth(lbl);
                const bx = in.x + @divFloor(in.w - lw, 2);
                const glow = if (is_sel) 0.75 + 0.25 * T.pulse(c.ms, 1100) else 0.0;
                const fg = if (is_sel) T.bg else if (primary) hue else T.dim;
                const bg = if (is_sel) T.mix(T.card_hi, hue, glow) else T.mix(T.card, hue, if (primary) 0.12 else 0.05);
                _ = c.put(bx, y, lbl, .{ .fg = fg, .bg = bg, .bold = true });
                continue;
            },
            .load, .quit => {
                c.putCenter(
                    in.x,
                    in.w,
                    y,
                    menuLabel(r),
                    .{ .fg = if (is_sel) T.text else T.faint, .bg = bgc, .bold = is_sel },
                );
                continue;
            },
            else => {},
        }

        const active = menuActive(app, r);
        _ = c.put(
            in.x + 2,
            y,
            menuLabel(r),
            .{ .fg = if (!active) T.ghost else if (is_sel) T.text else T.dim, .bg = bgc, .bold = is_sel },
        );
        const val = c.print(" {s} ", .{menuValue(app, c, r)});
        const vw = G.textWidth(val);
        const vx = in.right() - 4 - vw;
        _ = c.put(
            vx,
            y,
            val,
            .{
                .fg = if (!active) T.faint else if (is_sel) T.bg else T.text,
                .bg = if (is_sel and active) T.accent else if (active) T.border else T.card_hi,
                .bold = is_sel,
            },
        );
        c.cell(vx - 2, y, "‹", .{ .fg = if (is_sel) T.accent else T.faint, .bg = bgc });
        c.cell(in.right() - 2, y, "›", .{ .fg = if (is_sel) T.accent else T.faint, .bg = bgc });
        c.hits.add(.{ .x = vx - 3, .y = y, .w = 2, .h = 1 }, .{ .key = 'h' });

        if (app.has_game and nextGameOnly(r)) {
            // differs from the game in progress: flag it
            const g = app.game;
            const differs = switch (r) {
                .opening => g.opening != app.settings.opening,
                .captures => g.board.rules.captures != app.settings.captures,
                else => g.board.rules.double_three != app.settings.double_three,
            };

            if (differs) {
                _ = c.put(in.x + 2 + G.textWidth(menuLabel(r)) + 1, y, "*", .{ .fg = T.warn, .bg = bgc, .bold = true });
            }
        }

        if (r == .think) {
            // budget meter vs the 0.5 s limit
            const ms: f32 = @floatFromInt(app.budgetMs());
            const mx = in.x + 20;
            const mw: i32 = vx - 4 - mx;

            if (mw > 4) {
                c.gradientBar(
                    mx,
                    y,
                    mw,
                    ms / 500.0,
                    T.good,
                    T.heat(@min(1, ms / 500.0)),
                    T.ghost,
                );
            }
        }
    }

    // description of the selected row
    const desc = menuDesc(app, c, rows[sel]);
    const desc_c = c.clipped(.{ .x = 1, .y = 0, .w = W - 2, .h = H });
    const dw = G.textWidth(desc) + 2;
    desc_c.fill(.{ .x = @divFloor(W - dw, 2), .y = cr.bottom(), .w = dw, .h = 1 }, .{ .bg = T.bg });
    desc_c.putCenter(0, W, cr.bottom(), desc, .{ .fg = T.cyan, .bg = T.bg });

    // feedback of menu actions (e.g. a failed load)
    if (app.fx.toast_len > 0) {
        const col: Color = switch (app.fx.toast_kind) {
            .info => T.text,
            .good => T.good,
            .bad => T.bad,
            .ai => T.ai,
        };
        const t = c.print(" {s} ", .{app.fx.toastText()});
        const tw = G.textWidth(t);
        desc_c.fill(
            .{ .x = @divFloor(W - tw, 2), .y = cr.bottom() + 1, .w = tw, .h = 1 },
            .{ .bg = T.mix(T.card_hi, col, 0.18) },
        );
        desc_c.putCenter(0, W, cr.bottom() + 1, t, .{ .fg = col, .bg = T.mix(T.card_hi, col, 0.18), .bold = true });
    }

    const ky = cr.bottom() + 2;

    if (app.has_game) {
        capsCentered(c, ky, &.{
            .{ .key = "↑↓", .label = "select", .cp = 'j' },
            .{ .key = "←→", .label = "change", .cp = 'l' },
            .{ .key = "esc", .label = "back to game", .cp = 'C', .hot = true },
            .{ .key = "S", .label = "new game", .cp = 'S' },
            .{ .key = "?", .label = "help", .cp = '?' },
            .{ .key = "q", .label = "quit", .cp = 'q' },
        });
    } else {
        capsCentered(c, ky, &.{
            .{ .key = "↑↓", .label = "select", .cp = 'j' },
            .{ .key = "←→", .label = "change", .cp = 'l' },
            .{ .key = "↵", .label = "start", .cp = 'S', .hot = true },
            .{ .key = "o", .label = "load", .cp = 'o' },
            .{ .key = "?", .label = "help", .cp = '?' },
            .{ .key = "q", .label = "quit", .cp = 'q' },
        });
    }

    if (engine_line) {
        c.putCenter(
            0,
            W,
            ky + 2,
            c.print(
                "engine · {d} threads · PVS alpha-beta + threat search · {s}",
                .{ app.engine.workers.len, if (app.usesNetwork()) "NNUE evaluation" else "pattern evaluation" },
            ),
            .{ .fg = T.faint, .bg = T.bg },
        );
    }
}

// ---- help --------------------------------------------------------------------------------

fn help(app: *const App, c: *const Canvas) void {
    _ = app;
    ambient(c);
    const W = c.width();
    const H = c.height();
    const w: i32 = @min(84, W - 2);
    const h: i32 = @min(34, H - 2);
    const r: Rect = .{ .x = @divFloor(W - w, 2), .y = @divFloor(H - h, 2), .w = w, .h = h };
    c.fill(.{ .x = r.x + 1, .y = r.y + 1, .w = w, .h = h }, .{ .bg = T.bg_deep });
    const in = c.card(r, "HELP", "any key to go back", .{ .title_color = T.accent });
    const cv = c.clipped(in.inset(-1, 0));
    const bgc = T.card;
    var y = in.y + 1;
    const head = struct {
        fn f(cnv: *const Canvas, x: i32, yy: i32, t: []const u8) void {
            _ = cnv.put(x, yy, t, .{ .fg = T.accent, .bg = T.card, .bold = true });
        }
    }.f;

    head(&cv, in.x, y, "CONTROLS");
    y += 1;
    const keys = [_][2][]const u8{
        .{ "arrows · hjkl · mouse", "move the cursor (hover previews the stone)" },
        .{ "enter · space · click", "place a stone" },
        .{ "u · r", "undo / redo" },
        .{ "g · c", "ask the engine for a hint ◆ / jump to it" },
        .{ "d", "analysis overlay: best line, heat map, weak stones" },
        .{ "b · w · p", "swap openings: take Black / White / place two" },
        .{ "p", "pause / resume (AI vs AI)" },
        .{ "z · v", "board size / hide the result banner" },
        .{ "s · o", "save / load gomoku_save.txt" },
        .{ "q · esc", "menu (esc or Continue goes back to the game)" },
        .{ "n · ctrl-c", "new game · quit" },
    };

    for (keys) |k| {
        if (y >= in.bottom()) {
            break;
        }

        _ = cv.put(in.x + 2, y, k[0], .{ .fg = T.text, .bg = bgc, .bold = true });
        _ = cv.put(in.x + 26, y, k[1], .{ .fg = T.dim, .bg = bgc });
        y += 1;
    }

    y += 1;
    head(&cv, in.x, y, "ON THE BOARD");
    y += 1;
    const wood = T.mix(T.wood_a, T.wood_b, 0.4);
    const Legend = struct { g: []const u8, fg: Color, bg: Color, d: []const u8 };
    const legend = [_]Legend{
        .{ .g = "●", .fg = T.last_mark, .bg = T.stone_black, .d = "last move played" },
        .{
            .g = "1",
            .fg = T.stone_white,
            .bg = T.mix(wood, T.stone_black, 0.6),
            .d = "engine's best line: move 1, 2, 3… (ghost stones)",
        },
        .{
            .g = " ",
            .fg = T.text,
            .bg = T.mix(wood, T.good, 0.6),
            .d = "candidate moves being searched (greener = better)",
        },
        .{ .g = "◆", .fg = T.good, .bg = wood, .d = "suggested move (hint)" },
        .{ .g = "×", .fg = T.bad, .bg = wood, .d = "forbidden: double free-three for the side to move" },
        .{ .g = "•", .fg = T.bad, .bg = T.stone_white, .d = "stone that can be captured next move" },
        .{
            .g = "●",
            .fg = T.stone_black,
            .bg = T.mix(wood, T.win_glow, 0.7),
            .d = "winning five (red glow: five that can still be broken)",
        },
    };

    for (legend) |l| {
        if (y >= in.bottom()) {
            break;
        }

        cv.cell(in.x + 2, y, "▐", .{ .fg = wood, .bg = bgc });
        cv.cell(in.x + 3, y, l.g, .{ .fg = l.fg, .bg = l.bg, .bold = true });
        cv.cell(in.x + 4, y, "▌", .{ .fg = wood, .bg = bgc });
        _ = cv.put(in.x + 7, y, l.d, .{ .fg = T.dim, .bg = bgc });
        y += 1;
    }

    y += 1;
    head(&cv, in.x, y, "RULES");
    y += 1;
    const rules = [_][]const u8{
        "Five or more in a row wins, unless the opponent can break it by a capture.",
        "Flanking exactly two enemy stones captures them; 10 captured stones win.",
        "A move creating two free-threes is forbidden (unless it captures).",
        "The board frame glows while the engine thinks; the timer tracks the 0.5 s limit.",
    };

    for (rules) |l| {
        if (y >= in.bottom()) {
            break;
        }

        _ = cv.put(in.x + 2, y, l, .{ .fg = T.dim, .bg = bgc });
        y += 1;
    }
}
