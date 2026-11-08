//! Dashboard cards next to the board. Cards flow top-down into one or more
//! columns, each taking its preferred height when there is room, its
//! minimum otherwise, and are dropped (lowest priority first) when nothing
//! fits.
//!
//!   MATCH       players, captures, whose turn / game status
//!   AI TIMER    live timer vs the 0.5 s limit, statistics, history
//!   SEARCH      depth (>= 10 check), nodes, speed, evaluation gauge
//!   EVALUATION  win-probability graph over the game
//!   THINKING    principal variation + candidate moves (hover = show)
//!   MOVES       move list (hover = show)

const std = @import("std");
const Board = @import("../engine/Board.zig");
const eval = @import("../engine/eval.zig");
const app_mod = @import("app.zig");
const board_view = @import("board_view.zig");
const T = @import("theme.zig");
const G = @import("gfx.zig");
const F = @import("fmt.zig");

const App = app_mod.App;
const Canvas = G.Canvas;
const Rect = G.Rect;
const Color = T.Color;
const NONE = Board.no_move;

// ---- helpers -------------------------------------------------------------------

fn colorName(c: Board.Color) []const u8 {
    return if (c == .black) "Black" else "White";
}

/// Little wooden chip with a stone on it: " ● ".
fn chip(c: *const Canvas, x: i32, y: i32, col: Board.Color) i32 {
    const wood = T.mix(T.wood_a, T.wood_b, 0.4);
    c.cell(x, y, "▐", .{ .fg = wood, .bg = c.bgAt(x, y) });
    c.cell(x + 1, y, "●", .{ .fg = if (col == .black) T.stone_black else T.stone_white, .bg = wood, .bold = true });
    c.cell(x + 2, y, "▌", .{ .fg = wood, .bg = c.bgAt(x + 2, y) });
    return x + 3;
}

fn stoneGlyph(col: Board.Color) []const u8 {
    return if (col == .black) "●" else "○";
}

fn name(c: *const Canvas, m: Board.Move) []const u8 {
    const buf = c.a.alloc(u8, 4) catch {
        return "?";
    };
    return board_view.moveName(buf, m);
}

/// Status LED color for a card title: pulsing when the engine is busy.
fn led(c: *const Canvas, on: bool, col: Color) Color {
    return if (on) T.mix(T.faint, col, 0.45 + 0.55 * T.pulse(c.ms, 900)) else T.faint;
}

pub const EvalView = struct {
    text: []const u8,
    color: Color,
    /// probability that Black wins
    black: f32,
};

/// Evaluation `score` (for `side` to move) as text + Black win probability.
pub fn evalView(c: *const Canvas, score: i32, side: Board.Color) EvalView {
    if (eval.isMate(score)) {
        const winner = if (score > 0) side else side.opponent();
        const plies: i32 = if (score > 0) eval.mate_value - score else eval.mate_value + score;
        const n: i32 = @max(1, @divTrunc(plies + 1, 2));
        return .{
            .text = c.print("forced win for {s} in {d}", .{ colorName(winner), n }),
            .color = T.accent,
            .black = if (winner == .black) 1 else 0,
        };
    }

    const bs: i32 = if (side == .black) score else -score;
    const who: []const u8 = if (@abs(bs) < 60) "balanced" else if (bs > 0) "Black is better" else "White is better";
    return .{
        .text = c.print("{s}  {s}", .{ F.points(c.a, bs), who }),
        .color = T.text,
        .black = @floatCast(F.winProb(bs)),
    };
}

/// Black win probability of the latest search (0.5 when unknown).
pub fn blackProb(app: *const App) f32 {
    if (!app.have_info) {
        return 0.5;
    }

    const s = app.last_info.score;

    if (eval.isMate(s)) {
        return if ((s > 0) == (app.info_side == .black)) 1 else 0;
    }

    const bs: i32 = if (app.info_side == .black) s else -s;
    return @floatCast(F.winProb(bs));
}

fn shortScore(c: *const Canvas, score: i32) []const u8 {
    if (@abs(score) > eval.mate_value) {
        return "  --";
    }

    if (score >= eval.mate_threshold) {
        return c.print("win {d}", .{@max(1, @divTrunc(eval.mate_value - score + 1, 2))});
    }

    if (score <= -eval.mate_threshold) {
        return c.print("loss {d}", .{@max(1, @divTrunc(eval.mate_value + score + 1, 2))});
    }

    return F.points(c.a, score);
}

// ---- cards -----------------------------------------------------------------------

fn match(c: *const Canvas, app: *const App, r: Rect) void {
    const g = app.game;
    const b = &g.board;
    const sr = &app.series;
    const note = if (app.settings.endlessOn() and sr.games > 0)
        c.print(
            "series ● {d} ○ {d} = {d} · avg {d} moves",
            .{ sr.wins[0], sr.wins[1], sr.draws, sr.total_plies / sr.games },
        )
    else
        app.settings.mode.name();
    const in = c.card(r, "MATCH", note, .{});
    const cv = c.clipped(in.inset(-1, 0));
    const over = g.isOver();

    for ([_]Board.Color{ .black, .white }, 0..) |col, i| {
        const y = in.y + @as(i32, @intCast(i));

        if (y >= in.bottom()) {
            break;
        }

        const active = !over and b.side_to_move == col and g.pendingChoice() == .none;
        const winner = over and g.maybe_winner != null and g.maybe_winner.? == col;

        if (active or winner) {
            cv.fill(.{ .x = in.x - 1, .y = y, .w = in.w + 2, .h = 1 }, .{ .bg = T.card_hi });
        }

        const is_ai = std.mem.eql(u8, app.ownerName(col), "AI");
        const hue = if (is_ai) T.ai else T.accent;

        if (active) {
            cv.cell(
                in.x - 1,
                y,
                "▌",
                .{ .fg = T.mix(T.card_hi, hue, 0.5 + 0.5 * T.pulse(cv.ms, 1000)), .bg = T.card_hi },
            );
        }

        if (winner) {
            cv.cell(in.x - 1, y, "★", .{ .fg = T.win_glow, .bg = T.card_hi, .bold = true });
        }

        var x = chip(&cv, in.x, y, col);
        x = cv.put(x + 1, y, colorName(col), .{ .fg = T.text, .bg = cv.bgAt(x + 1, y), .bold = active or winner });
        _ = cv.put(in.x + 11, y, app.ownerName(col), .{ .fg = hue, .bg = cv.bgAt(in.x + 11, y) });

        // captures: one pip per pair, the 5th wins
        const need = b.rules.capture_win_pairs;
        const have = b.captured_pairs[col.index()];
        // pips spaced out when there is room, packed otherwise
        const step: i32 = if (in.w >= 40) 2 else 1;
        const px = in.right() - @as(i32, need) * step - 4;

        if (b.rules.captures) {
            const flash = fxFlash(cv.ms, app.fx.cap_flash[col.index()]);
            var k: u8 = 0;

            while (k < need) : (k += 1) {
                const filled = k < have;
                var pc = if (filled) (if (have + 1 >= need) T.bad else T.accent) else T.ghost;

                if (filled and k + 1 == have) {
                    pc = T.mix(pc, T.text, flash);
                }

                cv.cell(
                    px + @as(i32, k) * step,
                    y,
                    if (filled) "◆" else "◇",
                    .{ .fg = pc, .bg = cv.bgAt(px, y), .bold = filled },
                );
            }

            cv.putRight(
                in.right(),
                y,
                cv.print("{d}/{d}", .{ have, need }),
                .{ .fg = if (have > 0) T.text else T.faint, .bg = cv.bgAt(in.right() - 1, y) },
            );
        } else {
            cv.putRight(in.right(), y, "no captures", .{ .fg = T.faint, .bg = cv.bgAt(in.right() - 1, y) });
        }
    }

    if (in.h >= 3) {
        status(&cv, app, in.x, in.y + 2, in.w);
    }
}

fn fxFlash(now: u64, t: u64) f32 {
    if (t == 0) {
        return 0;
    }

    const age = now -| t;

    if (age > 900) {
        return 0;
    }

    return 1 - @as(f32, @floatFromInt(age)) / 900.0;
}

/// One line telling what is going on.
pub fn status(c: *const Canvas, app: *const App, x: i32, y: i32, w: i32) void {
    _ = w;
    const g = app.game;
    const bgc = c.bgAt(x, y);

    if (g.isOver()) {
        const txt = if (g.maybe_winner) |wc| c.print("★ {s} wins · {s}", .{ colorName(wc), switch (g.outcome) {
            .five => "five in a row",
            .capture => "10 stones captured",
            else => "",
        } }) else "Draw · the board is full";
        _ = c.put(x, y, txt, .{ .fg = T.mix(T.win_glow, T.accent, T.pulse(c.ms, 1400)), .bg = bgc, .bold = true });
        return;
    }

    if (g.pendingChoice() != .none) {
        _ = c.put(
            x,
            y,
            c.print("◇ {s}: choose a color", .{app.actorName()}),
            .{ .fg = T.accent, .bg = bgc, .bold = true },
        );
        return;
    }

    if (g.pendingFive() != NONE) {
        _ = c.put(
            x,
            y,
            c.print("! Five on board · {s} must capture", .{colorName(g.board.side_to_move)}),
            .{ .fg = T.mix(T.bad, T.warn, T.pulse(c.ms, 600)), .bg = bgc, .bold = true },
        );
        return;
    }

    if (app.paused) {
        _ = c.put(x, y, "‖ Paused · p to resume", .{ .fg = T.warn, .bg = bgc, .bold = true });
        return;
    }

    if (app.thinking.load(.acquire) and !app.job_is_suggestion) {
        _ = c.put(
            x,
            y,
            c.print("{s} AI is thinking for {s}", .{ G.spinner(c.ms), colorName(g.board.side_to_move) }),
            .{ .fg = T.ai, .bg = bgc, .bold = true },
        );
        return;
    }

    if (app.humanCanPlay()) {
        const who = app.actorName();
        const you = std.mem.eql(u8, who, "You");
        _ = c.put(
            x,
            y,
            c.print("▸ {s}{s} move · click or ↵", .{ if (you) "Your" else who, if (you) "" else "'s" }),
            .{ .fg = T.good, .bg = bgc, .bold = true },
        );
    }
}

fn timer(c: *const Canvas, app: *const App, r: Rect) void {
    const live = app.thinking.load(.acquire);
    const in = c.card(
        r,
        "AI TIMER",
        c.print("budget {d} ms · limit 500 ms", .{app.budgetMs()}),
        .{ .title_color = if (live) T.ai else T.dim, .active = live, .icon = "●", .icon_color = led(c, live, T.ai) },
    );
    const cv = c.clipped(in.inset(-1, 0));
    const bgc = cv.bgAt(in.x, in.y);
    const searched = live or app.think_count > 0;
    const elapsed = if (live) app.thinkElapsedMs() else app.last_think_ms;
    const lim: f32 = @floatFromInt(app_mod.time_limit_ms);
    const frac: f32 = @as(f32, @floatFromInt(elapsed)) / lim;
    // the gauge spans the limit, or the whole budget when it is longer
    const span: f32 = @max(lim, @as(f32, @floatFromInt(app.budgetMs())));
    const over = elapsed > app_mod.time_limit_ms;
    const col: Color = if (!searched) T.faint else if (live) T.heat(frac) else if (over) T.bad else T.good;

    var buf: [16]u8 = undefined;
    const digits = cv.a.dupe(u8, F.secsDigits(&buf, elapsed)) catch "-";
    const state: []const u8 = if (live) (if (app.job_is_suggestion) "COMPUTING HINT" else "THINKING") else if (!searched) "WAITING" else if (over) "TOO SLOW" else "DONE";
    const sym: []const u8 = if (live) G.spinner(cv.ms) else if (!searched) "·" else if (over) "✗" else "✓";
    const wide = in.w >= 40;
    const who: []const u8 = if (!searched)
        (if (wide) "no search yet" else "no search")
    else if (wide or !app.info_is_suggestion)
        cv.print(
            "{s} {s}{s}",
            .{ stoneGlyph(app.info_side), colorName(app.info_side), if (app.info_is_suggestion) " · hint" else "" },
        )
    else
        cv.print("{s} hint", .{stoneGlyph(app.info_side)});

    var y = in.y;

    if (in.h >= 6) {
        var x = G.bigText(
            &cv,
            in.x,
            y,
            digits,
            .{ .fg = col, .bg = bgc, .bold = true },
            false,
        );
        _ = cv.put(x, y + 2, "s", .{ .fg = col, .bg = bgc, .bold = true });
        x += 3;
        _ = cv.put(x, y, cv.print("{s} {s}", .{ sym, state }), .{ .fg = col, .bg = bgc, .bold = true });
        _ = cv.put(x, y + 1, who, .{ .fg = T.dim, .bg = bgc });

        if (live and app.have_info) {
            _ = cv.put(x, y + 2, cv.print("depth {d} reached", .{app.last_info.depth}), .{ .fg = T.dim, .bg = bgc });
        } else if (searched) {
            _ = cv.put(
                x,
                y + 2,
                if (over) (if (wide) "over the 0.5 s limit" else "> 0.5 s") else cv.print(
                    "{d}%{s}",
                    .{ @as(u32, @intFromFloat(@round(@min(frac, 10) * 100))), if (wide) " of the limit" else "" },
                ),
                .{ .fg = if (over) T.bad else T.faint, .bg = bgc },
            );
        }

        y += 3;
    } else {
        var x = cv.put(in.x, y, cv.print("{s} s", .{digits}), .{ .fg = col, .bg = bgc, .bold = true });
        x = cv.put(x + 2, y, cv.print("{s} {s}", .{ sym, state }), .{ .fg = col, .bg = bgc });
        _ = cv.put(x + 2, y, who, .{ .fg = T.dim, .bg = bgc });
        y += 1;
    }

    // gauge: elapsed vs the 0.5 s limit, tick at the soft budget
    if (y < in.bottom()) {
        const w = in.w;
        const gf = @as(f32, @floatFromInt(elapsed)) / span;
        cv.gradientBar(
            in.x,
            y,
            w,
            if (searched) gf else 0,
            T.good,
            if (frac > 0.8) T.bad else T.heat(
                @min(1, frac),
            ),
            T.ghost,
        );
        const wf: f32 = @floatFromInt(w - 1);
        const bx = in.x + @as(i32, @intFromFloat(@as(f32, @floatFromInt(app.budgetMs())) / span * wf));
        cv.cell(bx, y, "▏", .{ .fg = T.text, .bg = cv.bgAt(bx, y) });

        if (span > lim) {
            const lx = in.x + @as(i32, @intFromFloat(lim / span * wf));
            cv.cell(lx, y, "┃", .{ .fg = T.bad, .bg = cv.bgAt(lx, y) });
        }

        y += 1;
    }

    // statistics
    if (y < in.bottom()) {
        const none = app.think_count == 0;
        const avg = if (none) 0 else app.total_think_ms / app.think_count;
        var x = in.x;
        const items = [_]struct { l: []const u8, v: u64 }{
            .{ .l = "last", .v = app.last_think_ms },
            .{ .l = "avg", .v = avg },
            .{ .l = "max", .v = app.max_think_ms },
        };

        for (items) |it| {
            x = cv.put(x, y, it.l, .{ .fg = T.faint, .bg = bgc });
            x = cv.put(
                x + 1,
                y,
                if (none) "--" else F.secs(cv.a, it.v),
                .{
                    .fg = if (none) T.faint else if (it.v > app_mod.time_limit_ms) T.bad else T.text,
                    .bg = bgc,
                    .bold = !none and std.mem.eql(u8, it.l, "avg"),
                },
            );
            x += 2;
        }

        const cnt = cv.print("n {d}", .{app.think_count});

        if (x + G.textWidth(cnt) <= in.right()) {
            cv.putRight(in.right(), y, cnt, .{ .fg = T.faint, .bg = bgc });
        }

        y += 1;
    }

    // history of recent searches (one bar each) + verdict
    if (y < in.bottom()) {
        const verdict_w: i32 = 13;
        const nbars: usize = @intCast(@max(0, in.w - verdict_w - 1));
        const n = @min(app.search_n, nbars, app.search_ms.len);
        var i: usize = 0;

        while (i < nbars) : (i += 1) {
            const x = in.x + @as(i32, @intCast(i));

            if (i < nbars - n) {
                cv.cell(x, y, "▁", .{ .fg = T.ghost, .bg = bgc });
                continue;
            }

            const k = app.search_n - n + (i - (nbars - n));
            const ms = app.search_ms[k % app.search_ms.len];
            const f = @as(f32, @floatFromInt(ms)) / lim;
            cv.spark(x, y, @as(f32, @floatFromInt(ms)) / span, .{ .fg = T.heat(@min(1, f)), .bg = bgc });
        }

        if (app.think_count > 0) {
            const avg = app.total_think_ms / app.think_count;
            const ok = avg <= app_mod.time_limit_ms;
            cv.putRight(
                in.right(),
                y,
                if (ok) "✓ avg < 0.5 s" else "✗ avg > 0.5 s",
                .{ .fg = if (ok) T.good else T.bad, .bg = bgc, .bold = true },
            );
        }
    }
}

fn search(c: *const Canvas, app: *const App, r: Rect) void {
    const live = app.thinking.load(.acquire);
    const inf = &app.last_info;
    const in = c.card(
        r,
        "SEARCH",
        c.print("{d} threads · PVS α-β{s}", .{ app.engine.workers.len, if (app.usesNetwork()) " · NNUE" else "" }),
        .{ .title_color = if (live) T.ai else T.dim, .icon = "●", .icon_color = led(c, live, T.ai) },
    );
    const cv = c.clipped(in.inset(-1, 0));
    const bgc = cv.bgAt(in.x, in.y);
    var y = in.y;

    if (!app.have_info) {
        _ = cv.put(
            in.x,
            y,
            if (live) cv.print("{s} first iteration…", .{G.spinner(cv.ms)}) else "waiting for the first search",
            .{ .fg = T.faint, .bg = bgc },
        );
        y += 1;
    } else if (inf.depth == 0 and inf.nodes == 0 and !live) {
        // empty board: the engine plays the centre without searching
        _ = cv.put(in.x, y, "opening move: centre, no search needed", .{ .fg = T.dim, .bg = bgc });
        y += 1;
    } else {
        // depth with the subject's >= 10 requirement
        var x = cv.put(in.x, y, "depth", .{ .fg = T.faint, .bg = bgc });
        x = cv.put(x + 1, y, cv.print("{d}", .{inf.depth}), .{ .fg = T.text, .bg = bgc, .bold = true });
        const ok = inf.depth >= 10 or eval.isMate(inf.score);
        x = cv.put(
            x + 1,
            y,
            if (inf.depth >= 10) " ≥10 ✓ " else if (ok) " mate ✓ " else " <10 ",
            .{ .fg = if (ok) T.bg else T.text, .bg = if (ok) T.good else T.warn, .bold = true },
        );
        x = cv.put(x + 2, y, "sel", .{ .fg = T.faint, .bg = bgc });
        _ = cv.put(x + 1, y, cv.print("{d}", .{inf.selective_depth}), .{ .fg = T.dim, .bg = bgc });
        cv.putRight(in.right(), y, cv.print("TT {d}%", .{inf.hashfull_permille / 10}), .{ .fg = T.dim, .bg = bgc });
        y += 1;

        if (y < in.bottom()) {
            x = cv.put(in.x, y, "nodes", .{ .fg = T.faint, .bg = bgc });
            x = cv.put(x + 1, y, F.si(cv.a, inf.nodes), .{ .fg = T.text, .bg = bgc, .bold = true });
            x = cv.put(x + 3, y, "speed", .{ .fg = T.faint, .bg = bgc });
            x = cv.put(x + 1, y, F.si(cv.a, inf.nodes_per_second), .{ .fg = T.text, .bg = bgc, .bold = true });
            _ = cv.put(x, y, if (inf.nodes_per_second < 1000) " n/s" else "n/s", .{ .fg = T.dim, .bg = bgc });
            y += 1;
        }
    }

    // animated evaluation gauge  ● 62% ███████░░░░ 38% ○
    if (y < in.bottom()) {
        const p = app.fx.eval_disp;
        const pb: u32 = @intFromFloat(@round(p * 100));
        var x = chip(&cv, in.x, y, .black);
        x = cv.put(x, y, cv.print("{d:>3}%", .{pb}), .{ .fg = T.text, .bg = bgc, .bold = true });
        const bw = in.w - 16;

        if (bw > 4) {
            cv.bar(
                x + 1,
                y,
                bw,
                p,
                T.rgb(0x3A3F55),
                T.stone_white_lo,
            );
            // center tick
            cv.cell(x + 1 + @divFloor(bw, 2), y, "┊", .{ .fg = T.faint, .bg = cv.bgAt(x + 1 + @divFloor(bw, 2), y) });
        }

        cv.putRight(in.right() - 3, y, cv.print("{d}%", .{100 - pb}), .{ .fg = T.text, .bg = bgc, .bold = true });
        _ = chip(&cv, in.right() - 3, y, .white);
        y += 1;
    }

    if (y < in.bottom() and app.have_info) {
        const ev = evalView(&cv, inf.score, app.info_side);
        cv.putCenter(in.x, in.w, y, ev.text, .{ .fg = ev.color, .bg = bgc, .bold = eval.isMate(inf.score) });
    }
}

fn graph(c: *const Canvas, app: *const App, r: Rect) void {
    const in = c.card(r, "EVALUATION", "Black win chance per move", .{});
    const cv = c.clipped(in.inset(-1, 0));
    const bgc = cv.bgAt(in.x, in.y);
    const ply: usize = app.game.ply();
    const n = @min(ply + 1, app.eval_hist.len);
    var any = false;

    for (app.eval_hist[0..n]) |v| {
        if (!std.math.isNan(v)) {
            any = true;
        }
    }

    _ = cv.put(in.x, in.y, "●", .{ .fg = T.dim, .bg = bgc });
    _ = cv.put(in.x, in.bottom() - 1, "○", .{ .fg = T.dim, .bg = bgc });
    const area: Rect = .{ .x = in.x + 2, .y = in.y, .w = in.w - 2, .h = in.h };

    if (!any) {
        G.plot(
            &cv,
            area,
            &.{},
            0.5,
            T.ai,
            T.accent,
            bgc,
        );
        cv.putCenter(
            area.x,
            area.w,
            in.y + @divFloor(in.h, 2) - @as(i32, if (@mod(in.h, 2) == 0) 1 else 0),
            " no evaluation yet ",
            .{ .fg = T.faint, .bg = bgc },
        );
        return;
    }

    G.plot(
        &cv,
        area,
        app.eval_hist[0..n],
        0.5,
        T.rgb(0xA9B4D6),
        T.rgb(0xEADFC4),
        bgc,
    );
}

fn thinking(c: *const Canvas, app: *App, r: Rect) void {
    const live = app.thinking.load(.acquire);
    const in = c.card(
        r,
        if (live) "THINKING" else "BEST LINE",
        if (app.show_analysis) "overlay on · d" else "overlay off · d",
        .{ .title_color = if (live) T.pv else T.dim, .icon = "●", .icon_color = led(c, live, T.pv) },
    );
    const cv = c.clipped(in.inset(-1, 0));
    const bgc = cv.bgAt(in.x, in.y);
    const inf = &app.last_info;

    if (!app.have_info or inf.principal_variation_length == 0) {
        _ = cv.put(in.x, in.y, "the engine's plan appears here", .{ .fg = T.faint, .bg = bgc });
        return;
    }

    // principal variation as chips
    var x = in.x;
    var y = in.y;
    var side = app.info_side;
    const pv_rows: i32 = if (in.h >= 8) 2 else 1;

    for (inf.principal_variation[0..@min(inf.principal_variation_length, inf.principal_variation.len)], 0..) |m, i| {
        const nm = name(&cv, m);
        const w: i32 = G.textWidth(nm) + 3;

        if (x + w > in.right()) {
            y += 1;
            x = in.x;

            if (y >= in.y + pv_rows) {
                break;
            }
        }

        const first = i == 0;
        x = cv.put(x, y, stoneGlyph(side), .{ .fg = if (side == .black) T.dim else T.text, .bg = bgc });
        const st: G.Style = .{ .fg = if (first) T.bg else T.pv, .bg = if (first) T.pv else bgc, .bold = true };
        const x0 = x;
        x = cv.put(x, y, if (first) cv.print(" {s} ", .{nm}) else cv.print("{s} ", .{nm}), st);
        cv.hits.add(.{ .x = x0 - 1, .y = y, .w = x - x0 + 1, .h = 1 }, .{ .point = m });
        x += 1;
        side = side.opponent();
    }

    y = in.y + pv_rows;

    if (y >= in.bottom()) {
        return;
    }

    // candidate moves
    const n = @min(inf.top_move_count, inf.top_moves.len, @as(usize, @intCast(in.bottom() - y)));
    const best = if (inf.top_move_count > 0) inf.top_moves[0].score else 0;

    for (inf.top_moves[0..n], 0..) |t, i| {
        const row = y + @as(i32, @intCast(i));
        const hot = app.hover_point == t.move;

        if (hot) {
            cv.fill(.{ .x = in.x - 1, .y = row, .w = in.w + 2, .h = 1 }, .{ .bg = T.card_hi });
        }

        const rb = cv.bgAt(in.x, row);
        const top = i == 0;
        _ = cv.put(in.x, row, cv.print("{d}", .{i + 1}), .{ .fg = T.faint, .bg = rb });
        _ = cv.put(in.x + 2, row, name(&cv, t.move), .{ .fg = if (top) T.good else T.text, .bg = rb, .bold = top });
        const prob: f32 = if (@abs(
            t.score,
        ) > eval.mate_value) 0 else if (t.score >= eval.mate_threshold) 1 else if (t.score <= -eval.mate_threshold) 0 else @floatCast(
            F.winProb(t.score),
        );
        const bw = in.w - 20;

        if (bw > 3) {
            const diff: f32 = @floatFromInt(std.math.clamp(@as(i64, best) - t.score, 0, 800));
            // bar = strength relative to the best move (full = best)
            const rel = 0.08 + 0.92 * (1 - diff / 800.0);
            cv.bar(
                in.x + 7,
                row,
                bw,
                rel,
                if (top) T.good else T.mix(T.good, T.cyan, diff / 800.0),
                T.ghost,
            );
        }

        cv.putRight(in.right() - 5, row, shortScore(&cv, t.score), .{ .fg = if (top) T.good else T.dim, .bg = rb });
        cv.putRight(
            in.right(),
            row,
            cv.print("{d}%", .{@as(u32, @intFromFloat(@round(prob * 100)))}),
            .{ .fg = T.faint, .bg = rb },
        );
        cv.hits.add(.{ .x = in.x - 1, .y = row, .w = in.w + 2, .h = 1 }, .{ .point = t.move });
    }
}

fn moves(c: *const Canvas, app: *const App, r: Rect) void {
    const b = &app.game.board;
    const list = b.playedMoves();
    const in = c.card(r, "MOVES", c.print("{d} played · u undo · r redo", .{list.len}), .{});
    const cv = c.clipped(in.inset(-1, 0));
    const bgc = cv.bgAt(in.x, in.y);

    if (list.len == 0) {
        _ = cv.put(in.x, in.y, "no moves yet", .{ .fg = T.faint, .bg = bgc });
        return;
    }

    var i: usize = list.len;
    var y = in.y;

    while (i > 0 and y < in.bottom()) : (y += 1) {
        i -= 1;
        const m = list[i];
        const side: Board.Color = if (i % 2 == 0) .black else .white;
        const st = if (i < app.move_stats.len) app.move_stats[i] else app_mod.MoveStat{};
        const newest = i + 1 == list.len;
        const hot = app.hover_point == m;

        if (hot) {
            cv.fill(.{ .x = in.x - 1, .y = y, .w = in.w + 2, .h = 1 }, .{ .bg = T.card_hi });
        }

        const rb = cv.bgAt(in.x, y);
        _ = cv.put(in.x, y, cv.print("{d:>3}", .{i + 1}), .{ .fg = T.faint, .bg = rb });
        _ = cv.put(in.x + 4, y, stoneGlyph(side), .{ .fg = if (side == .black) T.dim else T.text, .bg = rb });
        _ = cv.put(in.x + 6, y, name(&cv, m), .{ .fg = if (newest) T.accent else T.text, .bg = rb, .bold = newest });

        if (st.known) {
            _ = cv.put(
                in.x + 11,
                y,
                if (st.by_ai) "AI" else "human",
                .{ .fg = if (st.by_ai) T.ai else T.faint, .bg = rb },
            );
            const tcol: Color = if (st.by_ai) (if (st.ms > app_mod.time_limit_ms) T.bad else T.text) else T.faint;
            cv.putRight(in.x + 26, y, F.secs(cv.a, st.ms), .{ .fg = tcol, .bg = rb });

            if (st.by_ai and st.depth > 0) {
                _ = cv.put(in.x + 27, y, cv.print("d{d}", .{st.depth}), .{ .fg = T.faint, .bg = rb });
            }
        }

        if (i < b.captured_stone_counts.len and b.captured_stone_counts[i] > 0) {
            cv.putRight(
                in.right(),
                y,
                cv.print("+{d}", .{b.captured_stone_counts[i]}),
                .{ .fg = T.bad, .bg = rb, .bold = true },
            );
        }

        cv.hits.add(.{ .x = in.x - 1, .y = y, .w = in.w + 2, .h = 1 }, .{ .point = m });
    }
}

// ---- flow layout -----------------------------------------------------------------------

const Kind = enum { match, timer, search, graph, thinking, moves };

const Spec = struct {
    kind: Kind,
    min: i32,
    pref: i32,
    fill: bool = false,
    /// with several columns, start this card in the next one
    col_break: bool = false,
};

pub fn draw(c: *const Canvas, app: *App, cols: []const Rect) void {
    const think_pref: i32 = 2 + 2 + @as(i32, @intCast(@max(@min(app.last_info.top_move_count, 8), 3)));
    var specs = [_]Spec{
        .{ .kind = .match, .min = 5, .pref = 5 },
        .{ .kind = .timer, .min = 5, .pref = 8 },
        .{ .kind = .search, .min = 5, .pref = 6 },
        .{ .kind = .graph, .min = 5, .pref = 7 },
        .{ .kind = .thinking, .min = 5, .pref = think_pref },
        .{ .kind = .moves, .min = 4, .pref = 4, .fill = true },
    };

    if (cols.len > 1) {
        // two columns: live search on the left, history on the right
        std.mem.swap(Spec, &specs[3], &specs[4]);
        specs[4].col_break = true;
        specs[4].pref = 12;
    }

    var ci: usize = 0;
    var y: i32 = if (cols.len > 0) cols[0].y else 0;

    for (specs, 0..) |s, si| {
        if (s.col_break and ci == 0 and cols.len > 1) {
            ci = 1;
            y = cols[1].y;
        }

        while (ci < cols.len) {
            const col = cols[ci];
            const rem = col.bottom() - y;
            // prefer leaving room for the next card's minimum in this column
            const next_min: i32 = if (si + 1 < specs.len) specs[si + 1].min else 0;
            var h: i32 = 0;

            if (rem >= s.min) {
                if (s.fill) {
                    h = rem;
                } else if (rem >= s.pref + next_min) {
                    h = s.pref;
                } else if (rem - next_min >= s.min) {
                    h = rem - next_min;
                } else {
                    h = @min(rem, s.pref);
                }
            }

            if (h > 0) {
                const rr: Rect = .{ .x = col.x, .y = y, .w = col.w, .h = h };
                const cc = c.clipped(rr);

                switch (s.kind) {
                    .match => {
                        match(&cc, app, rr);
                    },
                    .timer => {
                        timer(&cc, app, rr);
                    },
                    .search => {
                        search(&cc, app, rr);
                    },
                    .graph => {
                        graph(&cc, app, rr);
                    },
                    .thinking => {
                        thinking(&cc, app, rr);
                    },
                    .moves => {
                        moves(&cc, app, rr);
                    },
                }

                y += h;
                break;
            }

            ci += 1;

            if (ci < cols.len) {
                y = cols[ci].y;
            }
        }

        if (ci >= cols.len) {
            return;
        }
    }
}
