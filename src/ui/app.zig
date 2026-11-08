//! Terminal UI (libvaxis): state, game flow, input and the main loop.
//!
//! Files:
//!  * app.zig        : this file (state, engine glue, input, main loop)
//!  * screens.zig    : layout, header / footer, menu, help, modals, toasts
//!  * board_view.zig : the goban and its effects
//!  * sidebar.zig    : dashboard cards (timer, search, graph, best line…)
//!  * fx.zig         : animation state
//!  * gfx.zig / theme.zig / fmt.zig : primitives, palette, formatting
//!
//! Threads:
//!  * main thread : event loop + rendering (never blocks on the engine)
//!  * vaxis input : reads the tty, posts key / mouse / resize events
//!  * ticker      : posts a `tick` every 33 ms (animations, live timer)
//!  * ai thread   : one per search; runs Engine.think and posts `ai_info`
//!                  (each completed iteration) and `ai_done` events.

const std = @import("std");
const vaxis = @import("vaxis");
const Engine = @import("../engine/Engine.zig");
const Game = @import("../engine/Game.zig");
const Board = @import("../engine/Board.zig");
const eval = @import("../engine/eval.zig");
const screens = @import("screens.zig");
const board_view = @import("board_view.zig");
const sidebar = @import("sidebar.zig");
const fx_mod = @import("fx.zig");
const G = @import("gfx.zig");
const F = @import("fmt.zig");

const SearchInfo = Engine.SearchInfo;
const Move = Board.Move;
const NONE = Board.no_move;

const Event = union(enum) {
    key_press: vaxis.Key,
    mouse: vaxis.Mouse,
    winsize: vaxis.Winsize,
    tick,
    ai_info: AiMsg,
    ai_done: AiMsg,
};

const AiMsg = struct {
    id: u32,
    info: SearchInfo,
};

pub const Mode = enum {
    pva, // player vs AI
    hotseat, // two humans + move suggestion
    ava, // AI vs AI

    pub fn name(m: Mode) []const u8 {
        return switch (m) {
            .pva => "Player vs AI",
            .hotseat => "Hotseat",
            .ava => "AI vs AI",
        };
    }
};

/// Budgets above 450 ms exceed the subject's 0.5 s limit: they exist to watch
/// stronger play (the timer flags them honestly).
pub const think_times = [_]u32{
    100,
    250,
    400,
    450,
    1000,
    2000,
    5000,
    10000,
};
/// The subject's limit: an AI move must take less than half a second.
pub const time_limit_ms: u64 = 500;

/// Catch-up time: an AI that sees itself behind (win chance below ~42%)
/// keeps searching past its budget, up to this many budgets...
pub const CATCH_UP_FACTOR: u32 = 10;
/// ...bounded below and above (ms).
pub const CATCH_UP_MIN_MS: u32 = 3000;
pub const CATCH_UP_MAX_MS: u32 = 30000;
/// Score (side to move) under which the AI is considered behind.
pub const CATCH_UP_BELOW: i32 = -250;

pub const Settings = struct {
    mode: Mode = .pva,
    human_first: bool = true,
    opening: Game.Opening = .standard,
    think_idx: usize = 2,
    captures: bool = true,
    double_three: bool = true,
    /// AI vs AI: start a new game automatically after each result
    endless: bool = false,
    /// catch-up time: an AI that is behind thinks longer, until it finds a
    /// move that keeps the game competitive
    catch_up: bool = false,

    pub fn endlessOn(s: *const Settings) bool {
        return s.mode == .ava and s.endless;
    }

    pub fn catchUpOn(s: *const Settings) bool {
        return s.mode != .hotseat and s.catch_up;
    }
};

/// Results of the current endless series.
pub const Series = struct {
    games: u32 = 0,
    wins: [2]u32 = .{ 0, 0 },
    draws: u32 = 0,
    total_plies: u64 = 0,
    longest: u32 = 0,
};

/// Pause between two games of an endless series (result stays on screen).
pub const ENDLESS_PAUSE_MS: u64 = 0;

/// Save file, in the working directory.
pub const save_path = "gomoku_save.txt";

pub const Screen = enum { menu, game, help };

/// Menu rows, top to bottom.
pub const MenuRow = enum {
    resume_game,
    mode,
    human_first,
    opening,
    think,
    captures,
    double_three,
    endless,
    catch_up,
    start,
    load,
    quit,

    pub fn isSetting(r: MenuRow) bool {
        return @backingInt(r) >= @backingInt(MenuRow.mode) and @backingInt(r) <= @backingInt(MenuRow.catch_up);
    }
};

pub const MENU_MAX = std.enums.values(MenuRow).len;

/// Board scale preference (z key).
pub const Zoom = enum { auto, compact };

/// Per-ply record for the move list.
pub const MoveStat = struct {
    ms: u32 = 0,
    depth: u8 = 0,
    by_ai: bool = false,
    known: bool = false,
};

const LoopT = vaxis.Loop(Event);

const AiJob = struct {
    app: *App,
    board: Board,
    pending: Move,
    limits: Engine.Limits,
    id: u32,
};

pub const App = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    loop: ?*LoopT,
    engine: *Engine,
    game: *Game,
    settings: Settings = .{},
    screen: Screen = .menu,
    help_back: Screen = .game,
    menu_sel: usize = 0,
    /// a game has been started or loaded (the menu offers "Continue")
    has_game: bool = false,
    menu_t: u64 = 0,
    cursor_x: usize = 9,
    cursor_y: usize = 9,
    cursor_on_board: bool = true,
    zoom: Zoom = .auto,

    // AI state
    thinking: std.atomic.Value(bool) = .init(false),
    /// set by joinAi(): the search thread must not post its result
    ai_cancel: std.atomic.Value(bool) = .init(false),
    think_start: std.Io.Timestamp = undefined,
    job_id: u32 = 0,
    job_is_suggestion: bool = false,
    ai_thread: ?std.Thread = null,
    /// result of a search run on the main thread (thread spawn failure)
    sync_result: ?AiMsg = null,
    job: AiJob = undefined,
    /// latest search report (live while thinking, final afterwards)
    last_info: SearchInfo = .empty,
    have_info: bool = false,
    /// side to move in the position `last_info` refers to
    info_side: Board.Color = .black,
    info_is_suggestion: bool = false,
    suggestion: Move = NONE,

    // AI timer statistics (every search of this game: moves and hints)
    last_think_ms: u64 = 0,
    total_think_ms: u64 = 0,
    max_think_ms: u64 = 0,
    think_count: u64 = 0,
    /// ring buffer of recent search times
    search_ms: [48]u32 = @splat(0),
    search_n: usize = 0,
    /// time + depth of every ply (indexed by ply)
    move_stats: [Board.max_history]MoveStat = @splat(.{}),
    /// Black win probability measured before each ply (NaN = no search)
    eval_hist: [Board.max_history]f32 = @splat(std.math.nan(f32)),
    turn_start: std.Io.Timestamp = undefined,

    series: Series = .{},

    show_analysis: bool = true,
    hide_banner: bool = false,
    paused: bool = false,
    quit: bool = false,
    running: std.atomic.Value(bool) = .init(true),

    // rendering / effects
    t0: std.Io.Timestamp = undefined,
    fx: fx_mod.Fx = .{},
    hits: G.Hits = .{},
    hover_point: Move = NONE,
    /// board geometry of the last frame (mouse hit testing)
    geo: ?board_view.Geo = null,
    name_buf: [8]u8 = undefined,

    pub fn ms(self: *const App) u64 {
        return self.sinceMs(self.t0);
    }

    fn toast(self: *App, kind: fx_mod.ToastKind, comptime fmt: []const u8, args: anytype) void {
        self.fx.toast(kind, self.ms(), fmt, args);
    }

    fn now(self: *const App) std.Io.Timestamp {
        return std.Io.Clock.awake.now(self.io);
    }

    fn sinceMs(self: *const App, t: std.Io.Timestamp) u64 {
        const ns = t.untilNow(self.io, .awake).toNanoseconds();
        return @intCast(@max(@divTrunc(ns, std.time.ns_per_ms), 0));
    }

    // ---- engine glue -------------------------------------------------------

    fn onInfo(ctx: ?*anyopaque, info: *const SearchInfo) void {
        const job: *AiJob = @ptrCast(@alignCast(ctx.?));

        if (info.is_final) {
            return; // the final result is posted by the job thread
        }

        if (job.app.loop) |l| {
            _ = l.tryPostEvent(.{ .ai_info = .{ .id = job.id, .info = info.* } }) catch {};
        }
    }

    fn aiThreadMain(job: *AiJob) void {
        const app = job.app;
        const info = app.engine.think(&job.board, job.pending, job.limits);
        app.thinking.store(false, .release);
        const l = app.loop orelse {
            return;
        };

        // Never block on a full queue: the main thread may be waiting in
        // joinAi() for this thread, which would deadlock. Retry until posted
        // or cancelled.
        while (!app.ai_cancel.load(.acquire)) {
            const posted = l.tryPostEvent(.{ .ai_done = .{ .id = job.id, .info = info } }) catch {
                return;
            };

            if (posted) {
                return;
            }

            app.io.sleep(.fromMilliseconds(2), .awake) catch {};
        }
    }

    /// Stop and join the running search (if any). Its result, possibly
    /// still queued, is discarded (the job id changes).
    fn joinAi(self: *App) void {
        if (self.ai_thread) |t| {
            self.ai_cancel.store(true, .release);
            self.engine.stop();
            t.join();
            self.ai_thread = null;
        }

        self.job_id +%= 1;
        self.thinking.store(false, .release);
    }

    pub fn budgetMs(self: *const App) u32 {
        return think_times[self.settings.think_idx % think_times.len];
    }

    /// Longest an AI move may take in catch-up mode.
    pub fn catchUpMs(self: *const App) u32 {
        return std.math.clamp(
            self.budgetMs() * CATCH_UP_FACTOR,
            CATCH_UP_MIN_MS,
            @max(CATCH_UP_MAX_MS, self.budgetMs()),
        );
    }

    fn startThink(self: *App, suggestion: bool) void {
        self.joinAi();

        if (self.game.isOver()) {
            return;
        }

        self.job_id +%= 1;
        const budget = self.budgetMs();
        var limits: Engine.Limits = .withTime(budget * 2 / 3, budget);
        limits.root_min_centre_distance = self.game.openingMinCentreDistance();

        if (!suggestion and self.settings.catchUpOn()) {
            // behind: keep searching until a competitive move turns up
            limits.catch_up_ms = self.catchUpMs();
            limits.catch_up_below = CATCH_UP_BELOW;
        }

        self.job = .{
            .app = self,
            .board = self.game.board,
            .pending = self.game.pendingFive(),
            .limits = limits,
            .id = self.job_id,
        };
        self.job_is_suggestion = suggestion;
        self.have_info = false;
        self.last_info = .empty;
        self.info_side = self.game.board.side_to_move;
        self.info_is_suggestion = suggestion;
        self.suggestion = NONE;
        self.engine.reporter = .{ .maybe_context = &self.job, .maybe_on_info = onInfo };
        self.think_start = self.now();
        self.ai_cancel.store(false, .release);
        self.thinking.store(true, .release);
        self.ai_thread = std.Thread.spawn(.{ .stack_size = 16 << 20 }, aiThreadMain, .{&self.job}) catch blk: {
            // could not spawn: think synchronously (still correct, the UI
            // freezes for one budget); the result is handled by the main loop
            self.engine.reporter = .silent;
            const info = self.engine.think(&self.job.board, self.job.pending, self.job.limits);
            self.thinking.store(false, .release);
            self.sync_result = .{ .id = self.job_id, .info = info };
            break :blk null;
        };
    }

    pub fn thinkElapsedMs(self: *const App) u64 {
        return self.sinceMs(self.think_start);
    }

    /// Each rule variant has its own network; some may have none.
    pub fn usesNetwork(self: *const App) bool {
        return Engine.isNetworkAvailable(self.game.board.rules.variant());
    }

    pub fn isAiTurn(self: *const App) bool {
        if (self.game.isOver() or self.paused) {
            return false;
        }

        return switch (self.settings.mode) {
            .pva => self.game.currentPlayer() == aiPlayer(self),
            .hotseat => false,
            .ava => true,
        };
    }

    pub fn aiPlayer(self: *const App) u1 {
        return if (self.settings.human_first) 1 else 0;
    }

    /// Display name of the player owning a color ("You", "AI", "Player 1").
    pub fn ownerName(self: *const App, c: Board.Color) []const u8 {
        return self.playerName(self.game.player_of_color[c.index()]);
    }

    /// Display name of whoever must act now.
    pub fn actorName(self: *const App) []const u8 {
        return self.playerName(self.game.currentPlayer());
    }

    fn playerName(self: *const App, p: u1) []const u8 {
        return switch (self.settings.mode) {
            .pva => if (p == self.aiPlayer()) "AI" else "You",
            .hotseat => if (p == 0) "Player 1" else "Player 2",
            .ava => "AI",
        };
    }

    /// Decide what happens next after any state change.
    fn advance(self: *App) void {
        if (self.game.isOver()) {
            self.joinAi();
            return;
        }

        if (self.thinking.load(.acquire)) {
            return;
        }

        const choice = self.game.pendingChoice();

        if (choice != .none) {
            if (self.isAiTurn()) {
                self.aiChoose();
            }

            return; // human: wait for b / w / p
        }

        if (self.isAiTurn()) {
            self.startThink(false);
        } else if (self.settings.mode == .hotseat) {
            self.startThink(true); // move suggestion for the human to move
        }
    }

    /// AI decision in swap openings: a short search for the side to move,
    /// then take the color that stands better.
    fn aiChoose(self: *App) void {
        self.engine.reporter = .silent;
        var choice_limits: Engine.Limits = .withTime(150, 200);
        choice_limits.min_depth = 6;
        const info = self.engine.think(&self.game.board, NONE, choice_limits);
        const stm = self.game.board.side_to_move;
        const stm_better = info.score >= 0;
        const take: Game.ChoiceAnswer = if (stm_better) (if (stm == .white) .white else .black) else (if (stm == .white) .black else .white);
        self.game.applyChoice(take);
        var pbuf: [32]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&pbuf);
        self.toast(
            .ai,
            "AI chose {s} (eval {s} for the side to move)",
            .{ colorName(if (take == .white) .white else .black), F.points(fba.allocator(), info.score) },
        );
        self.turn_start = self.now();
        self.advance();
    }

    fn playMove(self: *App, m: Move, by_ai: bool, ai_ms: u64, depth: i32) void {
        const ply = self.game.ply();
        const human_ms = self.sinceMs(self.turn_start);
        const l = self.game.play(m);

        if (l != .ok) {
            self.toast(.bad, "{s}: {s}", .{ board_view.moveName(&self.name_buf, m), l.message() });
            return;
        }

        if (ply < self.move_stats.len) {
            self.move_stats[ply] = .{
                .ms = @intCast(@min(if (by_ai) ai_ms else human_ms, std.math.maxInt(u32))),
                .depth = @intCast(std.math.clamp(depth, 0, 255)),
                .by_ai = by_ai,
                .known = true,
            };
        }

        self.turn_start = self.now();
        self.suggestion = NONE;
        self.hover_point = NONE;

        // effects: placement ripple, captured stones dissolving, "+2" label
        const t = self.ms();
        const b = &self.game.board;
        self.fx.placed(m, t);
        const ncap = b.captured_stone_counts[ply];

        if (ncap > 0) {
            const victim = b.side_to_move.stoneCode();

            for (b.lastCapturedStones()) |s| {
                self.fx.captured(s, victim, t);
            }

            var buf: [8]u8 = undefined;
            self.fx.float(m, std.fmt.bufPrint(&buf, "+{d}", .{ncap}) catch "+", victim, t);
            self.fx.cap_flash[b.side_to_move.opponent().index()] = t;
        }

        if (self.game.isOver()) {
            self.fx.over_t = t;
            self.hide_banner = false;

            if (self.settings.mode == .ava) {
                const sr = &self.series;
                sr.games += 1;

                if (self.game.maybe_winner) |wc| {
                    sr.wins[wc.index()] += 1;
                } else {
                    sr.draws += 1;
                }

                sr.total_plies += self.game.ply();
                sr.longest = @max(sr.longest, self.game.ply());
            }

            const w = self.game.maybe_winner;

            switch (self.game.outcome) {
                .five => {
                    self.toast(.good, "{s} wins with five in a row", .{if (w) |c| colorName(c) else "nobody"});
                },
                .capture => {
                    self.toast(.good, "{s} wins by capturing 10 stones", .{if (w) |c| colorName(c) else "nobody"});
                },
                else => {
                    self.toast(.info, "Draw: the board is full", .{});
                },
            }
        } else if (self.game.pendingFive() != NONE) {
            self.toast(
                .bad,
                "Five in a row, but {s} can still break it by a capture",
                .{colorName(self.game.sideToMove())},
            );
        } else if (ncap > 0) {
            self.toast(
                .info,
                "{s} captured a pair ({d}/{d})",
                .{
                    colorName(b.side_to_move.opponent()),
                    b.captured_pairs[b.side_to_move.opponent().index()],
                    b.rules.capture_win_pairs,
                },
            );
        }

        self.advance();
    }

    fn resetStats(self: *App) void {
        self.have_info = false;
        self.last_info = .empty;
        self.suggestion = NONE;
        self.last_think_ms = 0;
        self.total_think_ms = 0;
        self.max_think_ms = 0;
        self.think_count = 0;
        self.search_n = 0;
        self.move_stats = @splat(.{});
        self.eval_hist = @splat(std.math.nan(f32));
        self.fx.clearBoard();
        self.hide_banner = false;
    }

    /// New game from the menu: also starts a fresh endless series.
    fn newSeries(self: *App) void {
        self.series = .{};
        self.newGame();
    }

    /// Two random stones near the centre (first within 2 of it, second next to it).
    fn randomStart(self: *App, salt: u64) void {
        var prng = std.Random.DefaultPrng.init(self.ms() *% 0x9E3779B97F4A7C15 +% salt);
        const r = prng.random();
        const x0 = r.intRangeAtMost(usize, 7, 11);
        const y0 = r.intRangeAtMost(usize, 7, 11);

        if (self.game.play(Board.moveAt(x0, y0)) != .ok) {
            return;
        }

        var tries: usize = 0;

        while (tries < 32) : (tries += 1) {
            const x = @as(i32, @intCast(x0)) + r.intRangeAtMost(i32, -1, 1);
            const y = @as(i32, @intCast(y0)) + r.intRangeAtMost(i32, -1, 1);

            if (x < 0 or y < 0 or x >= Board.size or y >= Board.size) {
                continue;
            }

            if (self.game.play(Board.moveAt(@intCast(x), @intCast(y))) == .ok) {
                break;
            }
        }
    }

    fn newGame(self: *App) void {
        self.joinAi();
        self.game.reset(
            .{
                .captures = self.settings.captures,
                .double_three = self.settings.double_three,
                .capture_win_pairs = Board.Rules.standard.capture_win_pairs,
            },
            self.settings.opening,
        );
        self.engine.newGame();
        self.resetStats();
        self.paused = false;
        self.screen = .game;
        self.has_game = true;
        self.cursor_x = 9;
        self.cursor_y = 9;
        self.turn_start = self.now();

        if (self.settings.endlessOn() and self.series.games > 0) {
            // the engine is deterministic: vary the games of a series with a
            // random 2-stone opening near the centre (Standard opening only)
            if (self.settings.opening == .standard) {
                self.randomStart(self.series.games);
            }

            self.toast(.ai, "Game {d} of the endless series · random opening", .{self.series.games + 1});
        } else {
            self.toast(
                .info,
                "New game · {s} · {s} opening",
                .{ self.settings.mode.name(), self.settings.opening.name() },
            );
        }

        self.advance();
    }

    fn undo(self: *App) void {
        self.joinAi();

        if (self.game.ply() == 0) {
            self.toast(.info, "Nothing to undo", .{});
            return;
        }

        if (self.settings.mode == .pva) {
            // undo back to the human's turn
            _ = self.game.undo();

            while (self.game.ply() > 0 and self.isAiTurn()) {
                _ = self.game.undo();
            }
        } else {
            _ = self.game.undo();
        }

        self.suggestion = NONE;
        self.have_info = false;
        self.fx.clearBoard();
        self.hide_banner = false;
        self.paused = self.settings.mode == .ava;
        self.turn_start = self.now();
        self.toast(
            .info,
            "Undo · back to move {d}{s}",
            .{ self.game.ply() + 1, if (self.paused) " · paused, p resumes" else "" },
        );
        // hotseat: new suggestion; Player vs AI with the AI moving first:
        // undoing its only move gives it the move again
        self.advance();
    }

    fn redo(self: *App) void {
        self.joinAi();
        const ok = self.game.redo();

        if (self.settings.mode == .pva) {
            while (self.isAiTurn() and self.game.redo()) {}
        }

        self.turn_start = self.now();
        self.suggestion = NONE;
        self.have_info = false;

        if (ok) {
            self.fx.placed(self.game.board.lastMove(), self.ms());

            if (self.game.isOver()) {
                self.fx.over_t = self.ms();
                self.hide_banner = false;
            }

            self.toast(.info, "Redo", .{});
        } else {
            self.toast(.info, "Nothing to redo", .{});
        }

        self.advance();
    }

    fn save(self: *App) void {
        var buf: [32 * 1024]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        self.game.writeRecord(&w) catch {
            self.toast(.bad, "Save failed: game too long", .{});
            return;
        };
        std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = save_path, .data = w.buffered() }) catch |e| {
            self.toast(.bad, "Save failed: {s}", .{@errorName(e)});
            return;
        };
        self.toast(.good, "Saved {d} moves to {s}", .{ self.game.ply(), save_path });
    }

    fn load(self: *App) void {
        self.joinAi();

        if (!self.loadFile()) {
            // nothing loaded: the current game (if any) goes on
            if (self.screen == .game) {
                self.advance();
            }
        }
    }

    /// Apply the saved swap decision if one is pending; false if it's missing.
    fn replayChoice(self: *App, header: []const u8) bool {
        const c = self.game.pendingChoice();

        if (c == .none) {
            return true;
        }

        const ans = Game.headerChoice(header, (c).answerIndex()) orelse {
            return false;
        };

        if (ans == .place_two and c != .swap2_first) {
            return false;
        }

        self.game.applyChoice(ans);
        return true;
    }

    /// Replace the game by the saved one; false (and a toast) on failure.
    fn loadFile(self: *App) bool {
        var buf: [32 * 1024]u8 = undefined;
        const data = std.Io.Dir.cwd().readFile(self.io, save_path, &buf) catch |e| {
            switch (e) {
                error.FileNotFound => {
                    self.toast(.bad, "Load failed: no {s} (s saves a game)", .{save_path});
                },
                else => {
                    self.toast(.bad, "Load failed: {s}", .{@errorName(e)});
                },
            }

            return false;
        };
        var lines = std.mem.tokenizeAny(u8, data, "\r\n");
        const header = lines.next() orelse {
            self.toast(.bad, "Load failed: empty file", .{});
            return false;
        };

        if (!std.mem.startsWith(u8, header, "gomoku ")) {
            self.toast(.bad, "Load failed: not a Gomoku save file", .{});
            return false;
        }

        // "gomoku <Opening name> captures=<bool> double_three=<bool>"
        var opening: Game.Opening = .standard;
        const rest = header["gomoku ".len..];

        for (std.enums.values(Game.Opening)) |o| {
            const nm = o.name();

            if (std.mem.startsWith(u8, rest, nm) and (rest.len == nm.len or rest[nm.len] == ' ')) {
                opening = o;
            }
        }

        // swap decisions are replayed from the header (choice1= / choice2=);
        // old saves without them fall back to free play
        const swap_lost = (opening == .swap or opening == .swap2) and Game.headerChoice(header, 0) == null;

        if (swap_lost) {
            opening = .standard;
        }

        self.settings.captures = std.mem.indexOf(u8, header, "captures=false") == null;
        self.settings.double_three = std.mem.indexOf(u8, header, "double_three=false") == null;
        self.settings.opening = opening;
        self.game.reset(
            .{
                .captures = self.settings.captures,
                .double_three = self.settings.double_three,
                .capture_win_pairs = Board.Rules.standard.capture_win_pairs,
            },
            opening,
        );
        self.engine.newGame();
        self.resetStats();
        self.series = .{};
        var n: usize = 0;
        var bad = false;

        while (lines.next()) |line| {
            const ln = std.mem.trim(u8, line, " \t");

            if (ln.len == 0) {
                continue;
            }

            if (!self.replayChoice(header)) {
                bad = true;
                break;
            }

            var it = std.mem.splitScalar(u8, ln, ',');
            const xs = std.mem.trim(u8, it.next() orelse "", " ");
            const ys = std.mem.trim(u8, it.next() orelse "", " ");
            const x = std.fmt.parseInt(usize, xs, 10) catch {
                bad = true;
                break;
            };
            const y = std.fmt.parseInt(usize, ys, 10) catch {
                bad = true;
                break;
            };

            if (x >= Board.size or y >= Board.size or self.game.isOver() or self.game.play(Board.moveAt(x, y)) != .ok) {
                bad = true;
                break;
            }

            n += 1;
        }

        if (!bad) {
            _ = self.replayChoice(header); // a decision made after the last stone
        }

        self.screen = .game;
        self.has_game = true;
        self.paused = self.settings.mode == .ava;
        self.turn_start = self.now();
        self.cursor_x = 9;
        self.cursor_y = 9;

        if (self.game.ply() > 0) {
            const last = self.game.board.lastMove();
            self.cursor_x = Board.columnOf(last);
            self.cursor_y = Board.rowOf(last);
        }

        if (self.game.isOver()) {
            self.fx.over_t = self.ms();
        }

        if (bad) {
            self.toast(.bad, "Loaded {d} moves, then hit an invalid line", .{n});
        } else if (swap_lost) {
            self.toast(.good, "Loaded {d} moves (swap choices not saved: Standard opening)", .{n});
        } else {
            self.toast(
                .good,
                "Loaded {d} moves · {s} opening{s}",
                .{ n, opening.name(), if (self.paused) " · p to resume" else "" },
            );
        }

        self.advance();
        return true;
    }

    // ---- events --------------------------------------------------------------

    /// Take a new search report. An iteration interrupted by the time limit
    /// may report no root-move scores: keep the previous iteration's list.
    fn mergeInfo(self: *App, info: SearchInfo) void {
        var inf = info;

        if (self.have_info) {
            const old = &self.last_info;

            if (inf.top_move_count == 0 and old.top_move_count > 0) {
                inf.top_moves = old.top_moves;
                inf.top_move_count = old.top_move_count;
            }

            if (inf.depth < old.depth) {
                inf.depth = old.depth;
            }

            if (inf.selective_depth < old.selective_depth) {
                inf.selective_depth = old.selective_depth;
            }
        }

        // list the move actually chosen first (the top list comes from the
        // main thread, the best move may come from a helper thread)
        const n = @min(inf.top_move_count, inf.top_moves.len);

        for (inf.top_moves[0..n], 0..) |t, k| {
            if (t.move != inf.best_move or k == 0) {
                continue;
            }

            std.mem.rotate(Engine.RootScore, inf.top_moves[0 .. k + 1], k);
            break;
        }

        self.last_info = inf;
        self.have_info = true;
    }

    fn handleAiResult(self: *App, msg: AiMsg, final: bool) void {
        if (msg.id != self.job_id) {
            return; // stale (undo / new game)
        }

        self.mergeInfo(msg.info);

        if (!final) {
            return;
        }

        if (self.ai_thread) |t| {
            t.join();
            self.ai_thread = null;
        }

        const t_ms = msg.info.time_ms;
        self.last_think_ms = t_ms;
        self.total_think_ms += t_ms;
        self.max_think_ms = @max(self.max_think_ms, t_ms);
        self.think_count += 1;
        self.search_ms[self.search_n % self.search_ms.len] = @intCast(@min(t_ms, std.math.maxInt(u32)));
        self.search_n += 1;
        const ply = self.job.board.ply;

        if (ply < self.eval_hist.len) {
            self.eval_hist[ply] = sidebar.blackProb(self);
        }

        if (self.job_is_suggestion) {
            self.suggestion = msg.info.best_move;

            if (self.settings.mode != .hotseat) {
                self.toast(
                    .ai,
                    "Hint: {s} (press c to jump there)",
                    .{board_view.moveName(&self.name_buf, msg.info.best_move)},
                );
            }

            return;
        }

        const m = msg.info.best_move;

        if (self.settings.catchUpOn() and t_ms > self.budgetMs() and !eval.isMate(msg.info.score)) {
            if (msg.info.score >= CATCH_UP_BELOW) {
                self.toast(
                    .ai,
                    "{s} dug {d:.1} s for a fighting move",
                    .{ self.ownerName(self.game.sideToMove()), @as(f32, @floatFromInt(t_ms)) / 1000 },
                );
            } else {
                self.toast(
                    .ai,
                    "{s} searched {d:.1} s: still behind",
                    .{ self.ownerName(self.game.sideToMove()), @as(f32, @floatFromInt(t_ms)) / 1000 },
                );
            }
        }

        if (m != NONE) {
            if (self.settings.mode == .ava) {
                self.cursor_x = Board.columnOf(m);
                self.cursor_y = Board.rowOf(m);
            }

            self.playMove(m, true, t_ms, msg.info.depth);
        }
    }

    fn handleKey(self: *App, key: vaxis.Key) void {
        if (key.matches('c', .{ .ctrl = true })) {
            self.quit = true;
            return;
        }

        switch (self.screen) {
            .menu => {
                self.menuKey(key);
            },
            .help => {
                self.screen = self.help_back;
            },
            .game => {
                self.gameKey(key);
            },
        }
    }

    /// Rows of the menu currently shown ("Continue" only when a game exists).
    pub fn menuRows(self: *const App, buf: *[MENU_MAX]MenuRow) []const MenuRow {
        var n: usize = 0;

        for (std.enums.values(MenuRow)) |r| {
            if (r == .resume_game and !self.has_game) {
                continue;
            }

            buf[n] = r;
            n += 1;
        }

        return buf[0..n];
    }

    /// Selected menu row (the selection index is clamped to the list).
    pub fn menuRow(self: *const App) MenuRow {
        var buf: [MENU_MAX]MenuRow = undefined;
        const rows = self.menuRows(&buf);
        return rows[@min(self.menu_sel, rows.len - 1)];
    }

    fn openMenu(self: *App) void {
        self.joinAi();
        self.screen = .menu;
        self.menu_t = self.ms();
        self.menu_sel = 0; // "Continue" when a game exists
        self.fx.toast_len = 0;
        self.fx.menu_disp = 0;
    }

    /// Back to the current game from the menu. Mode, AI time, side and
    /// catch-up changes apply immediately; rule changes wait for a new game.
    fn resumeGame(self: *App) void {
        if (!self.has_game) {
            return;
        }

        self.screen = .game;

        if (self.settings.mode != .ava) {
            self.paused = false;
        }

        self.turn_start = self.now();
        const s = &self.settings;
        const r = self.game.board.rules;

        if (r.captures != s.captures or r.double_three != s.double_three or self.game.opening != s.opening) {
            self.toast(.info, "Rule changes apply to the next game (n)", .{});
        } else if (self.game.isOver()) {
            self.toast(.info, "Game over · n for a new game, u to undo", .{});
        } else {
            self.toast(.info, "Resumed · {s}", .{s.mode.name()});
        }

        self.advance();
    }

    fn menuKey(self: *App, key: vaxis.Key) void {
        var buf: [MENU_MAX]MenuRow = undefined;
        const rows = self.menuRows(&buf);
        const n = rows.len;

        if (self.menu_sel >= n) {
            self.menu_sel = n - 1;
        }

        if (key.matches(vaxis.Key.up, .{}) or key.matches('k', .{}) or key.matches(vaxis.Key.tab, .{ .shift = true })) {
            self.menu_sel = (self.menu_sel + n - 1) % n;
        } else if (key.matches(vaxis.Key.down, .{}) or key.matches('j', .{}) or key.matches(vaxis.Key.tab, .{})) {
            self.menu_sel = (self.menu_sel + 1) % n;
        } else if (key.matches(vaxis.Key.home, .{})) {
            self.menu_sel = 0;
        } else if (key.matches(vaxis.Key.end, .{})) {
            self.menu_sel = n - 1;
        } else if (key.matches(vaxis.Key.escape, .{})) {
            if (self.has_game) {
                self.resumeGame();
            } else {
                self.quit = true;
            }
        } else if (key.matches('q', .{})) {
            self.quit = true;
        } else if (key.matches('?', .{})) {
            self.help_back = .menu;
            self.screen = .help;
        } else if (key.matches('S', .{})) {
            self.newSeries();
        } else if (key.matches('C', .{})) {
            self.resumeGame();
        } else if (key.matches('o', .{})) {
            self.load();
        } else if (key.matches(vaxis.Key.enter, .{}) or key.matches(vaxis.Key.space, .{}) or
            key.matches(vaxis.Key.left, .{}) or key.matches(
            vaxis.Key.right,
            .{},
        ) or key.matches('h', .{}) or key.matches('l', .{})) {
            const back = key.matches(vaxis.Key.left, .{}) or key.matches('h', .{});
            const activate = key.matches(vaxis.Key.enter, .{}) or key.matches(vaxis.Key.space, .{});
            const s = &self.settings;
            const nm = std.enums.values(Mode).len;
            const no = std.enums.values(Game.Opening).len;

            switch (rows[self.menu_sel]) {
                .resume_game => {
                    if (activate) {
                        self.resumeGame();
                    }
                },
                .mode => {
                    s.mode = @fromBackingInt(
                        @intCast((@as(usize, @backingInt(s.mode)) + (if (back) nm - 1 else 1)) % nm),
                    );
                },
                .human_first => {
                    s.human_first = !s.human_first;
                },
                .opening => {
                    s.opening = @fromBackingInt(
                        @intCast((@as(usize, @backingInt(s.opening)) + (if (back) no - 1 else 1)) % no),
                    );
                },
                .think => {
                    s.think_idx = (s.think_idx % think_times.len + (if (back) think_times.len - 1 else 1)) % think_times.len;
                },
                .captures => {
                    s.captures = !s.captures;
                },
                .double_three => {
                    s.double_three = !s.double_three;
                },
                .endless => {
                    s.endless = !s.endless;
                },
                .catch_up => {
                    s.catch_up = !s.catch_up;
                },
                .start => {
                    if (activate) {
                        self.newSeries();
                    }
                },
                .load => {
                    if (activate) {
                        self.load();
                    }
                },
                .quit => {
                    if (activate) {
                        self.quit = true;
                    }
                },
            }
        }
    }

    pub fn humanCanPlay(self: *const App) bool {
        return !self.game.isOver() and !self.isAiTurn() and self.game.pendingChoice() == .none and
            (self.settings.mode != .ava);
    }

    fn humanPlayAtCursor(self: *App) void {
        if (!self.humanCanPlay()) {
            if (self.game.isOver()) {
                self.toast(.bad, "The game is over · n for a new game, u to undo", .{});
            } else if (self.game.pendingChoice() != .none and !self.isAiTurn()) {
                self.toast(.bad, "Choose a color first: b (Black) / w (White)", .{});
            } else if (self.isAiTurn()) {
                self.toast(.ai, "Wait, the AI is thinking", .{});
            }

            return;
        }

        const m = Board.moveAt(self.cursor_x, self.cursor_y);
        const l = self.game.legality(m);

        if (l != .ok) {
            self.toast(.bad, "{s}: {s}", .{ board_view.moveName(&self.name_buf, m), l.message() });
            return;
        }

        self.joinAi();
        self.playMove(m, false, 0, 0);
    }

    fn gameKey(self: *App, key: vaxis.Key) void {
        if (key.matches('q', .{}) or key.matches(vaxis.Key.escape, .{})) {
            self.openMenu();
            return;
        }

        const arrows = [_]struct { k: u21, alt: u21, dx: i32, dy: i32 }{
            .{ .k = vaxis.Key.up, .alt = 'k', .dx = 0, .dy = -1 },
            .{ .k = vaxis.Key.down, .alt = 'j', .dx = 0, .dy = 1 },
            .{ .k = vaxis.Key.left, .alt = 'h', .dx = -1, .dy = 0 },
            .{ .k = vaxis.Key.right, .alt = 'l', .dx = 1, .dy = 0 },
        };

        for (arrows) |a| {
            if (key.matches(a.k, .{}) or key.matches(a.alt, .{})) {
                const nx = std.math.clamp(@as(i32, @intCast(self.cursor_x)) + a.dx, 0, Board.size - 1);
                const ny = std.math.clamp(@as(i32, @intCast(self.cursor_y)) + a.dy, 0, Board.size - 1);
                self.cursor_x = @intCast(nx);
                self.cursor_y = @intCast(ny);
                self.cursor_on_board = true;
                return;
            }
        }

        if (key.matches(vaxis.Key.enter, .{}) or key.matches(vaxis.Key.space, .{})) {
            self.humanPlayAtCursor();
        } else if (key.matches('u', .{})) {
            self.undo();
        } else if (key.matches('r', .{})) {
            self.redo();
        } else if (key.matches('n', .{})) {
            self.newGame();
        } else if (key.matches('d', .{})) {
            self.show_analysis = !self.show_analysis;
            self.toast(
                .ai,
                "Analysis overlay {s}",
                .{if (self.show_analysis) "on: best line, candidates, capturable stones" else "off"},
            );
        } else if (key.matches('v', .{})) {
            self.hide_banner = !self.hide_banner;
        } else if (key.matches('z', .{})) {
            self.zoom = if (self.zoom == .auto) .compact else .auto;
            self.toast(.info, "Board size: {s}", .{if (self.zoom == .auto) "large when it fits" else "compact"});
        } else if (key.matches('s', .{})) {
            self.save();
        } else if (key.matches('o', .{})) {
            self.load();
        } else if (key.matches('?', .{})) {
            self.help_back = .game;
            self.screen = .help;
        } else if (key.matches('g', .{})) {
            if (self.humanCanPlay()) {
                self.startThink(true);
            } else {
                self.toast(.info, "Hints are for the human to move", .{});
            }
        } else if (key.matches('c', .{})) {
            if (self.suggestion != NONE) {
                self.cursor_x = Board.columnOf(self.suggestion);
                self.cursor_y = Board.rowOf(self.suggestion);
                self.cursor_on_board = true;
            }
        } else if (key.matches('p', .{})) {
            if (self.game.pendingChoice() == .swap2_first and !self.isAiTurn()) {
                self.game.applyChoice(.place_two);
                self.toast(.info, "Place two more stones (one of each color)", .{});
                self.advance();
            } else if (self.settings.mode == .ava and !self.game.isOver()) {
                self.paused = !self.paused;

                if (!self.paused) {
                    self.advance();
                } else {
                    self.joinAi();
                }

                self.toast(.info, "{s}", .{if (self.paused) "Paused · p to resume" else "Resumed"});
            }
        } else if (key.matches('b', .{}) or key.matches('w', .{})) {
            if (self.game.pendingChoice() != .none and !self.isAiTurn()) {
                const black = key.matches('b', .{});
                const who = self.actorName();
                self.game.applyChoice(if (black) .black else .white);
                self.toast(.good, "{s} chose {s}", .{ who, if (black) "Black" else "White" });
                self.turn_start = self.now();
                self.advance();
            }
        }
    }

    fn handleMouse(self: *App, mouse: vaxis.Mouse) void {
        const x: i32 = mouse.col;
        const y: i32 = mouse.row;
        const act = self.hits.at(x, y);
        const click = mouse.type == .press and mouse.button == .left;

        self.hover_point = NONE;

        if (act) |a| {
            switch (a) {
                .point => |m| {
                    self.hover_point = m;
                },
                .menu => |i| {
                    if (mouse.type == .motion or click) {
                        self.menu_sel = i;
                    }
                },
                else => {},
            }
        }

        if (self.screen == .menu and mouse.type == .press) {
            if (mouse.button == .wheel_up) {
                self.menuKey(.{ .codepoint = vaxis.Key.up });
            }

            if (mouse.button == .wheel_down) {
                self.menuKey(.{ .codepoint = vaxis.Key.down });
            }
        }

        var on_board = false;

        if (self.screen == .game) {
            if (self.geo) |g| {
                if (act != null and act.? == .key and act.?.key == 0) {
                    if (board_view.hit(g, x, y)) |p| {
                        self.cursor_x = p[0];
                        self.cursor_y = p[1];
                        self.cursor_on_board = true;
                        on_board = true;
                    }
                }
            }
        }

        if (!click) {
            return;
        }

        if (self.screen == .help) {
            self.screen = self.help_back;
            return;
        }

        const a = act orelse {
            return;
        };

        switch (a) {
            .key => |cp| {
                if (cp == 0) {
                    if (on_board) {
                        self.humanPlayAtCursor();
                    }
                } else {
                    self.handleKey(.{ .codepoint = cp });
                }
            },
            .menu => |i| {
                self.menu_sel = i;
                self.menuKey(.{ .codepoint = if (self.menuRow().isSetting()) vaxis.Key.right else vaxis.Key.enter });
            },
            .point => |m| {
                self.cursor_x = Board.columnOf(m);
                self.cursor_y = Board.rowOf(m);
                self.cursor_on_board = true;
            },
        }
    }
};

pub fn colorName(c: Board.Color) []const u8 {
    return if (c == .black) "Black" else "White";
}

fn tickerMain(app: *App) void {
    while (app.running.load(.acquire)) {
        app.io.sleep(.fromMilliseconds(33), .awake) catch {};

        if (app.loop) |l| {
            _ = l.tryPostEvent(.tick) catch {};
        }
    }
}

pub fn run(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    const engine = try Engine.create(gpa, io, .default);
    defer engine.destroy();
    const game = try gpa.create(Game);
    defer gpa.destroy(game);
    game.reset(.standard, .standard);

    var tty_buf: [4096]u8 = undefined;
    var tty = try vaxis.Tty.init(io, &tty_buf);
    defer tty.deinit();
    var vx = try vaxis.init(io, gpa, init.environ_map, .{});
    defer vx.deinit(gpa, tty.writer());

    var loop: LoopT = .init(io, &tty, &vx);
    try loop.installResizeHandler();
    defer loop.uninstallResizeHandler();
    try loop.start();
    defer loop.stop();

    try vx.enterAltScreen(tty.writer());
    try vx.queryTerminal(tty.writer(), .fromSeconds(1));
    try vx.setMouseMode(tty.writer(), true);
    vx.setTitle(tty.writer(), "Gomoku") catch {};

    const app = try gpa.create(App);
    defer gpa.destroy(app);
    app.* = .{ .gpa = gpa, .io = io, .loop = &loop, .engine = engine, .game = game };
    app.t0 = app.now();
    app.turn_start = app.t0;
    app.think_start = app.t0;
    const ticker = std.Thread.spawn(.{}, tickerMain, .{app}) catch null;

    defer {
        app.running.store(false, .release);

        if (ticker) |t| {
            t.join();
        }
    }

    defer app.joinAi();

    var frame_arena: std.heap.ArenaAllocator = .init(gpa);
    defer frame_arena.deinit();

    while (!app.quit) {
        const event = loop.nextEvent() catch {
            break;
        };

        switch (event) {
            .key_press => |key| {
                app.handleKey(key);
            },
            .mouse => |m| {
                app.handleMouse(vx.translateMouse(m));
            },
            .winsize => |ws| {
                vx.resize(gpa, tty.writer(), ws) catch {};
            },
            .tick => {},
            .ai_info => |msg| {
                app.handleAiResult(msg, false);
            },
            .ai_done => |msg| {
                app.handleAiResult(msg, true);
            },
        }

        if (app.quit) {
            break;
        }

        // drain queued ticks so a slow frame never builds a backlog
        while (loop.tryEvent() catch null) |ev| {
            switch (ev) {
                .tick => {},
                .key_press => |key| {
                    app.handleKey(key);
                },
                .mouse => |m| {
                    app.handleMouse(vx.translateMouse(m));
                },
                .winsize => |ws| {
                    vx.resize(gpa, tty.writer(), ws) catch {};
                },
                .ai_info => |msg| {
                    app.handleAiResult(msg, false);
                },
                .ai_done => |msg| {
                    app.handleAiResult(msg, true);
                },
            }
        }

        if (app.sync_result) |msg| {
            app.sync_result = null;
            app.handleAiResult(msg, true);
        }

        if (app.quit) {
            break;
        }

        const t = app.ms();

        // endless series: next game once the result has been shown
        if (app.screen == .game and app.settings.endlessOn() and app.game.isOver() and !app.paused and t -| app.fx.over_t > ENDLESS_PAUSE_MS) {
            app.newGame();
        }

        app.fx.update(t, sidebar.blackProb(app), @floatFromInt(app.menu_sel));
        _ = frame_arena.reset(.retain_capacity);
        app.hits.len = 0;
        const canvas: G.Canvas = .{ .win = vx.window(), .a = frame_arena.allocator(), .hits = &app.hits, .ms = t };
        screens.draw(app, &canvas);
        vx.render(tty.writer()) catch {};
    }
}

test {
    _ = @import("fmt.zig");
    _ = @import("gfx.zig");
    _ = @import("fx.zig");
    _ = @import("theme.zig");
    _ = @import("board_view.zig");
    _ = @import("sidebar.zig");
    _ = @import("screens.zig");
}
