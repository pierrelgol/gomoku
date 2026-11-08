const std = @import("std");
const bench = @import("tools/bench.zig");
const datagen = @import("tools/datagen.zig");
const match = @import("tools/match.zig");
const Board = @import("engine/Board.zig");

const usage =
    \\Usage: Gomoku [option...]
    \\
    \\With no option: interactive game in the terminal (Player vs AI,
    \\hotseat with move suggestions, AI vs AI). Press ? in the game for help.
    \\
    \\Engine tools:
    \\  --bench                 search benchmark on fixed positions
    \\  --fixed                 fixed-depth benchmark (--depth N, --pos N)
    \\  --selfplay              engine self-play games
    \\  --threads N             search threads (0 = all cores)
    \\  --time MS               time per move (soft limit; hard = 1.5x)
    \\  --games N               number of games / pairs
    \\  --moves N               moves per game (self-play)
    \\  --depth N               search depth (--fixed)
    \\  --pos N                 single benchmark position
    \\  --trace                 verbose search trace
    \\
    \\Training data and matches:
    \\  --gendata               generate training data
    \\  --nodes N               nodes per move (--gendata)
    \\  --seed N                random seed
    \\  --out DIR               output directory (--gendata)
    \\  --no-captures           play without captures (--gendata, --match, checks)
    \\  --no-double-three       allow double free-threes (--gendata, --match, checks)
    \\  --match                 engine A vs engine B match
    \\  --match-nodes N         nodes per move in the match (0 = use --time)
    \\  --a-net                 engine A uses the network evaluation
    \\  --a-pol / --b-pol       engine A / B uses the policy move ordering
    \\  --b-classic             engine B uses the classical evaluation
    \\  --nncheck FILE          check a network file
    \\  --polcheck FILE         check a policy file (--limit N positions)
    \\  --limit N               position limit (--polcheck)
    \\  --polmult N             policy score multiplier
    \\  --pol-lmr OFF DIV MAX   policy-driven late move reductions
    \\  --pol-prune N           policy pruning threshold
    \\
    \\  -h, --help              show this help
    \\
;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const io = init.io;
    const gpa = init.gpa;

    var mode: enum { ui, bench, selfplay, fixed, gendata, match, nncheck, polcheck } = .ui;
    var rules: Board.Rules = .standard;
    var mo: match.Options = .{ .rules = .standard };
    var path: []const u8 = "";
    var limit: usize = 0;
    var opts: bench.Options = .{};
    var dg: datagen.Options = .{ .rules = .standard };
    var i: usize = 1;

    while (i < args.len) : (i += 1) {
        const a = args[i];

        if (std.mem.eql(u8, a, "--gendata")) {
            mode = .gendata;
        } else if (std.mem.eql(u8, a, "--no-captures")) {
            rules.captures = false;
        } else if (std.mem.eql(u8, a, "--no-double-three")) {
            rules.double_three = false;
        } else if (std.mem.eql(u8, a, "--match")) {
            mode = .match;
        } else if (std.mem.eql(u8, a, "--match-nodes") and i + 1 < args.len) {
            i += 1;
            mo.nodes = std.fmt.parseInt(u64, args[i], 10) catch 0;
        } else if (std.mem.eql(u8, a, "--nncheck") and i + 1 < args.len) {
            mode = .nncheck;
            i += 1;
            path = args[i];
        } else if (std.mem.eql(u8, a, "--polcheck") and i + 1 < args.len) {
            mode = .polcheck;
            i += 1;
            path = args[i];
        } else if (std.mem.eql(u8, a, "--limit") and i + 1 < args.len) {
            i += 1;
            limit = std.fmt.parseInt(usize, args[i], 10) catch 0;
        } else if (std.mem.eql(u8, a, "--a-net")) {
            mo.a_net = true;
        } else if (std.mem.eql(u8, a, "--a-pol")) {
            mo.a_pol = true;
        } else if (std.mem.eql(u8, a, "--b-pol")) {
            mo.b_pol = true;
        } else if (std.mem.eql(u8, a, "--b-classic")) {
            mo.b_net = false;
        } else if (std.mem.eql(u8, a, "--pol-lmr") and i + 3 < args.len) {
            const pm = @import("engine/policy.zig");
            pm.tuning.reduction_offset = std.fmt.parseInt(i32, args[i + 1], 10) catch 512;
            pm.tuning.reduction_divisor = std.fmt.parseInt(i32, args[i + 2], 10) catch 512;
            pm.tuning.max_reduction = std.fmt.parseInt(i32, args[i + 3], 10) catch 0;
            i += 3;
        } else if (std.mem.eql(u8, a, "--pol-prune") and i + 1 < args.len) {
            i += 1;
            @import("engine/policy.zig").tuning.prune_margin = std.fmt.parseInt(i32, args[i], 10) catch 0;
        } else if (std.mem.eql(u8, a, "--polmult") and i + 1 < args.len) {
            i += 1;
            @import("engine/policy.zig").tuning.score_multiplier = std.fmt.parseInt(i32, args[i], 10) catch 16;
        } else if (std.mem.eql(u8, a, "--nodes") and i + 1 < args.len) {
            i += 1;
            dg.nodes = std.fmt.parseInt(u64, args[i], 10) catch dg.nodes;
        } else if (std.mem.eql(u8, a, "--seed") and i + 1 < args.len) {
            i += 1;
            dg.seed = std.fmt.parseInt(u64, args[i], 10) catch dg.seed;
        } else if (std.mem.eql(u8, a, "--out") and i + 1 < args.len) {
            i += 1;
            dg.out = args[i];
        } else if (std.mem.eql(u8, a, "--bench")) {
            mode = .bench;
        } else if (std.mem.eql(u8, a, "--fixed")) {
            mode = .fixed;
        } else if (std.mem.eql(u8, a, "--pos") and i + 1 < args.len) {
            i += 1;
            opts.pos = std.fmt.parseInt(usize, args[i], 10) catch null;
        } else if (std.mem.eql(u8, a, "--depth") and i + 1 < args.len) {
            i += 1;
            opts.depth = std.fmt.parseInt(i32, args[i], 10) catch 10;
        } else if (std.mem.eql(u8, a, "--trace")) {
            opts.trace = true;
        } else if (std.mem.eql(u8, a, "--selfplay")) {
            mode = .selfplay;
        } else if (std.mem.eql(u8, a, "--threads") and i + 1 < args.len) {
            i += 1;
            opts.threads = std.fmt.parseInt(usize, args[i], 10) catch 0;
        } else if (std.mem.eql(u8, a, "--time") and i + 1 < args.len) {
            i += 1;
            opts.soft_ms = std.fmt.parseInt(u32, args[i], 10) catch opts.soft_ms;
            opts.hard_ms = opts.soft_ms * 3 / 2;
        } else if (std.mem.eql(u8, a, "--games") and i + 1 < args.len) {
            i += 1;
            opts.games = std.fmt.parseInt(usize, args[i], 10) catch opts.games;
        } else if (std.mem.eql(u8, a, "--moves") and i + 1 < args.len) {
            i += 1;
            opts.moves = std.fmt.parseInt(usize, args[i], 10) catch opts.moves;
        } else if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            std.debug.print("{s}", .{usage});
            return;
        } else {
            std.debug.print("Gomoku: unknown option or missing value: '{s}'\n\n{s}", .{ a, usage });
            std.process.exit(2);
        }
    }

    switch (mode) {
        .bench => {
            try bench.run(gpa, io, opts);
        },
        .selfplay => {
            try bench.selfplay(gpa, io, opts);
        },
        .fixed => {
            try bench.fixedDepth(gpa, io, opts);
        },
        .gendata => {
            dg.threads = opts.threads;
            dg.games = opts.games;
            dg.rules = rules;
            try datagen.run(gpa, io, dg);
        },
        .match => {
            mo.threads = opts.threads;
            mo.pairs = opts.games;
            mo.soft_ms = opts.soft_ms;
            mo.hard_ms = opts.hard_ms;
            mo.seed = dg.seed;
            mo.rules = rules;
            try match.run(gpa, io, mo);
        },
        .nncheck => {
            try match.nncheck(gpa, io, path, rules);
        },
        .polcheck => {
            try match.polcheck(gpa, io, path, limit, rules);
        },
        .ui => {
            const ui = @import("ui/app.zig");
            ui.run(init) catch |e| {
                const name = @errorName(e);

                if (std.mem.eql(u8, name, "NoDevice") or std.mem.eql(
                    u8,
                    name,
                    "NotATerminal",
                ) or std.mem.eql(u8, name, "FileNotFound")) {
                    std.debug.print(
                        "Gomoku: no terminal available ({s}).\nRun it from an interactive terminal; --help lists the other modes.\n",
                        .{name},
                    );
                } else {
                    std.debug.print("Gomoku: terminal error: {s}\n", .{name});
                }

                std.process.exit(1);
            };
        },
    }
}

test {
    _ = @import("engine/pattern.zig");
    _ = @import("engine/Board.zig");
    _ = @import("engine/movepick.zig");
    _ = @import("engine/Game.zig");
    _ = @import("engine/search.zig");
    _ = @import("ui/app.zig"); // app.zig's own test block pulls in the other ui files
}
