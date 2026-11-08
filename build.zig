const std = @import("std");

const Variant = @import("src/engine/Board.zig").Rules.Variant;
const variant_count = std.enums.values(Variant).len;

const EmbeddedFiles = struct {
    pattern_tables: std.Build.LazyPath,
    networks: [variant_count]std.Build.LazyPath,
    policies: [variant_count]std.Build.LazyPath,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.option(std.builtin.Optimize, "optimize", "Optimization mode (default: fast)") orelse .fast;
    const test_optimize = b.option(std.builtin.Optimize, "test-optimize", "Optimization mode for tests (default: safe)") orelse .safe;
    const nets_directory = b.option([]const u8, "nets", "Directory with <variant>.nnue / <variant>.pol (default assets)") orelse "assets";

    const embedded_files: EmbeddedFiles = .{
        .pattern_tables = addPatternTablesStep(b),
        .networks = variantFiles(b, nets_directory, "nnue"),
        .policies = variantFiles(b, nets_directory, "pol"),
    };
    const vaxis = b.dependency("vaxis", .{ .target = target, .optimize = optimize });

    const exe_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .omit_frame_pointer = optimize == .fast,
        .imports = &.{.{ .name = "vaxis", .module = vaxis.module("vaxis") }},
    });
    addEmbeddedFiles(exe_module, embedded_files);

    const exe = b.addExecutable(.{
        .name = "Gomoku",
        .root_module = exe_module,
    });
    b.installArtifact(exe);
    b.getInstallStep().dependOn(&b.addInstallFile(embedded_files.pattern_tables, "share/pattern_tables.bin").step);

    const run_command = b.addRunArtifact(exe);
    run_command.step.dependOn(b.getInstallStep());
    run_command.addPassthruArgs();
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_command.step);

    const bench_command = b.addRunArtifact(exe);
    bench_command.addArg("--bench");
    bench_command.addPassthruArgs();
    const bench_step = b.step("bench", "Run the engine benchmark");
    bench_step.dependOn(&bench_command.step);

    // Tests default to a safe build so the asserts guarding the incremental
    // board state actually run.
    const test_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = test_optimize,
        .imports = &.{.{ .name = "vaxis", .module = vaxis.module("vaxis") }},
    });
    addEmbeddedFiles(test_module, embedded_files);

    const tests = b.addTest(.{ .root_module = test_module });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);

    const check_step = b.step("check", "Check compilation");
    check_step.dependOn(&exe.step);
}

/// Runs the table generator on the host and returns the blob it writes.
fn addPatternTablesStep(b: *std.Build) std.Build.LazyPath {
    const pattern_module = b.createModule(.{
        .root_source_file = b.path("src/engine/pattern.zig"),
        .target = b.graph.host,
        .optimize = .fast,
    });
    // pattern.zig embeds the tables it is about to generate. The generator
    // never reads them, so an empty stand-in breaks the cycle.
    pattern_module.addAnonymousImport("pattern_tables", .{ .root_source_file = b.path("assets/empty.bin") });

    const generator = b.addExecutable(.{
        .name = "gen_tables",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools/gen_tables.zig"),
            .target = b.graph.host,
            .optimize = .fast,
            .imports = &.{.{ .name = "pattern", .module = pattern_module }},
        }),
    });

    const run_generator = b.addRunArtifact(generator);
    return run_generator.addOutputFileArg("pattern_tables.bin");
}

/// A variant without its file gets the empty stand-in, so the engine falls
/// back to the classical code for it instead of failing the build.
fn variantFiles(b: *std.Build, directory: []const u8, extension: []const u8) [variant_count]std.Build.LazyPath {
    var files: [variant_count]std.Build.LazyPath = undefined;
    for (std.enums.values(Variant), &files) |variant, *file| {
        const path = b.fmt("{s}/{s}.{s}", .{ directory, @tagName(variant), extension });
        file.* = b.path("assets/empty.bin");
        const opened = b.root.openFile(b.graph.io, path, .{}) catch {
            continue;
        };
        opened.close(b.graph.io);
        file.* = b.path(path);
    }
    return files;
}

fn addEmbeddedFiles(module: *std.Build.Module, embedded_files: EmbeddedFiles) void {
    module.addAnonymousImport("pattern_tables", .{ .root_source_file = embedded_files.pattern_tables });
    for (std.enums.values(Variant), embedded_files.networks, embedded_files.policies) |variant, network, policy| {
        module.addAnonymousImport(module.owner.fmt("nnue_{s}", .{@tagName(variant)}), .{ .root_source_file = network });
        module.addAnonymousImport(module.owner.fmt("policy_{s}", .{@tagName(variant)}), .{ .root_source_file = policy });
    }
}
