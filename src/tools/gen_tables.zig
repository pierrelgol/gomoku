//! Build-time generator for the pattern tables.
//!
//! Classifying all 64K line windows recursively is too slow to do at comptime
//! and pointless to redo at every start, so the build runs this once natively
//! and the engine embeds the resulting blob. The build system only reruns it
//! when pattern.zig changes.

const std = @import("std");
const pattern = @import("pattern");

pub fn main(init: std.process.Init) !void {
    const arguments = try init.minimal.args.toSlice(init.arena.allocator());

    if (arguments.len < 2) {
        return error.MissingOutputPath;
    }

    const output_path = arguments[1];

    // 192 KB of tables would be a risky amount of stack, so they go on the heap.
    const allocator = init.gpa;
    const line_table = try allocator.create([1 << 16]u16);
    defer allocator.destroy(line_table);
    const combine_table = try allocator.create([1 << 16]pattern.CellInfo);
    defer allocator.destroy(combine_table);

    line_table.* = pattern.buildLineTable();
    combine_table.* = pattern.buildCombineTable();

    const io = init.io;
    const file = try std.Io.Dir.cwd().createFile(io, output_path, .{});
    defer file.close(io);

    var buffer: [1 << 16]u8 = undefined;
    var file_writer = file.writer(io, &buffer);
    try file_writer.interface.writeAll(std.mem.sliceAsBytes(line_table));
    try file_writer.interface.writeAll(std.mem.sliceAsBytes(combine_table));
    try file_writer.interface.flush();
}
