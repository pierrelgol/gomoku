//! Number / time formatting helpers for the UI (std only, allocation into the
//! per-frame arena; every function falls back to a static string on OOM).

const std = @import("std");

/// Compact SI form: 999 -> "999", 12345 -> "12.3 k", 1234567 -> "1.23 M".
pub fn si(a: std.mem.Allocator, v: u64) []const u8 {
    const f: f64 = @floatFromInt(v);

    if (v < 1000) {
        return std.fmt.allocPrint(a, "{d}", .{v}) catch "?";
    }

    const units = [_]struct { div: f64, suf: []const u8 }{
        .{ .div = 1e3, .suf = "k" },
        .{ .div = 1e6, .suf = "M" },
        .{ .div = 1e9, .suf = "G" },
        .{ .div = 1e12, .suf = "T" },
    };
    var k: usize = 0;

    while (k + 1 < units.len and f >= units[k + 1].div * 0.9995) {
        k += 1;
    }

    const x = f / units[k].div;

    if (x < 9.995) {
        return std.fmt.allocPrint(a, "{d:.2} {s}", .{ x, units[k].suf }) catch "?";
    }

    if (x < 99.95) {
        return std.fmt.allocPrint(a, "{d:.1} {s}", .{ x, units[k].suf }) catch "?";
    }

    return std.fmt.allocPrint(a, "{d:.0} {s}", .{ x, units[k].suf }) catch "?";
}

/// Milliseconds as seconds: 412 -> "0.412 s", 61234 -> "61.2 s".
pub fn secs(a: std.mem.Allocator, ms: u64) []const u8 {
    if (ms < 10_000) {
        return std.fmt.allocPrint(a, "{d}.{d:0>3} s", .{ ms / 1000, ms % 1000 }) catch "?";
    }

    if (ms < 1_000_000) {
        return std.fmt.allocPrint(a, "{d}.{d} s", .{ ms / 1000, (ms % 1000) / 100 }) catch "?";
    }

    return std.fmt.allocPrint(a, "{d} s", .{ms / 1000}) catch "?";
}

/// Digits only (for the big timer): 412 -> "0.412", 12345 -> "12.34".
pub fn secsDigits(buf: []u8, ms: u64) []const u8 {
    if (ms < 10_000) {
        return std.fmt.bufPrint(buf, "{d}.{d:0>3}", .{ ms / 1000, ms % 1000 }) catch "-";
    }

    if (ms < 100_000) {
        return std.fmt.bufPrint(buf, "{d}.{d:0>2}", .{ ms / 1000, (ms % 1000) / 10 }) catch "-";
    }

    return std.fmt.bufPrint(buf, "{d}", .{ms / 1000}) catch "-";
}

/// Signed evaluation in "points" (1/100 of the internal unit): 345 -> "+3.45".
pub fn points(a: std.mem.Allocator, v: i32) []const u8 {
    const mag: u32 = @abs(v);
    const sign: []const u8 = if (v > 0) "+" else if (v < 0) "-" else " ";
    return std.fmt.allocPrint(a, "{s}{d}.{d:0>2}", .{ sign, mag / 100, mag % 100 }) catch "?";
}

/// Logistic win probability (0..1) of an evaluation for the side it refers to.
pub fn winProb(v: i32) f64 {
    const x: f64 = @floatFromInt(v);
    return 1.0 / (1.0 + @exp(-x / 800.0)); // engine units: 800 = 1 logit (network output scale K)
}

test si {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("999", si(a, 999));
    try std.testing.expectEqualStrings("1.00 k", si(a, 1000));
    try std.testing.expectEqualStrings("12.3 k", si(a, 12345));
    try std.testing.expectEqualStrings("1.23 M", si(a, 1234567));
    try std.testing.expectEqualStrings("1.00 M", si(a, 999999));
    try std.testing.expectEqualStrings("456 M", si(a, 456_000_000));
}

test secs {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("0.412 s", secs(a, 412));
    try std.testing.expectEqualStrings("1.005 s", secs(a, 1005));
    try std.testing.expectEqualStrings("61.2 s", secs(a, 61234));
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("0.045", secsDigits(&buf, 45));
    try std.testing.expectEqualStrings("+3.45", points(a, 345));
    try std.testing.expectEqualStrings("-0.05", points(a, -5));
}
