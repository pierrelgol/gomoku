//! Palette and color math. A dark, low-contrast chrome around a warm wooden
//! goban; one warm accent (amber) for the human side of things and one cool
//! accent (blue) for everything the engine does.

const std = @import("std");
const vaxis = @import("vaxis");
pub const Color = vaxis.Color;

pub fn rgb(comptime hex: u24) Color {
    return .{ .rgb = .{ @truncate(hex >> 16), @truncate(hex >> 8), @truncate(hex) } };
}

// ---- chrome
pub const bg = rgb(0x0F1117);
pub const bg_deep = rgb(0x090A0E);
pub const card = rgb(0x161923);
pub const card_hi = rgb(0x1E2230);
pub const border = rgb(0x2A2F3E);
pub const border_hi = rgb(0x3D4459);

pub const text = rgb(0xE6E8EF);
pub const dim = rgb(0x8B92A6);
pub const faint = rgb(0x4B5166);
pub const ghost = rgb(0x2C3141);

pub const accent = rgb(0xF5B961); // amber: you, highlights
pub const ai = rgb(0x7AA2F7); // blue: engine activity
pub const pv = rgb(0xBB9AF7); // violet: principal variation
pub const good = rgb(0x9ECE6A);
pub const warn = rgb(0xE0AF68);
pub const bad = rgb(0xF7768E);
pub const cyan = rgb(0x7DCFFF);

// ---- goban
pub const wood_a = rgb(0xE2B26A);
pub const wood_b = rgb(0xD29E52);
pub const line = rgb(0x6E4B24);
pub const label = rgb(0x5A3A16);
pub const stone_black = rgb(0x1A1B22);
pub const stone_black_hi = rgb(0x3A3D4A);
pub const stone_white = rgb(0xF5F2EA);
pub const stone_white_lo = rgb(0xC9C3B5);
pub const last_mark = rgb(0xF0503C);
pub const win_glow = rgb(0xFFE07A);

pub fn mix(a: Color, b: Color, t_in: f32) Color {
    const t = std.math.clamp(t_in, 0, 1);
    const ar = toRgb(a);
    const br = toRgb(b);
    var out: [3]u8 = undefined;

    for (0..3) |i| {
        const fa: f32 = @floatFromInt(ar[i]);
        const fb: f32 = @floatFromInt(br[i]);
        out[i] = @intFromFloat(std.math.clamp(fa + (fb - fa) * t, 0, 255));
    }

    return .{ .rgb = out };
}

pub fn toRgb(c: Color) [3]u8 {
    return switch (c) {
        .rgb => |v| v,
        else => .{ 0, 0, 0 },
    };
}

/// Smooth 0..1..0 pulse with the given period (ms).
pub fn pulse(ms: u64, period: u64) f32 {
    const ph: f32 = @floatFromInt(ms % period);
    const p: f32 = @floatFromInt(period);
    return 0.5 - 0.5 * @cos(ph / p * std.math.tau);
}

/// Color for a fraction of the 0.5 s time limit: green -> amber -> red.
pub fn heat(frac: f32) Color {
    if (frac < 0.6) {
        return mix(good, warn, frac / 0.6);
    }

    return mix(warn, bad, (frac - 0.6) / 0.4);
}

pub fn easeOut(t_in: f32) f32 {
    const t = std.math.clamp(t_in, 0, 1);
    return 1 - (1 - t) * (1 - t) * (1 - t);
}
