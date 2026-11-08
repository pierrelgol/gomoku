//! Animation state: short-lived, time-stamped effects (stone placement,
//! captures, floating labels, toasts) and smoothed values (eval gauge, menu
//! highlight). All times are milliseconds of the UI clock.

const std = @import("std");
const Board = @import("../engine/Board.zig");

const Move = Board.Move;
const NONE = Board.no_move;

pub const PLACE_MS: u64 = 650;
pub const CAPTURE_MS: u64 = 1100;
pub const FLOAT_MS: u64 = 1300;
pub const TOAST_MS: u64 = 3200;

pub const Captured = struct { m: Move, code: u8, t: u64 };
pub const Float = struct { m: Move, t: u64, text: [8]u8, len: u8, code: u8 };

pub const ToastKind = enum { info, good, bad, ai };

pub const Fx = struct {
    place_m: Move = NONE,
    place_t: u64 = 0,

    caps: [16]Captured = undefined,
    ncaps: usize = 0,
    floats: [8]Float = undefined,
    nfloats: usize = 0,
    /// time of the last capture per capturing color (pip flash)
    cap_flash: [2]u64 = .{ 0, 0 },

    toast_buf: [192]u8 = undefined,
    toast_len: usize = 0,
    toast_t: u64 = 0,
    toast_kind: ToastKind = .info,

    /// game end time (banner fade-in), 0 = not over
    over_t: u64 = 0,

    // smoothed values
    eval_disp: f32 = 0.5,
    menu_disp: f32 = 0,
    last_ms: u64 = 0,

    pub fn clearBoard(f: *Fx) void {
        f.place_m = NONE;
        f.ncaps = 0;
        f.nfloats = 0;
        f.over_t = 0;
    }

    pub fn placed(f: *Fx, m: Move, now: u64) void {
        f.place_m = m;
        f.place_t = now;
    }

    pub fn captured(f: *Fx, m: Move, code: u8, now: u64) void {
        if (f.ncaps == f.caps.len) {
            std.mem.copyForwards(Captured, f.caps[0 .. f.caps.len - 1], f.caps[1..]);
            f.ncaps -= 1;
        }

        f.caps[f.ncaps] = .{ .m = m, .code = code, .t = now };
        f.ncaps += 1;
    }

    pub fn float(f: *Fx, m: Move, text: []const u8, code: u8, now: u64) void {
        if (f.nfloats == f.floats.len) {
            std.mem.copyForwards(Float, f.floats[0 .. f.floats.len - 1], f.floats[1..]);
            f.nfloats -= 1;
        }

        var fl: Float = .{ .m = m, .t = now, .text = undefined, .len = @intCast(@min(text.len, 8)), .code = code };
        @memcpy(fl.text[0..fl.len], text[0..fl.len]);
        f.floats[f.nfloats] = fl;
        f.nfloats += 1;
    }

    pub fn toast(f: *Fx, kind: ToastKind, now: u64, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(&f.toast_buf, fmt, args) catch blk: {
            // too long: cut on a UTF-8 boundary and end with "..."
            var k = f.toast_buf.len - 3;

            while (k > 0 and (f.toast_buf[k] & 0xC0) == 0x80) {
                k -= 1;
            }

            @memcpy(f.toast_buf[k .. k + 3], "...");
            break :blk f.toast_buf[0 .. k + 3];
        };
        f.toast_len = s.len;
        f.toast_t = now;
        f.toast_kind = kind;
    }

    pub fn toastText(f: *const Fx) []const u8 {
        return f.toast_buf[0..f.toast_len];
    }

    /// Drop finished effects; ease smoothed values toward their targets.
    pub fn update(f: *Fx, now: u64, eval_target: f32, menu_target: f32) void {
        const dt: f32 = @floatFromInt(@min(now -| f.last_ms, 200));
        f.last_ms = now;
        const k_eval = 1 - @exp(-dt / 140.0);
        const k_menu = 1 - @exp(-dt / 55.0);
        f.eval_disp += (eval_target - f.eval_disp) * k_eval;
        f.menu_disp += (menu_target - f.menu_disp) * k_menu;

        if (@abs(f.menu_disp - menu_target) < 0.02) {
            f.menu_disp = menu_target;
        }

        var i: usize = 0;

        while (i < f.ncaps) {
            if (now -| f.caps[i].t > CAPTURE_MS) {
                std.mem.copyForwards(Captured, f.caps[i .. f.ncaps - 1], f.caps[i + 1 .. f.ncaps]);
                f.ncaps -= 1;
            } else {
                i += 1;
            }
        }

        i = 0;

        while (i < f.nfloats) {
            if (now -| f.floats[i].t > FLOAT_MS) {
                std.mem.copyForwards(Float, f.floats[i .. f.nfloats - 1], f.floats[i + 1 .. f.nfloats]);
                f.nfloats -= 1;
            } else {
                i += 1;
            }
        }

        if (f.toast_len > 0 and now -| f.toast_t > TOAST_MS) {
            f.toast_len = 0;
        }
    }

    /// Progress 0..1 of an effect started at `t` lasting `dur` (null = done).
    pub fn progress(now: u64, t: u64, dur: u64) ?f32 {
        const age = now -| t;

        if (age >= dur) {
            return null;
        }

        return @as(f32, @floatFromInt(age)) / @as(f32, @floatFromInt(dur));
    }
};

test "toast truncation keeps valid UTF-8" {
    var f: Fx = .{};
    var long: [400]u8 = undefined;
    var i: usize = 0;

    while (i < long.len) : (i += 2) {
        @memcpy(long[i .. i + 2], "·");
    }

    f.toast(.info, 0, "{s}", .{&long});
    try std.testing.expect(std.unicode.utf8ValidateSlice(f.toastText()));
    try std.testing.expect(std.mem.endsWith(u8, f.toastText(), "..."));
}
