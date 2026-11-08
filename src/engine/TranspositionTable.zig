//! Transposition table shared by every search thread, without locks.
//!
//! Each slot stores `key ^ entry` next to `entry`. When two threads write the
//! same slot at once, a reader can see one word from each store; the XOR check
//! then fails and the slot just reads as a miss. That is the Hyatt/Mann
//! lockless trick, and it is far cheaper than a lock per probe. Four slots make
//! a 64-byte bucket so a probe touches exactly one cache line.

const std = @import("std");
const eval = @import("eval.zig");
const Board = @import("Board.zig");

const Self = @This();

buckets: []Bucket,
generation: u5,
maybe_allocator: ?std.mem.Allocator,

pub const Bound = enum(u2) { none = 0, upper = 1, lower = 2, exact = 3 };

pub const Entry = packed struct(u64) {
    move: u16,
    value: i16,
    static_eval: i16,
    depth: u8,
    bound: Bound,
    is_pv: bool,
    generation: u5,
};

const Slot = extern struct {
    checksum: u64,
    entry: u64,
};

pub const Bucket = extern struct {
    slots: [4]Slot align(64),
};

comptime {
    std.debug.assert(@sizeOf(Bucket) == 64);
}

/// Used when even the smallest heap table cannot be allocated, so creating an
/// engine never fails just because memory is tight.
var fallback_buckets: [1 << 14]Bucket = undefined;

/// Allocates `megabytes` of table, halving on OOM until we reach the size of
/// the static fallback.
pub fn init(allocator: std.mem.Allocator, megabytes: usize) Self {
    const requested_bytes = @max(megabytes, 1) * (1 << 20);
    var bucket_count: usize = std.math.floorPowerOfTwo(usize, requested_bytes / @sizeOf(Bucket));

    while (bucket_count > fallback_buckets.len) : (bucket_count /= 2) {
        const buckets = allocator.alignedAlloc(Bucket, .@"64", bucket_count) catch {
            continue;
        };
        var table: Self = .{ .buckets = buckets, .generation = 0, .maybe_allocator = allocator };
        table.clear();
        return table;
    }

    var table: Self = .{ .buckets = &fallback_buckets, .generation = 0, .maybe_allocator = null };
    table.clear();
    return table;
}

pub fn deinit(self: *Self) void {
    if (self.maybe_allocator) |allocator| {
        allocator.free(self.buckets);
    }

    self.* = undefined;
}

pub fn clear(self: *Self) void {
    @memset(std.mem.sliceAsBytes(self.buckets), 0);
    self.generation = 0;
}

pub fn newSearch(self: *Self) void {
    self.generation +%= 1;
}

inline fn bucketFor(self: *const Self, key: u64) *Bucket {
    // Multiply-shift maps the key onto any table size without a modulo, which
    // matters because the size is not a power of two after an OOM fallback.
    const index: usize = @intCast((@as(u128, key) * self.buckets.len) >> 64);
    return &self.buckets[index];
}

pub inline fn prefetch(self: *const Self, key: u64) void {
    @prefetch(self.bucketFor(key), .{ .rw = .read, .locality = 3, .cache = .data });
}

/// Checks all four slots of the bucket with one vector load. The load races
/// with writers on purpose: a torn slot fails the XOR check like any miss.
pub fn probe(self: *const Self, key: u64) ?Entry {
    const bucket = self.bucketFor(key);
    const words: @Vector(8, u64) = @as(*const volatile @Vector(8, u64), @ptrCast(bucket)).*;
    const checksums = @shuffle(u64, words, undefined, @Vector(4, i32){ 0, 2, 4, 6 });
    const entries = @shuffle(u64, words, undefined, @Vector(4, i32){ 1, 3, 5, 7 });

    const key_matches = checksums ^ entries == @as(@Vector(4, u64), @splat(key));
    const is_used = entries != @as(@Vector(4, u64), @splat(0));
    const hit_mask: u4 = @bitCast(key_matches & is_used);

    if (hit_mask == 0) {
        return null;
    }

    const entry_words: [4]u64 = entries;
    return @bitCast(entry_words[@ctz(hit_mask)]);
}

pub fn store(
    self: *Self,
    key: u64,
    value: i32,
    static_eval: i32,
    depth: i32,
    bound: Bound,
    move: Board.Move,
    is_pv: bool,
) void {
    const bucket = self.bucketFor(key);
    var victim_index: usize = 0;
    var victim_worth: i32 = std.math.maxInt(i32);
    var move_to_keep: u16 = move;

    inline for (0..4) |slot_index| {
        const checksum = @atomicLoad(u64, &bucket.slots[slot_index].checksum, .monotonic);
        const entry_word = @atomicLoad(u64, &bucket.slots[slot_index].entry, .monotonic);
        const entry: Entry = @bitCast(entry_word);

        if (entry_word == 0 or checksum ^ entry_word == key) {
            if (entry_word != 0) {
                // A much deeper bound from this search is worth more than a
                // fresh shallow one, unless the new one is exact.
                if (bound != .exact and entry.depth > depth + 2 and entry.generation == self.generation) {
                    return;
                }

                if (move == Board.no_move) {
                    move_to_keep = entry.move;
                }
            }

            victim_index = slot_index;
            victim_worth = std.math.minInt(i32);
            break;
        }

        const age: i32 = @as(u5, self.generation -% entry.generation);
        const worth = @as(i32, entry.depth) - 4 * age;

        if (worth < victim_worth) {
            victim_worth = worth;
            victim_index = slot_index;
        }
    }

    const new_entry: Entry = .{
        .move = move_to_keep,
        .value = @intCast(std.math.clamp(value, -32000, 32000)),
        .static_eval = @intCast(std.math.clamp(static_eval, -32000, 32000)),
        .depth = @intCast(std.math.clamp(depth, 0, 255)),
        .bound = bound,
        .is_pv = is_pv,
        .generation = self.generation,
    };
    const new_entry_word: u64 = @bitCast(new_entry);
    @atomicStore(u64, &bucket.slots[victim_index].checksum, key ^ new_entry_word, .monotonic);
    @atomicStore(u64, &bucket.slots[victim_index].entry, new_entry_word, .monotonic);
}

/// Permille of slots written by the current search, sampled over the first
/// 250 buckets the way UCI engines report "hashfull".
pub fn hashfullPermille(self: *const Self) u32 {
    const sampled_bucket_count: usize = @min(self.buckets.len, 250);

    var current_count: u32 = 0;

    for (self.buckets[0..sampled_bucket_count]) |*bucket| {
        for (bucket.slots) |slot| {
            const entry: Entry = @bitCast(slot.entry);

            if (slot.entry != 0 and entry.generation == self.generation) {
                current_count += 1;
            }
        }
    }

    return current_count * 1000 / @as(u32, @intCast(sampled_bucket_count * 4));
}

/// Mate scores are stored as distance from the node rather than from the root,
/// because the same position can be reached at different plies.
pub inline fn toStoredValue(value: i32, ply: i32) i32 {
    if (value >= eval.mate_threshold) {
        return value + ply;
    }

    if (value <= -eval.mate_threshold) {
        return value - ply;
    }

    return value;
}

pub inline fn fromStoredValue(stored_value: i32, ply: i32) i32 {
    if (stored_value >= eval.mate_threshold) {
        return stored_value - ply;
    }

    if (stored_value <= -eval.mate_threshold) {
        return stored_value + ply;
    }

    return stored_value;
}
