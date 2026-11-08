//! A real game on top of Board: who won and how (five, capture win, the
//! endgame capture rule), legality with a reason the UI can show, undo and
//! redo, and the opening rules (Standard, Pro, Long Pro, Swap, Swap2) with the
//! player roles they create.

const std = @import("std");
const Board = @import("Board.zig");

const Self = @This();
const Move = Board.Move;
const Color = Board.Color;

/// The player index (0 or 1) that owns each color.
pub const PlayerIndex = u1;

board: Board,
opening: Opening,
/// pending_five_history[ply] is the breakable five made by the move before
/// that ply, or no_move. Kept per ply so undo and redo restore it for free.
pending_five_history: [Board.max_history + 1]Move,
outcome: Outcome,
maybe_winner: ?Color,
redo_moves: [Board.max_history]Move,
redo_count: usize,
player_of_color: [2]PlayerIndex,
is_choice_made: [2]bool,
chose_place_two: bool,
/// What was answered at each swap decision, kept for save files.
maybe_choice_answers: [2]?ChoiceAnswer,

pub const Opening = enum {
    standard,
    pro,
    long_pro,
    swap,
    swap2,

    pub fn name(opening: Opening) []const u8 {
        return switch (opening) {
            .standard => "Standard",
            .pro => "Pro",
            .long_pro => "Long Pro",
            .swap => "Swap",
            .swap2 => "Swap2",
        };
    }
};

pub const Outcome = enum {
    ongoing,
    five,
    capture,
    draw,
};

pub const Legality = enum {
    ok,
    occupied,
    off_board,
    double_three,
    opening_centre,
    opening_distance,
    game_over,
    awaiting_choice,

    pub fn message(self: Legality) []const u8 {
        return switch (self) {
            .ok => "ok",
            .occupied => "intersection occupied",
            .off_board => "off board",
            .double_three => "forbidden: double free-three",
            .opening_centre => "opening rule: first stone must be at the centre",
            .opening_distance => "opening rule: third stone too close to the centre",
            .game_over => "game is over",
            .awaiting_choice => "waiting for a color choice",
        };
    }
};

/// A decision the swap openings ask for before the next stone.
pub const Choice = enum {
    none,
    /// Swap: player 2 picks black or white after three stones.
    swap,
    /// Swap2: player 2 picks black, white, or to place two more stones.
    swap2_first,
    /// Swap2: after those two stones, player 1 picks the color.
    swap2_second,

    /// Which slot of maybe_choice_answers this decision fills.
    pub fn answerIndex(choice: Choice) usize {
        if (choice == .swap2_second) {
            return 1;
        }

        return 0;
    }
};

pub const ChoiceAnswer = enum {
    black,
    white,
    place_two,

    fn chosenColor(answer: ChoiceAnswer) Color {
        if (answer == .white) {
            return .white;
        }

        return .black;
    }
};

pub fn reset(self: *Self, rules: Board.Rules, opening: Opening) void {
    self.board.init(rules);
    self.opening = opening;
    self.pending_five_history[0] = Board.no_move;
    self.outcome = .ongoing;
    self.maybe_winner = null;
    self.redo_count = 0;
    self.player_of_color = .{ 0, 1 };
    self.is_choice_made = .{ false, false };
    self.chose_place_two = false;
    self.maybe_choice_answers = .{ null, null };
}

pub fn ply(self: *const Self) u16 {
    return self.board.ply;
}

pub fn sideToMove(self: *const Self) Color {
    return self.board.side_to_move;
}

/// The breakable five the side to move has to answer, or no_move.
pub fn pendingFive(self: *const Self) Move {
    return self.pending_five_history[self.board.ply];
}

pub fn isOver(self: *const Self) bool {
    return self.outcome != .ongoing;
}

pub fn pendingChoice(self: *const Self) Choice {
    const current_ply = self.board.ply;

    switch (self.opening) {
        .swap => {
            if (current_ply == 3 and !self.is_choice_made[0]) {
                return .swap;
            }
        },
        .swap2 => {
            if (current_ply == 3 and !self.is_choice_made[0]) {
                return .swap2_first;
            }

            if (current_ply == 5 and self.chose_place_two and !self.is_choice_made[1]) {
                return .swap2_second;
            }
        },
        else => {},
    }

    return .none;
}

/// The player who has to act now, whether by placing a stone or by choosing.
/// During the swap openings that is not simply the owner of the side to move.
pub fn currentPlayer(self: *const Self) PlayerIndex {
    const current_ply = self.board.ply;

    switch (self.opening) {
        .swap => {
            if (current_ply < 3) {
                return 0;
            }

            if (!self.is_choice_made[0]) {
                return 1;
            }
        },
        .swap2 => {
            if (current_ply < 3) {
                return 0;
            }

            if (!self.is_choice_made[0]) {
                return 1;
            }

            if (self.chose_place_two) {
                if (current_ply < 5) {
                    return 1;
                }

                if (!self.is_choice_made[1]) {
                    return 0;
                }
            }
        },
        else => {},
    }

    return self.player_of_color[self.board.side_to_move.index()];
}

/// Applies the current player's answer to the pending swap decision.
pub fn applyChoice(self: *Self, answer: ChoiceAnswer) void {
    const choice = self.pendingChoice();
    const chooser = self.currentPlayer();

    switch (choice) {
        .none => {
            return;
        },
        .swap, .swap2_first => {
            self.maybe_choice_answers[0] = answer;
            self.is_choice_made[0] = true;

            if (answer == .place_two and choice == .swap2_first) {
                self.chose_place_two = true;
                return;
            }
        },
        .swap2_second => {
            self.maybe_choice_answers[1] = answer;
            self.is_choice_made[1] = true;
        },
    }

    const chosen_color = answer.chosenColor();
    self.player_of_color[chosen_color.index()] = chooser;
    self.player_of_color[chosen_color.opponent().index()] = ~chooser;
}

/// How far from the centre the next stone must be under Pro (3) and Long Pro
/// (4); the rule only applies to black's second stone. The engine gets it as a
/// root restriction through Limits.root_min_centre_distance.
pub fn openingMinCentreDistance(self: *const Self) u8 {
    if (self.board.ply != 2) {
        return 0;
    }

    return switch (self.opening) {
        .pro => 3,
        .long_pro => 4,
        else => 0,
    };
}

pub fn legality(self: *const Self, move: Move) Legality {
    if (self.outcome != .ongoing) {
        return .game_over;
    }

    if (self.pendingChoice() != .none) {
        return .awaiting_choice;
    }

    if (move >= Board.padded_cell_count or !Board.isOnBoard(move)) {
        return .off_board;
    }

    if (!self.board.isEmpty(move)) {
        return .occupied;
    }

    switch (self.opening) {
        .pro, .long_pro => {
            if (self.board.ply == 0 and move != Board.moveAt(9, 9)) {
                return .opening_centre;
            }

            if (self.board.ply == 2) {
                const x = Board.columnOf(move);
                const y = Board.rowOf(move);
                const centre_distance = @max(@max(x, 9) - @min(x, 9), @max(y, 9) - @min(y, 9));

                if (centre_distance < self.openingMinCentreDistance()) {
                    return .opening_distance;
                }
            }
        },
        else => {},
    }

    if (self.board.isForbidden(move, self.board.side_to_move)) {
        return .double_three;
    }

    return .ok;
}

fn declareWinner(self: *Self, outcome: Outcome, winner: Color) void {
    self.outcome = outcome;
    self.maybe_winner = winner;
}

fn applyMove(self: *Self, move: Move) void {
    const board = &self.board;
    const mover = board.side_to_move;
    const opponent_pending_five = self.pending_five_history[board.ply];

    board.makeMove(move);
    self.pending_five_history[board.ply] = Board.no_move;

    if (board.captured_pairs[mover.index()] >= board.rules.capture_win_pairs) {
        self.declareWinner(.capture, mover);
        return;
    }

    // The opponent's breakable five survived our answer, so it wins.
    if (opponent_pending_five != Board.no_move and board.hasFiveAnywhere(mover.opponent().stoneCode())) {
        self.declareWinner(.five, mover.opponent());
        return;
    }

    if (board.fiveDirectionThrough(move) != null) {
        if (!board.isFiveBreakable(move)) {
            self.declareWinner(.five, mover);
            return;
        }

        self.pending_five_history[board.ply] = move;
    }

    if (board.stone_count >= Board.cell_count) {
        self.outcome = .draw;
    }
}

/// Plays `move` if it is legal, which also forgets any redo history.
pub fn play(self: *Self, move: Move) Legality {
    const move_legality = self.legality(move);

    if (move_legality != .ok) {
        return move_legality;
    }

    self.applyMove(move);
    self.redo_count = 0;
    return .ok;
}

pub fn undo(self: *Self) bool {
    if (self.board.ply == 0) {
        return false;
    }

    const move = self.board.lastMove();
    self.board.undoMove();

    if (self.redo_count < self.redo_moves.len) {
        self.redo_moves[self.redo_count] = move;
        self.redo_count += 1;
    }

    self.outcome = .ongoing;
    self.maybe_winner = null;

    // Swap decisions are tied to the ply they were taken at, so undoing past
    // that ply has to forget them too.
    const current_ply = self.board.ply;

    if (current_ply < 3) {
        self.is_choice_made = .{ false, false };
        self.chose_place_two = false;
        self.player_of_color = .{ 0, 1 };
    } else if (current_ply < 5 and self.opening == .swap2) {
        self.is_choice_made[1] = false;
    }

    return true;
}

pub fn redo(self: *Self) bool {
    if (self.redo_count == 0) {
        return false;
    }

    const move = self.redo_moves[self.redo_count - 1];

    if (self.legality(move) != .ok) {
        return false;
    }

    self.redo_count -= 1;
    self.applyMove(move);
    return true;
}

/// Writes the game as a save file: a header line with the rules and swap
/// answers, then one "x,y" line per move (0-based).
pub fn writeRecord(self: *const Self, writer: *std.Io.Writer) !void {
    try writer.print(
        "gomoku {s} captures={} double_three={}",
        .{ self.opening.name(), self.board.rules.captures, self.board.rules.double_three },
    );

    for (self.maybe_choice_answers, 1..) |maybe_answer, choice_number| {
        if (maybe_answer) |answer| {
            try writer.print(" choice{d}={s}", .{ choice_number, @tagName(answer) });
        }
    }

    try writer.writeAll("\n");

    for (self.board.playedMoves()) |move| {
        try writer.print("{d},{d}\n", .{ Board.columnOf(move), Board.rowOf(move) });
    }
}

/// Reads swap decision `answer_index` (0 first, 1 second) back from a save-file header.
pub fn headerChoice(header: []const u8, answer_index: usize) ?ChoiceAnswer {
    std.debug.assert(answer_index < 2);

    const keys = [2][]const u8{ " choice1=", " choice2=" };
    const key = keys[answer_index];
    const key_start = std.mem.indexOf(u8, header, key) orelse {
        return null;
    };
    const value = header[key_start + key.len ..];
    const value_end = std.mem.indexOfScalar(u8, value, ' ') orelse value.len;
    return std.meta.stringToEnum(ChoiceAnswer, value[0..value_end]);
}

/// Games embed a Board, which is too big for the test runner's stack.
fn testGame() *Self {
    const Static = struct {
        var game: Self = undefined;
    };
    return &Static.game;
}

fn playAll(game: *Self, coordinates: []const [2]usize) !void {
    for (coordinates) |point| {
        try std.testing.expectEqual(Legality.ok, game.play(Board.moveAt(point[0], point[1])));
    }
}

test "save header keeps swap decisions" {
    const game = testGame();
    game.reset(.standard, .swap2);
    try playAll(game, &.{ .{ 9, 9 }, .{ 10, 10 }, .{ 8, 10 } });
    game.applyChoice(.place_two);
    try playAll(game, &.{ .{ 11, 11 }, .{ 7, 7 } });
    game.applyChoice(.white);

    var buffer: [256]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try game.writeRecord(&writer);
    const header = std.mem.sliceTo(writer.buffered(), '\n');

    try std.testing.expectEqual(ChoiceAnswer.place_two, headerChoice(header, 0).?);
    try std.testing.expectEqual(ChoiceAnswer.white, headerChoice(header, 1).?);
}

test "five wins when unbreakable" {
    const game = testGame();
    game.reset(.standard, .standard);

    for (0..4) |x| {
        try playAll(game, &.{ .{ x, 0 }, .{ x, 10 } });
    }

    try playAll(game, &.{.{ 4, 0 }});
    try std.testing.expectEqual(Outcome.five, game.outcome);
    try std.testing.expectEqual(Color.black, game.maybe_winner.?);

    try std.testing.expect(game.undo());
    try std.testing.expectEqual(Outcome.ongoing, game.outcome);
}

test "pro opening restrictions" {
    const game = testGame();
    game.reset(.standard, .pro);

    try std.testing.expectEqual(Legality.opening_centre, game.play(Board.moveAt(0, 0)));
    try std.testing.expectEqual(Legality.ok, game.play(Board.moveAt(9, 9)));
    try std.testing.expectEqual(Legality.ok, game.play(Board.moveAt(10, 10)));
    try std.testing.expectEqual(Legality.opening_distance, game.play(Board.moveAt(11, 11)));
    try std.testing.expectEqual(Legality.ok, game.play(Board.moveAt(12, 9)));
}

test "swap2 roles" {
    const game = testGame();
    game.reset(.standard, .swap2);
    try playAll(game, &.{ .{ 9, 9 }, .{ 10, 9 }, .{ 9, 10 } });
    try std.testing.expectEqual(Choice.swap2_first, game.pendingChoice());
    try std.testing.expectEqual(@as(PlayerIndex, 1), game.currentPlayer());

    game.applyChoice(.place_two);
    try std.testing.expectEqual(@as(PlayerIndex, 1), game.currentPlayer());
    try playAll(game, &.{ .{ 10, 10 }, .{ 8, 8 } });
    try std.testing.expectEqual(Choice.swap2_second, game.pendingChoice());
    try std.testing.expectEqual(@as(PlayerIndex, 0), game.currentPlayer());

    // Player 1 takes white, and white is to move at ply 5.
    game.applyChoice(.white);
    try std.testing.expectEqual(@as(PlayerIndex, 0), game.currentPlayer());
}
