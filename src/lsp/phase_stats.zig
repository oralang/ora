const std = @import("std");

pub const Phase = enum {
    lex,
    parse,
    ast_lower,
    item_index,
    resolve,
    const_eval,
    type_check,
    formatter,
};

pub const Stats = struct {
    lex_builds: usize = 0,
    parse_builds: usize = 0,
    ast_lower_builds: usize = 0,
    item_index_builds: usize = 0,
    resolve_builds: usize = 0,
    const_eval_builds: usize = 0,
    type_check_builds: usize = 0,
    formatter_builds: usize = 0,

    pub fn record(self: *Stats, phase: Phase) void {
        switch (phase) {
            .lex => increment(&self.lex_builds),
            .parse => increment(&self.parse_builds),
            .ast_lower => increment(&self.ast_lower_builds),
            .item_index => increment(&self.item_index_builds),
            .resolve => increment(&self.resolve_builds),
            .const_eval => increment(&self.const_eval_builds),
            .type_check => increment(&self.type_check_builds),
            .formatter => increment(&self.formatter_builds),
        }
    }
};

pub fn record(stats: ?*Stats, phase: Phase) void {
    if (stats) |s| s.record(phase);
}

fn increment(value: *usize) void {
    value.* = if (value.* == std.math.maxInt(usize)) value.* else value.* + 1;
}
