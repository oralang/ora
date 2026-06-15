pub const DiagnosticSource = enum {
    lexer,
    parser,
    sema,
};

pub const Severity = enum {
    err,
    warning,
    information,
    hint,
};

pub const Position = struct {
    line: u32,
    character: u32,
};

pub const Range = struct {
    start: Position,
    end: Position,
};
