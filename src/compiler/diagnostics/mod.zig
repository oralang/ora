const std = @import("std");
const source = @import("../source/mod.zig");

pub const Severity = enum {
    Error,
    Warning,
    Note,
    Help,
};

pub const Label = struct {
    location: source.SourceLocation,
    message: []const u8,
};

pub const Diagnostic = struct {
    severity: Severity,
    message: []const u8,
    code: ?[]const u8,
    labels: []Label,
};

pub const DiagnosticList = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayList(Diagnostic),

    pub fn init(allocator: std.mem.Allocator) DiagnosticList {
        return .{
            .allocator = allocator,
            .items = .{},
        };
    }

    pub fn deinit(self: *DiagnosticList) void {
        for (self.items.items) |diag| {
            self.allocator.free(diag.message);
            if (diag.code) |code| self.allocator.free(code);
            for (diag.labels) |label| {
                self.allocator.free(label.message);
            }
            self.allocator.free(diag.labels);
        }
        self.items.deinit(self.allocator);
    }

    pub fn clear(self: *DiagnosticList) void {
        self.deinit();
        self.* = DiagnosticList.init(self.allocator);
    }

    pub fn append(self: *DiagnosticList, severity: Severity, message: []const u8, location: source.SourceLocation) !void {
        try self.appendWithCode(severity, null, message, &[_]Label{.{
            .location = location,
            .message = "here",
        }});
    }

    pub fn appendWithCode(self: *DiagnosticList, severity: Severity, code: ?[]const u8, message: []const u8, labels: []const Label) !void {
        const owned_message = try self.allocator.dupe(u8, message);
        errdefer self.allocator.free(owned_message);

        const owned_code = if (code) |value| try self.allocator.dupe(u8, value) else null;
        errdefer if (owned_code) |value| self.allocator.free(value);

        const owned_labels = try self.allocator.alloc(Label, labels.len);
        errdefer self.allocator.free(owned_labels);

        for (labels, 0..) |label, index| {
            owned_labels[index] = .{
                .location = label.location,
                .message = try self.allocator.dupe(u8, label.message),
            };
        }
        errdefer {
            for (owned_labels) |label| self.allocator.free(label.message);
        }

        try self.items.append(self.allocator, .{
            .severity = severity,
            .message = owned_message,
            .code = owned_code,
            .labels = owned_labels,
        });
    }

    pub fn appendError(self: *DiagnosticList, message: []const u8, location: source.SourceLocation) !void {
        try self.append(.Error, message, location);
    }

    pub fn appendNote(self: *DiagnosticList, message: []const u8, location: source.SourceLocation) !void {
        try self.append(.Note, message, location);
    }

    pub fn isEmpty(self: *const DiagnosticList) bool {
        return self.items.items.len == 0;
    }

    pub fn len(self: *const DiagnosticList) usize {
        return self.items.items.len;
    }
};
