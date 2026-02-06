//! Comptime Environment
//!
//! Manages the evaluation context including:
//! - Lexical scopes with slot-based bindings
//! - Global constants
//! - Heap for aggregates (with COW)

const std = @import("std");
const value = @import("value.zig");
const heap_mod = @import("heap.zig");
const limits = @import("limits.zig");

const CtValue = value.CtValue;
const SlotId = value.SlotId;
const HeapId = value.HeapId;
const CtHeap = heap_mod.CtHeap;
const EvalConfig = limits.EvalConfig;
const EvalStats = limits.EvalStats;

/// Lexical scope
pub const Scope = struct {
    /// Bindings in this scope: name -> slot
    bindings: std.StringHashMap(SlotId),

    /// Whether this is a loop scope (for break/continue)
    is_loop: bool = false,

    pub fn init(allocator: std.mem.Allocator) Scope {
        return .{
            .bindings = std.StringHashMap(SlotId).init(allocator),
            .is_loop = false,
        };
    }

    pub fn deinit(self: *Scope) void {
        self.bindings.deinit();
    }
};

/// Comptime evaluation environment
pub const CtEnv = struct {
    allocator: std.mem.Allocator,

    /// Configuration for limits
    config: EvalConfig,

    /// Statistics tracked during evaluation
    stats: EvalStats,

    /// Lexical scopes (innermost last)
    scopes: std.ArrayList(Scope),

    /// Global constants: name -> slot
    globals: std.StringHashMap(SlotId),

    /// Slot storage (all values live here)
    slots: std.ArrayList(CtValue),

    /// Heap for aggregates (with COW support)
    heap: CtHeap,

    pub fn init(allocator: std.mem.Allocator, config: EvalConfig) CtEnv {
        var env = CtEnv{
            .allocator = allocator,
            .config = config,
            .stats = .{},
            .scopes = .empty,
            .globals = std.StringHashMap(SlotId).init(allocator),
            .slots = .empty,
            .heap = CtHeap.init(allocator),
        };

        // Start with a root scope
        env.pushScope(false) catch {};

        return env;
    }

    pub fn deinit(self: *CtEnv) void {
        for (self.scopes.items) |*scope| {
            scope.deinit();
        }
        self.scopes.deinit(self.allocator);
        self.globals.deinit();
        self.slots.deinit(self.allocator);
        self.heap.deinit();
    }

    // ========================================================================
    // Scope Management
    // ========================================================================

    /// Push a new scope
    pub fn pushScope(self: *CtEnv, is_loop: bool) !void {
        var scope = Scope.init(self.allocator);
        scope.is_loop = is_loop;
        try self.scopes.append(self.allocator, scope);
    }

    /// Pop the current scope
    pub fn popScope(self: *CtEnv) void {
        if (self.scopes.items.len > 1) { // Keep at least root scope
            var scope = self.scopes.pop() orelse return;
            scope.deinit();
        }
    }

    /// Get current scope depth
    pub fn scopeDepth(self: *const CtEnv) usize {
        return self.scopes.items.len;
    }

    /// Check if we're in a loop scope
    pub fn inLoop(self: *const CtEnv) bool {
        var i = self.scopes.items.len;
        while (i > 0) {
            i -= 1;
            if (self.scopes.items[i].is_loop) return true;
        }
        return false;
    }

    // ========================================================================
    // Binding Management
    // ========================================================================

    /// Bind a name in the current (top) scope
    pub fn bind(self: *CtEnv, name: []const u8, val: CtValue) !SlotId {
        const slot = try self.allocSlot(val);
        const top = &self.scopes.items[self.scopes.items.len - 1];
        try top.bindings.put(name, slot);
        return slot;
    }

    /// Define a global constant
    pub fn defineGlobal(self: *CtEnv, name: []const u8, val: CtValue) !SlotId {
        const slot = try self.allocSlot(val);
        try self.globals.put(name, slot);
        return slot;
    }

    /// Lookup a binding: scopes first (innermost to outermost), then globals
    pub fn lookup(self: *const CtEnv, name: []const u8) ?SlotId {
        // Check scopes (innermost first)
        var i = self.scopes.items.len;
        while (i > 0) {
            i -= 1;
            if (self.scopes.items[i].bindings.get(name)) |slot| {
                return slot;
            }
        }
        // Fallback to globals
        return self.globals.get(name);
    }

    /// Check if a name is bound in the current scope only
    pub fn isBoundInCurrentScope(self: *const CtEnv, name: []const u8) bool {
        if (self.scopes.items.len == 0) return false;
        const top = &self.scopes.items[self.scopes.items.len - 1];
        return top.bindings.contains(name);
    }

    // ========================================================================
    // Slot Management
    // ========================================================================

    /// Allocate a new slot with the given value
    fn allocSlot(self: *CtEnv, val: CtValue) !SlotId {
        const slot: SlotId = @intCast(self.slots.items.len);
        try self.slots.append(self.allocator, val);
        return slot;
    }

    /// Read a value from a slot
    pub fn read(self: *const CtEnv, slot: SlotId) CtValue {
        return self.slots.items[slot];
    }

    /// Write a value to a slot
    pub fn write(self: *CtEnv, slot: SlotId, val: CtValue) void {
        self.slots.items[slot] = val;
    }

    /// Update a slot with a new value, handling aggregate sharing
    pub fn update(self: *CtEnv, slot: SlotId, val: CtValue) void {
        const old = self.slots.items[slot];
        // If assigning a heap-backed value, mark it as shared
        if (val.isHeapBacked()) {
            if (val.getHeapId()) |heap_id| {
                self.heap.markShared(heap_id);
            }
        }
        // If the old value was heap-backed, it might now be orphaned
        // (but we don't GC during evaluation, so just overwrite)
        _ = old;
        self.slots.items[slot] = val;
    }

    // ========================================================================
    // Convenience Methods
    // ========================================================================

    /// Bind and return the value for chaining
    pub fn bindValue(self: *CtEnv, name: []const u8, val: CtValue) !CtValue {
        _ = try self.bind(name, val);
        return val;
    }

    /// Lookup and read a value by name
    pub fn lookupValue(self: *const CtEnv, name: []const u8) ?CtValue {
        if (self.lookup(name)) |slot| {
            return self.read(slot);
        }
        return null;
    }

    /// Get slot count
    pub fn slotCount(self: *const CtEnv) usize {
        return self.slots.items.len;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CtEnv basic binding" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    _ = try env.bind("x", .{ .integer = 42 });
    const val = env.lookupValue("x");
    try std.testing.expect(val != null);
    try std.testing.expectEqual(@as(u256, 42), val.?.integer);
}

test "CtEnv scope shadowing" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    _ = try env.bind("x", .{ .integer = 1 });

    try env.pushScope(false);
    _ = try env.bind("x", .{ .integer = 2 });

    // Inner scope sees shadowed value
    try std.testing.expectEqual(@as(u256, 2), env.lookupValue("x").?.integer);

    env.popScope();

    // Outer scope sees original value
    try std.testing.expectEqual(@as(u256, 1), env.lookupValue("x").?.integer);
}

test "CtEnv globals" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    _ = try env.defineGlobal("CONST", .{ .integer = 100 });

    try env.pushScope(false);
    // Global is visible from inner scope
    try std.testing.expectEqual(@as(u256, 100), env.lookupValue("CONST").?.integer);

    // Local shadows global
    _ = try env.bind("CONST", .{ .integer = 999 });
    try std.testing.expectEqual(@as(u256, 999), env.lookupValue("CONST").?.integer);

    env.popScope();

    // Global still has original value
    try std.testing.expectEqual(@as(u256, 100), env.lookupValue("CONST").?.integer);
}

test "CtEnv slot mutation" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    const slot = try env.bind("x", .{ .integer = 10 });
    try std.testing.expectEqual(@as(u256, 10), env.read(slot).integer);

    env.write(slot, .{ .integer = 20 });
    try std.testing.expectEqual(@as(u256, 20), env.read(slot).integer);
}

test "CtEnv loop detection" {
    var env = CtEnv.init(std.testing.allocator, .{});
    defer env.deinit();

    try std.testing.expect(!env.inLoop());

    try env.pushScope(true); // loop scope
    try std.testing.expect(env.inLoop());

    try env.pushScope(false); // nested non-loop
    try std.testing.expect(env.inLoop()); // still in a loop

    env.popScope();
    env.popScope();

    try std.testing.expect(!env.inLoop());
}
