//! Test Arena - Simple test memory management
//!
//! Provides basic arena allocation for tests. For AST testing, use the
//! existing AstArena from src/ast/ast_arena.zig instead.

const std = @import("std");
const Allocator = std.mem.Allocator;

// For AST tests, import AstArena directly from the ora module in your tests

/// Simple test arena - just a wrapper around std.heap.ArenaAllocator
pub const TestArena = struct {
    arena: std.heap.ArenaAllocator,

    /// Initialize a new test arena
    pub fn init(backing_allocator: Allocator, _: bool) TestArena {
        return TestArena{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
        };
    }

    /// Deinitialize the test arena
    pub fn deinit(self: *TestArena) void {
        self.arena.deinit();
    }

    /// Get the arena allocator
    pub fn allocator(self: *TestArena) Allocator {
        return self.arena.allocator();
    }

    /// Reset the arena, keeping capacity for reuse
    pub fn reset(self: *TestArena) void {
        _ = self.arena.reset(.retain_capacity);
    }

    /// Get basic memory statistics (simplified)
    pub fn getMemoryStats(self: *TestArena) BasicMemoryStats {
        _ = self;
        return BasicMemoryStats{
            .current_allocated = 0, // Arena doesn't track this easily
        };
    }
};

/// Basic memory statistics for simple test arena
pub const BasicMemoryStats = struct {
    current_allocated: u64,
};

/// Memory leak information (simplified)
pub const MemoryLeak = struct {
    address: usize,
    size: usize,
    timestamp: i64,
};

// Tests
test "TestArena basic functionality" {
    var arena = TestArena.init(std.testing.allocator, false);
    defer arena.deinit();

    const allocator = arena.allocator();
    const memory = try allocator.alloc(u8, 100);
    _ = memory;

    arena.reset();
}

// AstArena tests should be in files that import the ora module
