// ============================================================================
// Ora AST Arena
//
// - All AST nodes allocated from this arena become invalid after `deinit` or reset.
// - Do NOT hold references to nodes beyond the arena's lifetime.
// - Not thread-safe; intended for single-threaded AST construction. For now is fine.
// - No use-after-reset protection unless debug checks are added elsewhere.
//
// ============================================================================
const std = @import("std");

/// Memory statistics for tracking allocations and usage
pub const MemoryStats = struct {
    /// Current memory usage in bytes
    current_usage: usize = 0,
    /// Peak memory usage in bytes
    peak_usage: usize = 0,
    /// Total number of allocations
    allocation_count: usize = 0,
    /// Total number of nodes created
    node_count: usize = 0,
    /// Number of resets performed
    reset_count: usize = 0,
};

/// Error type for AstArena operations
pub const AstArenaError = error{
    /// Allocation failed due to out of memory
    OutOfMemory,
    /// Node creation failed due to invalid type
    InvalidNodeType,
    /// Arena was reset while nodes are still in use
    ArenaResetWithActiveNodes,
};

/// Arena-based allocator for AST nodes with statistics tracking
pub const AstArena = struct {
    /// The underlying arena allocator
    arena: std.heap.ArenaAllocator,
    /// Memory statistics
    stats: MemoryStats,
    /// Debug mode flag
    debug_mode: bool,

    /// Initialize a new AstArena with the given backing allocator
    pub fn init(backing_allocator: std.mem.Allocator) AstArena {
        return AstArena{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .stats = MemoryStats{},
            .debug_mode = false,
        };
    }

    /// Initialize a new AstArena with debug mode enabled
    pub fn initDebug(backing_allocator: std.mem.Allocator) AstArena {
        var arena = AstArena.init(backing_allocator);
        arena.debug_mode = true;
        return arena;
    }

    /// Deinitialize the arena and free all allocated memory
    pub fn deinit(self: *AstArena) void {
        self.reset() catch {};
        self.arena.deinit();
    }

    /// Get the allocator interface for this arena
    pub fn allocator(self: *AstArena) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Create a new node of the given type
    /// The node is allocated in the arena and tracked in statistics
    pub fn createNode(self: *AstArena, comptime T: type) !*T {
        // Allocate memory for the node
        const node = try self.arena.allocator().create(T);

        // Update statistics
        self.stats.allocation_count += 1;
        self.stats.node_count += 1;
        self.stats.current_usage += @sizeOf(T);
        self.stats.peak_usage = @max(self.stats.peak_usage, self.stats.current_usage);

        return node;
    }

    /// Create a slice of items in the arena
    pub fn createSlice(self: *AstArena, comptime T: type, count: usize) ![]T {
        const slice = try self.arena.allocator().alloc(T, count);

        // Update statistics
        self.stats.allocation_count += 1;
        self.stats.current_usage += @sizeOf(T) * count;
        self.stats.peak_usage = @max(self.stats.peak_usage, self.stats.current_usage);

        return slice;
    }

    /// Create a duplicate of a string in the arena
    pub fn createString(self: *AstArena, string: []const u8) ![]const u8 {
        const dup = try self.arena.allocator().dupe(u8, string);

        // Update statistics
        self.stats.allocation_count += 1;
        self.stats.current_usage += string.len;
        self.stats.peak_usage = @max(self.stats.peak_usage, self.stats.current_usage);

        return dup;
    }

    /// Reset the arena, freeing all allocated memory
    /// This will invalidate all nodes created by this arena
    pub fn reset(self: *AstArena) !void {
        // In debug mode, we check if there are still active nodes
        if (self.debug_mode and self.stats.node_count > 0) {
            return AstArenaError.ArenaResetWithActiveNodes;
        }

        // Reset the arena
        self.arena.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.arena.child_allocator);

        // Update statistics
        self.stats.current_usage = 0;
        self.stats.node_count = 0;
        self.stats.reset_count += 1;
    }

    /// Get the current memory statistics
    pub fn getStats(self: *const AstArena) MemoryStats {
        return self.stats;
    }

    /// Enable or disable debug mode
    pub fn setDebugMode(self: *AstArena, enable: bool) void {
        self.debug_mode = enable;
    }
};
