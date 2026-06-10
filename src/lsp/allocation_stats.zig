const std = @import("std");

const Allocator = std.mem.Allocator;

pub const Scope = enum {
    unscoped,
    request_protocol,
    response,
    cache_build,
    temp_analysis,
};

const scope_count = std.meta.fields(Scope).len;

pub const ScopedStats = struct {
    alloc_calls: usize = 0,
    resize_calls: usize = 0,
    remap_calls: usize = 0,
    free_calls: usize = 0,
    bytes_allocated: usize = 0,
    bytes_freed: usize = 0,
};

pub const Stats = struct {
    alloc_calls: usize = 0,
    resize_calls: usize = 0,
    remap_calls: usize = 0,
    free_calls: usize = 0,
    bytes_allocated: usize = 0,
    bytes_freed: usize = 0,
    bytes_live: usize = 0,
    bytes_peak: usize = 0,
    scopes: [scope_count]ScopedStats = [_]ScopedStats{.{}} ** scope_count,

    pub fn scope(self: *const Stats, allocation_scope: Scope) ScopedStats {
        return self.scopes[@intFromEnum(allocation_scope)];
    }
};

pub const CountingAllocator = struct {
    backing: Allocator,
    stats: Stats = .{},
    current_scope: Scope = .unscoped,

    pub fn init(backing: Allocator) CountingAllocator {
        return .{ .backing = backing };
    }

    pub fn allocator(self: *CountingAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    pub fn setScope(self: *CountingAllocator, scope: Scope) void {
        self.current_scope = scope;
    }

    pub fn beginScope(self: *CountingAllocator, scope: Scope) ScopeGuard {
        const previous = self.current_scope;
        self.current_scope = scope;
        return .{ .allocator = self, .previous = previous };
    }

    const vtable: Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.stats.alloc_calls += 1;
        self.scopeStats().alloc_calls += 1;
        self.addLiveBytes(len);
        return result;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        if (!self.backing.rawResize(memory, alignment, new_len, ret_addr)) return false;
        self.stats.resize_calls += 1;
        self.scopeStats().resize_calls += 1;
        self.adjustLiveForResize(memory.len, new_len);
        return true;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing.rawRemap(memory, alignment, new_len, ret_addr) orelse return null;
        self.stats.remap_calls += 1;
        self.scopeStats().remap_calls += 1;
        self.adjustLiveForResize(memory.len, new_len);
        return result;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.stats.free_calls += 1;
        self.scopeStats().free_calls += 1;
        self.stats.bytes_freed = addSat(self.stats.bytes_freed, memory.len);
        self.scopeStats().bytes_freed = addSat(self.scopeStats().bytes_freed, memory.len);
        self.stats.bytes_live = if (memory.len > self.stats.bytes_live) 0 else self.stats.bytes_live - memory.len;
        self.backing.rawFree(memory, alignment, ret_addr);
    }

    fn adjustLiveForResize(self: *CountingAllocator, old_len: usize, new_len: usize) void {
        if (new_len >= old_len) {
            self.addLiveBytes(new_len - old_len);
        } else {
            const delta = old_len - new_len;
            self.stats.bytes_freed = addSat(self.stats.bytes_freed, delta);
            self.scopeStats().bytes_freed = addSat(self.scopeStats().bytes_freed, delta);
            self.stats.bytes_live = if (delta > self.stats.bytes_live) 0 else self.stats.bytes_live - delta;
        }
    }

    fn addLiveBytes(self: *CountingAllocator, bytes: usize) void {
        self.stats.bytes_allocated = addSat(self.stats.bytes_allocated, bytes);
        self.scopeStats().bytes_allocated = addSat(self.scopeStats().bytes_allocated, bytes);
        self.stats.bytes_live = addSat(self.stats.bytes_live, bytes);
        self.stats.bytes_peak = @max(self.stats.bytes_peak, self.stats.bytes_live);
    }

    fn scopeStats(self: *CountingAllocator) *ScopedStats {
        return &self.stats.scopes[@intFromEnum(self.current_scope)];
    }
};

pub const ScopeGuard = struct {
    allocator: *CountingAllocator,
    previous: Scope,

    pub fn deinit(self: ScopeGuard) void {
        self.allocator.current_scope = self.previous;
    }
};

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
