//! Fixture Cache - Performance-optimized fixture loading and caching
//!
//! Provides a caching system for test fixtures to improve test performance
//! by avoiding repeated file I/O operations.

const std = @import("std");
const Allocator = std.mem.Allocator;
const TestFixture = @import("fixtures.zig").TestFixture;
const FixtureMetadata = @import("fixtures.zig").FixtureMetadata;
const FixtureCategory = @import("fixtures.zig").FixtureCategory;

/// Cached fixture entry with metadata
const CachedFixture = struct {
    fixture: TestFixture,
    last_accessed: i64,
    access_count: u32,

    pub fn touch(self: *CachedFixture) void {
        self.last_accessed = std.time.milliTimestamp();
        self.access_count += 1;
    }
};

/// Fixture cache configuration
pub const CacheConfig = struct {
    /// Maximum number of fixtures to cache
    max_cached_fixtures: u32 = 100,
    /// Maximum memory usage for cached fixtures (in bytes)
    max_memory_usage: u64 = 10 * 1024 * 1024, // 10MB
    /// Time after which unused fixtures are evicted (in milliseconds)
    eviction_timeout_ms: i64 = 5 * 60 * 1000, // 5 minutes
    /// Enable LRU eviction when cache is full
    enable_lru_eviction: bool = true,
};

/// High-performance fixture cache with LRU eviction
pub const FixtureCache = struct {
    allocator: Allocator,
    config: CacheConfig,
    cache: std.HashMap([]const u8, CachedFixture, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    current_memory_usage: u64,

    pub fn init(allocator: Allocator, config: CacheConfig) FixtureCache {
        return FixtureCache{
            .allocator = allocator,
            .config = config,
            .cache = std.HashMap([]const u8, CachedFixture, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .current_memory_usage = 0,
        };
    }

    pub fn deinit(self: *FixtureCache) void {
        // Free all cached fixtures
        var iterator = self.cache.iterator();
        while (iterator.next()) |entry| {
            var cached_fixture = entry.value_ptr;
            cached_fixture.fixture.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.cache.deinit();
    }

    /// Get fixture from cache or load if not cached
    pub fn getFixture(self: *FixtureCache, name: []const u8) !TestFixture {
        // Check if fixture is in cache
        if (self.cache.getPtr(name)) |cached_fixture| {
            cached_fixture.touch();
            return cached_fixture.fixture;
        }

        // Load fixture from source
        const fixture = try self.loadFixtureFromSource(name);

        // Add to cache if there's space
        try self.cacheFixture(name, fixture);

        return fixture;
    }

    /// Preload fixtures for a category
    pub fn preloadCategory(self: *FixtureCache, category: FixtureCategory) !void {
        const fixture_names = getCategoryFixtureNames(category);

        for (fixture_names) |name| {
            _ = try self.getFixture(name);
        }
    }

    /// Clear cache and free memory
    pub fn clear(self: *FixtureCache) void {
        var iterator = self.cache.iterator();
        while (iterator.next()) |entry| {
            var cached_fixture = entry.value_ptr;
            cached_fixture.fixture.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.cache.clearRetainingCapacity();
        self.current_memory_usage = 0;
    }

    /// Get cache statistics
    pub fn getStats(self: *FixtureCache) CacheStats {
        var total_access_count: u64 = 0;
        var oldest_access: i64 = std.time.milliTimestamp();
        var newest_access: i64 = 0;

        var iterator = self.cache.iterator();
        while (iterator.next()) |entry| {
            const cached_fixture = entry.value_ptr.*;
            total_access_count += cached_fixture.access_count;
            oldest_access = @min(oldest_access, cached_fixture.last_accessed);
            newest_access = @max(newest_access, cached_fixture.last_accessed);
        }

        return CacheStats{
            .cached_fixtures = @intCast(self.cache.count()),
            .memory_usage_bytes = self.current_memory_usage,
            .total_access_count = total_access_count,
            .oldest_access_time = oldest_access,
            .newest_access_time = newest_access,
        };
    }

    /// Evict old or unused fixtures
    pub fn evictStale(self: *FixtureCache) !void {
        const current_time = std.time.milliTimestamp();
        var to_evict = std.ArrayList([]const u8).init(self.allocator);
        defer to_evict.deinit();

        var iterator = self.cache.iterator();
        while (iterator.next()) |entry| {
            const cached_fixture = entry.value_ptr.*;
            const age = current_time - cached_fixture.last_accessed;

            if (age > self.config.eviction_timeout_ms) {
                try to_evict.append(entry.key_ptr.*);
            }
        }

        // Evict stale fixtures
        for (to_evict.items) |name| {
            try self.evictFixture(name);
        }
    }

    /// Cache a fixture with eviction if necessary
    fn cacheFixture(self: *FixtureCache, name: []const u8, fixture: TestFixture) !void {
        const fixture_size = estimateFixtureSize(fixture);

        // Check if we need to evict fixtures to make space
        if (self.cache.count() >= self.config.max_cached_fixtures or
            self.current_memory_usage + fixture_size > self.config.max_memory_usage)
        {
            try self.evictLeastRecentlyUsed();
        }

        // Create cached fixture entry
        const cached_fixture = CachedFixture{
            .fixture = fixture,
            .last_accessed = std.time.milliTimestamp(),
            .access_count = 1,
        };

        // Store in cache
        const name_copy = try self.allocator.dupe(u8, name);
        try self.cache.put(name_copy, cached_fixture);
        self.current_memory_usage += fixture_size;
    }

    /// Evict least recently used fixture
    fn evictLeastRecentlyUsed(self: *FixtureCache) !void {
        if (self.cache.count() == 0) return;

        var oldest_time: i64 = std.time.milliTimestamp();
        var oldest_name: ?[]const u8 = null;

        var iterator = self.cache.iterator();
        while (iterator.next()) |entry| {
            const cached_fixture = entry.value_ptr.*;
            if (cached_fixture.last_accessed < oldest_time) {
                oldest_time = cached_fixture.last_accessed;
                oldest_name = entry.key_ptr.*;
            }
        }

        if (oldest_name) |name| {
            try self.evictFixture(name);
        }
    }

    /// Evict specific fixture from cache
    fn evictFixture(self: *FixtureCache, name: []const u8) !void {
        if (self.cache.fetchRemove(name)) |entry| {
            var cached_fixture = entry.value;
            const fixture_size = estimateFixtureSize(cached_fixture.fixture);

            cached_fixture.fixture.deinit(self.allocator);
            self.allocator.free(entry.key);
            self.current_memory_usage -= fixture_size;
        }
    }

    /// Load fixture from source (file system or embedded data)
    fn loadFixtureFromSource(self: *FixtureCache, name: []const u8) !TestFixture {
        // Try to load from file system first
        const file_path = try std.fmt.allocPrint(self.allocator, "tests/fixtures/{s}.ora", .{name});
        defer self.allocator.free(file_path);

        const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
            return switch (err) {
                error.FileNotFound => error.FixtureNotFound,
                else => err,
            };
        };
        defer file.close();

        const content = try file.readToEndAlloc(self.allocator, 1024 * 1024); // 1MB limit

        return TestFixture{
            .metadata = FixtureMetadata{
                .name = try self.allocator.dupe(u8, name),
                .description = "Loaded from file",
                .category = .complex_programs,
                .tags = &.{},
                .expected_result = .success,
            },
            .content = content,
            .content_owned = true,
        };
    }
};

/// Cache statistics
pub const CacheStats = struct {
    cached_fixtures: u32,
    memory_usage_bytes: u64,
    total_access_count: u64,
    oldest_access_time: i64,
    newest_access_time: i64,

    pub fn getHitRate(self: CacheStats, total_requests: u64) f64 {
        if (total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_access_count)) / @as(f64, @floatFromInt(total_requests));
    }

    pub fn getMemoryUsageMB(self: CacheStats) f64 {
        return @as(f64, @floatFromInt(self.memory_usage_bytes)) / (1024.0 * 1024.0);
    }

    pub fn format(self: CacheStats, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        try writer.print("Cache Stats: {d} fixtures, {d:.2}MB memory, {d} total accesses", .{
            self.cached_fixtures,
            self.getMemoryUsageMB(),
            self.total_access_count,
        });
    }
};

/// Estimate memory usage of a fixture
fn estimateFixtureSize(fixture: TestFixture) u64 {
    var size: u64 = 0;

    // Content size
    size += fixture.content.len;

    // Metadata size (approximate)
    size += fixture.metadata.name.len;
    size += fixture.metadata.description.len;
    for (fixture.metadata.tags) |tag| {
        size += tag.len;
    }

    // Overhead for struct and pointers
    size += @sizeOf(TestFixture) + @sizeOf(FixtureMetadata);

    return size;
}

/// Get fixture names for a category
fn getCategoryFixtureNames(category: FixtureCategory) []const []const u8 {
    return switch (category) {
        .valid_tokens => &.{ "identifiers", "numbers", "strings", "operators", "keywords" },
        .error_cases => &.{ "unterminated_string", "invalid_hex", "invalid_binary", "unexpected_character" },
        .complex_programs => &.{ "simple_token", "nested_expressions", "error_handling" },
        .expressions => &.{ "binary_expressions", "function_calls", "field_access" },
        .statements => &.{ "variable_declarations", "control_flow" },
        .declarations => &.{ "function_declarations", "struct_declarations" },
        .contracts => &.{ "basic_contract", "contract_with_events" },
    };
}

// Tests
test "FixtureCache basic functionality" {
    var cache = FixtureCache.init(std.testing.allocator, CacheConfig{});
    defer cache.deinit();

    // This test would need actual fixture files to work properly
    // For now, just test the cache structure
    const stats = cache.getStats();
    try std.testing.expect(stats.cached_fixtures == 0);
    try std.testing.expect(stats.memory_usage_bytes == 0);
}

test "CacheStats calculations" {
    const stats = CacheStats{
        .cached_fixtures = 5,
        .memory_usage_bytes = 1024 * 1024, // 1MB
        .total_access_count = 100,
        .oldest_access_time = 0,
        .newest_access_time = 1000,
    };

    try std.testing.expect(stats.getMemoryUsageMB() == 1.0);
    try std.testing.expect(stats.getHitRate(200) == 0.5);
}
