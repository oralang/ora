const std = @import("std");
const Allocator = std.mem.Allocator;

/// RuneSet is a set of Unicode codepoints (runes)
/// Wraps a sorted slice of u21 codepoints for efficient membership testing
pub const RuneSet = struct {
    runes: []const u21, // Sorted, immutable, not owned by this struct

    /// Create a RuneSet from an array of integers
    /// Allocates new memory to store the runes
    /// Caller is responsible for freeing the memory using deinit()
    pub fn fromInts(allocator: Allocator, ints: []const i32) !RuneSet {
        const runes = try allocator.alloc(u21, ints.len);
        for (ints, 0..) |val, i| {
            runes[i] = @intCast(val);
        }
        return RuneSet{ .runes = runes };
    }

    /// Create a RuneSet from an existing slice (does not allocate)
    /// The slice must remain valid for the lifetime of the RuneSet
    pub fn fromSlice(runes: []const u21) RuneSet {
        return RuneSet{ .runes = runes };
    }

    /// Check if the set contains a codepoint using binary search
    /// O(log n) complexity
    pub fn contains(self: *const RuneSet, cp: u21) bool {
        return binarySearch(self.runes, cp);
    }

    /// Filter the set based on a predicate function
    /// Creates a new RuneSet with only elements matching the predicate
    /// Caller is responsible for freeing the returned RuneSet using deinit()
    pub fn filter(self: *const RuneSet, allocator: Allocator, predicate: *const fn (u21) bool) !RuneSet {
        var filtered: std.ArrayListUnmanaged(u21) = .{};
        defer filtered.deinit(allocator);

        for (self.runes) |r| {
            if (predicate(r)) {
                try filtered.append(allocator, r);
            }
        }

        return RuneSet{ .runes = try filtered.toOwnedSlice(allocator) };
    }

    /// Convert the RuneSet to an array (clones the internal slice)
    /// Caller is responsible for freeing the returned slice
    pub fn toArray(self: *const RuneSet, allocator: Allocator) ![]u21 {
        return try allocator.dupe(u21, self.runes);
    }

    /// Free the memory used by this RuneSet
    /// Only call this if the RuneSet was created with fromInts() or filter()
    pub fn deinit(self: *const RuneSet, allocator: Allocator) void {
        allocator.free(self.runes);
    }
};

/// Binary search for u21 in a sorted slice
/// Returns true if the key is found, false otherwise
fn binarySearch(items: []const u21, key: u21) bool {
    var left: usize = 0;
    var right: usize = items.len;

    while (left < right) {
        const mid = left + (right - left) / 2;
        if (items[mid] < key) {
            left = mid + 1;
        } else if (items[mid] > key) {
            right = mid;
        } else {
            return true;
        }
    }
    return false;
}
