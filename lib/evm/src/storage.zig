/// Storage management for the EVM
/// Handles persistent storage, transient storage (EIP-1153), and original storage tracking
const std = @import("std");
const primitives = @import("voltaire");
const host = @import("host.zig");
const errors = @import("errors.zig");
const log = @import("logger.zig");

/// Re-export StorageKey from primitives for convenience.
pub const StorageKey = primitives.State.StorageKey;
/// Backwards compatibility alias for StorageKey (prefer StorageKey for new code).
pub const StorageSlotKey = StorageKey;

/// Storage manager - handles all storage operations for the EVM
pub const Storage = struct {
    /// Persistent storage (current transaction state)
    storage: std.AutoHashMap(StorageSlotKey, u256),
    /// Original storage values (snapshot at transaction start)
    original_storage: std.AutoHashMap(StorageSlotKey, u256),
    /// Transient storage (EIP-1153, cleared at transaction boundaries)
    transient: std.AutoHashMap(StorageSlotKey, u256),
    /// Host interface (optional, for external state backends)
    host: ?host.HostInterface,
    /// Arena allocator for transaction-scoped memory
    allocator: std.mem.Allocator,

    /// Initialize storage manager
    ///
    /// Creates a new storage manager for handling persistent and transient storage operations.
    ///
    /// Parameters:
    ///   - allocator: Arena allocator for transaction-scoped memory
    ///   - h: Optional host interface for external state backend
    ///   - injector: Optional storage injector for async data fetching
    ///
    /// Returns: Initialized Storage instance
    pub fn init(allocator: std.mem.Allocator, h: ?host.HostInterface) Storage {
        return Storage{
            .storage = std.AutoHashMap(StorageSlotKey, u256).init(allocator),
            .original_storage = std.AutoHashMap(StorageSlotKey, u256).init(allocator),
            .transient = std.AutoHashMap(StorageSlotKey, u256).init(allocator),
            .host = h,
            .allocator = allocator,
        };
    }

    /// Clean up storage maps when using a non-arena allocator.
    pub fn deinit(self: *Storage) void {
        self.storage.deinit();
        self.original_storage.deinit();
        self.transient.deinit();
    }

    /// Get storage value
    ///
    /// Retrieves the current value from a contract's persistent storage slot.
    ///
    /// Behavior depends on configuration:
    ///   - With host interface: Delegates to host.getStorage()
    ///   - Without host: Uses internal HashMap, returns 0 for empty slots
    ///
    /// Parameters:
    ///   - address: Contract address
    ///   - slot: Storage slot (u256)
    ///
    /// Returns: Current storage value (0 if slot is empty)
    ///
    pub fn get(self: *Storage, address: primitives.Address, slot: u256) !u256 {
        const key = StorageSlotKey{ .address = address.bytes, .slot = slot };

        if (self.host) |h| {
            return h.getStorage(address, slot);
        }
        return self.storage.get(key) orelse 0;
    }

    /// Set storage value
    ///
    /// Sets a contract's persistent storage slot to a new value. Automatically tracks
    /// the original value (before transaction modifications) for SSTORE gas calculations.
    ///
    /// EVM semantics: Setting a slot to 0 deletes it (per EVM specification).
    ///
    /// Parameters:
    ///   - address: Contract address
    ///   - slot: Storage slot (u256)
    ///   - value: New value to set (0 = delete slot)
    ///
    /// Returns: void
    ///
    /// Errors:
    ///   - OutOfMemory: If original_storage tracking allocation fails
    pub fn set(self: *Storage, address: primitives.Address, slot: u256, value: u256) !void {
        const key = StorageSlotKey{ .address = address.bytes, .slot = slot };

        // Track original value on first write in transaction
        if (!self.original_storage.contains(key)) {
            const current = if (self.host) |h|
                h.getStorage(address, slot)
            else
                self.storage.get(key) orelse 0;
            try self.original_storage.put(key, current);
        }

        if (self.host) |h| {
            h.setStorage(address, slot, value);
            return;
        }

        // EVM spec: storage slots with value 0 should be deleted, not stored
        if (value == 0) {
            _ = self.storage.remove(key);
        } else {
            try self.storage.put(key, value);
        }
    }

    /// Seed persistent storage without recording transaction-local original values.
    ///
    /// This is intended for test and pre-state setup, where we need the current cache
    /// to reflect an initial world state before any SSTORE tracking begins.
    pub fn put_in_cache(self: *Storage, address: primitives.Address, slot: u256, value: u256) !void {
        const key = StorageSlotKey{ .address = address.bytes, .slot = slot };

        if (self.host) |h| {
            h.setStorage(address, slot, value);
        }

        if (value == 0) {
            _ = self.storage.remove(key);
        } else {
            try self.storage.put(key, value);
        }
    }

    /// Get original storage value (before transaction modifications)
    ///
    /// Returns the storage value as it existed at the start of the transaction,
    /// before any SSTORE modifications. Used for SSTORE gas calculation (EIP-2200, EIP-2929).
    ///
    /// Parameters:
    ///   - address: Contract address
    ///   - slot: Storage slot (u256)
    ///
    /// Returns: Original storage value (0 if slot was empty at transaction start)
    pub fn get_original(self: *Storage, address: primitives.Address, slot: u256) u256 {
        const key = StorageSlotKey{ .address = address.bytes, .slot = slot };
        // If we have tracked the original, return it
        if (self.original_storage.get(key)) |original| {
            return original;
        }
        // Otherwise return current value (unchanged in this transaction)
        // Use host if available
        if (self.host) |h| {
            return h.getStorage(address, slot);
        }
        return self.storage.get(key) orelse 0;
    }

    /// Get transient storage value (EIP-1153)
    ///
    /// Retrieves a value from transient storage (Cancun+ hardfork, EIP-1153).
    /// Transient storage is cleared at transaction boundaries and is cheaper than persistent storage.
    ///
    /// Parameters:
    ///   - address: Contract address
    ///   - slot: Transient storage slot (u256)
    ///
    /// Returns: Current transient value (0 if slot is empty)
    pub fn get_transient(self: *Storage, address: primitives.Address, slot: u256) u256 {
        const key = StorageSlotKey{ .address = address.bytes, .slot = slot };
        return self.transient.get(key) orelse 0;
    }

    /// Set transient storage value (EIP-1153)
    ///
    /// Sets a transient storage slot (Cancun+ hardfork, EIP-1153). Setting to 0 deletes the slot.
    /// Transient storage is cleared automatically at transaction boundaries.
    ///
    /// Parameters:
    ///   - address: Contract address
    ///   - slot: Transient storage slot (u256)
    ///   - value: New value (0 = delete slot)
    ///
    /// Returns: void
    ///
    /// Errors:
    ///   - OutOfMemory: If HashMap allocation fails
    pub fn set_transient(self: *Storage, address: primitives.Address, slot: u256, value: u256) !void {
        const key = StorageSlotKey{ .address = address.bytes, .slot = slot };
        if (value == 0) {
            _ = self.transient.remove(key);
        } else {
            try self.transient.put(key, value);
        }
    }

    /// Clear transient storage (called at transaction boundaries)
    ///
    /// Clears all transient storage (EIP-1153). Must be called at the end of each transaction.
    /// Transient storage does NOT persist across transaction boundaries.
    ///
    /// Parameters: self
    ///
    /// Returns: void
    pub fn clear_transient(self: *Storage) void {
        self.transient.clearRetainingCapacity();
    }
};

// Tests
test "Storage: init creates empty storage" {
    const allocator = std.testing.allocator;
    var storage = Storage.init(allocator, null);
    defer storage.deinit();

    try std.testing.expectEqual(@as(usize, 0), storage.storage.count());
    try std.testing.expectEqual(@as(usize, 0), storage.original_storage.count());
    try std.testing.expectEqual(@as(usize, 0), storage.transient.count());
}

test "Storage: get returns 0 for empty slot" {
    const allocator = std.testing.allocator;
    var storage = Storage.init(allocator, null);
    defer storage.deinit();

    const addr = primitives.Address{ .bytes = [_]u8{0xab} ** 20 };
    const value = try storage.get(addr, 123);
    try std.testing.expectEqual(@as(u256, 0), value);
}

test "Storage: set and get persistent storage" {
    const allocator = std.testing.allocator;
    var storage = Storage.init(allocator, null);
    defer storage.deinit();

    const addr = primitives.Address{ .bytes = [_]u8{0xab} ** 20 };
    const slot: u256 = 42;
    const value: u256 = 0x1234567890abcdef;

    // Set value
    try storage.set(addr, slot, value);

    // Get value back
    const retrieved = try storage.get(addr, slot);
    try std.testing.expectEqual(value, retrieved);

    // Verify original storage was tracked
    const original = storage.get_original(addr, slot);
    try std.testing.expectEqual(@as(u256, 0), original);
}

test "Storage: set to zero deletes slot" {
    const allocator = std.testing.allocator;
    var storage = Storage.init(allocator, null);
    defer storage.deinit();

    const addr = primitives.Address{ .bytes = [_]u8{0xab} ** 20 };
    const slot: u256 = 42;

    // Set non-zero value
    try storage.set(addr, slot, 999);
    try std.testing.expectEqual(@as(u256, 999), try storage.get(addr, slot));

    // Set to zero (should delete)
    try storage.set(addr, slot, 0);
    try std.testing.expectEqual(@as(u256, 0), try storage.get(addr, slot));
}

test "Storage: get_original returns current for unmodified slots" {
    const allocator = std.testing.allocator;
    var storage = Storage.init(allocator, null);
    defer storage.deinit();

    const addr = primitives.Address{ .bytes = [_]u8{0xab} ** 20 };
    const slot: u256 = 42;

    // Set initial value
    try storage.set(addr, slot, 100);

    // First write tracks original as 0
    const original = storage.get_original(addr, slot);
    try std.testing.expectEqual(@as(u256, 0), original);

    // Modify again
    try storage.set(addr, slot, 200);

    // Original should still be 0 (first value before transaction)
    try std.testing.expectEqual(@as(u256, 0), storage.get_original(addr, slot));
}

test "Storage: transient storage get/set" {
    const allocator = std.testing.allocator;
    var storage = Storage.init(allocator, null);
    defer storage.deinit();

    const addr = primitives.Address{ .bytes = [_]u8{0xcd} ** 20 };
    const slot: u256 = 789;
    const value: u256 = 0xdeadbeef;

    // Get empty transient slot
    try std.testing.expectEqual(@as(u256, 0), storage.get_transient(addr, slot));

    // Set transient value
    try storage.set_transient(addr, slot, value);

    // Get transient value back
    try std.testing.expectEqual(value, storage.get_transient(addr, slot));

    // Verify it doesn't affect persistent storage
    try std.testing.expectEqual(@as(u256, 0), try storage.get(addr, slot));
}

test "Storage: transient storage set to zero deletes slot" {
    const allocator = std.testing.allocator;
    var storage = Storage.init(allocator, null);
    defer storage.deinit();

    const addr = primitives.Address{ .bytes = [_]u8{0xef} ** 20 };
    const slot: u256 = 100;

    // Set transient value
    try storage.set_transient(addr, slot, 12345);
    try std.testing.expectEqual(@as(u256, 12345), storage.get_transient(addr, slot));

    // Set to zero (should delete)
    try storage.set_transient(addr, slot, 0);
    try std.testing.expectEqual(@as(u256, 0), storage.get_transient(addr, slot));
}

test "Storage: clear_transient removes all transient storage" {
    const allocator = std.testing.allocator;
    var storage = Storage.init(allocator, null);
    defer storage.deinit();

    const addr1 = primitives.Address{ .bytes = [_]u8{0x11} ** 20 };
    const addr2 = primitives.Address{ .bytes = [_]u8{0x22} ** 20 };

    // Set multiple transient values
    try storage.set_transient(addr1, 1, 100);
    try storage.set_transient(addr1, 2, 200);
    try storage.set_transient(addr2, 1, 300);

    // Verify they exist
    try std.testing.expectEqual(@as(u256, 100), storage.get_transient(addr1, 1));
    try std.testing.expectEqual(@as(u256, 200), storage.get_transient(addr1, 2));
    try std.testing.expectEqual(@as(u256, 300), storage.get_transient(addr2, 1));

    // Clear transient storage
    storage.clear_transient();

    // Verify all are gone
    try std.testing.expectEqual(@as(u256, 0), storage.get_transient(addr1, 1));
    try std.testing.expectEqual(@as(u256, 0), storage.get_transient(addr1, 2));
    try std.testing.expectEqual(@as(u256, 0), storage.get_transient(addr2, 1));
}

test "Storage: multiple addresses with same slot" {
    const allocator = std.testing.allocator;
    var storage = Storage.init(allocator, null);
    defer storage.deinit();

    const addr1 = primitives.Address{ .bytes = [_]u8{0xaa} ** 20 };
    const addr2 = primitives.Address{ .bytes = [_]u8{0xbb} ** 20 };
    const slot: u256 = 1;

    // Set different values for same slot on different addresses
    try storage.set(addr1, slot, 111);
    try storage.set(addr2, slot, 222);

    // Verify they're independent
    try std.testing.expectEqual(@as(u256, 111), try storage.get(addr1, slot));
    try std.testing.expectEqual(@as(u256, 222), try storage.get(addr2, slot));
}
