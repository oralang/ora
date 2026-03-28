/// EIP-2929 Warm/Cold Access Tracking
///
/// This module manages the warm/cold access state for addresses and storage slots
/// according to EIP-2929. It's EVM-specific logic, not a primitive type.
const std = @import("std");
const primitives = @import("voltaire");
const Address = primitives.Address.Address;
const StorageKey = primitives.State.StorageKey;
const AccessList = primitives.AccessList.AccessList;
const gas_constants = primitives.GasConstants;
const Allocator = std.mem.Allocator;

/// Context for hashing StorageKey in hash maps
const StorageKeyContext = struct {
    pub fn hash(_: @This(), key: StorageKey) u32 {
        var hasher = std.hash.Wyhash.init(0);
        key.hash(&hasher);
        return @truncate(hasher.final());
    }
    pub fn eql(_: @This(), a: StorageKey, b: StorageKey, _: usize) bool {
        return StorageKey.eql(a, b);
    }
};

/// Manages warm/cold access state for EIP-2929
pub const AccessListManager = struct {
    allocator: Allocator,
    // Address uses AutoHashMap (20-byte array, auto-hash is sufficient).
    warm_addresses: std.AutoHashMap(Address, void),
    // StorageKey uses ArrayHashMap with a custom context (address + slot).
    warm_storage_slots: std.ArrayHashMap(StorageKey, void, StorageKeyContext, false),

    /// Initialize empty access list manager
    pub fn init(allocator: Allocator) AccessListManager {
        return .{
            .allocator = allocator,
            .warm_addresses = std.AutoHashMap(Address, void).init(allocator),
            .warm_storage_slots = std.ArrayHashMap(StorageKey, void, StorageKeyContext, false).init(allocator),
        };
    }

    /// Clean up resources
    pub fn deinit(self: *AccessListManager) void {
        self.warm_addresses.deinit();
        self.warm_storage_slots.deinit();
    }

    /// Access an address and return gas cost (warm=100, cold=2600)
    /// EIP-2929: First access is cold, subsequent accesses are warm
    pub fn access_address(self: *AccessListManager, addr: Address) !u64 {
        const entry = try self.warm_addresses.getOrPut(addr);
        return if (entry.found_existing)
            gas_constants.WarmStorageReadCost
        else
            gas_constants.ColdAccountAccessCost;
    }

    /// Access a storage slot and return gas cost (warm=100, cold=2100)
    /// EIP-2929: First access is cold, subsequent accesses are warm
    pub fn access_storage_slot(self: *AccessListManager, addr: Address, slot: u256) !u64 {
        const key = StorageKey{ .address = addr.bytes, .slot = slot };
        const entry = try self.warm_storage_slots.getOrPut(key);
        return if (entry.found_existing)
            gas_constants.WarmStorageReadCost
        else
            gas_constants.ColdSloadCost;
    }

    /// Pre-warm multiple addresses (marks them as already accessed)
    pub fn pre_warm_addresses(self: *AccessListManager, addresses: []const Address) !void {
        for (addresses) |addr| {
            _ = try self.warm_addresses.getOrPut(addr);
        }
    }

    /// Pre-warm multiple storage slots (marks them as already accessed)
    pub fn pre_warm_storage_slots(self: *AccessListManager, slots: []const StorageKey) !void {
        for (slots) |slot| {
            _ = try self.warm_storage_slots.getOrPut(slot);
        }
    }

    /// Pre-warm from EIP-2930 access list
    pub fn pre_warm_from_access_list(self: *AccessListManager, access_list: AccessList) !void {
        for (access_list) |entry| {
            // Pre-warm address
            _ = try self.warm_addresses.getOrPut(entry.address);

            // Pre-warm storage keys (convert Hash to u256)
            for (entry.storage_keys) |key_hash| {
                const slot = std.mem.readInt(u256, &key_hash, .big);
                const key = StorageKey{ .address = entry.address.bytes, .slot = slot };
                _ = try self.warm_storage_slots.getOrPut(key);
            }
        }
    }

    /// Compatibility wrapper for camelCase API expected by spec runner
    /// Delegates to snake_case implementation to avoid duplicate logic
    pub fn preWarmFromAccessList(self: *AccessListManager, access_list: AccessList) !void {
        return self.pre_warm_from_access_list(access_list);
    }
    /// Check if address is warm
    pub fn is_address_warm(self: *const AccessListManager, addr: Address) bool {
        return self.warm_addresses.contains(addr);
    }

    /// Check if storage slot is warm
    pub fn is_storage_slot_warm(self: *const AccessListManager, addr: Address, slot: u256) bool {
        const key = StorageKey{ .address = addr.bytes, .slot = slot };
        return self.warm_storage_slots.contains(key);
    }

    /// Clear all warm sets (call at transaction boundaries).
    /// Warm state does not persist across transactions per EIP-2929.
    pub fn clear(self: *AccessListManager) void {
        self.warm_addresses.clearRetainingCapacity();
        self.warm_storage_slots.clearRetainingCapacity();
    }

    /// Create snapshot for nested call revert handling.
    /// On successful call completion, discard the snapshot without restoring.
    pub fn snapshot(self: *const AccessListManager) !AccessListSnapshot {
        var addr_snapshot = std.AutoHashMap(Address, void).init(self.allocator);
        errdefer addr_snapshot.deinit();
        var addr_it = self.warm_addresses.iterator();
        while (addr_it.next()) |entry| {
            try addr_snapshot.put(entry.key_ptr.*, {});
        }

        var slot_snapshot = std.ArrayHashMap(StorageKey, void, StorageKeyContext, false).init(self.allocator);
        errdefer slot_snapshot.deinit();
        var slot_it = self.warm_storage_slots.iterator();
        while (slot_it.next()) |entry| {
            _ = try slot_snapshot.put(entry.key_ptr.*, {});
        }

        return .{
            .addresses = addr_snapshot,
            .slots = slot_snapshot,
        };
    }

    /// Restore from snapshot (for nested call reverts).
    pub fn restore(self: *AccessListManager, snap: AccessListSnapshot) !void {
        var new_addresses = std.AutoHashMap(Address, void).init(self.allocator);
        errdefer new_addresses.deinit();
        var addr_it = snap.addresses.iterator();
        while (addr_it.next()) |entry| {
            try new_addresses.put(entry.key_ptr.*, {});
        }

        var new_slots = std.ArrayHashMap(StorageKey, void, StorageKeyContext, false).init(self.allocator);
        errdefer new_slots.deinit();
        var slot_it = snap.slots.iterator();
        while (slot_it.next()) |entry| {
            _ = try new_slots.put(entry.key_ptr.*, {});
        }

        self.warm_addresses.deinit();
        self.warm_storage_slots.deinit();
        self.warm_addresses = new_addresses;
        self.warm_storage_slots = new_slots;
    }
};

/// Snapshot of warm sets for nested call revert handling.
pub const AccessListSnapshot = struct {
    addresses: std.AutoHashMap(Address, void),
    slots: std.ArrayHashMap(StorageKey, void, StorageKeyContext, false),

    /// Release snapshot resources.
    pub fn deinit(self: *AccessListSnapshot) void {
        self.addresses.deinit();
        self.slots.deinit();
    }
};

test "AccessListManager: init creates empty warm sets" {
    const allocator = std.testing.allocator;
    var manager = AccessListManager.init(allocator);
    defer manager.deinit();

    try std.testing.expectEqual(@as(usize, 0), manager.warm_addresses.count());
    try std.testing.expectEqual(@as(usize, 0), manager.warm_storage_slots.count());
}

test "AccessListManager: access_address returns cold then warm" {
    const allocator = std.testing.allocator;
    var manager = AccessListManager.init(allocator);
    defer manager.deinit();

    const addr = Address{ .bytes = [_]u8{1} ** 20 };
    const cold = try manager.access_address(addr);
    try std.testing.expectEqual(gas_constants.ColdAccountAccessCost, cold);

    const warm = try manager.access_address(addr);
    try std.testing.expectEqual(gas_constants.WarmStorageReadCost, warm);
}

test "AccessListManager: access_storage_slot returns cold then warm" {
    const allocator = std.testing.allocator;
    var manager = AccessListManager.init(allocator);
    defer manager.deinit();

    const addr = Address{ .bytes = [_]u8{2} ** 20 };
    const slot: u256 = 42;
    const cold = try manager.access_storage_slot(addr, slot);
    try std.testing.expectEqual(gas_constants.ColdSloadCost, cold);

    const warm = try manager.access_storage_slot(addr, slot);
    try std.testing.expectEqual(gas_constants.WarmStorageReadCost, warm);
}

test "AccessListManager: pre_warm_addresses marks warm" {
    const allocator = std.testing.allocator;
    var manager = AccessListManager.init(allocator);
    defer manager.deinit();

    const addr1 = Address{ .bytes = [_]u8{1} ** 20 };
    const addr2 = Address{ .bytes = [_]u8{2} ** 20 };
    const addresses = [_]Address{ addr1, addr2 };

    try manager.pre_warm_addresses(&addresses);

    try std.testing.expect(manager.is_address_warm(addr1));
    try std.testing.expect(manager.is_address_warm(addr2));
}

test "AccessListManager: pre_warm_storage_slots marks warm" {
    const allocator = std.testing.allocator;
    var manager = AccessListManager.init(allocator);
    defer manager.deinit();

    const addr = Address{ .bytes = [_]u8{3} ** 20 };
    const slot1: u256 = 1;
    const slot2: u256 = 2;
    const slots = [_]StorageKey{
        .{ .address = addr.bytes, .slot = slot1 },
        .{ .address = addr.bytes, .slot = slot2 },
    };

    try manager.pre_warm_storage_slots(&slots);

    try std.testing.expect(manager.is_storage_slot_warm(addr, slot1));
    try std.testing.expect(manager.is_storage_slot_warm(addr, slot2));
}

test "AccessListManager: pre_warm_from_access_list marks addresses and slots" {
    const allocator = std.testing.allocator;
    var manager = AccessListManager.init(allocator);
    defer manager.deinit();

    const addr = Address{ .bytes = [_]u8{4} ** 20 };
    const slot_hash: primitives.Hash.Hash = [_]u8{0xaa} ** 32;
    const access_list = [_]primitives.AccessList.AccessListEntry{
        .{
            .address = addr,
            .storage_keys = &[_]primitives.Hash.Hash{slot_hash},
        },
    };

    try manager.pre_warm_from_access_list(&access_list);

    try std.testing.expect(manager.is_address_warm(addr));
    const slot = std.mem.readInt(u256, &slot_hash, .big);
    try std.testing.expect(manager.is_storage_slot_warm(addr, slot));
}

test "AccessListManager: clear resets warm state" {
    const allocator = std.testing.allocator;
    var manager = AccessListManager.init(allocator);
    defer manager.deinit();

    const addr = Address{ .bytes = [_]u8{5} ** 20 };
    const slot: u256 = 7;
    _ = try manager.access_address(addr);
    _ = try manager.access_storage_slot(addr, slot);

    manager.clear();

    try std.testing.expect(!manager.is_address_warm(addr));
    try std.testing.expect(!manager.is_storage_slot_warm(addr, slot));
}

test "AccessListManager: snapshot and restore" {
    const allocator = std.testing.allocator;
    var manager = AccessListManager.init(allocator);
    defer manager.deinit();

    const addr1 = Address{ .bytes = [_]u8{6} ** 20 };
    const addr2 = Address{ .bytes = [_]u8{7} ** 20 };

    _ = try manager.access_address(addr1);

    var snap = try manager.snapshot();
    defer snap.deinit();

    _ = try manager.access_address(addr2);
    try std.testing.expect(manager.is_address_warm(addr2));

    try manager.restore(snap);
    try std.testing.expect(manager.is_address_warm(addr1));
    try std.testing.expect(!manager.is_address_warm(addr2));
}
