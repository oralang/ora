const std = @import("std");
const evm_mod = @import("ora_evm");
const types = @import("types.zig");

const StorageSlotKey = struct {
    address: types.Address,
    slot: u256,
};

pub const HarnessHost = struct {
    allocator: std.mem.Allocator,
    balances: std.AutoHashMap(types.Address, u256),
    code: std.AutoHashMap(types.Address, []const u8),
    storage: std.AutoHashMap(StorageSlotKey, u256),
    nonces: std.AutoHashMap(types.Address, u64),
    fatal: ?HostFatal = null,

    const HostFatal = enum {
        out_of_memory,
    };

    pub fn init(allocator: std.mem.Allocator) HarnessHost {
        return .{
            .allocator = allocator,
            .balances = std.AutoHashMap(types.Address, u256).init(allocator),
            .code = std.AutoHashMap(types.Address, []const u8).init(allocator),
            .storage = std.AutoHashMap(StorageSlotKey, u256).init(allocator),
            .nonces = std.AutoHashMap(types.Address, u64).init(allocator),
        };
    }

    pub fn deinit(self: *HarnessHost) void {
        var it = self.code.valueIterator();
        while (it.next()) |code| self.allocator.free(code.*);
        self.balances.deinit();
        self.code.deinit();
        self.storage.deinit();
        self.nonces.deinit();
    }

    pub fn check(self: *const HarnessHost) !void {
        if (self.fatal != null) return error.HostMutationFailed;
    }

    fn recordFatal(self: *HarnessHost, fatal: HostFatal) void {
        if (self.fatal == null) self.fatal = fatal;
    }

    pub fn hostInterface(self: *HarnessHost) evm_mod.HostInterface {
        return .{
            .ptr = self,
            .vtable = &.{
                .getBalance = getBalance,
                .setBalance = setBalanceVTable,
                .getCode = getCode,
                .setCode = setCodeVTable,
                .getStorage = getStorage,
                .setStorage = setStorageVTable,
                .getNonce = getNonceVTable,
                .setNonce = setNonceVTable,
            },
        };
    }

    pub fn setBalance(self: *HarnessHost, address: types.Address, balance: u256) !void {
        try self.balances.put(address, balance);
    }

    pub fn setNonce(self: *HarnessHost, address: types.Address, nonce: u64) !void {
        if (nonce == 0) {
            _ = self.nonces.remove(address);
        } else {
            try self.nonces.put(address, nonce);
        }
    }

    pub fn getStorageSlot(self: *HarnessHost, address: types.Address, slot: u256) u256 {
        return self.storage.get(.{ .address = address, .slot = slot }) orelse 0;
    }

    pub fn getCodeForAddress(self: *HarnessHost, address: types.Address) []const u8 {
        return self.code.get(address) orelse &.{};
    }

    fn getBalance(ptr: *anyopaque, address: types.Address) u256 {
        const self: *HarnessHost = @ptrCast(@alignCast(ptr));
        return self.balances.get(address) orelse 0;
    }

    fn setBalanceVTable(ptr: *anyopaque, address: types.Address, balance: u256) void {
        const self: *HarnessHost = @ptrCast(@alignCast(ptr));
        self.balances.put(address, balance) catch self.recordFatal(.out_of_memory);
    }

    fn getCode(ptr: *anyopaque, address: types.Address) []const u8 {
        const self: *HarnessHost = @ptrCast(@alignCast(ptr));
        return self.code.get(address) orelse &.{};
    }

    fn setCodeVTable(ptr: *anyopaque, address: types.Address, code: []const u8) void {
        const self: *HarnessHost = @ptrCast(@alignCast(ptr));
        if (self.code.fetchRemove(address)) |kv| self.allocator.free(kv.value);
        if (code.len == 0) return;
        const owned = self.allocator.dupe(u8, code) catch {
            self.recordFatal(.out_of_memory);
            return;
        };
        self.code.put(address, owned) catch {
            self.allocator.free(owned);
            self.recordFatal(.out_of_memory);
        };
    }

    fn getStorage(ptr: *anyopaque, address: types.Address, slot: u256) u256 {
        const self: *HarnessHost = @ptrCast(@alignCast(ptr));
        return self.getStorageSlot(address, slot);
    }

    fn setStorageVTable(ptr: *anyopaque, address: types.Address, slot: u256, value: u256) void {
        const self: *HarnessHost = @ptrCast(@alignCast(ptr));
        const key = StorageSlotKey{ .address = address, .slot = slot };
        if (value == 0) {
            _ = self.storage.remove(key);
        } else {
            self.storage.put(key, value) catch self.recordFatal(.out_of_memory);
        }
    }

    fn getNonceVTable(ptr: *anyopaque, address: types.Address) u64 {
        const self: *HarnessHost = @ptrCast(@alignCast(ptr));
        return self.nonces.get(address) orelse 0;
    }

    fn setNonceVTable(ptr: *anyopaque, address: types.Address, nonce: u64) void {
        const self: *HarnessHost = @ptrCast(@alignCast(ptr));
        self.setNonce(address, nonce) catch self.recordFatal(.out_of_memory);
    }
};
