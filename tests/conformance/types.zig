const std = @import("std");
const evm_mod = @import("ora_evm");
const primitives = @import("voltaire");

pub const ORA_BINARY_REL = "zig-out/bin/ora";
pub const CONFORMANCE_DIR_REL = "tests/conformance";
pub const DEFAULT_GAS: u64 = 30_000_000;

pub const Address = primitives.Address.Address;
pub const Evm = evm_mod.Evm(.{});

pub const ArgValue = union(enum) {
    literal: []const u8,
    boolean: bool,
};

pub const ExpectedStaticReturn = struct {
    spec_type: []const u8,
    value: ArgValue,
};

pub const ExpectedOutcome = union(enum) {
    returns_empty,
    returns_static: ExpectedStaticReturn,
    /// Call must succeed; return bytes are not checked. For state-effect
    /// scenarios where the return encoding is irrelevant or untrusted.
    succeeds_any,
    reverts_empty,
    reverts_selector: [4]u8,
    reverts_data: []const u8,
};

pub const StorageAssertion = struct {
    slot: u256,
    value: u256,
};

pub const LogAssertion = struct {
    topics: []u256,
    data: []u8,

    pub fn deinit(self: LogAssertion, allocator: std.mem.Allocator) void {
        allocator.free(self.topics);
        allocator.free(self.data);
    }
};

pub const DeploySpec = struct {
    caller: Address,
    value: u256,
    args: []ArgValue,
    /// Repo-relative .ora source path; defaults to the spec's sibling <stem>.ora.
    source: ?[]const u8 = null,

    pub fn deinit(self: DeploySpec, allocator: std.mem.Allocator) void {
        allocator.free(self.args);
    }
};

pub const CallSpec = struct {
    @"fn": []const u8,
    caller: Address,
    value: u256,
    args: []ArgValue,
    outcome: ExpectedOutcome,
    storage: []StorageAssertion,
    logs: []LogAssertion,

    pub fn deinit(self: CallSpec, allocator: std.mem.Allocator) void {
        allocator.free(self.args);
        allocator.free(self.storage);
        for (self.logs) |log| log.deinit(allocator);
        allocator.free(self.logs);
    }
};

pub const Spec = struct {
    deploy: DeploySpec,
    calls: []CallSpec,

    pub fn deinit(self: Spec, allocator: std.mem.Allocator) void {
        self.deploy.deinit(allocator);
        for (self.calls) |call| call.deinit(allocator);
        allocator.free(self.calls);
    }
};
