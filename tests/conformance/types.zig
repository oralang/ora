const std = @import("std");
const evm_mod = @import("ora_evm");
const primitives = evm_mod.primitives;

pub const ORA_BINARY_REL = "zig-out/bin/ora";
pub const CONFORMANCE_DIR_REL = "tests/conformance";
pub const DEFAULT_GAS: u64 = 30_000_000;

pub const Address = primitives.Address.Address;
pub const Evm = evm_mod.Evm(.{});

pub const ArgValue = union(enum) {
    literal: []const u8,
    boolean: bool,
    /// `@name` — resolved at execution time to the deployed address of the
    /// named contract (primary or a `[[contract]]` secondary). Enables
    /// cross-contract wiring (e.g. handing the victim the attacker's address).
    contract_ref: []const u8,
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
    /// Call must revert; revert data is not checked (adversarial robustness).
    reverts_any,
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
    /// When true, skip per-call event-log assertions (for app gas benchmarks
    /// that care about gas/returns/state, not exact emitted event bytes).
    ignore_logs: bool = false,

    pub fn deinit(self: DeploySpec, allocator: std.mem.Allocator) void {
        allocator.free(self.args);
    }
};

pub const CallSpec = struct {
    /// Target contract by name; null = the primary `[deploy]` contract.
    /// Names come from `[[contract]]` sections (and "self"/primary is implicit).
    to: ?[]const u8 = null,
    /// Either @"fn"+args (typed) or calldata (raw hostile bytes) is set, never both.
    @"fn": ?[]const u8,
    /// Raw calldata bytes; when set, the call bypasses ABI encoding. Such calls
    /// can only assert `succeeds`/`reverts` (no function ABI to type returns).
    calldata: ?[]const u8,
    caller: Address,
    value: u256,
    args: []ArgValue,
    outcome: ExpectedOutcome,
    gas_max: ?u64,
    storage: []StorageAssertion,
    logs: []LogAssertion,

    pub fn deinit(self: CallSpec, allocator: std.mem.Allocator) void {
        allocator.free(self.args);
        allocator.free(self.storage);
        for (self.logs) |log| log.deinit(allocator);
        allocator.free(self.logs);
    }
};

/// A named secondary contract deployed alongside the primary `[deploy]`.
/// Declared via `[[contract]]` sections. `source` is required (its own .ora).
pub const ContractSpec = struct {
    name: []const u8,
    source: []const u8,
    caller: Address,
    value: u256,
    args: []ArgValue,

    pub fn deinit(self: ContractSpec, allocator: std.mem.Allocator) void {
        allocator.free(self.args);
    }
};

pub const Spec = struct {
    deploy: DeploySpec,
    /// Secondary contracts (`[[contract]]`), deployed after the primary in
    /// declaration order so earlier ones can be referenced by later args.
    secondary: []ContractSpec = &.{},
    calls: []CallSpec,

    pub fn deinit(self: Spec, allocator: std.mem.Allocator) void {
        self.deploy.deinit(allocator);
        for (self.secondary) |c| c.deinit(allocator);
        allocator.free(self.secondary);
        for (self.calls) |call| call.deinit(allocator);
        allocator.free(self.calls);
    }
};
