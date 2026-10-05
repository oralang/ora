//! Diagnostic lifecycle inputs. No verifier consumes this inventory as a proof.
const std = @import("std");
const builtin = @import("ora_types").builtin;

pub const Digest = [32]u8;
pub const Bounds = struct {
    kind: enum { MinValue, MaxValue, InRange },
    minimum: u256,
    maximum: u256,
};
pub const Refusal = enum { unsupported_type, missing_type, width_mismatch, unsupported_initialization };
pub const Claim = struct {
    owner: []const u8,
    root: []const u8,
    ordinal: u32,
    type_id: ?u32,
    bounds: ?Bounds,
    zero_base: bool,
    refusal: ?Refusal,
};
pub const Event = struct {
    ordinal: u32,
    operation: []const u8,
    kind: enum { write, call, unknown },
    target: ?[]const u8,
    refusal: ?enum { unresolved_call, external_interference, unknown_effect } = null,
    hir_sha256: Digest,
};
pub const Function = struct {
    owner: []const u8,
    module_path: []const u8,
    symbol: []const u8,
    ordinal: u32,
    runtime: bool,
    constructor: bool,
    hir_sha256: Digest,
    events: []const Event,
};

pub const FunctionBinding = struct {
    function_ordinal: u32,
    function_id: u32,
    function_vc_sha256: Digest,
};

pub fn validateBindings(inventory: Inventory, bindings: []const FunctionBinding) !void {
    try inventory.validate();
    for (bindings, 0..) |binding, index| {
        if (binding.function_id == 0 or emptyDigest(binding.function_vc_sha256)) return error.InvalidLifecycleBinding;
        for (bindings[0..index]) |previous| if (previous.function_id == binding.function_id or previous.function_ordinal == binding.function_ordinal)
            return error.DuplicateLifecycleBinding;
        const found = for (inventory.functions) |function| {
            if (function.ordinal == binding.function_ordinal) break true;
        } else false;
        if (!found) return error.ForeignLifecycleBinding;
    }
    for (inventory.functions) |function| {
        if (!function.runtime and !function.constructor) continue;
        for (bindings) |binding| {
            if (binding.function_ordinal == function.ordinal) break;
        } else return error.MissingLifecycleBinding;
    }
}

pub const Inventory = struct {
    module_sha256: Digest,
    claims: []const Claim,
    functions: []const Function,

    pub fn validate(self: Inventory) !void {
        if (emptyDigest(self.module_sha256)) return error.InvalidLifecycleDigest;
        for (self.claims, 0..) |claim, index| {
            if (claim.owner.len == 0 or claim.root.len == 0) return error.InvalidLifecycleOwner;
            for (self.claims[0..index]) |previous| {
                if (previous.ordinal == claim.ordinal or
                    (std.mem.eql(u8, previous.owner, claim.owner) and std.mem.eql(u8, previous.root, claim.root))) return error.DuplicateLifecycleClaim;
            }
            if (claim.refusal != null) continue;
            const info = builtin.integerInfoByComptimeTypeId(claim.type_id orelse return error.InvalidLifecycleType) orelse return error.InvalidLifecycleType;
            const bounds = claim.bounds orelse return error.InvalidLifecycleBounds;
            if (!claim.zero_base) return error.InvalidLifecycleInitialization;
            if (info.width < 256 and (bounds.minimum >> @as(u8, @intCast(info.width)) != 0 or bounds.maximum >> @as(u8, @intCast(info.width)) != 0)) return error.InvalidLifecycleBounds;
        }
        for (self.functions, 0..) |function, index| {
            if (function.owner.len == 0 or function.module_path.len == 0 or function.symbol.len == 0) return error.InvalidLifecycleOwner;
            if (emptyDigest(function.hir_sha256)) return error.InvalidLifecycleDigest;
            for (self.functions[0..index]) |previous| {
                if (previous.ordinal == function.ordinal or std.mem.eql(u8, previous.symbol, function.symbol)) return error.AmbiguousLifecycleFunction;
            }
            for (function.events, 0..) |event, event_index| {
                if (event.operation.len == 0 or emptyDigest(event.hir_sha256)) return error.InvalidLifecycleEvent;
                if (event_index != 0 and function.events[event_index - 1].ordinal >= event.ordinal) return error.DuplicateLifecycleEvent;
            }
        }
    }

    pub fn canonical(self: Inventory, allocator: std.mem.Allocator) ![]u8 {
        try self.validate();
        var output = std.Io.Writer.Allocating.init(allocator);
        errdefer output.deinit();
        try output.writer.writeAll("StorageLifecycleInventory-v1\n");
        try std.json.Stringify.value(self, .{}, &output.writer);
        return output.toOwnedSlice();
    }

    pub fn sha256(self: Inventory, allocator: std.mem.Allocator) !Digest {
        const bytes = try self.canonical(allocator);
        defer allocator.free(bytes);
        return hash(bytes);
    }
};

pub fn hash(bytes: []const u8) Digest {
    var result: Digest = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &result, .{});
    return result;
}

fn emptyDigest(digest: Digest) bool {
    return std.mem.allEqual(u8, &digest, 0);
}
