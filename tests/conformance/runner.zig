const std = @import("std");
const testing = std.testing;
const evm_mod = @import("ora_evm");
const abi = @import("abi.zig");
const abi_doc = @import("abi_doc.zig");
const host_mod = @import("host.zig");
const spec_mod = @import("spec.zig");
const types = @import("types.zig");

pub const HarnessHost = host_mod.HarnessHost;
pub const PropertyRuntime = struct {
    allocator: std.mem.Allocator,
    host: *HarnessHost,
    evm: *types.Evm,
    abi: *const abi_doc.AbiDoc,
    contract_address: types.Address,
    caller: types.Address,

    pub fn call(self: *PropertyRuntime, signature: []const u8, args: []const types.ArgValue) !types.Evm.CallResult {
        const function_abi = try self.abi.findFunction(signature);
        defer function_abi.deinit(self.allocator);
        const resolved_args = try resolveContractAddressArgs(self.allocator, function_abi.inputs, args, self.contract_address);
        defer deinitResolvedContractAddressArgs(self.allocator, args, resolved_args);
        const encoded_args = try abi.encodeArgs(self.allocator, function_abi.inputs, resolved_args);
        defer self.allocator.free(encoded_args);

        var calldata: std.ArrayList(u8) = .empty;
        defer calldata.deinit(self.allocator);
        try calldata.appendSlice(self.allocator, &function_abi.selector);
        try calldata.appendSlice(self.allocator, encoded_args);

        const result = self.evm.call(.{ .call = .{
            .caller = self.caller,
            .to = self.contract_address,
            .value = 0,
            .input = calldata.items,
            .gas = types.DEFAULT_GAS,
        } });
        try self.host.check();
        return result;
    }

    /// Send raw calldata bytes (no ABI encoding) — for the structured fuzzer.
    pub fn callRaw(self: *PropertyRuntime, calldata: []const u8) !types.Evm.CallResult {
        const result = self.evm.call(.{ .call = .{
            .caller = self.caller,
            .to = self.contract_address,
            .value = 0,
            .input = calldata,
            .gas = types.DEFAULT_GAS,
        } });
        try self.host.check();
        return result;
    }
};

pub fn runOraEmit(allocator: std.mem.Allocator, source_path: []const u8, output_dir: []const u8) !void {
    try runOraEmitWithExtraArgs(allocator, source_path, output_dir, &.{"--no-verify"});
}

pub fn runOraEmitWithExtraArgs(
    allocator: std.mem.Allocator,
    source_path: []const u8,
    output_dir: []const u8,
    extra_args: []const []const u8,
) !void {
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    try argv.appendSlice(allocator, &.{
        types.ORA_BINARY_REL,
        "emit",
        "--emit=abi,bytecode",
        "--output-dir",
        output_dir,
    });
    try argv.appendSlice(allocator, extra_args);
    try argv.append(allocator, source_path);

    var process_io = std.Io.Threaded.init(allocator, .{
        .async_limit = .nothing,
        .concurrent_limit = .nothing,
    });
    defer process_io.deinit();

    const result = try std.process.run(allocator, process_io.io(), .{
        .argv = argv.items,
        .stdout_limit = std.Io.Limit.limited(8 * 1024 * 1024),
        .stderr_limit = std.Io.Limit.limited(8 * 1024 * 1024),
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("ora emit failed with code {d}\nstdout:\n{s}\nstderr:\n{s}\n", .{ code, result.stdout, result.stderr });
            return error.OraEmitFailed;
        },
        else => {
            std.debug.print("ora emit terminated abnormally: {}\n", .{result.term});
            return error.OraEmitFailed;
        },
    }
}

pub fn pathFromTmpAlloc(allocator: std.mem.Allocator, tmp: std.testing.TmpDir, rel_path: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, rel_path });
}

const SkipList = struct {
    buffer: []u8 = &.{},
    entries: [][]const u8 = &.{},

    fn deinit(self: *SkipList, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        allocator.free(self.buffer);
        self.* = .{};
    }

    fn contains(self: *const SkipList, file_name: []const u8) bool {
        for (self.entries) |entry| {
            if (std.mem.eql(u8, entry, file_name)) return true;
        }
        return false;
    }
};

pub fn checkCorpusSidecars(allocator: std.mem.Allocator, corpus_dir_path: []const u8) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var skip_list = try loadSkipList(allocator, corpus_dir_path);
    defer skip_list.deinit(allocator);

    var dir = try std.Io.Dir.cwd().openDir(io, corpus_dir_path, .{ .iterate = true });
    defer dir.close(io);

    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".ora")) continue;
        if (skip_list.contains(entry.name)) continue;

        const stem = entry.name[0 .. entry.name.len - ".ora".len];
        const spec_name = try std.fmt.allocPrint(allocator, "{s}.spec.toml", .{stem});
        defer allocator.free(spec_name);
        dir.access(io, spec_name, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.MissingSidecar,
            else => return err,
        };
    }
}

fn loadSkipList(allocator: std.mem.Allocator, corpus_dir_path: []const u8) !SkipList {
    const skip_path = try std.fs.path.join(allocator, &.{ corpus_dir_path, "SKIP" });
    defer allocator.free(skip_path);
    const buffer = std.Io.Dir.cwd().readFileAlloc(std.Io.Threaded.global_single_threaded.io(), skip_path, allocator, std.Io.Limit.limited(1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => return .{},
        else => return err,
    };
    errdefer allocator.free(buffer);

    var entries: std.ArrayList([]const u8) = .empty;
    errdefer entries.deinit(allocator);

    var lines = std.mem.splitScalar(u8, buffer, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, stripComment(line), " \t\r");
        if (trimmed.len == 0) continue;
        if (!std.mem.endsWith(u8, trimmed, ".ora")) return error.InvalidSkipEntry;
        try entries.append(allocator, trimmed);
    }

    return .{ .buffer = buffer, .entries = try entries.toOwnedSlice(allocator) };
}

fn stripComment(line: []const u8) []const u8 {
    var in_string = false;
    var escaped = false;
    for (line, 0..) |c, i| {
        if (escaped) {
            escaped = false;
            continue;
        }
        if (in_string and c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (!in_string and c == '#') return line[0..i];
    }
    return line;
}

fn readArtifacts(allocator: std.mem.Allocator, output_dir: []const u8, stem: []const u8) !struct {
    bytecode: []u8,
    abi_path: []u8,
} {
    const hex_name = try std.fmt.allocPrint(allocator, "{s}.hex", .{stem});
    const abi_name = try std.fmt.allocPrint(allocator, "{s}.abi.json", .{stem});
    const hex_path = try std.fs.path.join(allocator, &.{ output_dir, hex_name });
    const abi_path = try std.fs.path.join(allocator, &.{ output_dir, abi_name });
    const hex = try std.Io.Dir.cwd().readFileAlloc(std.Io.Threaded.global_single_threaded.io(), hex_path, allocator, std.Io.Limit.limited(16 * 1024 * 1024));
    return .{ .bytecode = try abi.parseHexBytes(allocator, hex), .abi_path = abi_path };
}

pub fn runConformanceSpec(allocator: std.mem.Allocator, source_path: []const u8, spec_path: []const u8) !void {
    return runConformanceSpecImpl(allocator, source_path, spec_path, null, &.{"--no-verify"});
}

pub fn runConformanceSpecWithExtraArgs(
    allocator: std.mem.Allocator,
    source_path: []const u8,
    spec_path: []const u8,
    extra_args: []const []const u8,
) !void {
    return runConformanceSpecImpl(allocator, source_path, spec_path, null, extra_args);
}

pub fn runConformanceSpecDifferential(
    allocator: std.mem.Allocator,
    source_path: []const u8,
    spec_path: []const u8,
) !void {
    try runConformanceSpecWithExtraArgs(allocator, source_path, spec_path, &.{});
    try runConformanceSpecWithExtraArgs(allocator, source_path, spec_path, &.{"--keep-proved-checks"});
}

/// Like runConformanceSpec but records numeric metrics (deploy gas, per-call
/// gas, and per-contract bytecode size) into `metrics`. Still asserts all spec
/// outcomes, so it only measures on a green corpus.
pub fn runConformanceSpecMetrics(allocator: std.mem.Allocator, source_path: []const u8, spec_path: []const u8, metrics: MetricSink) !void {
    return runConformanceSpecImpl(allocator, source_path, spec_path, metrics, &.{"--no-verify"});
}

pub fn runConformanceSpecMetricsWithExtraArgs(
    allocator: std.mem.Allocator,
    source_path: []const u8,
    spec_path: []const u8,
    metrics: MetricSink,
    extra_args: []const []const u8,
) !void {
    return runConformanceSpecImpl(allocator, source_path, spec_path, metrics, extra_args);
}

fn runConformanceSpecImpl(
    allocator: std.mem.Allocator,
    source_path: []const u8,
    spec_path: []const u8,
    metrics: ?MetricSink,
    extra_args: []const []const u8,
) !void {
    var run_arena = std.heap.ArenaAllocator.init(allocator);
    defer run_arena.deinit();
    const arena = run_arena.allocator();

    const spec_text = try std.Io.Dir.cwd().readFileAlloc(std.Io.Threaded.global_single_threaded.io(), spec_path, arena, std.Io.Limit.limited(1024 * 1024));
    var parsed = try spec_mod.parse(arena, spec_text);
    defer parsed.deinit();

    // [deploy] source = "..." points at a repo-relative contract (e.g. an
    // ora-example app) instead of the spec's sibling .ora.
    const effective_source = parsed.value.deploy.source orelse source_path;
    if (parsed.value.deploy.source) |declared| {
        std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), declared, .{}) catch {
            std.debug.print("declared deploy source not found: {s}\n", .{declared});
            return error.DeclaredSourceMissing;
        };
    }

    const io = std.Io.Threaded.global_single_threaded.io();
    // Scoped per process as well as per spec: the conformance suite and the
    // metrics harness walk the same corpus as sibling gate steps, and a
    // shared deterministic path lets one process's entry/exit deleteTree
    // vaporize the other's artifacts mid-read (observed as FileNotFound on
    // abi.json and as vanished metrics rows).
    const tmp_path = try std.fmt.allocPrint(arena, ".zig-cache/tmp/conformance-{x}-{d}", .{ std.hash.Wyhash.hash(0, spec_path), std.c.getpid() });
    std.Io.Dir.cwd().deleteTree(io, tmp_path) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, tmp_path) catch {};
    try std.Io.Dir.cwd().createDirPath(io, tmp_path);
    const out_path = try std.fmt.allocPrint(arena, "{s}/out", .{tmp_path});
    try std.Io.Dir.cwd().createDir(io, out_path, .default_dir);

    // Compile the primary contract, then every [[contract]] secondary. Each
    // .ora has a distinct stem so all artifacts coexist in one out dir.
    var compiled: std.ArrayList(CompiledContract) = .empty;
    const primary = try compileOne(arena, effective_source, out_path, extra_args);
    try testing.expectError(error.UnknownFunction, primary.doc.findFunction("missing()"));
    if (metrics) |sink| try sink.record("__bytecode_bytes", primary.bytecode.len);
    try compiled.append(arena, .{
        .name = "self",
        .caller = parsed.value.deploy.caller,
        .value = parsed.value.deploy.value,
        .args = parsed.value.deploy.args,
        .bytecode = primary.bytecode,
        .doc = primary.doc,
    });

    for (parsed.value.secondary) |c| {
        std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), c.source, .{}) catch {
            std.debug.print("secondary contract source not found: {s}\n", .{c.source});
            return error.DeclaredSourceMissing;
        };
        const built = try compileOne(arena, c.source, out_path, extra_args);
        try compiled.append(arena, .{
            .name = c.name,
            .caller = c.caller,
            .value = c.value,
            .args = c.args,
            .bytecode = built.bytecode,
            .doc = built.doc,
        });
    }

    try executeSpec(arena, parsed.value, compiled.items, metrics);
}

const CompiledContract = struct {
    name: []const u8,
    caller: types.Address,
    value: u256,
    args: []types.ArgValue,
    bytecode: []const u8,
    doc: abi_doc.AbiDoc,
};

fn compileOne(
    arena: std.mem.Allocator,
    source_path: []const u8,
    out_path: []const u8,
    extra_args: []const []const u8,
) !struct {
    bytecode: []const u8,
    doc: abi_doc.AbiDoc,
} {
    try runOraEmitWithExtraArgs(arena, source_path, out_path, extra_args);
    const stem = sourceStem(source_path) orelse return error.InvalidSourcePath;
    const artifacts = try readArtifacts(arena, out_path, stem);
    const doc = try abi_doc.AbiDoc.load(arena, artifacts.abi_path);
    return .{ .bytecode = artifacts.bytecode, .doc = doc };
}

fn sourceStem(source_path: []const u8) ?[]const u8 {
    const base = std.fs.path.basename(source_path);
    if (!std.mem.endsWith(u8, base, ".ora")) return null;
    return base[0 .. base.len - ".ora".len];
}

pub fn collectSpecNames(allocator: std.mem.Allocator, corpus_dir_path: []const u8) ![][]const u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, corpus_dir_path, .{ .iterate = true });
    defer dir.close(io);

    var specs: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (specs.items) |name| allocator.free(name);
        specs.deinit(allocator);
    }

    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".spec.toml")) continue;
        try specs.append(allocator, try allocator.dupe(u8, entry.name));
    }

    std.mem.sort([]const u8, specs.items, {}, lessThanString);
    return try specs.toOwnedSlice(allocator);
}

fn lessThanString(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

pub fn freeStringList(allocator: std.mem.Allocator, list: [][]const u8) void {
    for (list) |item| allocator.free(item);
    allocator.free(list);
}

/// Optional metrics sink: when present, the runner records numeric metrics —
/// per-call metered gas and per-contract bytecode size — for the metrics
/// benchmark. Keys are duped into the sink's allocator so they outlive the arena.
pub const MetricSample = struct { key: []const u8, value: u64 };
pub const MetricSink = struct {
    allocator: std.mem.Allocator,
    list: *std.ArrayList(MetricSample),

    fn record(self: MetricSink, key: []const u8, value: u64) !void {
        try self.list.append(self.allocator, .{ .key = try self.allocator.dupe(u8, key), .value = value });
    }
};

pub const PropertyRuntimeBundle = struct {
    allocator: std.mem.Allocator,
    host: *HarnessHost,
    evm: *types.Evm,
    doc: *abi_doc.AbiDoc,
    contract_address: types.Address,
    caller: types.Address,

    pub fn deinit(self: *PropertyRuntimeBundle) void {
        self.evm.deinit();
        self.host.deinit();
        self.doc.deinit();
    }

    pub fn runtime(self: *PropertyRuntimeBundle) PropertyRuntime {
        return .{
            .allocator = self.allocator,
            .host = self.host,
            .evm = self.evm,
            .abi = self.doc,
            .contract_address = self.contract_address,
            .caller = self.caller,
        };
    }
};

fn findContract(contracts: []const CompiledContract, name: []const u8) ?*const CompiledContract {
    for (contracts) |*c| {
        if (std.mem.eql(u8, c.name, name)) return c;
    }
    return null;
}

fn executeSpec(allocator: std.mem.Allocator, spec: types.Spec, contracts: []const CompiledContract, metrics: ?MetricSink) !void {
    var host = HarnessHost.init(allocator);
    defer host.deinit();
    try host.setBalance(spec.deploy.caller, std.math.maxInt(u256));
    try host.setNonce(spec.deploy.caller, 1);

    var evm: types.Evm = undefined;
    try evm.init(allocator, host.hostInterface(), .OSAKA, evm_mod.deterministicBlockContext(), spec.deploy.caller, 0, null);
    defer evm.deinit();

    try evm.initTransactionState(null);
    try evm.preWarmTransaction(spec.deploy.caller);

    // Deploy primary first, then each secondary in declaration order, all from
    // the same origin (sequential nonces → distinct deterministic addresses).
    // Each contract's constructor args may reference an already-deployed
    // contract by @name (resolved against the map built so far).
    var addresses = std.StringHashMap(types.Address).init(allocator);
    defer addresses.deinit();

    for (contracts, 0..) |c, i| {
        // inner_create does not bump the deployer nonce for top-level creates
        // (it expects the runner to). Advance it per deploy so each contract
        // gets a distinct CREATE address instead of colliding on nonce 1.
        try host.setNonce(spec.deploy.caller, @intCast(1 + i));

        const ctor = try c.doc.findConstructor();
        defer ctor.deinit(allocator);
        const resolved = try resolveAddressArgs(allocator, ctor.inputs, c.args, null, &addresses);
        defer deinitResolvedAddressArgs(allocator, c.args, resolved);
        const ctor_args = try abi.encodeArgs(allocator, ctor.inputs, resolved);
        defer allocator.free(ctor_args);

        var init_code: std.ArrayList(u8) = .empty;
        defer init_code.deinit(allocator);
        try init_code.appendSlice(allocator, c.bytecode);
        try init_code.appendSlice(allocator, ctor_args);

        const create_result = try evm.inner_create(c.value, init_code.items, types.DEFAULT_GAS, null);
        try host.check();
        if (!create_result.success) {
            std.debug.print("contract deployment failed: {s}\n", .{c.name});
        }
        try testing.expect(create_result.success);
        if (metrics) |sink| {
            const deploy_key = if (i == 0)
                "__deploy_gas"
            else
                try std.fmt.allocPrint(allocator, "__deploy_gas:{s}", .{c.name});
            try sink.record(deploy_key, types.DEFAULT_GAS - create_result.gas_left);
        }
        try testing.expect(host.getCodeForAddress(create_result.address).len > 0);
        try addresses.put(c.name, create_result.address);
    }

    const primary_address = addresses.get("self").?;

    for (spec.calls) |call| {
        const target_address = if (call.to) |name|
            (addresses.get(name) orelse return error.UnknownContractTarget)
        else
            primary_address;
        const target_doc = if (call.to) |name|
            &(findContract(contracts, name) orelse return error.UnknownContractTarget).doc
        else
            &contracts[0].doc;

        // Raw-calldata calls bypass the ABI; typed calls
        // resolve a function and encode args. A raw call has no function_abi,
        // so its outcome is restricted to succeeds/reverts (enforced at parse).
        var function_abi: ?abi_doc.FunctionAbi = null;
        defer if (function_abi) |fa| fa.deinit(allocator);

        var calldata: std.ArrayList(u8) = .empty;
        defer calldata.deinit(allocator);

        if (call.calldata) |raw| {
            try calldata.appendSlice(allocator, raw);
        } else {
            const fa = try target_doc.findFunction(call.@"fn".?);
            function_abi = fa;
            const resolved_args = try resolveAddressArgs(allocator, fa.inputs, call.args, target_address, &addresses);
            defer deinitResolvedAddressArgs(allocator, call.args, resolved_args);
            const encoded_args = try abi.encodeArgs(allocator, fa.inputs, resolved_args);
            defer allocator.free(encoded_args);
            try calldata.appendSlice(allocator, &fa.selector);
            try calldata.appendSlice(allocator, encoded_args);
        }

        const result = evm.call(.{ .call = .{
            .caller = call.caller,
            .to = target_address,
            .value = call.value,
            .input = calldata.items,
            .gas = types.DEFAULT_GAS,
        } });
        try host.check();
        if (call.gas_max) |max| {
            try testing.expect(result.gasConsumed(types.DEFAULT_GAS) <= max);
        }
        if (metrics) |sink| {
            const label = call.@"fn" orelse "calldata";
            try sink.record(label, result.gasConsumed(types.DEFAULT_GAS));
        }

        switch (call.outcome) {
            .returns_empty => {
                const fa = function_abi orelse return error.RawCalldataNeedsSucceedsOrReverts;
                try testing.expect(result.success);
                if (fa.outputs.len != 0) return error.UnsupportedReturnType;
                try testing.expectEqual(@as(usize, 0), result.output.len);
            },
            .returns_static => |expected| {
                const fa = function_abi orelse return error.RawCalldataNeedsSucceedsOrReverts;
                try testing.expect(result.success);
                if (fa.outputs.len != 1) return error.UnsupportedReturnType;
                if (!abi.specTypeMatchesAbiWire(expected.spec_type, fa.outputs[0])) return error.UnsupportedReturnType;
                if (try abi.isSingleStaticWord(allocator, fa.outputs[0])) {
                    try testing.expectEqual(@as(usize, 32), result.output.len);
                    try abi.expectStaticReturn(fa.outputs[0], expected.value, result.output[0..32]);
                } else {
                    const expected_args = [_]types.ArgValue{expected.value};
                    const expected_output = try abi.encodeArgs(allocator, fa.outputs, &expected_args);
                    defer allocator.free(expected_output);
                    try testing.expectEqualSlices(u8, expected_output, result.output);
                }
            },
            .succeeds_any => {
                try testing.expect(result.success);
            },
            .reverts_any => {
                try testing.expect(!result.success);
            },
            .reverts_empty => {
                try testing.expect(!result.success);
                try testing.expectEqual(@as(usize, 0), result.output.len);
            },
            .reverts_selector => |expected| {
                try testing.expect(!result.success);
                try testing.expect(result.output.len >= 4);
                try testing.expectEqualSlices(u8, &expected, result.output[0..4]);
            },
            .reverts_data => |expected| {
                try testing.expect(!result.success);
                try testing.expectEqualSlices(u8, expected, result.output);
            },
        }

        if (!spec.deploy.ignore_logs) {
            try testing.expectEqual(call.logs.len, result.logs.len);
            for (call.logs, result.logs) |expected_log, actual_log| {
                try testing.expectEqualSlices(u256, expected_log.topics, actual_log.topics);
                try testing.expectEqualSlices(u8, expected_log.data, actual_log.data);
            }
        }

        for (call.storage) |assertion| {
            try testing.expectEqual(assertion.value, host.getStorageSlot(target_address, assertion.slot));
        }
    }
}

fn resolveContractAddressArgs(
    allocator: std.mem.Allocator,
    wires: []const []const u8,
    args: []const types.ArgValue,
    contract_address: types.Address,
) ![]types.ArgValue {
    if (wires.len != args.len) return error.ArgumentCountMismatch;
    const resolved = try allocator.alloc(types.ArgValue, args.len);
    var initialized: usize = 0;
    errdefer {
        freeResolvedContractAddressLiterals(allocator, args[0..initialized], resolved[0..initialized]);
        allocator.free(resolved);
    }
    for (args, 0..) |arg, i| {
        resolved[i] = switch (arg) {
            .literal => |literal| if (std.mem.eql(u8, literal, "$contract")) blk: {
                if (!std.mem.eql(u8, wires[i], "address")) return error.UnsupportedArgType;
                break :blk .{ .literal = try allocAddressLiteral(allocator, contract_address) };
            } else arg,
            .boolean => arg,
            .contract_ref => return error.UnsupportedArgType,
        };
        initialized += 1;
    }
    return resolved;
}

fn deinitResolvedContractAddressArgs(allocator: std.mem.Allocator, original: []const types.ArgValue, resolved: []const types.ArgValue) void {
    freeResolvedContractAddressLiterals(allocator, original, resolved);
    allocator.free(resolved);
}

fn freeResolvedContractAddressLiterals(allocator: std.mem.Allocator, original: []const types.ArgValue, resolved: []const types.ArgValue) void {
    for (original, resolved) |original_arg, resolved_arg| {
        const original_literal = switch (original_arg) {
            .literal => |literal| literal,
            .boolean => continue,
            .contract_ref => continue,
        };
        if (!std.mem.eql(u8, original_literal, "$contract")) continue;
        switch (resolved_arg) {
            .literal => |literal| allocator.free(@constCast(literal)),
            .boolean => {},
            .contract_ref => {},
        }
    }
}

/// Generalized arg resolution for multi-contract specs: `$contract` → the
/// call's target address (`self_address`), `@name` → the named contract's
/// deployed address (looked up in `addresses`). Used by executeSpec.
fn resolveAddressArgs(
    allocator: std.mem.Allocator,
    wires: []const []const u8,
    args: []const types.ArgValue,
    self_address: ?types.Address,
    addresses: *const std.StringHashMap(types.Address),
) ![]types.ArgValue {
    if (wires.len != args.len) return error.ArgumentCountMismatch;
    const resolved = try allocator.alloc(types.ArgValue, args.len);
    var initialized: usize = 0;
    errdefer {
        freeResolvedAddressLiterals(allocator, args[0..initialized], resolved[0..initialized]);
        allocator.free(resolved);
    }
    for (args, 0..) |arg, i| {
        resolved[i] = switch (arg) {
            .literal => |literal| if (std.mem.eql(u8, literal, "$contract")) blk: {
                if (!std.mem.eql(u8, wires[i], "address")) return error.UnsupportedArgType;
                const self = self_address orelse return error.NoSelfAddress;
                break :blk .{ .literal = try allocAddressLiteral(allocator, self) };
            } else arg,
            .contract_ref => |name| blk: {
                if (!std.mem.eql(u8, wires[i], "address")) return error.UnsupportedArgType;
                const addr = addresses.get(name) orelse return error.UnknownContractRef;
                break :blk .{ .literal = try allocAddressLiteral(allocator, addr) };
            },
            .boolean => arg,
        };
        initialized += 1;
    }
    return resolved;
}

fn deinitResolvedAddressArgs(allocator: std.mem.Allocator, original: []const types.ArgValue, resolved: []const types.ArgValue) void {
    freeResolvedAddressLiterals(allocator, original, resolved);
    allocator.free(resolved);
}

fn freeResolvedAddressLiterals(allocator: std.mem.Allocator, original: []const types.ArgValue, resolved: []const types.ArgValue) void {
    for (original, resolved) |original_arg, resolved_arg| {
        const was_resolved = switch (original_arg) {
            .literal => |literal| std.mem.eql(u8, literal, "$contract"),
            .contract_ref => true,
            .boolean => false,
        };
        if (!was_resolved) continue;
        switch (resolved_arg) {
            .literal => |literal| allocator.free(@constCast(literal)),
            else => {},
        }
    }
}

fn allocAddressLiteral(allocator: std.mem.Allocator, address: types.Address) ![]u8 {
    const hex_chars = "0123456789abcdef";
    const out = try allocator.alloc(u8, 42);
    out[0] = '0';
    out[1] = 'x';
    for (address.bytes, 0..) |byte, i| {
        out[2 + i * 2] = hex_chars[byte >> 4];
        out[3 + i * 2] = hex_chars[byte & 0x0f];
    }
    return out;
}

pub fn compileAndRunPropertySource(
    allocator: std.mem.Allocator,
    stem: []const u8,
    source: []const u8,
    comptime run: fn (*PropertyRuntime) anyerror!void,
) !void {
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var run_arena = std.heap.ArenaAllocator.init(allocator);
    defer run_arena.deinit();
    const arena = run_arena.allocator();

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "out", .default_dir);

    const source_name = try std.fmt.allocPrint(arena, "{s}.ora", .{stem});
    try tmp.dir.writeFile(io, .{ .sub_path = source_name, .data = source });

    const source_path = try pathFromTmpAlloc(arena, tmp, source_name);
    const out_path = try pathFromTmpAlloc(arena, tmp, "out");

    var bundle = try compileAndDeployPropertySource(arena, stem, source_path, out_path, &.{"--no-verify"});
    defer bundle.deinit();
    var runtime = bundle.runtime();
    try run(&runtime);
}

pub fn compileAndRunPropertySourceDifferential(
    allocator: std.mem.Allocator,
    stem: []const u8,
    source: []const u8,
    lhs_args: []const []const u8,
    rhs_args: []const []const u8,
    comptime run: fn (*PropertyRuntime, *PropertyRuntime) anyerror!void,
) !void {
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var run_arena = std.heap.ArenaAllocator.init(allocator);
    defer run_arena.deinit();
    const arena = run_arena.allocator();

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "lhs", .default_dir);
    try tmp.dir.createDir(io, "rhs", .default_dir);

    const source_name = try std.fmt.allocPrint(arena, "{s}.ora", .{stem});
    try tmp.dir.writeFile(io, .{ .sub_path = source_name, .data = source });

    const source_path = try pathFromTmpAlloc(arena, tmp, source_name);
    const lhs_out = try pathFromTmpAlloc(arena, tmp, "lhs");
    const rhs_out = try pathFromTmpAlloc(arena, tmp, "rhs");

    var lhs = try compileAndDeployPropertySource(arena, stem, source_path, lhs_out, lhs_args);
    defer lhs.deinit();
    var rhs = try compileAndDeployPropertySource(arena, stem, source_path, rhs_out, rhs_args);
    defer rhs.deinit();

    var lhs_runtime = lhs.runtime();
    var rhs_runtime = rhs.runtime();
    try run(&lhs_runtime, &rhs_runtime);
}

fn compileAndDeployPropertySource(
    arena: std.mem.Allocator,
    stem: []const u8,
    source_path: []const u8,
    out_path: []const u8,
    extra_args: []const []const u8,
) !PropertyRuntimeBundle {
    try runOraEmitWithExtraArgs(arena, source_path, out_path, extra_args);

    const artifacts = try readArtifacts(arena, out_path, stem);
    const doc = try arena.create(abi_doc.AbiDoc);
    doc.* = try abi_doc.AbiDoc.load(arena, artifacts.abi_path);
    errdefer doc.deinit();

    const caller = try types.Address.fromHex("0x1000000000000000000000000000000000000000");
    const host = try arena.create(HarnessHost);
    host.* = HarnessHost.init(arena);
    errdefer host.deinit();
    try host.setBalance(caller, std.math.maxInt(u256));
    try host.setNonce(caller, 1);

    const evm = try arena.create(types.Evm);
    try evm.init(arena, host.hostInterface(), .OSAKA, evm_mod.deterministicBlockContext(), caller, 0, null);
    errdefer evm.deinit();

    try evm.initTransactionState(null);
    try evm.preWarmTransaction(caller);

    const constructor_abi = try doc.findConstructor();
    defer constructor_abi.deinit(arena);
    const deploy_args = try abi.encodeArgs(arena, constructor_abi.inputs, &.{});

    var init_code: std.ArrayList(u8) = .empty;
    defer init_code.deinit(arena);
    try init_code.appendSlice(arena, artifacts.bytecode);
    try init_code.appendSlice(arena, deploy_args);

    const create_result = try evm.inner_create(0, init_code.items, types.DEFAULT_GAS, null);
    try host.check();
    try testing.expect(create_result.success);
    try testing.expect(host.getCodeForAddress(create_result.address).len > 0);

    return .{
        .allocator = arena,
        .host = host,
        .evm = evm,
        .doc = doc,
        .contract_address = create_result.address,
        .caller = caller,
    };
}
