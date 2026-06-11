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

        var calldata = std.ArrayList(u8){};
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
};

pub fn runOraEmit(allocator: std.mem.Allocator, source_path: []const u8, output_dir: []const u8) !void {
    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{
            types.ORA_BINARY_REL,
            "emit",
            "--no-verify",
            "--emit=abi,bytecode",
            "--output-dir",
            output_dir,
            source_path,
        },
        .max_output_bytes = 8 * 1024 * 1024,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| if (code != 0) {
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
    var skip_list = try loadSkipList(allocator, corpus_dir_path);
    defer skip_list.deinit(allocator);

    var dir = try std.fs.cwd().openDir(corpus_dir_path, .{ .iterate = true });
    defer dir.close();

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".ora")) continue;
        if (skip_list.contains(entry.name)) continue;

        const stem = entry.name[0 .. entry.name.len - ".ora".len];
        const spec_name = try std.fmt.allocPrint(allocator, "{s}.spec.toml", .{stem});
        defer allocator.free(spec_name);
        dir.access(spec_name, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.MissingSidecar,
            else => return err,
        };
    }
}

fn loadSkipList(allocator: std.mem.Allocator, corpus_dir_path: []const u8) !SkipList {
    const skip_path = try std.fs.path.join(allocator, &.{ corpus_dir_path, "SKIP" });
    defer allocator.free(skip_path);
    const buffer = std.fs.cwd().readFileAlloc(allocator, skip_path, 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => return .{},
        else => return err,
    };
    errdefer allocator.free(buffer);

    var entries = std.ArrayList([]const u8){};
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
    const hex = try std.fs.cwd().readFileAlloc(allocator, hex_path, 16 * 1024 * 1024);
    return .{ .bytecode = try abi.parseHexBytes(allocator, hex), .abi_path = abi_path };
}

pub fn runConformanceSpec(allocator: std.mem.Allocator, source_path: []const u8, spec_path: []const u8) !void {
    var run_arena = std.heap.ArenaAllocator.init(allocator);
    defer run_arena.deinit();
    const arena = run_arena.allocator();

    const spec_text = try std.fs.cwd().readFileAlloc(arena, spec_path, 1024 * 1024);
    var parsed = try spec_mod.parse(arena, spec_text);
    defer parsed.deinit();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makeDir("out");
    const out_path = try pathFromTmpAlloc(arena, tmp, "out");

    try runOraEmit(arena, source_path, out_path);

    const stem = sourceStem(source_path) orelse return error.InvalidSourcePath;
    const artifacts = try readArtifacts(arena, out_path, stem);

    var doc = try abi_doc.AbiDoc.load(arena, artifacts.abi_path);
    defer doc.deinit();
    try testing.expectError(error.UnknownFunction, doc.findFunction("missing()"));

    try executeSpec(arena, parsed.value, artifacts.bytecode, &doc);
}

fn sourceStem(source_path: []const u8) ?[]const u8 {
    const base = std.fs.path.basename(source_path);
    if (!std.mem.endsWith(u8, base, ".ora")) return null;
    return base[0 .. base.len - ".ora".len];
}

pub fn collectSpecNames(allocator: std.mem.Allocator, corpus_dir_path: []const u8) ![][]const u8 {
    var dir = try std.fs.cwd().openDir(corpus_dir_path, .{ .iterate = true });
    defer dir.close();

    var specs = std.ArrayList([]const u8){};
    errdefer {
        for (specs.items) |name| allocator.free(name);
        specs.deinit(allocator);
    }

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
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

fn executeSpec(allocator: std.mem.Allocator, spec: types.Spec, bytecode: []const u8, doc: *const abi_doc.AbiDoc) !void {
    var host = HarnessHost.init(allocator);
    defer host.deinit();
    try host.setBalance(spec.deploy.caller, std.math.maxInt(u256));
    try host.setNonce(spec.deploy.caller, 1);

    var evm: types.Evm = undefined;
    try evm.init(allocator, host.hostInterface(), .CANCUN, evm_mod.deterministicBlockContext(), spec.deploy.caller, 0, null);
    defer evm.deinit();

    try evm.initTransactionState(null);
    try evm.preWarmTransaction(spec.deploy.caller);
    const constructor_abi = try doc.findConstructor();
    defer constructor_abi.deinit(allocator);
    const deploy_args = try abi.encodeArgs(allocator, constructor_abi.inputs, spec.deploy.args);

    var init_code = std.ArrayList(u8){};
    defer init_code.deinit(allocator);
    try init_code.appendSlice(allocator, bytecode);
    try init_code.appendSlice(allocator, deploy_args);

    const create_result = try evm.inner_create(spec.deploy.value, init_code.items, types.DEFAULT_GAS, null);
    try host.check();
    try testing.expect(create_result.success);

    const contract_address = create_result.address;
    try testing.expect(host.getCodeForAddress(contract_address).len > 0);

    for (spec.calls) |call| {
        const function_abi = try doc.findFunction(call.@"fn");
        defer function_abi.deinit(allocator);
        const resolved_args = try resolveContractAddressArgs(allocator, function_abi.inputs, call.args, contract_address);
        defer deinitResolvedContractAddressArgs(allocator, call.args, resolved_args);
        const encoded_args = try abi.encodeArgs(allocator, function_abi.inputs, resolved_args);
        defer allocator.free(encoded_args);

        var calldata = std.ArrayList(u8){};
        defer calldata.deinit(allocator);
        try calldata.appendSlice(allocator, &function_abi.selector);
        try calldata.appendSlice(allocator, encoded_args);

        const result = evm.call(.{ .call = .{
            .caller = call.caller,
            .to = contract_address,
            .value = call.value,
            .input = calldata.items,
            .gas = types.DEFAULT_GAS,
        } });
        try host.check();

        switch (call.outcome) {
            .returns_empty => {
                try testing.expect(result.success);
                if (function_abi.outputs.len != 0) return error.UnsupportedReturnType;
                try testing.expectEqual(@as(usize, 0), result.output.len);
            },
            .returns_static => |expected| {
                try testing.expect(result.success);
                if (function_abi.outputs.len != 1) return error.UnsupportedReturnType;
                if (!abi.specTypeMatchesAbiWire(expected.spec_type, function_abi.outputs[0])) return error.UnsupportedReturnType;
                if (try abi.isSingleStaticWord(allocator, function_abi.outputs[0])) {
                    try testing.expectEqual(@as(usize, 32), result.output.len);
                    try abi.expectStaticReturn(function_abi.outputs[0], expected.value, result.output[0..32]);
                    continue;
                }
                const expected_args = [_]types.ArgValue{expected.value};
                const expected_output = try abi.encodeArgs(allocator, function_abi.outputs, &expected_args);
                defer allocator.free(expected_output);
                try testing.expectEqualSlices(u8, expected_output, result.output);
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

        try testing.expectEqual(call.logs.len, result.logs.len);
        for (call.logs, result.logs) |expected_log, actual_log| {
            try testing.expectEqualSlices(u256, expected_log.topics, actual_log.topics);
            try testing.expectEqualSlices(u8, expected_log.data, actual_log.data);
        }

        for (call.storage) |assertion| {
            try testing.expectEqual(assertion.value, host.getStorageSlot(contract_address, assertion.slot));
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
        };
        if (!std.mem.eql(u8, original_literal, "$contract")) continue;
        switch (resolved_arg) {
            .literal => |literal| allocator.free(@constCast(literal)),
            .boolean => {},
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
    std.fs.cwd().access(types.ORA_BINARY_REL, .{}) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };

    var run_arena = std.heap.ArenaAllocator.init(allocator);
    defer run_arena.deinit();
    const arena = run_arena.allocator();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.makeDir("out");

    const source_name = try std.fmt.allocPrint(arena, "{s}.ora", .{stem});
    try tmp.dir.writeFile(.{ .sub_path = source_name, .data = source });

    const source_path = try pathFromTmpAlloc(arena, tmp, source_name);
    const out_path = try pathFromTmpAlloc(arena, tmp, "out");

    try runOraEmit(arena, source_path, out_path);

    const artifacts = try readArtifacts(arena, out_path, stem);
    var doc = try abi_doc.AbiDoc.load(arena, artifacts.abi_path);
    defer doc.deinit();

    const caller = try types.Address.fromHex("0x1000000000000000000000000000000000000000");
    var host = HarnessHost.init(arena);
    defer host.deinit();
    try host.setBalance(caller, std.math.maxInt(u256));
    try host.setNonce(caller, 1);

    var evm: types.Evm = undefined;
    try evm.init(arena, host.hostInterface(), .CANCUN, evm_mod.deterministicBlockContext(), caller, 0, null);
    defer evm.deinit();

    try evm.initTransactionState(null);
    try evm.preWarmTransaction(caller);

    const constructor_abi = try doc.findConstructor();
    defer constructor_abi.deinit(arena);
    const deploy_args = try abi.encodeArgs(arena, constructor_abi.inputs, &.{});

    var init_code = std.ArrayList(u8){};
    defer init_code.deinit(arena);
    try init_code.appendSlice(arena, artifacts.bytecode);
    try init_code.appendSlice(arena, deploy_args);

    const create_result = try evm.inner_create(0, init_code.items, types.DEFAULT_GAS, null);
    try host.check();
    try testing.expect(create_result.success);
    try testing.expect(host.getCodeForAddress(create_result.address).len > 0);

    var runtime = PropertyRuntime{
        .allocator = arena,
        .host = &host,
        .evm = &evm,
        .abi = &doc,
        .contract_address = create_result.address,
        .caller = caller,
    };
    try run(&runtime);
}
