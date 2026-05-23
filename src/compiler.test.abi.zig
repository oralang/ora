const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const compiler = ora_root.compiler;
const mlir = @import("mlir_c_api").c;
const z3_verification = @import("ora_z3_verification");

const h = @import("compiler.test.helpers.zig");
const compileText = h.compileText;
const renderHirTextForSource = h.renderHirTextForSource;
const renderOraMlirForSource = h.renderOraMlirForSource;
const renderSirTextForModule = h.renderSirTextForModule;
const compilePackage = h.compilePackage;
const expectOraToSirConverts = h.expectOraToSirConverts;
const expectNoResidualOraRuntimeOps = h.expectNoResidualOraRuntimeOps;
const VerificationProbeSummary = h.VerificationProbeSummary;
const expectVerificationProbeEquivalent = h.expectVerificationProbeEquivalent;
const verifyExampleWithoutDegradation = h.verifyExampleWithoutDegradation;
const verifyTextWithoutDegradation = h.verifyTextWithoutDegradation;
const verifyTextWithoutDegradationWithTimeout = h.verifyTextWithoutDegradationWithTimeout;
const firstChildNodeOfKind = h.firstChildNodeOfKind;
const nthChildNodeOfKind = h.nthChildNodeOfKind;
const containsNodeOfKind = h.containsNodeOfKind;
const findVariablePatternByName = h.findVariablePatternByName;
const diagnosticMessagesContain = h.diagnosticMessagesContain;
const countDiagnosticMessages = h.countDiagnosticMessages;
const DiagnosticProbePhase = h.DiagnosticProbePhase;
const expectDiagnosticProbeContains = h.expectDiagnosticProbeContains;
const containsEffectSlot = h.containsEffectSlot;
const containsKeyedEffectSlot = h.containsKeyedEffectSlot;
const nthDescendantNodeOfKind = h.nthDescendantNodeOfKind;
const nthDescendantNodeOfKindInner = h.nthDescendantNodeOfKindInner;

fn expectAbiEncodeReturnBytes(source_text: []const u8, function_name: []const u8, expected_hex: []const u8) !void {
    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const value_index = try rootFunctionReturnValueIndex(&compilation, function_name);
    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.diagnostics.isEmpty());
    try expectHexBytes(expected_hex, consteval.values[value_index].?.fixed_bytes);
}

fn expectHexBytes(expected_hex: []const u8, actual: []const u8) !void {
    const hex = if (std.mem.startsWith(u8, expected_hex, "0x")) expected_hex[2..] else expected_hex;
    try testing.expectEqual(@as(usize, 0), hex.len % 2);
    const expected = try testing.allocator.alloc(u8, hex.len / 2);
    defer testing.allocator.free(expected);
    for (expected, 0..) |*byte, index| {
        const hi = try std.fmt.charToDigit(hex[index * 2], 16);
        const lo = try std.fmt.charToDigit(hex[index * 2 + 1], 16);
        byte.* = @intCast((hi << 4) | lo);
    }
    try testing.expectEqualSlices(u8, expected, actual);
}

fn expectedHexByteLen(expected_hex: []const u8) !usize {
    const hex = if (std.mem.startsWith(u8, expected_hex, "0x")) expected_hex[2..] else expected_hex;
    try testing.expectEqual(@as(usize, 0), hex.len % 2);
    return hex.len / 2;
}

fn functionSlice(sir_text: []const u8, function_name: []const u8) ![]const u8 {
    const header = try std.fmt.allocPrint(testing.allocator, "fn {s}:", .{function_name});
    defer testing.allocator.free(header);
    const start = std.mem.indexOf(u8, sir_text, header) orelse return error.TestUnexpectedResult;
    const search_from = start + header.len;
    const rel_end = std.mem.indexOfPos(u8, sir_text, search_from, "\nfn ");
    const end = rel_end orelse sir_text.len;
    return sir_text[start..end];
}

fn collectTokens(line: []const u8, out: [][]const u8) usize {
    var count: usize = 0;
    var it = std.mem.tokenizeAny(u8, line, " \t\r");
    while (it.next()) |token| {
        if (count >= out.len) break;
        out[count] = token;
        count += 1;
    }
    return count;
}

fn parseSirIntLiteral(token: []const u8) ?u256 {
    if (std.mem.startsWith(u8, token, "0x")) {
        return std.fmt.parseInt(u256, token[2..], 16) catch null;
    }
    return std.fmt.parseInt(u256, token, 10) catch null;
}

fn lowBitsMask(comptime T: type, bits: usize) T {
    if (bits >= @bitSizeOf(T)) return ~@as(T, 0);
    return (@as(T, 1) << @intCast(bits)) - 1;
}

fn signExtendU256(byte_index: u256, value: u256) u256 {
    if (byte_index >= 32) return value;
    const byte_index_usize: usize = @intCast(byte_index);
    const value_bits = (byte_index_usize + 1) * 8;
    if (value_bits >= 256) return value;
    const sign_bit_shift: u8 = @intCast(value_bits - 1);
    const low_mask = lowBitsMask(u256, value_bits);
    const low = value & low_mask;
    if (((value >> sign_bit_shift) & 1) == 0) return low;
    return low | ~low_mask;
}

fn valueForToken(values: *const std.StringHashMap(u256), token: []const u8) ?u256 {
    return parseSirIntLiteral(token) orelse values.get(token);
}

fn writeU256WordClipped(buffer: []u8, offset: usize, value: u256) void {
    for (0..32) |index| {
        const absolute = offset + index;
        if (absolute >= buffer.len) break;
        const shift: u8 = @intCast((31 - index) * 8);
        buffer[absolute] = @intCast((value >> shift) & 0xff);
    }
}

fn createOraMlirContext() mlir.MlirContext {
    const ctx = mlir.oraContextCreate();
    const registry = mlir.oraDialectRegistryCreate();
    mlir.oraRegisterAllDialects(registry);
    mlir.oraContextAppendDialectRegistry(ctx, registry);
    mlir.oraDialectRegistryDestroy(registry);
    mlir.oraContextLoadAllAvailableDialects(ctx);
    _ = mlir.oraDialectRegister(ctx);
    return ctx;
}

fn parseOraModule(ctx: mlir.MlirContext, text: []const u8) !mlir.MlirModule {
    const module = mlir.oraModuleCreateParse(ctx, mlir.oraStringRefCreate(text.ptr, text.len));
    if (mlir.oraModuleIsNull(module)) return error.TestUnexpectedResult;
    return module;
}

// This is a narrow interpreter for SIR emitted by the OraToSIR ABI materializer.
// Its operation semantics for const/large_const/add/mul/and/or/shl/signextend,
// mload256, and mstore256 must stay aligned with SIR runtime semantics.
fn extractAbiBytesFromSir(sir_text: []const u8, function_name: []const u8, payload_len: usize, prefix_len: usize) ![]u8 {
    const fn_text = try functionSlice(sir_text, function_name);
    const total_len = payload_len + prefix_len;

    const Pointer = struct {
        base: []const u8,
        offset: usize,
        size: ?usize = null,
    };
    const Store = struct {
        base: []const u8,
        offset: usize,
        value: u256,
    };
    const PointerStore = struct {
        base: []const u8,
        offset: usize,
        value: Pointer,
    };
    const SirBlock = struct {
        name: []const u8,
        inputs: []const []const u8,
        outputs: []const []const u8,
        first_line: usize,
        last_line: usize,
    };

    var values = std.StringHashMap(u256).init(testing.allocator);
    defer values.deinit();
    var pointers = std.StringHashMap(Pointer).init(testing.allocator);
    defer pointers.deinit();
    var stores = std.ArrayList(Store){};
    defer stores.deinit(testing.allocator);
    var pointer_stores = std.ArrayList(PointerStore){};
    defer pointer_stores.deinit(testing.allocator);
    var selected_base: ?[]const u8 = null;

    const Eval = struct {
        values: *std.StringHashMap(u256),
        pointers: *std.StringHashMap(Pointer),
        stores: *std.ArrayList(Store),
        pointer_stores: *std.ArrayList(PointerStore),
        selected_base: *?[]const u8,

        fn executeLine(self: *@This(), tokens: []const []const u8) !void {
            if (tokens.len == 0) return;

            if (std.mem.eql(u8, tokens[0], "mstore256") and tokens.len >= 3) {
                const ptr = self.pointers.get(tokens[1]) orelse return;
                if (valueForToken(self.values, tokens[2])) |value| {
                    try self.stores.append(testing.allocator, .{ .base = ptr.base, .offset = ptr.offset, .value = value });
                } else if (self.pointers.get(tokens[2])) |stored_ptr| {
                    try self.pointer_stores.append(testing.allocator, .{ .base = ptr.base, .offset = ptr.offset, .value = stored_ptr });
                }
                return;
            }

            if (std.mem.eql(u8, tokens[0], "mcopy") and tokens.len >= 4) {
                const dst = self.pointers.get(tokens[1]) orelse return;
                const src = self.pointers.get(tokens[2]) orelse return;
                const len = valueForToken(self.values, tokens[3]) orelse return;
                if (len > std.math.maxInt(usize)) return;
                const copy_len: usize = @intCast(len);
                var chunk_offset: usize = 0;
                while (chunk_offset < copy_len) : (chunk_offset += 32) {
                    var word: u256 = 0;
                    for (0..32) |byte_index| {
                        const relative = chunk_offset + byte_index;
                        word <<= 8;
                        if (relative >= copy_len) continue;
                        const absolute = src.offset + relative;
                        var store_index = self.stores.items.len;
                        while (store_index > 0) {
                            store_index -= 1;
                            const store = self.stores.items[store_index];
                            if (!std.mem.eql(u8, store.base, src.base)) continue;
                            if (absolute < store.offset or absolute >= store.offset + 32) continue;
                            const shift: u8 = @intCast((31 - (absolute - store.offset)) * 8);
                            word |= @intCast((store.value >> shift) & 0xff);
                            break;
                        }
                    }
                    try self.stores.append(testing.allocator, .{ .base = dst.base, .offset = dst.offset + chunk_offset, .value = word });
                }
                return;
            }

            if (tokens.len < 4 or !std.mem.eql(u8, tokens[1], "=")) return;
            const name = tokens[0];
            const op = tokens[2];

            if (std.mem.eql(u8, op, "const") or std.mem.eql(u8, op, "large_const")) {
                if (parseSirIntLiteral(tokens[3])) |value| try self.values.put(name, value);
                return;
            }

            if (std.mem.eql(u8, op, "malloc")) {
                const size = valueForToken(self.values, tokens[3]) orelse return;
                if (size > std.math.maxInt(usize)) return;
                try self.pointers.put(name, .{ .base = name, .offset = 0, .size = @intCast(size) });
                return;
            }

            if (std.mem.eql(u8, op, "mload256") and tokens.len >= 4) {
                const ptr = self.pointers.get(tokens[3]) orelse return;
                var pointer_store_index = self.pointer_stores.items.len;
                while (pointer_store_index > 0) {
                    pointer_store_index -= 1;
                    const store = self.pointer_stores.items[pointer_store_index];
                    if (std.mem.eql(u8, store.base, ptr.base) and store.offset == ptr.offset) {
                        try self.pointers.put(name, store.value);
                        break;
                    }
                }
                if (self.pointers.contains(name)) return;

                var index = self.stores.items.len;
                while (index > 0) {
                    index -= 1;
                    const store = self.stores.items[index];
                    if (std.mem.eql(u8, store.base, ptr.base) and store.offset == ptr.offset) {
                        try self.values.put(name, store.value);
                        break;
                    }
                }
                return;
            }

            if (std.mem.eql(u8, op, "add") and tokens.len >= 5) {
                if (self.pointers.get(tokens[3])) |base_ptr| {
                    if (valueForToken(self.values, tokens[4])) |offset| {
                        if (offset <= std.math.maxInt(usize)) {
                            try self.pointers.put(name, .{
                                .base = base_ptr.base,
                                .offset = base_ptr.offset + @as(usize, @intCast(offset)),
                                .size = null,
                            });
                        }
                    }
                } else if (valueForToken(self.values, tokens[3])) |lhs| {
                    if (valueForToken(self.values, tokens[4])) |rhs| {
                        try self.values.put(name, lhs +% rhs);
                    }
                }
                return;
            }

            if (std.mem.eql(u8, op, "and") and tokens.len >= 5) {
                const lhs = valueForToken(self.values, tokens[3]) orelse return;
                const rhs = valueForToken(self.values, tokens[4]) orelse return;
                try self.values.put(name, lhs & rhs);
                return;
            }

            if (std.mem.eql(u8, op, "or") and tokens.len >= 5) {
                const lhs = valueForToken(self.values, tokens[3]) orelse return;
                const rhs = valueForToken(self.values, tokens[4]) orelse return;
                try self.values.put(name, lhs | rhs);
                return;
            }

            if (std.mem.eql(u8, op, "mul") and tokens.len >= 5) {
                const lhs = valueForToken(self.values, tokens[3]) orelse return;
                const rhs = valueForToken(self.values, tokens[4]) orelse return;
                try self.values.put(name, lhs *% rhs);
                return;
            }

            if (std.mem.eql(u8, op, "div") and tokens.len >= 5) {
                const lhs = valueForToken(self.values, tokens[3]) orelse return;
                const rhs = valueForToken(self.values, tokens[4]) orelse return;
                if (rhs == 0) return;
                try self.values.put(name, lhs / rhs);
                return;
            }

            if (std.mem.eql(u8, op, "lt") and tokens.len >= 5) {
                const lhs = valueForToken(self.values, tokens[3]) orelse return;
                const rhs = valueForToken(self.values, tokens[4]) orelse return;
                try self.values.put(name, if (lhs < rhs) 1 else 0);
                return;
            }

            if (std.mem.eql(u8, op, "shl") and tokens.len >= 5) {
                const shift = valueForToken(self.values, tokens[3]) orelse return;
                const value = valueForToken(self.values, tokens[4]) orelse return;
                if (shift >= 256) {
                    try self.values.put(name, 0);
                } else {
                    try self.values.put(name, value << @intCast(shift));
                }
                return;
            }

            if (std.mem.eql(u8, op, "signextend") and tokens.len >= 5) {
                const byte_index = valueForToken(self.values, tokens[3]) orelse return;
                const value = valueForToken(self.values, tokens[4]) orelse return;
                try self.values.put(name, signExtendU256(byte_index, value));
                return;
            }

            if ((std.mem.eql(u8, op, "staticcall") or std.mem.eql(u8, op, "call")) and tokens.len >= 6) {
                if (self.pointers.get(tokens[5])) |ptr| self.selected_base.* = ptr.base;
                return;
            }
        }
    };

    var all_lines = std.ArrayList([]const u8){};
    defer all_lines.deinit(testing.allocator);
    var lines = std.mem.splitScalar(u8, fn_text, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        try all_lines.append(testing.allocator, line);
    }

    var blocks = std.ArrayList(SirBlock){};
    defer {
        for (blocks.items) |block| {
            testing.allocator.free(block.inputs);
            testing.allocator.free(block.outputs);
        }
        blocks.deinit(testing.allocator);
    }
    var line_index: usize = 0;
    while (line_index < all_lines.items.len) : (line_index += 1) {
        const line = all_lines.items[line_index];
        if (!std.mem.endsWith(u8, line, "{")) continue;
        var token_buf: [16][]const u8 = undefined;
        const count = collectTokens(line, &token_buf);
        const tokens = token_buf[0..count];
        if (tokens.len == 0 or std.mem.eql(u8, tokens[0], "fn")) continue;

        var arrow_index: ?usize = null;
        for (tokens, 0..) |token, index| {
            if (std.mem.eql(u8, token, "->")) {
                arrow_index = index;
                break;
            }
        }
        const header_end = if (std.mem.eql(u8, tokens[tokens.len - 1], "{")) tokens.len - 1 else tokens.len;
        const input_end = arrow_index orelse header_end;
        const output_start = if (arrow_index) |arrow| arrow + 1 else header_end;
        const inputs = try testing.allocator.dupe([]const u8, tokens[1..input_end]);
        const outputs = try testing.allocator.dupe([]const u8, tokens[output_start..header_end]);
        const first_line = line_index + 1;
        var last_line = first_line;
        while (last_line < all_lines.items.len and !std.mem.eql(u8, all_lines.items[last_line], "}")) : (last_line += 1) {}
        try blocks.append(testing.allocator, .{
            .name = tokens[0],
            .inputs = inputs,
            .outputs = outputs,
            .first_line = first_line,
            .last_line = last_line,
        });
    }

    var eval: Eval = .{
        .values = &values,
        .pointers = &pointers,
        .stores = &stores,
        .pointer_stores = &pointer_stores,
        .selected_base = &selected_base,
    };

    const Interpreter = struct {
        lines: []const []const u8,
        blocks: []const SirBlock,
        eval: *Eval,

        fn blockIndex(self: *const @This(), name: []const u8) ?usize {
            for (self.blocks, 0..) |block, index| {
                if (std.mem.eql(u8, block.name, name)) return index;
            }
            return null;
        }

        fn bindInputs(self: *@This(), block: SirBlock, args: []const []const u8) !void {
            if (args.len != block.inputs.len) return error.TestUnexpectedResult;
            for (block.inputs, args) |input, arg| {
                if (self.eval.pointers.get(arg)) |ptr| {
                    try self.eval.pointers.put(input, ptr);
                } else if (valueForToken(self.eval.values, arg)) |value| {
                    try self.eval.values.put(input, value);
                } else {
                    return error.TestUnexpectedResult;
                }
            }
        }

        fn outputArgsForTarget(self: *const @This(), block: SirBlock, target: []const u8) ![]const []const u8 {
            const target_index = self.blockIndex(target) orelse return error.TestUnexpectedResult;
            const needed = self.blocks[target_index].inputs.len;
            if (needed > block.outputs.len) return error.TestUnexpectedResult;
            return block.outputs[block.outputs.len - needed ..];
        }

        fn executeBlock(self: *@This(), block_index: usize, args: []const []const u8) !void {
            const block = self.blocks[block_index];
            try self.bindInputs(block, args);

            var body_line = block.first_line;
            while (body_line < block.last_line) : (body_line += 1) {
                var token_buf: [16][]const u8 = undefined;
                const count = collectTokens(self.lines[body_line], &token_buf);
                const tokens = token_buf[0..count];
                if (tokens.len == 0) continue;

                if (std.mem.eql(u8, tokens[0], "return")) return;

                if (std.mem.eql(u8, tokens[0], "=>")) {
                    if (tokens.len >= 2 and std.mem.startsWith(u8, tokens[1], "@")) {
                        const target = tokens[1][1..];
                        const target_index = self.blockIndex(target) orelse return error.TestUnexpectedResult;
                        return self.executeBlock(target_index, try self.outputArgsForTarget(block, target));
                    }

                    if (tokens.len >= 6 and std.mem.eql(u8, tokens[2], "?") and std.mem.startsWith(u8, tokens[3], "@") and std.mem.startsWith(u8, tokens[5], "@")) {
                        const condition = valueForToken(self.eval.values, tokens[1]) orelse return error.TestUnexpectedResult;
                        const target = if (condition != 0) tokens[3][1..] else tokens[5][1..];
                        const target_index = self.blockIndex(target) orelse return error.TestUnexpectedResult;
                        return self.executeBlock(target_index, try self.outputArgsForTarget(block, target));
                    }

                    return error.TestUnexpectedResult;
                }

                try self.eval.executeLine(tokens);
                if (self.eval.selected_base.* != null) return;
            }
        }
    };

    var interpreter: Interpreter = .{
        .lines = all_lines.items,
        .blocks = blocks.items,
        .eval = &eval,
    };
    const entry_index = interpreter.blockIndex("entry") orelse return error.TestUnexpectedResult;
    try interpreter.executeBlock(entry_index, &.{});

    if (selected_base == null) {
        var pointer_it = pointers.iterator();
        while (pointer_it.next()) |entry| {
            const ptr = entry.value_ptr.*;
            // The ABI materializer's output allocation is identified by its exact
            // byte size. If future lowering emits scratch allocations with the same
            // size, this selector should become call-site-aware.
            if (ptr.size != null and ptr.size.? == total_len) {
                selected_base = entry.key_ptr.*;
                break;
            }
        }
    }
    const base = selected_base orelse return error.TestUnexpectedResult;

    const calldata = try testing.allocator.alloc(u8, total_len);
    errdefer testing.allocator.free(calldata);
    @memset(calldata, 0);
    for (stores.items) |store| {
        if (!std.mem.eql(u8, store.base, base)) continue;
        writeU256WordClipped(calldata, store.offset, store.value);
    }

    const payload = try testing.allocator.dupe(u8, calldata[prefix_len..]);
    testing.allocator.free(calldata);
    return payload;
}

fn extractRuntimeAbiPayloadFromSir(sir_text: []const u8, function_name: []const u8, payload_len: usize) ![]u8 {
    return extractAbiBytesFromSir(sir_text, function_name, payload_len, 4);
}

fn rootFunctionReturnValueIndex(compilation: anytype, function_name: []const u8) !usize {
    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    for (ast_file.root_items) |item_id| {
        const item = ast_file.item(item_id).*;
        if (item != .Function or !std.mem.eql(u8, item.Function.name, function_name)) continue;
        const body = ast_file.body(item.Function.body);
        const ret_stmt = ast_file.statement(body.statements[0]).Return;
        return ret_stmt.value.?.index();
    }
    return error.TestUnexpectedResult;
}

fn expectRuntimeAbiPayloadMatchesComptimeAndCast(
    comptime_source_text: []const u8,
    runtime_source_text: []const u8,
    runtime_function_name: []const u8,
    comptime_function_name: []const u8,
    expected_hex: []const u8,
) !void {
    try expectAbiEncodeReturnBytes(comptime_source_text, comptime_function_name, expected_hex);

    var compilation = try compileText(runtime_source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const hir_result = try compilation.db.lowerToHir(compilation.root_module_id);
    try testing.expect(mlir.oraConvertToSIR(hir_result.context, hir_result.module.raw_module, false));

    const rendered = try renderSirTextForModule(hir_result.context, hir_result.module.raw_module);
    defer testing.allocator.free(rendered);

    const payload = try extractRuntimeAbiPayloadFromSir(rendered, runtime_function_name, try expectedHexByteLen(expected_hex));
    defer testing.allocator.free(payload);
    try expectHexBytes(expected_hex, payload);
}

test "compiler abi generation uses compiler pipeline for public abi" {
    const source_text =
        \\contract Test {
        \\    storage var counter: u256;
        \\    error InvalidAmount(amount: u256);
        \\    log Transfer(indexed from: address, amount: u256);
        \\    pub fn viewFn() -> u256 { return counter; }
        \\    pub fn writeFn(amount: u256) { counter = amount; }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    try testing.expectEqual(@as(usize, 1), contract_abi.contract_count);
    try testing.expectEqualStrings("Test", contract_abi.contract_name);

    var saw_view = false;
    var saw_write = false;
    var saw_error = false;
    var saw_event = false;

    for (contract_abi.callables) |callable| {
        if (callable.kind == .function and std.mem.eql(u8, callable.name, "viewFn")) {
            saw_view = true;
            try testing.expect(callable.selector != null);
        }
        if (callable.kind == .function and std.mem.eql(u8, callable.name, "writeFn")) {
            saw_write = true;
            try testing.expect(callable.selector != null);
        }
        if (callable.kind == .@"error" and std.mem.eql(u8, callable.name, "InvalidAmount")) {
            saw_error = true;
            try testing.expectEqual(@as(usize, 1), callable.inputs.len);
        }
        if (callable.kind == .event and std.mem.eql(u8, callable.name, "Transfer")) {
            saw_event = true;
            try testing.expect(callable.selector == null);
        }
    }

    try testing.expect(saw_view);
    try testing.expect(saw_write);
    try testing.expect(saw_error);
    try testing.expect(saw_event);
}

test "compiler abi emits uint256 for enum without declared repr" {
    const source_text =
        \\enum Status { Active, Paused }
        \\
        \\contract C {
        \\    pub fn set(status: Status) -> Status {
        \\        return status;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "set")) continue;
        const input = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("uint256", input.wire_type.?);
        try testing.expectEqualStrings("uint256", output.wire_type.?);
        return;
    }
    return error.TestUnexpectedResult;
}

test "compiler abi emits fixed bytes public ABI types" {
    const source_text =
        \\contract C {
        \\    pub fn set(a: bytes1, b: bytes4, c: bytes20, d: bytes31, e: bytes32, value: u256) -> bytes31 {
        \\        return d;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "set")) continue;
        try testing.expectEqual(@as(usize, 6), callable.inputs.len);
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);
        try testing.expectEqualStrings("0x36a0470c", callable.selector.?);
        const b1_input = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        const b4_input = contract_abi.findType(callable.inputs[1].type_id) orelse return error.TestUnexpectedResult;
        const b20_input = contract_abi.findType(callable.inputs[2].type_id) orelse return error.TestUnexpectedResult;
        const b31_input = contract_abi.findType(callable.inputs[3].type_id) orelse return error.TestUnexpectedResult;
        const b32_input = contract_abi.findType(callable.inputs[4].type_id) orelse return error.TestUnexpectedResult;
        const value_input = contract_abi.findType(callable.inputs[5].type_id) orelse return error.TestUnexpectedResult;
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("bytes1", b1_input.wire_type.?);
        try testing.expectEqualStrings("bytes4", b4_input.wire_type.?);
        try testing.expectEqualStrings("bytes20", b20_input.wire_type.?);
        try testing.expectEqualStrings("bytes31", b31_input.wire_type.?);
        try testing.expectEqualStrings("bytes32", b32_input.wire_type.?);
        try testing.expectEqualStrings("uint256", value_input.wire_type.?);
        try testing.expectEqualStrings("bytes31", output.wire_type.?);
        return;
    }
    return error.TestUnexpectedResult;
}

test "compiler abi emits registry-backed refinement predicates" {
    const source_text =
        \\contract C {
        \\    pub fn configure(rate: BasisPoints<u256>, amount: NonZero<u256>, owner: NonZeroAddress) -> BasisPoints<u256> {
        \\        return rate;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "configure")) continue;
        try testing.expectEqual(@as(usize, 3), callable.inputs.len);
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);

        const rate = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        const amount = contract_abi.findType(callable.inputs[1].type_id) orelse return error.TestUnexpectedResult;
        const owner = contract_abi.findType(callable.inputs[2].type_id) orelse return error.TestUnexpectedResult;
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;

        try testing.expectEqualStrings("{\"kind\":\"range\",\"min\":\"0\",\"max\":\"10000\"}", rate.predicate_json.?);
        try testing.expectEqualStrings("{\"kind\":\"nonZero\"}", amount.predicate_json.?);
        try testing.expectEqualStrings("{\"kind\":\"nonZeroAddress\"}", owner.predicate_json.?);
        try testing.expectEqualStrings("{\"kind\":\"range\",\"min\":\"0\",\"max\":\"10000\"}", output.predicate_json.?);
        try testing.expectEqualStrings("uint256", rate.wire_type.?);
        try testing.expectEqualStrings("uint256", amount.wire_type.?);
        try testing.expectEqualStrings("address", owner.wire_type.?);
        return;
    }
    return error.TestUnexpectedResult;
}

test "compiler abi keeps anonymous structs distinct from tuples" {
    const source_text =
        \\contract Test {
        \\    pub fn viewFn() -> struct { amount: u256, ok: bool } {
        \\        return .{ .amount = 1, .ok = true };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_struct_output = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "viewFn")) continue;
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(@as(usize, 2), output.fields.len);
        try testing.expectEqual(@as(usize, 0), output.components.len);
        try testing.expectEqualStrings("amount", output.fields[0].name);
        try testing.expectEqualStrings("ok", output.fields[1].name);
        saw_struct_output = true;
    }
    try testing.expect(saw_struct_output);
}

test "compiler abi preserves named struct return types" {
    const source_text =
        \\contract Test {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    pub fn viewFn() -> Pair {
        \\        return Pair { left: 1, right: true };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_struct_output = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "viewFn")) continue;
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);
        const output = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Pair", output.name.?);
        try testing.expectEqual(@as(usize, 2), output.fields.len);
        try testing.expectEqual(@as(usize, 0), output.components.len);
        try testing.expectEqualStrings("left", output.fields[0].name);
        try testing.expectEqualStrings("right", output.fields[1].name);
        saw_struct_output = true;
    }
    try testing.expect(saw_struct_output);
}

test "compiler abi preserves named struct parameter types" {
    const source_text =
        \\contract Test {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    pub fn consume(pair: Pair) -> bool {
        \\        return pair.right;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_struct_input = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "consume")) continue;
        try testing.expectEqual(@as(usize, 1), callable.inputs.len);
        const input = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Pair", input.name.?);
        try testing.expectEqual(@as(usize, 2), input.fields.len);
        try testing.expectEqual(@as(usize, 0), input.components.len);
        try testing.expectEqualStrings("left", input.fields[0].name);
        try testing.expectEqualStrings("right", input.fields[1].name);
        saw_struct_input = true;
    }
    try testing.expect(saw_struct_input);
}

test "compiler abi preserves nested named struct return types" {
    const source_text =
        \\contract Test {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    struct Wrapper {
        \\        pair: Pair,
        \\        ok: bool,
        \\    }
        \\
        \\    pub fn viewFn() -> Wrapper {
        \\        return Wrapper { pair: Pair { left: 1, right: true }, ok: true };
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_wrapper_output = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "viewFn")) continue;
        try testing.expectEqual(@as(usize, 1), callable.outputs.len);
        const wrapper = contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Wrapper", wrapper.name.?);
        try testing.expectEqual(@as(usize, 2), wrapper.fields.len);
        try testing.expectEqualStrings("pair", wrapper.fields[0].name);
        try testing.expectEqualStrings("ok", wrapper.fields[1].name);
        try testing.expectEqual(@as(usize, 0), wrapper.components.len);

        const pair = contract_abi.findType(wrapper.fields[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Pair", pair.name.?);
        try testing.expectEqual(@as(usize, 2), pair.fields.len);
        try testing.expectEqualStrings("left", pair.fields[0].name);
        try testing.expectEqualStrings("right", pair.fields[1].name);
        try testing.expectEqual(@as(usize, 0), pair.components.len);
        saw_wrapper_output = true;
    }
    try testing.expect(saw_wrapper_output);
}

test "compiler abi preserves nested named struct parameter types" {
    const source_text =
        \\contract Test {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    struct Wrapper {
        \\        pair: Pair,
        \\        ok: bool,
        \\    }
        \\
        \\    pub fn consume(wrapper: Wrapper) -> bool {
        \\        return wrapper.pair.right;
        \\    }
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    var contract_abi = try ora_root.abi.generateCompilerAbi(testing.allocator, &compilation);
    defer contract_abi.deinit();

    var saw_wrapper_input = false;
    for (contract_abi.callables) |callable| {
        if (callable.kind != .function or !std.mem.eql(u8, callable.name, "consume")) continue;
        try testing.expectEqual(@as(usize, 1), callable.inputs.len);
        const wrapper = contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Wrapper", wrapper.name.?);
        try testing.expectEqual(@as(usize, 2), wrapper.fields.len);
        try testing.expectEqualStrings("pair", wrapper.fields[0].name);
        try testing.expectEqualStrings("ok", wrapper.fields[1].name);
        try testing.expectEqual(@as(usize, 0), wrapper.components.len);

        const pair = contract_abi.findType(wrapper.fields[0].type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings("Pair", pair.name.?);
        try testing.expectEqual(@as(usize, 2), pair.fields.len);
        try testing.expectEqualStrings("left", pair.fields[0].name);
        try testing.expectEqualStrings("right", pair.fields[1].name);
        try testing.expectEqual(@as(usize, 0), pair.components.len);
        saw_wrapper_input = true;
    }
    try testing.expect(saw_wrapper_input);
}

test "compiler const eval supports typeName and string concat for ABI signatures" {
    const source_text =
        \\pub fn run() -> string {
        \\    return comptime {
        \\        "transfer(" + @typeName(address) + "," + @typeName(u256) + ")";
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(std.mem.eql(u8, "transfer(address,uint256)", consteval.values[ret_stmt.value.?.index()].?.string));
}

test "compiler const eval supports keccak256 for ABI selector hashes" {
    const source_text =
        \\pub fn run() -> u256 {
        \\    return comptime {
        \\        @keccak256("transfer(" + @typeName(address) + "," + @typeName(u256) + ")");
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    var hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("transfer(address,uint256)", &hash, .{});
    var hex: [64]u8 = undefined;
    for (hash, 0..) |byte, index| {
        hex[index * 2] = std.fmt.hex_charset[byte >> 4];
        hex[index * 2 + 1] = std.fmt.hex_charset[byte & 0x0f];
    }
    var expected = try std.math.big.int.Managed.init(testing.allocator);
    defer expected.deinit();
    try expected.setString(16, hex[0..]);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.values[ret_stmt.value.?.index()].?.integer.eql(expected));
}

test "compiler abiEncode encodes static scalar tuple matching Solidity abi.encode" {
    const source_text =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const selector: bytes4 = hex"abcdef12";
        \\        @abiEncode((@cast(u256, 1), true, 0x1234567890abcdef1234567890abcdef12345678, selector));
        \\    };
        \\}
    ;

    // cast abi-encode "f(uint256,bool,address,bytes4)" 1 true 0x1234567890abcdef1234567890abcdef12345678 0xabcdef12
    try expectAbiEncodeReturnBytes(source_text, "run", "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000001234567890abcdef1234567890abcdef12345678" ++
        "abcdef1200000000000000000000000000000000000000000000000000000000");
}

test "compiler abiEncode sign-extends signed integers" {
    const neg_one_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        @abiEncode(@cast(i256, -1));
        \\    };
        \\}
    ;
    // cast abi-encode "f(int256)" -- -1
    try expectAbiEncodeReturnBytes(neg_one_source, "run", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");

    const min_i8_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        @abiEncode(@cast(i8, -128));
        \\    };
        \\}
    ;
    // cast abi-encode "f(int8)" -- -128
    try expectAbiEncodeReturnBytes(min_i8_source, "run", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff80");

    const positive_i16_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        @abiEncode(@cast(i16, 258));
        \\    };
        \\}
    ;
    // cast abi-encode "f(int16)" 258
    try expectAbiEncodeReturnBytes(positive_i16_source, "run", "0000000000000000000000000000000000000000000000000000000000000102");
}

test "compiler abiEncode encodes static arrays structs enum bitfield and empty values" {
    const array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const values: [u256; 2] = [1, 2];
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(uint256[2])" "[1,2]"
    try expectAbiEncodeReturnBytes(array_source, "run", "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000002");

    const struct_source =
        \\struct Pair {
        \\    amount: u256,
        \\    ok: bool,
        \\}
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const value: Pair = Pair { amount: 7, ok: true };
        \\        @abiEncode(value);
        \\    };
        \\}
    ;
    // cast abi-encode "f((uint256,bool))" "(7,true)"
    try expectAbiEncodeReturnBytes(struct_source, "run", "0000000000000000000000000000000000000000000000000000000000000007" ++
        "0000000000000000000000000000000000000000000000000000000000000001");

    const enum_source =
        \\enum Status: u8 { Active, Paused }
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        @abiEncode(Status.Paused);
        \\    };
        \\}
    ;
    // cast abi-encode "f(uint8)" 1
    try expectAbiEncodeReturnBytes(enum_source, "run", "0000000000000000000000000000000000000000000000000000000000000001");

    const bitfield_source =
        \\bitfield Flags: u8 {
        \\    enabled: u1,
        \\    mode: u7,
        \\}
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        @abiEncode(@cast(Flags, 3));
        \\    };
        \\}
    ;
    // cast abi-encode "f(uint8)" 3
    try expectAbiEncodeReturnBytes(bitfield_source, "run", "0000000000000000000000000000000000000000000000000000000000000003");

    const empty_tuple_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        @abiEncode(());
        \\    };
        \\}
    ;
    // cast abi-encode "f()"
    try expectAbiEncodeReturnBytes(empty_tuple_source, "run", "");

    const void_source =
        \\fn noop() {}
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        @abiEncode(noop());
        \\    };
        \\}
    ;
    // void values encode to the same empty payload as a zero-argument ABI tuple.
    try expectAbiEncodeReturnBytes(void_source, "run", "");
}

test "compiler abiEncode single element tuple matches bare value" {
    const bare_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        @abiEncode(@cast(u256, 5));
        \\    };
        \\}
    ;
    const tuple_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        @abiEncode((@cast(u256, 5),));
        \\    };
        \\}
    ;
    const expected = "0000000000000000000000000000000000000000000000000000000000000005";
    // cast abi-encode "f(uint256)" 5
    try expectAbiEncodeReturnBytes(bare_source, "run", expected);
    // cast abi-encode "f(uint256)" 5; a single-element tuple is flattened by ABI static encoding.
    try expectAbiEncodeReturnBytes(tuple_source, "run", expected);
}

test "compiler abiEncode strips refinements to base ABI type" {
    const source_text =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const refined: MinValue<u256, 1> = @cast(MinValue<u256, 1>, 5);
        \\        @abiEncode(refined);
        \\    };
        \\}
    ;

    // cast abi-encode "f(uint256)" 5
    try expectAbiEncodeReturnBytes(source_text, "run", "0000000000000000000000000000000000000000000000000000000000000005");
}

test "compiler abiEncode encodes dynamic string and bytes values" {
    const empty_string_source =
        \\pub fn run() -> bytes {
        \\    return comptime { @abiEncode(""); };
        \\}
    ;
    // cast abi-encode "f(string)" ""
    try expectAbiEncodeReturnBytes(empty_string_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000000");

    const hello_source =
        \\pub fn run() -> bytes {
        \\    return comptime { @abiEncode("hello"); };
        \\}
    ;
    // cast abi-encode "f(string)" "hello"
    try expectAbiEncodeReturnBytes(hello_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000");

    const single_tuple_source =
        \\pub fn run() -> bytes {
        \\    return comptime { @abiEncode(("hello",)); };
        \\}
    ;
    // cast abi-encode "f(string)" "hello"; a single-element tuple is flattened by ABI dynamic encoding.
    try expectAbiEncodeReturnBytes(single_tuple_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000");

    const utf8_source =
        \\pub fn run() -> bytes {
        \\    return comptime { @abiEncode("hé"); };
        \\}
    ;
    // cast abi-encode "f(string)" "hé"
    try expectAbiEncodeReturnBytes(utf8_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000003" ++
        "68c3a90000000000000000000000000000000000000000000000000000000000");

    const bytes_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"deadbeef";
        \\        @abiEncode(payload);
        \\    };
        \\}
    ;
    // cast abi-encode "f(bytes)" 0xdeadbeef
    try expectAbiEncodeReturnBytes(bytes_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const empty_bytes_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"";
        \\        @abiEncode(payload);
        \\    };
        \\}
    ;
    // cast abi-encode "f(bytes)" 0x
    try expectAbiEncodeReturnBytes(empty_bytes_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000000");

    const aligned_bytes_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
        \\        @abiEncode(payload);
        \\    };
        \\}
    ;
    // cast abi-encode "f(bytes)" 0x000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
    try expectAbiEncodeReturnBytes(aligned_bytes_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000020" ++
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");

    const unaligned_bytes_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
        \\        @abiEncode(payload);
        \\    };
        \\}
    ;
    // cast abi-encode "f(bytes)" 0x000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20
    try expectAbiEncodeReturnBytes(unaligned_bytes_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000021" ++
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20" ++
        "00000000000000000000000000000000000000000000000000000000000000");
}

test "compiler abiEncode encodes mixed static and dynamic tuples" {
    const one_dynamic_source =
        \\pub fn run() -> bytes {
        \\    return comptime { @abiEncode((@cast(u256, 1), "hello")); };
        \\}
    ;
    // cast abi-encode "f(uint256,string)" 1 "hello"
    try expectAbiEncodeReturnBytes(one_dynamic_source, "run", "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000");

    const alternating_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"deadbeef";
        \\        @abiEncode(("hello", @cast(u256, 7), payload));
        \\    };
        \\}
    ;
    // cast abi-encode "f(string,uint256,bytes)" "hello" 7 0xdeadbeef
    try expectAbiEncodeReturnBytes(alternating_source, "run", "0000000000000000000000000000000000000000000000000000000000000060" ++
        "0000000000000000000000000000000000000000000000000000000000000007" ++
        "00000000000000000000000000000000000000000000000000000000000000a0" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const empty_dynamic_pair_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"";
        \\        @abiEncode(("", payload));
        \\    };
        \\}
    ;
    // cast abi-encode "f(string,bytes)" "" 0x
    try expectAbiEncodeReturnBytes(empty_dynamic_pair_source, "run", "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000060" ++
        "0000000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000000");

    const nested_dynamic_source =
        \\pub fn run() -> bytes {
        \\    return comptime { @abiEncode((@cast(u256, 1), ("hello", true))); };
        \\}
    ;
    // cast abi-encode "f(uint256,(string,bool))" 1 "(hello,true)"
    try expectAbiEncodeReturnBytes(nested_dynamic_source, "run", "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000");
}

test "compiler abiEncode encodes dynamic arrays with static elements" {
    const uint_array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[u256], [1, 2, 3]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(uint256[])" "[1,2,3]"
    try expectAbiEncodeReturnBytes(uint_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000003" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000003");

    const address_array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[address], [0x0000000000000000000000000000000000000001, 0x1234567890abcdef1234567890abcdef12345678]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(address[])" "[0x0000000000000000000000000000000000000001,0x1234567890abcdef1234567890abcdef12345678]"
    try expectAbiEncodeReturnBytes(address_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000001234567890abcdef1234567890abcdef12345678");

    const mixed_tuple_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[u256], [4, 5]);
        \\        @abiEncode((@cast(u256, 9), values));
        \\    };
        \\}
    ;
    // cast abi-encode "f(uint256,uint256[])" 9 "[4,5]"
    try expectAbiEncodeReturnBytes(mixed_tuple_source, "run", "0000000000000000000000000000000000000000000000000000000000000009" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "0000000000000000000000000000000000000000000000000000000000000005");
}

test "compiler abiEncode encodes dynamic arrays with dynamic elements" {
    const string_array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[string], ["a", "bb"]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(string[])" '["a","bb"]'
    try expectAbiEncodeReturnBytes(string_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000080" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000");

    const bytes_array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const first: bytes = hex"";
        \\        const second: bytes = hex"deadbeef";
        \\        const values = @cast(slice[bytes], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(bytes[])" "[0x,0xdeadbeef]"
    try expectAbiEncodeReturnBytes(bytes_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000060" ++
        "0000000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const nested_uint_array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const first = @cast(slice[u256], [1, 2]);
        \\        const second = @cast(slice[u256], [3]);
        \\        const values = @cast(slice[slice[u256]], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(uint256[][])" "[[1,2],[3]]"
    try expectAbiEncodeReturnBytes(nested_uint_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000a0" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000003");

    const fixed_string_array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const values: [string; 3] = ["a", "bb", "ccc"];
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(string[3])" '["a","bb","ccc"]'
    try expectAbiEncodeReturnBytes(fixed_string_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000060" ++
        "00000000000000000000000000000000000000000000000000000000000000a0" ++
        "00000000000000000000000000000000000000000000000000000000000000e0" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000003" ++
        "6363630000000000000000000000000000000000000000000000000000000000");

    const dynamic_tuple_array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const first = (@cast(u256, 1), "a");
        \\        const second = (@cast(u256, 2), "bb");
        \\        const values = @cast(slice[(u256, string)], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f((uint256,string)[])" '[(1,"a"),(2,"bb")]'
    try expectAbiEncodeReturnBytes(dynamic_tuple_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000c0" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000");
}

test "compiler abiEncode encodes M6 nested dynamic aggregates" {
    const nested_tuple_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"deadbeef";
        \\        @abiEncode((("hello", @cast(u256, 7)), payload));
        \\    };
        \\}
    ;
    // cast abi-encode "f((string,uint256),bytes)" "(hello,7)" 0xdeadbeef
    try expectAbiEncodeReturnBytes(nested_tuple_source, "run", "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000c0" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000007" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const deeply_nested_tuple_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"deadbeef";
        \\        @abiEncode(((((("hello", @cast(u256, 7)), payload), true), @cast(u256, 9)), false));
        \\    };
        \\}
    ;
    // cast abi-encode "f(((((string,uint256),bytes),bool),uint256),bool)" "((((hello,7),0xdeadbeef),true),9)" false
    try expectAbiEncodeReturnBytes(deeply_nested_tuple_source, "run", "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000009" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000c0" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000007" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const nested_string_arrays_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const first = @cast(slice[string], ["a", "bb"]);
        \\        const second = @cast(slice[string], []);
        \\        const values = @cast(slice[slice[string]], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(string[][])" "[[a,bb],[]]"
    try expectAbiEncodeReturnBytes(nested_string_arrays_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000120" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000080" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000000");

    const empty_string_array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[string], []);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(string[])" "[]"
    try expectAbiEncodeReturnBytes(empty_string_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000000");

    const single_empty_string_array_source =
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[string], [""]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f(string[])" '[""]'
    try expectAbiEncodeReturnBytes(single_empty_string_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000000");

    const dynamic_struct_source =
        \\struct Profile {
        \\    id: u256,
        \\    name: string,
        \\}
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const value: Profile = Profile { id: 1, name: "hello" };
        \\        @abiEncode(value);
        \\    };
        \\}
    ;
    // cast abi-encode "f((uint256,string))" "(1,hello)"
    try expectAbiEncodeReturnBytes(dynamic_struct_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000");

    const dynamic_struct_array_source =
        \\struct Profile {
        \\    id: u256,
        \\    name: string,
        \\}
        \\pub fn run() -> bytes {
        \\    return comptime {
        \\        const first: Profile = Profile { id: 1, name: "a" };
        \\        const second: Profile = Profile { id: 2, name: "bb" };
        \\        const values = @cast(slice[Profile], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    // cast abi-encode "f((uint256,string)[])" "[(1,a),(2,bb)]"
    try expectAbiEncodeReturnBytes(dynamic_struct_array_source, "run", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000c0" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000");
}

test "compiler runtime static ABI materializer matches comptime abiEncode and cast" {
    const scalar_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const tag: bytes4 = hex"abcdef12";
        \\        @abiEncode((@cast(u256, 1), true, 0x1234567890abcdef1234567890abcdef12345678, tag));
        \\    };
        \\}
    ;
    const scalar_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, amount: u256, ok: bool, owner: address, tag: bytes4) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const tag: bytes4 = hex"abcdef12";
        \\        return external<Sink>(target, gas: 50000).submit(@cast(u256, 1), true, 0x1234567890abcdef1234567890abcdef12345678, tag);
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint256,bool,address,bytes4)" 1 true 0x1234567890abcdef1234567890abcdef12345678 0xabcdef12
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(scalar_comptime_source, scalar_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000001234567890abcdef1234567890abcdef12345678" ++
        "abcdef1200000000000000000000000000000000000000000000000000000000");

    const signed_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode(@cast(i8, -128)); };
        \\}
    ;
    const signed_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, value: i8) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).submit(@cast(i8, -128));
        \\    }
        \\}
    ;
    // cast abi-encode "f(int8)" -- -128
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(signed_comptime_source, signed_runtime_source, "probe", "expected", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff80");

    const neg_i256_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode(@cast(i256, -1)); };
        \\}
    ;
    const neg_i256_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, value: i256) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).submit(@cast(i256, -1));
        \\    }
        \\}
    ;
    // cast abi-encode "f(int256)" -- -1
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(neg_i256_comptime_source, neg_i256_runtime_source, "probe", "expected", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");

    const positive_i16_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode(@cast(i16, 258)); };
        \\}
    ;
    const positive_i16_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, value: i16) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).submit(@cast(i16, 258));
        \\    }
        \\}
    ;
    // cast abi-encode "f(int16)" 258
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(positive_i16_comptime_source, positive_i16_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000102");

    const array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values: [u256; 2] = [1, 2];
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: [u256; 2]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values: [u256; 2] = [1, 2];
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint256[2])" "[1,2]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(array_comptime_source, array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000002");

    const struct_comptime_source =
        \\struct Pair {
        \\    amount: u256,
        \\    ok: bool,
        \\}
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const pair: Pair = Pair { amount: 7, ok: true };
        \\        @abiEncode(pair);
        \\    };
        \\}
    ;
    const struct_runtime_source =
        \\struct Pair {
        \\    amount: u256,
        \\    ok: bool,
        \\}
        \\extern trait Sink {
        \\    staticcall fn submit(self, pair: Pair) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const pair: Pair = Pair { amount: 7, ok: true };
        \\        return external<Sink>(target, gas: 50000).submit(pair);
        \\    }
        \\}
    ;
    // cast abi-encode "f((uint256,bool))" "(7,true)"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(struct_comptime_source, struct_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000007" ++
        "0000000000000000000000000000000000000000000000000000000000000001");

    const enum_comptime_source =
        \\enum Status: u8 { Active, Paused }
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode(Status.Paused); };
        \\}
    ;
    const enum_runtime_source =
        \\enum Status: u8 { Active, Paused }
        \\extern trait Sink {
        \\    staticcall fn submit(self, status: Status) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).submit(Status.Paused);
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint8)" 1
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(enum_comptime_source, enum_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000001");

    const bitfield_comptime_source =
        \\bitfield Flags: u8 {
        \\    enabled: u1,
        \\    mode: u7,
        \\}
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode(@cast(Flags, 3)); };
        \\}
    ;
    const bitfield_runtime_source =
        \\bitfield Flags: u8 {
        \\    enabled: u1,
        \\    mode: u7,
        \\}
        \\extern trait Sink {
        \\    staticcall fn submit(self, flags: Flags) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).submit(@cast(Flags, 3));
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint8)" 3
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(bitfield_comptime_source, bitfield_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000003");

    const empty_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode(()); };
        \\}
    ;
    const empty_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn ping(self) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).ping();
        \\    }
        \\}
    ;
    // cast abi-encode "f()"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(empty_comptime_source, empty_runtime_source, "probe", "expected", "");

    const void_comptime_source =
        \\fn noop() {}
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode(noop()); };
        \\}
    ;
    const void_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, value: void) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    fn noop() {}
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).submit(noop());
        \\    }
        \\}
    ;
    // void values encode to the same empty payload as a zero-argument ABI tuple.
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(void_comptime_source, void_runtime_source, "probe", "expected", "");

    const single_tuple_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode((@cast(u256, 5),)); };
        \\}
    ;
    const single_tuple_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, value: u256) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).submit(@cast(u256, 5));
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint256)" 5. Ora's extern trait parameter syntax does
    // not expose a distinct one-element tuple type, but both this runtime path
    // and @abiEncode((value,)) serialize through tuple(static(uint256)).
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(single_tuple_comptime_source, single_tuple_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000005");

    const refinement_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const refined: MinValue<u256, 1> = @cast(MinValue<u256, 1>, 5);
        \\        @abiEncode(refined);
        \\    };
        \\}
    ;
    const refinement_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, value: MinValue<u256, 1>) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const refined: MinValue<u256, 1> = @cast(MinValue<u256, 1>, 5);
        \\        return external<Sink>(target, gas: 50000).submit(refined);
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint256)" 5
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(refinement_comptime_source, refinement_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000005");
}

test "compiler plain runtime static ABI encode op matches cast" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  ora.contract @C {
        \\    func.func @encode() {
        \\      %value = arith.constant 5 : i256
        \\      %encoded = "ora.abi_encode"(%value) {layout = "tuple(static(uint256))"} : (i256) -> !ora.int<256, false>
        \\      ora.return
        \\    }
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraConvertToSIR(ctx, module, false));

    const rendered = try renderSirTextForModule(ctx, module);
    defer testing.allocator.free(rendered);

    const expected = "0000000000000000000000000000000000000000000000000000000000000005";
    const payload = try extractAbiBytesFromSir(rendered, "encode", try expectedHexByteLen(expected), 0);
    defer testing.allocator.free(payload);
    // cast abi-encode "f(uint256)" 5
    try expectHexBytes(expected, payload);
}

test "compiler runtime dynamic ABI materializer matches comptime abiEncode and cast" {
    const string_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode("hello"); };
        \\}
    ;
    const string_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, message: string) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).submit("hello");
        \\    }
        \\}
    ;
    // cast abi-encode "f(string)" "hello"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(string_comptime_source, string_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000");

    const bytes_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"deadbeef";
        \\        @abiEncode(payload);
        \\    };
        \\}
    ;
    const bytes_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, payload: bytes) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const payload: bytes = hex"deadbeef";
        \\        return external<Sink>(target, gas: 50000).submit(payload);
        \\    }
        \\}
    ;
    // cast abi-encode "f(bytes)" 0xdeadbeef
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(bytes_comptime_source, bytes_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const mixed_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"deadbeef";
        \\        @abiEncode(("hello", @cast(u256, 7), payload));
        \\    };
        \\}
    ;
    const mixed_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, message: string, value: u256, payload: bytes) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const payload: bytes = hex"deadbeef";
        \\        return external<Sink>(target, gas: 50000).submit("hello", @cast(u256, 7), payload);
        \\    }
        \\}
    ;
    // cast abi-encode "f(string,uint256,bytes)" "hello" 7 0xdeadbeef
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(mixed_comptime_source, mixed_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000060" ++
        "0000000000000000000000000000000000000000000000000000000000000007" ++
        "00000000000000000000000000000000000000000000000000000000000000a0" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const static_array_dynamic_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values: [u256; 2] = [1, 2];
        \\        @abiEncode((values, "hello"));
        \\    };
        \\}
    ;
    const static_array_dynamic_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: [u256; 2], message: string) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values: [u256; 2] = [1, 2];
        \\        return external<Sink>(target, gas: 50000).submit(values, "hello");
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint256[2],string)" "[1,2]" "hello"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(static_array_dynamic_comptime_source, static_array_dynamic_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000060" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000");

    const nested_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime { @abiEncode((@cast(u256, 1), ("hello", true))); };
        \\}
    ;
    const nested_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, amount: u256, pair: (string, bool)) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        return external<Sink>(target, gas: 50000).submit(@cast(u256, 1), ("hello", true));
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint256,(string,bool))" 1 "(hello,true)"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(nested_comptime_source, nested_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000");
}

test "compiler runtime dynamic array ABI materializer matches comptime abiEncode and cast" {
    const uint_array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[u256], [1, 2, 3]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const uint_array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[u256]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicArrayParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values = @cast(slice[u256], [1, 2, 3]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint256[])" "[1,2,3]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(uint_array_comptime_source, uint_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000003" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000003");

    const address_array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[address], [0x0000000000000000000000000000000000000001, 0x1234567890abcdef1234567890abcdef12345678]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const address_array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[address]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicArrayParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values = @cast(slice[address], [0x0000000000000000000000000000000000000001, 0x1234567890abcdef1234567890abcdef12345678]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(address[])" "[0x0000000000000000000000000000000000000001,0x1234567890abcdef1234567890abcdef12345678]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(address_array_comptime_source, address_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000001234567890abcdef1234567890abcdef12345678");

    const mixed_tuple_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[u256], [4, 5]);
        \\        @abiEncode((@cast(u256, 9), values));
        \\    };
        \\}
    ;
    const mixed_tuple_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, amount: u256, values: slice[u256]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicArrayParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values = @cast(slice[u256], [4, 5]);
        \\        return external<Sink>(target, gas: 50000).submit(@cast(u256, 9), values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint256,uint256[])" 9 "[4,5]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(mixed_tuple_comptime_source, mixed_tuple_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000009" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "0000000000000000000000000000000000000000000000000000000000000005");

    const pair_array_comptime_source =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[Pair], [
        \\            Pair { left: 1, right: 2 },
        \\            Pair { left: 3, right: 4 },
        \\        ]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const pair_array_runtime_source =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[Pair]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicArrayParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values = @cast(slice[Pair], [
        \\            Pair { left: 1, right: 2 },
        \\            Pair { left: 3, right: 4 },
        \\        ]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f((uint256,uint256)[])" "[(1,2),(3,4)]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(pair_array_comptime_source, pair_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000003" ++
        "0000000000000000000000000000000000000000000000000000000000000004");

    const string_array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[string], ["a", "bb"]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const string_array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[string]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicArrayParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values = @cast(slice[string], ["a", "bb"]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(string[])" '["a","bb"]'
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(string_array_comptime_source, string_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000080" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000");

    const bytes_array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const first: bytes = hex"";
        \\        const second: bytes = hex"deadbeef";
        \\        const values = @cast(slice[bytes], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const bytes_array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[bytes]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicArrayParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const first: bytes = hex"";
        \\        const second: bytes = hex"deadbeef";
        \\        const values = @cast(slice[bytes], [first, second]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(bytes[])" "[0x,0xdeadbeef]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(bytes_array_comptime_source, bytes_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000060" ++
        "0000000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const nested_uint_array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const first = @cast(slice[u256], [1, 2]);
        \\        const second = @cast(slice[u256], [3]);
        \\        const values = @cast(slice[slice[u256]], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const nested_uint_array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[slice[u256]]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicArrayParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const first = @cast(slice[u256], [1, 2]);
        \\        const second = @cast(slice[u256], [3]);
        \\        const values = @cast(slice[slice[u256]], [first, second]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(uint256[][])" "[[1,2],[3]]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(nested_uint_array_comptime_source, nested_uint_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000a0" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000003");

    const fixed_string_array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values: [string; 3] = ["a", "bb", "ccc"];
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const fixed_string_array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: [string; 3]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicArrayParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values: [string; 3] = ["a", "bb", "ccc"];
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(string[3])" '["a","bb","ccc"]'
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(fixed_string_array_comptime_source, fixed_string_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000060" ++
        "00000000000000000000000000000000000000000000000000000000000000a0" ++
        "00000000000000000000000000000000000000000000000000000000000000e0" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000003" ++
        "6363630000000000000000000000000000000000000000000000000000000000");

    const dynamic_tuple_array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const first = (@cast(u256, 1), "a");
        \\        const second = (@cast(u256, 2), "bb");
        \\        const values = @cast(slice[(u256, string)], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const dynamic_tuple_array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[(u256, string)]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiDynamicArrayParity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const first = (@cast(u256, 1), "a");
        \\        const second = (@cast(u256, 2), "bb");
        \\        const values = @cast(slice[(u256, string)], [first, second]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f((uint256,string)[])" '[(1,"a"),(2,"bb")]'
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(dynamic_tuple_array_comptime_source, dynamic_tuple_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000c0" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000");
}

test "compiler runtime M6 nested dynamic aggregates match comptime abiEncode and cast" {
    const nested_tuple_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"deadbeef";
        \\        @abiEncode((("hello", @cast(u256, 7)), payload));
        \\    };
        \\}
    ;
    const nested_tuple_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, inner: (string, u256), payload: bytes) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiM6Parity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const payload: bytes = hex"deadbeef";
        \\        return external<Sink>(target, gas: 50000).submit(("hello", @cast(u256, 7)), payload);
        \\    }
        \\}
    ;
    // cast abi-encode "f((string,uint256),bytes)" "(hello,7)" 0xdeadbeef
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(nested_tuple_comptime_source, nested_tuple_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000c0" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000007" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const deeply_nested_tuple_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const payload: bytes = hex"deadbeef";
        \\        @abiEncode(((((("hello", @cast(u256, 7)), payload), true), @cast(u256, 9)), false));
        \\    };
        \\}
    ;
    const deeply_nested_tuple_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, group: ((((string, u256), bytes), bool), u256), flag: bool) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiM6Parity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const payload: bytes = hex"deadbeef";
        \\        return external<Sink>(target, gas: 50000).submit((((("hello", @cast(u256, 7)), payload), true), @cast(u256, 9)), false);
        \\    }
        \\}
    ;
    // cast abi-encode "f(((((string,uint256),bytes),bool),uint256),bool)" "((((hello,7),0xdeadbeef),true),9)" false
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(deeply_nested_tuple_comptime_source, deeply_nested_tuple_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000009" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000c0" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000007" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000");

    const nested_string_arrays_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const first = @cast(slice[string], ["a", "bb"]);
        \\        const second = @cast(slice[string], []);
        \\        const values = @cast(slice[slice[string]], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const nested_string_arrays_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[slice[string]]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiM6Parity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const first = @cast(slice[string], ["a", "bb"]);
        \\        const second = @cast(slice[string], []);
        \\        const values = @cast(slice[slice[string]], [first, second]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(string[][])" "[[a,bb],[]]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(nested_string_arrays_comptime_source, nested_string_arrays_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000120" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000080" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000000");

    const empty_string_array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[string], []);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const empty_string_array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[string]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiM6Parity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values = @cast(slice[string], []);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(string[])" "[]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(empty_string_array_comptime_source, empty_string_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000000");

    const single_empty_string_array_comptime_source =
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const values = @cast(slice[string], [""]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const single_empty_string_array_runtime_source =
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[string]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiM6Parity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const values = @cast(slice[string], [""]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f(string[])" '[""]'
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(single_empty_string_array_comptime_source, single_empty_string_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000000");

    const dynamic_struct_comptime_source =
        \\struct Profile {
        \\    id: u256,
        \\    name: string,
        \\}
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const value: Profile = Profile { id: 1, name: "hello" };
        \\        @abiEncode(value);
        \\    };
        \\}
    ;
    const dynamic_struct_runtime_source =
        \\struct Profile {
        \\    id: u256,
        \\    name: string,
        \\}
        \\extern trait Sink {
        \\    staticcall fn submit(self, value: Profile) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiM6Parity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const value: Profile = Profile { id: 1, name: "hello" };
        \\        return external<Sink>(target, gas: 50000).submit(value);
        \\    }
        \\}
    ;
    // cast abi-encode "f((uint256,string))" "(1,hello)"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(dynamic_struct_comptime_source, dynamic_struct_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000");

    const dynamic_struct_array_comptime_source =
        \\struct Profile {
        \\    id: u256,
        \\    name: string,
        \\}
        \\pub fn expected() -> bytes {
        \\    return comptime {
        \\        const first: Profile = Profile { id: 1, name: "a" };
        \\        const second: Profile = Profile { id: 2, name: "bb" };
        \\        const values = @cast(slice[Profile], [first, second]);
        \\        @abiEncode(values);
        \\    };
        \\}
    ;
    const dynamic_struct_array_runtime_source =
        \\struct Profile {
        \\    id: u256,
        \\    name: string,
        \\}
        \\extern trait Sink {
        \\    staticcall fn submit(self, values: slice[Profile]) -> bool;
        \\}
        \\error ExternalCallFailed;
        \\contract RuntimeAbiM6Parity {
        \\    storage var target: address;
        \\    pub fn probe() -> !bool | ExternalCallFailed {
        \\        const first: Profile = Profile { id: 1, name: "a" };
        \\        const second: Profile = Profile { id: 2, name: "bb" };
        \\        const values = @cast(slice[Profile], [first, second]);
        \\        return external<Sink>(target, gas: 50000).submit(values);
        \\    }
        \\}
    ;
    // cast abi-encode "f((uint256,string)[])" "[(1,a),(2,bb)]"
    try expectRuntimeAbiPayloadMatchesComptimeAndCast(dynamic_struct_array_comptime_source, dynamic_struct_array_runtime_source, "probe", "expected", "0000000000000000000000000000000000000000000000000000000000000020" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "00000000000000000000000000000000000000000000000000000000000000c0" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000001" ++
        "6100000000000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000002" ++
        "6262000000000000000000000000000000000000000000000000000000000000");
}

test "compiler plain runtime dynamic ABI encode op matches cast" {
    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const text =
        \\module {
        \\  ora.contract @C {
        \\    func.func @encode() {
        \\      %message = ora.string.constant "hello" : !ora.string
        \\      %payload = ora.bytes.constant "0xdeadbeef" : !ora.bytes
        \\      %encoded = "ora.abi_encode"(%message, %payload) {layout = "tuple(dynamic(string),dynamic(bytes))"} : (!ora.string, !ora.bytes) -> !ora.int<256, false>
        \\      ora.return
        \\    }
        \\  }
        \\}
    ;
    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraConvertToSIR(ctx, module, false));

    const rendered = try renderSirTextForModule(ctx, module);
    defer testing.allocator.free(rendered);

    const expected = "0000000000000000000000000000000000000000000000000000000000000040" ++
        "0000000000000000000000000000000000000000000000000000000000000080" ++
        "0000000000000000000000000000000000000000000000000000000000000005" ++
        "68656c6c6f000000000000000000000000000000000000000000000000000000" ++
        "0000000000000000000000000000000000000000000000000000000000000004" ++
        "deadbeef00000000000000000000000000000000000000000000000000000000";
    const payload = try extractAbiBytesFromSir(rendered, "encode", try expectedHexByteLen(expected), 0);
    defer testing.allocator.free(payload);
    // cast abi-encode "f(string,bytes)" "hello" 0xdeadbeef
    try expectHexBytes(expected, payload);
}

test "compiler abiEncode emits exact diagnostics for unsupported cases" {
    const cases = [_]struct {
        source: []const u8,
        needle: []const u8,
    }{
        .{
            .source =
            \\pub fn run() -> bytes {
            \\    return comptime { @abiEncode(); };
            \\}
            ,
            .needle = "@abiEncode expects 1 argument, found 0",
        },
        .{
            .source =
            \\pub fn run() -> bytes {
            \\    return comptime { @abiEncode(1, 2); };
            \\}
            ,
            .needle = "@abiEncode expects 1 argument, found 2",
        },
        .{
            .source =
            \\pub fn run(table: map<address, u256>) -> bytes {
            \\    return comptime { @abiEncode(table); };
            \\}
            ,
            .needle = "@abiEncode: type 'map<address, u256>' has no ABI representation",
        },
    };

    for (cases) |case| {
        var compilation = try compileText(case.source);
        defer compilation.deinit();
        const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
        try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, case.needle));
    }
}

test "compiler corpus covers static abiEncode builtin" {
    var compilation = try compilePackage("ora-example/corpus/comptime/abi_encode_static.ora");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("AbiEncodeStaticCorpus").?;
    const contract = ast_file.item(contract_id).Contract;
    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.diagnostics.isEmpty());

    const Case = struct {
        name: []const u8,
        expected: i128,
    };
    const cases = [_]Case{
        .{
            .name = "scalar_tuple_len",
            // cast abi-encode "f(uint256,bool)" 1 true
            .expected = 64,
        },
        .{
            .name = "scalar_tuple_byte_sum",
            // First word last byte is 1, second word last byte is 1.
            .expected = 2,
        },
        .{
            .name = "signed_negative_first_byte",
            // cast abi-encode "f(int256)" -- -1
            .expected = 255,
        },
        .{
            .name = "static_struct_len",
            // cast abi-encode "f((uint256,bool))" "(7,true)"
            .expected = 64,
        },
        .{
            .name = "enum_value_last_byte",
            // cast abi-encode "f(uint8)" 1
            .expected = 1,
        },
        .{
            .name = "bitfield_value_last_byte",
            // cast abi-encode "f(uint8)" 3
            .expected = 3,
        },
        .{
            .name = "empty_tuple_len",
            // cast abi-encode "f()"
            .expected = 0,
        },
    };

    for (cases) |case| {
        var value_index: ?usize = null;
        for (contract.members) |member_id| {
            const item = ast_file.item(member_id).*;
            if (item != .Function or !std.mem.eql(u8, item.Function.name, case.name)) continue;
            const body = ast_file.body(item.Function.body);
            const ret_stmt = ast_file.statement(body.statements[0]).Return;
            value_index = ret_stmt.value.?.index();
            break;
        }
        try testing.expect(value_index != null);
        try testing.expectEqual(case.expected, try consteval.values[value_index.?].?.integer.toInt(i128));
    }
}

test "compiler corpus covers dynamic abiEncode builtin" {
    var compilation = try compilePackage("ora-example/corpus/comptime/abi_encode_dynamic.ora");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("AbiEncodeDynamicCorpus").?;
    const contract = ast_file.item(contract_id).Contract;
    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expect(consteval.diagnostics.isEmpty());

    const Case = struct {
        name: []const u8,
        expected: i128,
    };
    const cases = [_]Case{
        .{
            .name = "string_len",
            // cast abi-encode "f(string)" "hello"
            .expected = 96,
        },
        .{
            .name = "string_head_offset",
            // Single dynamic argument starts its tail at byte 32.
            .expected = 32,
        },
        .{
            .name = "string_payload_len",
            // "hello" is five UTF-8 bytes.
            .expected = 5,
        },
        .{
            .name = "string_first_byte",
            // 'h'
            .expected = 0x68,
        },
        .{
            .name = "bytes_aligned_len",
            // cast abi-encode "f(bytes)" 0x000102...1f
            .expected = 96,
        },
        .{
            .name = "mixed_tuple_second_offset",
            // cast abi-encode "f(uint256,string)" 1 "hello"
            .expected = 64,
        },
        .{
            .name = "empty_dynamic_pair_len",
            // cast abi-encode "f(string,bytes)" "" 0x
            .expected = 128,
        },
        .{
            .name = "dynamic_array_len",
            // cast abi-encode "f(uint256[])" "[1,2,3]"
            .expected = 160,
        },
        .{
            .name = "dynamic_array_head_offset",
            // Single dynamic array argument starts its tail at byte 32.
            .expected = 32,
        },
        .{
            .name = "dynamic_array_payload_len",
            // Three uint256 elements.
            .expected = 3,
        },
        .{
            .name = "dynamic_array_first_value",
            // First uint256 element.
            .expected = 1,
        },
        .{
            .name = "dynamic_address_array_second_tail_byte",
            // Last byte of 0x1234567890abcdef1234567890abcdef12345678.
            .expected = 0x78,
        },
    };

    for (cases) |case| {
        var value_index: ?usize = null;
        for (contract.members) |member_id| {
            const item = ast_file.item(member_id).*;
            if (item != .Function or !std.mem.eql(u8, item.Function.name, case.name)) continue;
            const body = ast_file.body(item.Function.body);
            const ret_stmt = ast_file.statement(body.statements[0]).Return;
            value_index = ret_stmt.value.?.index();
            break;
        }
        try testing.expect(value_index != null);
        try testing.expectEqual(case.expected, try consteval.values[value_index.?].?.integer.toInt(i128));
    }
}

test "compiler const eval supports selector builtin for trait methods" {
    const source_text =
        \\trait ERC20 {
        \\    fn transfer(self, to: address, value: u256) -> bool;
        \\}
        \\pub fn run() -> bytes4 {
        \\    return comptime {
        \\        @selector(ERC20.transfer);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    const selector_value = consteval.values[ret_stmt.value.?.index()].?.fixed_bytes;
    try testing.expectEqualSlices(u8, &.{ 0xa9, 0x05, 0x9c, 0xbb }, selector_value);
}

test "compiler const eval supports abiSignature builtin for trait methods" {
    const source_text =
        \\trait ERC20 {
        \\    fn transfer(self, to: address, value: u256) -> bool;
        \\}
        \\pub fn run() -> string {
        \\    return comptime {
        \\        @abiSignature(ERC20.transfer);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualStrings("transfer(address,uint256)", consteval.values[ret_stmt.value.?.index()].?.string);
}

test "compiler selector builtin rejects non-function arguments" {
    const source_text =
        \\pub fn run() -> bytes4 {
        \\    return comptime {
        \\        @selector(42);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "function reference"));
}

test "compiler selector builtin integrates with bytes4 assignments and comparison" {
    const source_text =
        \\trait ERC20 {
        \\    fn transfer(self, to: address, value: u256) -> bool;
        \\}
        \\pub fn run() -> bool {
        \\    return comptime {
        \\        const sel: bytes4 = @selector(ERC20.transfer);
        \\        sel == @selector(ERC20.transfer);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_body = ast_file.body(ast_file.expression(ret_stmt.value.?).Comptime.body);
    const decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.pattern_types[decl.pattern.index()].kind());
    try testing.expectEqual(@as(u8, 4), typecheck.pattern_types[decl.pattern.index()].type.fixed_bytes.len);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0xa9, 0x05, 0x9c, 0xbb }, consteval.values[decl.value.?.index()].?.fixed_bytes);
    try testing.expectEqual(true, consteval.values[ret_stmt.value.?.index()].?.boolean);
}

test "compiler const eval supports bytewise fixed-bytes xor" {
    const source_text =
        \\pub fn run() -> bytes4 {
        \\    return comptime {
        \\        const lhs: bytes4 = hex"01020304";
        \\        const rhs: bytes4 = hex"05060708";
        \\        lhs ^ rhs;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[0]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqual(@as(u8, 4), typecheck.exprType(ret_stmt.value.?).fixed_bytes.len);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0x04, 0x04, 0x04, 0x0c }, consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler trait method reflection exposes selectors" {
    const source_text =
        \\trait ERC165 {
        \\    fn supportsInterface(self, interface_id: bytes4) -> bool;
        \\}
        \\pub fn run() -> bytes4 {
        \\    return comptime {
        \\        const selector: bytes4 = @traitMethods(ERC165)[0].selector;
        \\        selector;
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;
    const comptime_body = ast_file.body(ast_file.expression(ret_stmt.value.?).Comptime.body);
    const selector_decl = ast_file.statement(comptime_body.statements[0]).VariableDecl;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqual(@as(u8, 4), typecheck.exprType(ret_stmt.value.?).fixed_bytes.len);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0x01, 0xff, 0xc9, 0xa7 }, consteval.values[selector_decl.value.?.index()].?.fixed_bytes);
}

test "compiler const eval supports eventTopic builtin for log declarations" {
    const source_text =
        \\log Transfer(indexed from: address, indexed to: address, value: u256);
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(Transfer);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqual(@as(u8, 32), typecheck.exprType(ret_stmt.value.?).fixed_bytes.len);

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Transfer(address,address,uint256)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eventTopic builtin rejects non-event arguments" {
    const source_text =
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(42);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "event reference"));
}

test "compiler eventTopic builtin supports contract-qualified log references" {
    const source_text =
        \\contract Events {
        \\    log Transfer(indexed from: address, indexed to: address, value: u256);
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(Events.Transfer);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Transfer(address,address,uint256)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eventTopic builtin uses pinned event_name metadata" {
    const source_text =
        \\log TransferV2(indexed from: address, indexed to: address, value: u256) {
        \\    pub const event_name = "Transfer";
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(TransferV2);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Transfer(address,address,uint256)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eventTopic builtin rejects non-string event_name metadata" {
    const source_text =
        \\log Transfer(indexed from: address, indexed to: address, value: u256) {
        \\    pub const event_name = 42;
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eventTopic(Transfer);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "event_name"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "string literal"));
}

test "compiler preserves pinned eip712_name metadata on structs" {
    const source_text =
        \\struct PermitV2 {
        \\    pub const eip712_name = "Permit";
        \\    owner: address,
        \\    spender: address,
        \\    value: u256,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const struct_item = ast_file.item(ast_file.root_items[0]).Struct;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(@as(usize, 1), struct_item.metadata.len);
    try testing.expectEqualStrings("Permit", compiler.hir.abi.eip712WireNameFromStructItem(ast_file, struct_item).?);
}

test "compiler eip712_name metadata falls back to struct identifier" {
    const source_text =
        \\struct Permit {
        \\    owner: address,
        \\    spender: address,
        \\    value: u256,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const struct_item = ast_file.item(ast_file.root_items[0]).Struct;

    try testing.expectEqualStrings("Permit", compiler.hir.abi.eip712WireNameFromStructItem(ast_file, struct_item).?);
}

test "compiler rejects non-string eip712_name metadata" {
    const source_text =
        \\struct Permit {
        \\    pub const eip712_name = 42;
        \\    owner: address,
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "eip712_name"));
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "string literal"));
}

test "compiler const eval supports eip712TypeHash builtin for ERC-2612 Permit" {
    const source_text =
        \\struct Permit {
        \\    pub const eip712_name = "Permit";
        \\    owner: address,
        \\    spender: address,
        \\    value: u256,
        \\    nonce: u256,
        \\    deadline: u256,
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(Permit);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());
    try testing.expectEqual(compiler.sema.TypeKind.fixed_bytes, typecheck.exprType(ret_stmt.value.?).kind());
    try testing.expectEqual(@as(u8, 32), typecheck.exprType(ret_stmt.value.?).fixed_bytes.len);

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eip712TypeHash builtin falls back to struct identifier" {
    const source_text =
        \\struct Permit {
        \\    owner: address,
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(Permit);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Permit(address owner)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eip712TypeHash builtin uses pinned eip712_name metadata" {
    const source_text =
        \\struct PermitV2 {
        \\    pub const eip712_name = "Permit";
        \\    owner: address,
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(PermitV2);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const function = ast_file.item(ast_file.root_items[1]).Function;
    const body = ast_file.body(function.body);
    const ret_stmt = ast_file.statement(body.statements[0]).Return;

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Permit(address owner)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[ret_stmt.value.?.index()].?.fixed_bytes);
}

test "compiler eip712TypeHash builtin rejects non-struct arguments" {
    const source_text =
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(42);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "struct type"));
}

test "compiler eip712TypeHash builtin rejects nested struct fields" {
    const source_text =
        \\struct Permit {
        \\    owner: address,
        \\}
        \\
        \\struct Order {
        \\    permit: Permit,
        \\    deadline: u256,
        \\}
        \\
        \\pub fn run() -> bytes32 {
        \\    return comptime {
        \\        @eip712TypeHash(Order);
        \\    };
        \\}
    ;

    var compilation = try compileText(source_text);
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "nested struct fields"));
}

test "compiler corpus covers ABI selector and signature builtins" {
    var compilation = try compilePackage("ora-example/corpus/comptime/abi_builtins.ora");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("AbiBuiltinCorpus").?;
    const contract = ast_file.item(contract_id).Contract;

    var selector_index: ?usize = null;
    var signature_index: ?usize = null;
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function) continue;

        const function = item.Function;
        const body = ast_file.body(function.body);
        const ret_stmt = ast_file.statement(body.statements[0]).Return;
        if (std.mem.eql(u8, function.name, "transfer_selector")) {
            selector_index = ret_stmt.value.?.index();
        } else if (std.mem.eql(u8, function.name, "transfer_signature")) {
            signature_index = ret_stmt.value.?.index();
        }
    }

    try testing.expect(selector_index != null);
    try testing.expect(signature_index != null);

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0xa9, 0x05, 0x9c, 0xbb }, consteval.values[selector_index.?].?.fixed_bytes);
    try testing.expectEqualStrings("transfer(address,uint256)", consteval.values[signature_index.?].?.string);
}

test "compiler corpus covers ABI eventTopic builtin" {
    var compilation = try compilePackage("ora-example/corpus/comptime/event_topic.ora");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("EventTopicCorpus").?;
    const contract = ast_file.item(contract_id).Contract;

    var topic_index: ?usize = null;
    var qualified_topic_index: ?usize = null;
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function) continue;

        const body = ast_file.body(item.Function.body);
        const ret_stmt = ast_file.statement(body.statements[0]).Return;
        if (std.mem.eql(u8, item.Function.name, "transfer_topic")) {
            topic_index = ret_stmt.value.?.index();
        } else if (std.mem.eql(u8, item.Function.name, "transfer_topic_qualified")) {
            qualified_topic_index = ret_stmt.value.?.index();
        }
    }

    try testing.expect(topic_index != null);
    try testing.expect(qualified_topic_index != null);

    var expected: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("Transfer(address,address,uint256)", &expected, .{});

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[topic_index.?].?.fixed_bytes);
    try testing.expectEqualSlices(u8, expected[0..], consteval.values[qualified_topic_index.?].?.fixed_bytes);
}

test "compiler corpus covers std interfaceId helper" {
    var compilation = try compilePackage("ora-example/corpus/comptime/interface_id.ora");
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(typecheck.diagnostics.isEmpty());

    const contract_id = item_index.lookup("InterfaceIdCorpus").?;
    const contract = ast_file.item(contract_id).Contract;

    var erc165_index: ?usize = null;
    var mini_index: ?usize = null;
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function) continue;

        const function = item.Function;
        const body = ast_file.body(function.body);
        const ret_stmt = ast_file.statement(body.statements[0]).Return;
        if (std.mem.eql(u8, item.Function.name, "erc165_id")) {
            erc165_index = ret_stmt.value.?.index();
        } else if (std.mem.eql(u8, item.Function.name, "mini_id")) {
            mini_index = ret_stmt.value.?.index();
        }
    }

    try testing.expect(erc165_index != null);
    try testing.expect(mini_index != null);

    var balance_hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("balanceOf(address)", &balance_hash, .{});
    var transfer_hash: [32]u8 = undefined;
    std.crypto.hash.sha3.Keccak256.hash("transfer(address,uint256)", &transfer_hash, .{});
    const mini_expected = [_]u8{
        balance_hash[0] ^ transfer_hash[0],
        balance_hash[1] ^ transfer_hash[1],
        balance_hash[2] ^ transfer_hash[2],
        balance_hash[3] ^ transfer_hash[3],
    };

    const consteval = try compilation.db.constEval(compilation.root_module_id);
    try testing.expectEqualSlices(u8, &.{ 0x01, 0xff, 0xc9, 0xa7 }, consteval.values[erc165_index.?].?.fixed_bytes);
    try testing.expectEqualSlices(u8, mini_expected[0..], consteval.values[mini_index.?].?.fixed_bytes);
}

test "compiler corpus rejects selector misuse" {
    var compilation = try compilePackage("ora-example/corpus/comptime/fail_selector_non_function.ora");
    defer compilation.deinit();

    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    try testing.expect(diagnosticMessagesContain(&typecheck.diagnostics, "function reference"));
}

test "compiler emits ABI attrs for public contract entries" {
    const source_text =
        \\contract Entry {
        \\    pub fn init(owner: address) {}
        \\
        \\    pub fn run(owner: address, amount: u256) -> bool {
        \\        return owner == owner && amount > 0;
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.selector"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_params"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"address\""));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"uint256\""));
}

test "compiler emits ABI attrs for slice of struct containing bitfield" {
    const source_text =
        \\bitfield CustomFlags : u8 {
        \\    enabled: bool @bits(0..1);
        \\    code: u8 @bits(1..8);
        \\}
        \\
        \\struct Row {
        \\    head: u256;
        \\    flags: CustomFlags;
        \\    tail: u256;
        \\}
        \\
        \\contract Entries {
        \\    storage var rows: slice[Row];
        \\
        \\    pub fn set(values: slice[Row]) {
        \\        rows = values;
        \\    }
        \\}
    ;

    const rendered = try renderHirTextForSource(source_text);
    defer testing.allocator.free(rendered);

    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_params"));
    try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, "\"(uint256,uint8,uint256)[]\""));
}
