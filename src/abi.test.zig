// ============================================================================
// ABI Tests
// ============================================================================

const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");
const abi = ora_root.abi;
const abi_layout = ora_root.abi_layout;
const abi_layout_context = ora_root.abi_layout_context;
const abi_policy = ora_root.abi_policy;
const abi_comptime_decoder = ora_root.abi_comptime_decoder;
const abi_comptime_encoder = ora_root.abi_comptime_encoder;
const compiler = ora_root.compiler;
const ora_types = @import("ora_types");
const hir_module_lowering = compiler.hir.abi_layout_test_support;
const mlir = @import("mlir_c_api").c;

const AbiFixture = struct {
    compilation: compiler.driver.Compilation,
    contract_abi: abi.ContractAbi,

    fn deinit(self: *AbiFixture) void {
        self.contract_abi.deinit();
        self.compilation.deinit();
    }
};

fn generateAbiForSource(allocator: std.mem.Allocator, source: []const u8) !AbiFixture {
    var compilation = try compiler.compileSource(allocator, "abi-test.ora", source);
    errdefer compilation.deinit();
    const contract_abi = try abi.generateCompilerAbi(allocator, &compilation);
    return .{
        .compilation = compilation,
        .contract_abi = contract_abi,
    };
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

fn renderSirTextForModule(ctx: mlir.MlirContext, module: mlir.MlirModule) ![]u8 {
    const sir_text_ref = mlir.oraEmitSIRText(ctx, module);
    defer if (sir_text_ref.data != null) mlir.oraStringRefFree(sir_text_ref);
    if (sir_text_ref.data == null) return error.TestUnexpectedResult;
    return try testing.allocator.dupe(u8, sir_text_ref.data[0..sir_text_ref.length]);
}

fn serializeAbiLayoutForType(ty: compiler.sema.Type) ![]const u8 {
    const layout = try abi_layout.fromType(testing.allocator, ty);
    defer layout.deinit(testing.allocator);
    return abi_layout.serializeForMlirAttr(testing.allocator, layout);
}

const PublicPolicyTestProvider = struct {
    pub fn enumHasPayload(_: @This(), _: []const u8) bool {
        return false;
    }

    pub fn errorTypeHasPayload(_: @This(), _: compiler.sema.Type) bool {
        return false;
    }
};

const PublicPolicyTest = abi_policy.Policy(PublicPolicyTestProvider);

const PublicPolicyTestContext = struct {
    fn init() PublicPolicyTestContext {
        return .{};
    }

    fn deinit(_: *PublicPolicyTestContext) void {}

    fn policy(self: *PublicPolicyTestContext) PublicPolicyTest {
        _ = self;
        return .{
            .provider = .{},
        };
    }
};

test "public ABI policy static word count fails closed for non-builtin integers" {
    var context = PublicPolicyTestContext.init();
    defer context.deinit();

    var policy = context.policy();

    try testing.expectEqual(@as(?usize, 1), policy.staticWordCount(.{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } }));
    try testing.expectEqual(@as(?usize, 1), policy.staticWordCount(.{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } }));
    try testing.expectEqual(@as(?usize, null), policy.staticWordCount(.{ .integer = .{ .bits = 500, .signed = false, .spelling = "u500" } }));
    try testing.expectEqual(@as(?usize, null), policy.staticWordCount(.{ .comptime_integer = .{ .spelling = "u256" } }));
    try testing.expectEqual(@as(?usize, null), policy.staticWordCount(.{ .comptime_integer = .{} }));

    try testing.expectEqual(@as(?usize, 1), policy.staticWordCount(.{ .named = .{ .name = "u8" } }));
    try testing.expectEqual(@as(?usize, 1), policy.staticWordCount(.{ .named = .{ .name = "u256" } }));
    try testing.expectEqual(@as(?usize, null), policy.staticWordCount(.{ .named = .{ .name = "u500" } }));
}

test "public ABI policy Result dynamic arrays accept only canonical u256 integer elements" {
    var context = PublicPolicyTestContext.init();
    defer context.deinit();

    var policy = context.policy();

    const error_types = [_]compiler.sema.Type{.{ .named = .{ .name = "Failure" } }};
    const u256_element: compiler.sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const u500_element: compiler.sema.Type = .{ .integer = .{ .bits = 500, .signed = false, .spelling = "u500" } };
    const u256_slice: compiler.sema.Type = .{ .slice = .{ .element_type = &u256_element } };
    const u500_slice: compiler.sema.Type = .{ .slice = .{ .element_type = &u500_element } };

    try testing.expect(policy.planResultCarrier(.{ .error_union = .{ .payload_type = &u256_slice, .error_types = &error_types } }) != null);
    try testing.expect(policy.planResultCarrier(.{ .error_union = .{ .payload_type = &u500_slice, .error_types = &error_types } }) == null);
}

test "public ABI policy encode and decode fail closed for non-builtin integers" {
    var context = PublicPolicyTestContext.init();
    defer context.deinit();

    var policy = context.policy();

    const u8_type: compiler.sema.Type = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };
    const u256_type: compiler.sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const u500_type: compiler.sema.Type = .{ .integer = .{ .bits = 500, .signed = false, .spelling = "u500" } };
    const unresolved_integer: compiler.sema.Type = .{ .comptime_integer = .{ .spelling = "u256" } };

    try testing.expect(policy.supportsAbiEncode(u8_type));
    try testing.expect(policy.supportsAbiDecode(u256_type));
    try testing.expect(!policy.supportsAbiEncode(u500_type));
    try testing.expect(!policy.supportsAbiDecode(u500_type));
    try testing.expect(!policy.supportsAbiEncode(unresolved_integer));
    try testing.expect(!policy.supportsAbiDecode(unresolved_integer));

    try testing.expect(policy.supportsAbiEncode(.{ .named = .{ .name = "u8" } }));
    try testing.expect(policy.supportsAbiDecode(.{ .named = .{ .name = "u256" } }));
    try testing.expect(!policy.supportsAbiEncode(.{ .named = .{ .name = "u500" } }));
    try testing.expect(!policy.supportsAbiDecode(.{ .named = .{ .name = "u500" } }));
}

test "public ABI policy runtime decode fails closed for non-builtin integers" {
    var context = PublicPolicyTestContext.init();
    defer context.deinit();

    var policy = context.policy();

    const u8_type: compiler.sema.Type = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };
    const u256_type: compiler.sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const u500_type: compiler.sema.Type = .{ .integer = .{ .bits = 500, .signed = false, .spelling = "u500" } };
    const u256_slice: compiler.sema.Type = .{ .slice = .{ .element_type = &u256_type } };
    const u500_slice: compiler.sema.Type = .{ .slice = .{ .element_type = &u500_type } };

    try testing.expect(policy.supportsRuntimeAbiDecode(u8_type, .strict));
    try testing.expect(policy.supportsRuntimeAbiDecode(u256_type, .strict));
    try testing.expect(!policy.supportsRuntimeAbiDecode(u500_type, .strict));
    try testing.expect(!policy.supportsRuntimeAbiDecode(.{ .comptime_integer = .{} }, .strict));
    try testing.expect(policy.supportsRuntimeAbiDecode(.{ .named = .{ .name = "u256" } }, .strict));
    try testing.expect(!policy.supportsRuntimeAbiDecode(.{ .named = .{ .name = "u500" } }, .strict));

    try testing.expect(policy.supportsRuntimeAbiDecode(u256_slice, .strict));
    try testing.expect(policy.supportsRuntimeAbiDecode(u256_slice, .permissive));
    try testing.expect(!policy.supportsRuntimeAbiDecode(u500_slice, .strict));
    try testing.expect(!policy.supportsRuntimeAbiDecode(u500_slice, .permissive));
}

const ExpectedSirNeedleCount = struct {
    needle: []const u8,
    count: usize,
};

var noop_resolver_context: u8 = 0;

fn noopTypeIdForType(_: *anyopaque, _: compiler.sema.Type) anyerror!?u32 {
    return null;
}

fn noopStructFields(_: *anyopaque, _: []const u8) anyerror!?[]const compiler.sema.AnonymousStructField {
    return null;
}

fn noopEnumVariantCount(_: *anyopaque, _: []const u8) anyerror!?usize {
    return null;
}

fn noopAbiDecodeResolver() abi_comptime_decoder.TypeResolver {
    return .{
        .context = &noop_resolver_context,
        .typeIdForType = noopTypeIdForType,
        .structFields = noopStructFields,
        .enumVariantCount = noopEnumVariantCount,
    };
}

var n4a_resolver_context: u8 = 0;

fn n4aTypeIdForType(_: *anyopaque, ty: compiler.sema.Type) anyerror!?u32 {
    return switch (ty) {
        .enum_ => |named| if (std.mem.eql(u8, named.name, "Status")) 1 else null,
        else => null,
    };
}

fn n4aStructFields(_: *anyopaque, _: []const u8) anyerror!?[]const compiler.sema.AnonymousStructField {
    return null;
}

fn n4aEnumVariantCount(_: *anyopaque, name: []const u8) anyerror!?usize {
    return if (std.mem.eql(u8, name, "Status")) 2 else null;
}

fn n4aAbiDecodeResolver() abi_comptime_decoder.TypeResolver {
    return .{
        .context = &n4a_resolver_context,
        .typeIdForType = n4aTypeIdForType,
        .structFields = n4aStructFields,
        .enumVariantCount = n4aEnumVariantCount,
    };
}

var failing_struct_fields_context: u8 = 0;

fn failingStructFieldsTypeIdForType(_: *anyopaque, _: compiler.sema.Type) anyerror!?u32 {
    return 1;
}

fn failingStructFields(_: *anyopaque, _: []const u8) anyerror!?[]const compiler.sema.AnonymousStructField {
    return error.TestResolverFailure;
}

fn failingStructFieldsEnumVariantCount(_: *anyopaque, _: []const u8) anyerror!?usize {
    return null;
}

fn failingStructFieldsResolver() abi_comptime_decoder.TypeResolver {
    return .{
        .context = &failing_struct_fields_context,
        .typeIdForType = failingStructFieldsTypeIdForType,
        .structFields = failingStructFields,
        .enumVariantCount = failingStructFieldsEnumVariantCount,
    };
}

fn decodeHexBytes(allocator: std.mem.Allocator, hex_with_optional_prefix: []const u8) ![]u8 {
    const hex = if (std.mem.startsWith(u8, hex_with_optional_prefix, "0x")) hex_with_optional_prefix[2..] else hex_with_optional_prefix;
    try testing.expectEqual(@as(usize, 0), hex.len % 2);
    const bytes = try allocator.alloc(u8, hex.len / 2);
    errdefer allocator.free(bytes);
    for (bytes, 0..) |*byte, index| {
        const hi = try std.fmt.charToDigit(hex[index * 2], 16);
        const lo = try std.fmt.charToDigit(hex[index * 2 + 1], 16);
        byte.* = @intCast((hi << 4) | lo);
    }
    return bytes;
}

fn expectComptimeDecodeError(
    layout: abi_layout.LayoutNode,
    target_type: compiler.sema.Type,
    bytes: []const u8,
    resolver: abi_comptime_decoder.TypeResolver,
    expected: abi_comptime_decoder.DecodeError,
) !void {
    try expectComptimeDecodeErrorWithMode(layout, target_type, bytes, resolver, expected, .strict);
}

fn expectComptimeDecodeErrorWithMode(
    layout: abi_layout.LayoutNode,
    target_type: compiler.sema.Type,
    bytes: []const u8,
    resolver: abi_comptime_decoder.TypeResolver,
    expected: abi_comptime_decoder.DecodeError,
    mode: abi_comptime_decoder.DecodeMode,
) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var heap = ora_root.comptime_eval.CtHeap.init(allocator);
    defer heap.deinit();

    const decoded = try abi_comptime_decoder.decodeComptimeValueWithMode(
        allocator,
        &heap,
        resolver,
        layout,
        target_type,
        bytes,
        mode,
    );
    try testing.expectEqual(expected, decoded.err);
}

fn expectComptimeDecodeErrorHex(
    layout: abi_layout.LayoutNode,
    target_type: compiler.sema.Type,
    payload_hex: []const u8,
    resolver: abi_comptime_decoder.TypeResolver,
    expected: abi_comptime_decoder.DecodeError,
) !void {
    const bytes = try decodeHexBytes(testing.allocator, payload_hex);
    defer testing.allocator.free(bytes);
    try expectComptimeDecodeError(layout, target_type, bytes, resolver, expected);
}

fn expectComptimeDecodeErrorHexWithMode(
    layout: abi_layout.LayoutNode,
    target_type: compiler.sema.Type,
    payload_hex: []const u8,
    resolver: abi_comptime_decoder.TypeResolver,
    expected: abi_comptime_decoder.DecodeError,
    mode: abi_comptime_decoder.DecodeMode,
) !void {
    const bytes = try decodeHexBytes(testing.allocator, payload_hex);
    defer testing.allocator.free(bytes);
    try expectComptimeDecodeErrorWithMode(layout, target_type, bytes, resolver, expected, mode);
}

const NestedDecodeFixture = struct {
    layout: abi_layout.LayoutNode,
    target_type: compiler.sema.Type,
};

fn nestedTupleDecodeFixture(
    allocator: std.mem.Allocator,
    depth: usize,
    leaf_layout: abi_layout.LayoutNode,
    leaf_type: compiler.sema.Type,
) !NestedDecodeFixture {
    if (depth == 0) return .{ .layout = leaf_layout, .target_type = leaf_type };

    const inner = try nestedTupleDecodeFixture(allocator, depth - 1, leaf_layout, leaf_type);
    const layout_elements = try allocator.alloc(abi_layout.LayoutNode, 1);
    layout_elements[0] = inner.layout;
    const type_elements = try allocator.alloc(compiler.sema.Type, 1);
    type_elements[0] = inner.target_type;
    return .{
        .layout = .{ .tuple = .{ .path = .{}, .elements = layout_elements } },
        .target_type = .{ .tuple = type_elements },
    };
}

test "abi decode error ordinals are stable for runtime C++ mapping" {
    const DecodeError = abi_comptime_decoder.DecodeError;

    try testing.expectEqual(@as(u32, 0), @intFromEnum(DecodeError.truncated_buffer));
    try testing.expectEqual(@as(u32, 1), @intFromEnum(DecodeError.oversize_buffer));
    try testing.expectEqual(@as(u32, 2), @intFromEnum(DecodeError.buffer_size_exceeded));
    try testing.expectEqual(@as(u32, 3), @intFromEnum(DecodeError.non_canonical_padding));
    try testing.expectEqual(@as(u32, 4), @intFromEnum(DecodeError.invalid_bool_value));
    try testing.expectEqual(@as(u32, 5), @intFromEnum(DecodeError.invalid_address));
    try testing.expectEqual(@as(u32, 6), @intFromEnum(DecodeError.invalid_fixed_bytes));
    try testing.expectEqual(@as(u32, 7), @intFromEnum(DecodeError.enum_out_of_range));
    try testing.expectEqual(@as(u32, 8), @intFromEnum(DecodeError.depth_limit_exceeded));
    try testing.expectEqual(@as(u32, 9), @intFromEnum(DecodeError.array_length_exceeded));
    try testing.expectEqual(@as(u32, 10), @intFromEnum(DecodeError.refinement_violation));
    try testing.expectEqual(@as(u32, 11), @intFromEnum(DecodeError.non_canonical_encoding));
    try testing.expectEqual(@as(u32, 12), @intFromEnum(DecodeError.invalid_offset));
    try testing.expectEqual(@as(u32, 13), @intFromEnum(DecodeError.length_overflow));
    try testing.expectEqual(@as(u32, 14), @intFromEnum(DecodeError.string_length_exceeded));
}

test "abi comptime decoder propagates struct resolver failures" {
    const layout_elements = [_]abi_layout.LayoutNode{
        .{ .static_word = .{ .path = .{}, .encoding = .{ .uint = 256 } } },
    };
    const layout: abi_layout.LayoutNode = .{ .tuple = .{ .path = .{}, .elements = &layout_elements } };
    const target_type: compiler.sema.Type = .{ .struct_ = .{ .name = "Point" } };
    const bytes = [_]u8{0} ** 32;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var heap = ora_root.comptime_eval.CtHeap.init(allocator);
    defer heap.deinit();

    try testing.expectError(error.TestResolverFailure, abi_comptime_decoder.decodeComptimeValue(
        allocator,
        &heap,
        failingStructFieldsResolver(),
        layout,
        target_type,
        &bytes,
    ));
}

test "abi comptime decoder has malformed corpus fixture for every DecodeError variant" {
    const sema = compiler.sema;
    const resolver = n4aAbiDecodeResolver();

    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const u8_ty: sema.Type = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };
    const string_ty: sema.Type = .string;
    const u256_slice_ty: sema.Type = .{ .slice = .{ .element_type = &u256_ty } };
    const bytes4_ty: sema.Type = .{ .fixed_bytes = .{ .len = 4, .spelling = "bytes4" } };
    const status_ty: sema.Type = .{ .enum_ = .{ .name = "Status" } };
    const refinement_args = [_]ora_types.RefinementArg{
        .Type,
        .{ .Integer = .{ .text = "1" } },
    };
    const non_zero_ty: sema.Type = .{ .refinement = .{
        .name = "MinValue",
        .base_type = &u256_ty,
        .args = &refinement_args,
    } };

    const u256_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .uint = 256 } } };
    const u8_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .uint = 8 } } };
    const bool_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .bool } };
    const address_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .address } };
    const bytes4_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .fixed_bytes = 4 } } };
    const string_layout: abi_layout.LayoutNode = .{ .dynamic_bytes = .{ .path = .{}, .kind = .string } };
    const u256_array_element = try testing.allocator.create(abi_layout.LayoutNode);
    defer testing.allocator.destroy(u256_array_element);
    u256_array_element.* = u256_layout;
    const u256_slice_layout: abi_layout.LayoutNode = .{ .dynamic_array = .{ .path = .{}, .element = u256_array_element } };
    const status_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .uint = 8 } } };

    const word_zero = "0000000000000000000000000000000000000000000000000000000000000000";
    const word_one = "0000000000000000000000000000000000000000000000000000000000000001";
    const word_two = "0000000000000000000000000000000000000000000000000000000000000002";
    const word_32 = "0000000000000000000000000000000000000000000000000000000000000020";
    const word_2_64 = "0000000000000000000000000000000000000000000000010000000000000000";
    const word_32769 = "0000000000000000000000000000000000000000000000000000000000008001";
    const word_1048577 = "0000000000000000000000000000000000000000000000000000000000100001";

    // N4a corpus map:
    // - truncated_buffer: missing a required static word.
    // - oversize_buffer: valid static word followed by extra bytes.
    // - buffer_size_exceeded: input exceeds the decoder's global byte budget.
    // - non_canonical_padding: narrow integer with non-zero high bits.
    // - invalid_bool_value: bool word is neither 0 nor 1.
    // - invalid_address: address word has non-zero high 96 bits.
    // - invalid_fixed_bytes: bytesN word has non-zero trailing padding.
    // - enum_out_of_range: positional enum id is >= variant count.
    // - depth_limit_exceeded: nested tuple walk crosses MAX_DECODE_DEPTH.
    // - array_length_exceeded: dynamic array length is over MAX_ARRAY_LENGTH.
    // - refinement_violation: decoded value violates MinValue.
    // - non_canonical_encoding: top-level dynamic offset is not the canonical 32.
    // - invalid_offset: top-level offset word does not fit usize.
    // - length_overflow: dynamic length word does not fit usize.
    // - string_length_exceeded: string/bytes length is above MAX_STRING_LENGTH.
    try expectComptimeDecodeErrorHex(u256_layout, u256_ty, "", resolver, .truncated_buffer);
    try expectComptimeDecodeErrorHex(u256_layout, u256_ty, word_one ++ word_zero, resolver, .oversize_buffer);

    const too_large = try testing.allocator.alloc(u8, abi_comptime_decoder.MAX_BUFFER_SIZE + 1);
    defer testing.allocator.free(too_large);
    @memset(too_large, 0);
    try expectComptimeDecodeError(u256_layout, u256_ty, too_large, resolver, .buffer_size_exceeded);

    try expectComptimeDecodeErrorHex(u8_layout, u8_ty, "0000000000000000000000000000000000000000000000000000000000000100", resolver, .non_canonical_padding);
    try expectComptimeDecodeErrorHex(bool_layout, .bool, word_two, resolver, .invalid_bool_value);
    try expectComptimeDecodeErrorHex(address_layout, .address, "0100000000000000000000001234567890abcdef1234567890abcdef12345678", resolver, .invalid_address);
    try expectComptimeDecodeErrorHex(bytes4_layout, bytes4_ty, "aabbccdd00000000000000000000000000000000000000000000000000000001", resolver, .invalid_fixed_bytes);
    try expectComptimeDecodeErrorHex(status_layout, status_ty, word_two, resolver, .enum_out_of_range);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const nested = try nestedTupleDecodeFixture(arena.allocator(), abi_comptime_decoder.MAX_DECODE_DEPTH, u256_layout, u256_ty);
    try expectComptimeDecodeErrorHex(nested.layout, nested.target_type, word_one, resolver, .depth_limit_exceeded);

    try expectComptimeDecodeErrorHex(u256_slice_layout, u256_slice_ty, word_32 ++ word_32769, resolver, .array_length_exceeded);
    try expectComptimeDecodeErrorHex(u256_layout, non_zero_ty, word_zero, resolver, .refinement_violation);
    try expectComptimeDecodeErrorHex(string_layout, string_ty, word_zero, resolver, .non_canonical_encoding);
    try expectComptimeDecodeErrorHex(string_layout, string_ty, word_2_64, resolver, .invalid_offset);
    try expectComptimeDecodeErrorHex(string_layout, string_ty, word_32 ++ word_2_64, resolver, .length_overflow);
    try expectComptimeDecodeErrorHex(string_layout, string_ty, word_32 ++ word_1048577, resolver, .string_length_exceeded);
}

test "abi comptime decoder fails closed for refinement base with unresolved comptime integer" {
    const sema = compiler.sema;
    const resolver = noopAbiDecodeResolver();

    const unresolved_u256: sema.Type = .{ .comptime_integer = .{ .spelling = "u256" } };
    const refinement_args = [_]ora_types.RefinementArg{
        .Type,
        .{ .Integer = .{ .text = "1" } },
    };
    const refined_ty: sema.Type = .{ .refinement = .{
        .name = "MinValue",
        .base_type = &unresolved_u256,
        .args = &refinement_args,
    } };
    const u256_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .uint = 256 } } };
    const bytes = try decodeHexBytes(testing.allocator, "0000000000000000000000000000000000000000000000000000000000000001");
    defer testing.allocator.free(bytes);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var heap = ora_root.comptime_eval.CtHeap.init(allocator);
    defer heap.deinit();

    try testing.expectError(error.AbiDecoderInternalShapeMismatch, abi_comptime_decoder.decodeComptimeValue(
        allocator,
        &heap,
        resolver,
        u256_layout,
        refined_ty,
        bytes,
    ));
}

test "abi N5 permissive comptime decoder masks canonicality but preserves hard errors" {
    const sema = compiler.sema;
    const resolver = n4aAbiDecodeResolver();

    const u8_ty: sema.Type = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };
    const bool_ty: sema.Type = .bool;
    const address_ty: sema.Type = .address;
    const bytes4_ty: sema.Type = .{ .fixed_bytes = .{ .len = 4, .spelling = "bytes4" } };
    const string_ty: sema.Type = .string;
    const u8_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .uint = 8 } } };
    const bool_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .bool } };
    const address_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .address } };
    const bytes4_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .fixed_bytes = 4 } } };
    const string_layout: abi_layout.LayoutNode = .{ .dynamic_bytes = .{ .path = .{}, .kind = .string } };

    const noncanonical_u8 = try decodeHexBytes(testing.allocator, "0000000000000000000000000000000000000000000000000000000000000100");
    defer testing.allocator.free(noncanonical_u8);
    const invalid_bool = try decodeHexBytes(testing.allocator, "0000000000000000000000000000000000000000000000000000000000000002");
    defer testing.allocator.free(invalid_bool);
    const invalid_address = try decodeHexBytes(testing.allocator, "0100000000000000000000001234567890abcdef1234567890abcdef12345678");
    defer testing.allocator.free(invalid_address);
    const invalid_fixed = try decodeHexBytes(testing.allocator, "aabbccdd00000000000000000000000000000000000000000000000000000001");
    defer testing.allocator.free(invalid_fixed);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var heap = ora_root.comptime_eval.CtHeap.init(allocator);
    defer heap.deinit();

    const decoded_u8 = try abi_comptime_decoder.decodeComptimeValuePermissive(allocator, &heap, resolver, u8_layout, u8_ty, noncanonical_u8);
    try testing.expectEqual(@as(u256, 0), decoded_u8.ok.integer);

    const decoded_bool = try abi_comptime_decoder.decodeComptimeValuePermissive(allocator, &heap, resolver, bool_layout, bool_ty, invalid_bool);
    try testing.expect(decoded_bool.ok.boolean);

    const decoded_address = try abi_comptime_decoder.decodeComptimeValuePermissive(allocator, &heap, resolver, address_layout, address_ty, invalid_address);
    try testing.expectEqual(@as(u160, 0x1234567890abcdef1234567890abcdef12345678), decoded_address.ok.address);

    const decoded_fixed = try abi_comptime_decoder.decodeComptimeValuePermissive(allocator, &heap, resolver, bytes4_layout, bytes4_ty, invalid_fixed);
    try testing.expectEqualSlices(u8, &[_]u8{ 0xaa, 0xbb, 0xcc, 0xdd }, heap.getBytes(decoded_fixed.ok.bytes_ref));

    const word_zero = "0000000000000000000000000000000000000000000000000000000000000000";
    const word_32 = "0000000000000000000000000000000000000000000000000000000000000020";
    const word_64 = "0000000000000000000000000000000000000000000000000000000000000040";
    const word_one = "0000000000000000000000000000000000000000000000000000000000000001";
    const word_2_64 = "0000000000000000000000000000000000000000000000010000000000000000";
    const refinement_args = [_]ora_types.RefinementArg{
        .Type,
        .{ .Integer = .{ .text = "1" } },
    };
    const positive_u8_ty: sema.Type = .{ .refinement = .{
        .name = "MinValue",
        .base_type = &u8_ty,
        .args = &refinement_args,
    } };

    try expectComptimeDecodeErrorHexWithMode(u8_layout, positive_u8_ty, "0000000000000000000000000000000000000000000000000000000000000100", resolver, .refinement_violation, .permissive);
    try expectComptimeDecodeErrorHexWithMode(string_layout, string_ty, word_32, resolver, .truncated_buffer, .permissive);
    try expectComptimeDecodeErrorHexWithMode(string_layout, string_ty, word_32 ++ word_2_64, resolver, .length_overflow, .permissive);

    const trailing = try decodeHexBytes(testing.allocator, word_zero ++ word_zero);
    defer testing.allocator.free(trailing);
    const decoded_trailing = try abi_comptime_decoder.decodeComptimeValuePermissive(allocator, &heap, resolver, u8_layout, u8_ty, trailing);
    try testing.expectEqual(@as(u256, 0), decoded_trailing.ok.integer);

    const shifted_string = try decodeHexBytes(testing.allocator, word_64 ++ word_zero ++ word_one ++ "61ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    defer testing.allocator.free(shifted_string);
    const decoded_shifted = try abi_comptime_decoder.decodeComptimeValuePermissive(allocator, &heap, resolver, string_layout, string_ty, shifted_string);
    const reencoded_shifted = try abi_comptime_encoder.encodeComptimeValue(
        testing.allocator,
        &heap,
        string_layout,
        .{ .ct = decoded_shifted.ok },
    );
    defer testing.allocator.free(reencoded_shifted);
    try testing.expect(!std.mem.eql(u8, shifted_string, reencoded_shifted));
    try testing.expectEqual(@as(u8, 0x40), shifted_string[31]);
    try testing.expectEqual(@as(u8, 0x20), reencoded_shifted[31]);
}

fn expectDecodeReencodesCanonicalBytes(
    layout: abi_layout.LayoutNode,
    target_type: compiler.sema.Type,
    bytes: []const u8,
    resolver: abi_comptime_decoder.TypeResolver,
) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var heap = ora_root.comptime_eval.CtHeap.init(allocator);
    defer heap.deinit();

    const decoded = try abi_comptime_decoder.decodeComptimeValue(
        allocator,
        &heap,
        resolver,
        layout,
        target_type,
        bytes,
    );

    switch (decoded) {
        .err => {},
        .ok => |value| {
            const encoded = try abi_comptime_encoder.encodeComptimeValue(
                testing.allocator,
                &heap,
                layout,
                .{ .ct = value },
            );
            defer testing.allocator.free(encoded);
            try testing.expectEqualSlices(u8, bytes, encoded);
        },
    }
}

fn expectCanonicalValueRoundTrip(
    source_heap: *const ora_root.comptime_eval.CtHeap,
    layout: abi_layout.LayoutNode,
    target_type: compiler.sema.Type,
    value: ora_root.comptime_eval.CtValue,
    resolver: abi_comptime_decoder.TypeResolver,
) !void {
    const encoded = try abi_comptime_encoder.encodeComptimeValue(
        testing.allocator,
        source_heap,
        layout,
        .{ .ct = value },
    );
    defer testing.allocator.free(encoded);
    try expectDecodeReencodesCanonicalBytes(layout, target_type, encoded, resolver);
}

fn expectCanonicalValueRoundTripForType(
    source_heap: *const ora_root.comptime_eval.CtHeap,
    target_type: compiler.sema.Type,
    value: ora_root.comptime_eval.CtValue,
    resolver: abi_comptime_decoder.TypeResolver,
) !void {
    const layout = try abi_layout.fromType(testing.allocator, target_type);
    defer layout.deinit(testing.allocator);
    try expectCanonicalValueRoundTrip(source_heap, layout, target_type, value, resolver);
}

const N4bFuzzTarget = struct {
    layout: abi_layout.LayoutNode,
    target_type: compiler.sema.Type,
    resolver: abi_comptime_decoder.TypeResolver,
};

const N4bDeterministicBytes = struct {
    state: u64,

    fn next(self: *N4bDeterministicBytes) u8 {
        self.state = self.state *% 6364136223846793005 +% 1442695040888963407;
        return @truncate(self.state >> 56);
    }

    fn fill(self: *N4bDeterministicBytes, bytes: []u8) void {
        for (bytes) |*byte| byte.* = self.next();
    }
};

test "abi N4b comptime decode encode round-trip covers committed corpus shapes" {
    const sema = compiler.sema;
    const resolver = n4aAbiDecodeResolver();

    var heap = ora_root.comptime_eval.CtHeap.init(testing.allocator);
    defer heap.deinit();

    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const u8_ty: sema.Type = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };
    const i16_ty: sema.Type = .{ .integer = .{ .bits = 16, .signed = true, .spelling = "i16" } };
    const bytes4_ty: sema.Type = .{ .fixed_bytes = .{ .len = 4, .spelling = "bytes4" } };
    const u256_slice_ty: sema.Type = .{ .slice = .{ .element_type = &u256_ty } };
    const tuple_elems = [_]sema.Type{ u256_ty, .bool };
    const static_tuple_ty: sema.Type = .{ .tuple = &tuple_elems };
    const mixed_elems = [_]sema.Type{ u256_ty, .string };
    const mixed_tuple_ty: sema.Type = .{ .tuple = &mixed_elems };
    const bitfield_ty: sema.Type = .{ .bitfield = .{ .name = "Flags" } };
    const status_ty: sema.Type = .{ .enum_ = .{ .name = "Status" } };

    try expectCanonicalValueRoundTripForType(&heap, .void, .void_val, resolver);
    try expectCanonicalValueRoundTripForType(&heap, u256_ty, .{ .integer = 0x1234 }, resolver);
    try expectCanonicalValueRoundTripForType(&heap, u8_ty, .{ .integer = 0xa5 }, resolver);
    try expectCanonicalValueRoundTripForType(&heap, i16_ty, .{ .integer = std.math.maxInt(u256) }, resolver);
    try expectCanonicalValueRoundTripForType(&heap, .bool, .{ .boolean = true }, resolver);
    try expectCanonicalValueRoundTripForType(&heap, .address, .{ .address = 0x1234567890abcdef1234567890abcdef12345678 }, resolver);

    const bytes4_value: ora_root.comptime_eval.CtValue = .{ .bytes_ref = try heap.allocBytes(&.{ 0xaa, 0xbb, 0xcc, 0xdd }) };
    try expectCanonicalValueRoundTripForType(&heap, bytes4_ty, bytes4_value, resolver);

    const string_value: ora_root.comptime_eval.CtValue = .{ .string_ref = try heap.allocString("hello") };
    try expectCanonicalValueRoundTripForType(&heap, .string, string_value, resolver);

    const bytes_value: ora_root.comptime_eval.CtValue = .{ .bytes_ref = try heap.allocBytes(&.{ 0xde, 0xad, 0xbe, 0xef, 0x01 }) };
    try expectCanonicalValueRoundTripForType(&heap, .bytes, bytes_value, resolver);

    const slice_elems = [_]ora_root.comptime_eval.CtValue{
        .{ .integer = 1 },
        .{ .integer = 2 },
        .{ .integer = 3 },
    };
    const u256_slice_value: ora_root.comptime_eval.CtValue = .{ .slice_ref = try heap.allocSlice(&slice_elems) };
    try expectCanonicalValueRoundTripForType(&heap, u256_slice_ty, u256_slice_value, resolver);

    const tuple_values = [_]ora_root.comptime_eval.CtValue{
        .{ .integer = 42 },
        .{ .boolean = false },
    };
    const static_tuple_value: ora_root.comptime_eval.CtValue = .{ .tuple_ref = try heap.allocTuple(&tuple_values) };
    try expectCanonicalValueRoundTripForType(&heap, static_tuple_ty, static_tuple_value, resolver);

    const mixed_values = [_]ora_root.comptime_eval.CtValue{
        .{ .integer = 7 },
        string_value,
    };
    const mixed_tuple_value: ora_root.comptime_eval.CtValue = .{ .tuple_ref = try heap.allocTuple(&mixed_values) };
    try expectCanonicalValueRoundTripForType(&heap, mixed_tuple_ty, mixed_tuple_value, resolver);

    try expectCanonicalValueRoundTripForType(&heap, bitfield_ty, .{ .integer = 0xf0f0 }, resolver);

    const status_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .uint = 8 } } };
    try expectCanonicalValueRoundTrip(
        &heap,
        status_layout,
        status_ty,
        .{ .adt_val = .{ .type_id = 1, .variant_id = 1, .payload = null } },
        resolver,
    );
}

test "abi N4b bounded mutation fuzz either rejects or re-encodes exactly" {
    const sema = compiler.sema;
    const resolver = n4aAbiDecodeResolver();

    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const u8_ty: sema.Type = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };
    const bytes4_ty: sema.Type = .{ .fixed_bytes = .{ .len = 4, .spelling = "bytes4" } };
    const u256_slice_ty: sema.Type = .{ .slice = .{ .element_type = &u256_ty } };
    const tuple_elems = [_]sema.Type{ u256_ty, .bool };
    const static_tuple_ty: sema.Type = .{ .tuple = &tuple_elems };

    const u256_layout = try abi_layout.fromType(testing.allocator, u256_ty);
    defer u256_layout.deinit(testing.allocator);
    const u8_layout = try abi_layout.fromType(testing.allocator, u8_ty);
    defer u8_layout.deinit(testing.allocator);
    const bool_layout = try abi_layout.fromType(testing.allocator, .bool);
    defer bool_layout.deinit(testing.allocator);
    const address_layout = try abi_layout.fromType(testing.allocator, .address);
    defer address_layout.deinit(testing.allocator);
    const bytes4_layout = try abi_layout.fromType(testing.allocator, bytes4_ty);
    defer bytes4_layout.deinit(testing.allocator);
    const string_layout = try abi_layout.fromType(testing.allocator, .string);
    defer string_layout.deinit(testing.allocator);
    const bytes_layout = try abi_layout.fromType(testing.allocator, .bytes);
    defer bytes_layout.deinit(testing.allocator);
    const u256_slice_layout = try abi_layout.fromType(testing.allocator, u256_slice_ty);
    defer u256_slice_layout.deinit(testing.allocator);
    const static_tuple_layout = try abi_layout.fromType(testing.allocator, static_tuple_ty);
    defer static_tuple_layout.deinit(testing.allocator);
    const status_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .uint = 8 } } };
    const status_ty: sema.Type = .{ .enum_ = .{ .name = "Status" } };

    const targets = [_]N4bFuzzTarget{
        .{ .layout = u256_layout, .target_type = u256_ty, .resolver = resolver },
        .{ .layout = u8_layout, .target_type = u8_ty, .resolver = resolver },
        .{ .layout = bool_layout, .target_type = .bool, .resolver = resolver },
        .{ .layout = address_layout, .target_type = .address, .resolver = resolver },
        .{ .layout = bytes4_layout, .target_type = bytes4_ty, .resolver = resolver },
        .{ .layout = string_layout, .target_type = .string, .resolver = resolver },
        .{ .layout = bytes_layout, .target_type = .bytes, .resolver = resolver },
        .{ .layout = u256_slice_layout, .target_type = u256_slice_ty, .resolver = resolver },
        .{ .layout = static_tuple_layout, .target_type = static_tuple_ty, .resolver = resolver },
        .{ .layout = status_layout, .target_type = status_ty, .resolver = resolver },
    };

    var fuzz = N4bDeterministicBytes{ .state = 0x4e34625f61626931 };
    var buffer: [192]u8 = undefined;
    for (0..96) |index| {
        const target = targets[index % targets.len];
        const len = @as(usize, fuzz.next()) % buffer.len;
        fuzz.fill(buffer[0..len]);
        try expectDecodeReencodesCanonicalBytes(target.layout, target.target_type, buffer[0..len], target.resolver);
    }

    const mutated = [_]struct {
        layout: abi_layout.LayoutNode,
        target_type: sema.Type,
        hex: []const u8,
        expected: abi_comptime_decoder.DecodeError,
    }{
        .{
            .layout = bool_layout,
            .target_type = .bool,
            .hex = "0000000000000000000000000000000000000000000000000000000000000002",
            .expected = .invalid_bool_value,
        },
        .{
            .layout = u8_layout,
            .target_type = u8_ty,
            .hex = "0000000000000000000000000000000000000000000000000000000000000100",
            .expected = .non_canonical_padding,
        },
        .{
            .layout = address_layout,
            .target_type = .address,
            .hex = "0100000000000000000000001234567890abcdef1234567890abcdef12345678",
            .expected = .invalid_address,
        },
        .{
            .layout = bytes4_layout,
            .target_type = bytes4_ty,
            .hex = "aabbccdd00000000000000000000000000000000000000000000000000000001",
            .expected = .invalid_fixed_bytes,
        },
        .{
            .layout = string_layout,
            .target_type = .string,
            .hex = "0000000000000000000000000000000000000000000000000000000000000040",
            .expected = .non_canonical_encoding,
        },
        .{
            .layout = string_layout,
            .target_type = .string,
            .hex = "0000000000000000000000000000000000000000000000000000000000000020" ++
                "0000000000000000000000000000000000000000000000000000000000000001" ++
                "6100000000000000000000000000000000000000000000000000000000000001",
            .expected = .non_canonical_encoding,
        },
        .{
            .layout = u256_slice_layout,
            .target_type = u256_slice_ty,
            .hex = "0000000000000000000000000000000000000000000000000000000000000020" ++
                "0000000000000000000000000000000000000000000000000000000000008001",
            .expected = .array_length_exceeded,
        },
        .{
            .layout = status_layout,
            .target_type = status_ty,
            .hex = "0000000000000000000000000000000000000000000000000000000000000002",
            .expected = .enum_out_of_range,
        },
    };

    for (mutated) |case| {
        const bytes = try decodeHexBytes(testing.allocator, case.hex);
        defer testing.allocator.free(bytes);
        try expectComptimeDecodeError(case.layout, case.target_type, bytes, resolver, case.expected);
    }
}

test "abi comptime decoder prioritizes decode errors over oversize" {
    var heap = ora_root.comptime_eval.CtHeap.init(testing.allocator);
    defer heap.deinit();

    const bool_layout: abi_layout.LayoutNode = .{ .static_word = .{
        .path = .{},
        .encoding = .bool,
    } };
    const target_type: compiler.sema.Type = .{ .bool = {} };
    const resolver = noopAbiDecodeResolver();

    const invalid_bool_plus_extra =
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" ++
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02" ++
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" ++
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    const invalid = try abi_comptime_decoder.decodeComptimeValue(
        testing.allocator,
        &heap,
        resolver,
        bool_layout,
        target_type,
        invalid_bool_plus_extra,
    );
    try testing.expectEqual(abi_comptime_decoder.DecodeError.invalid_bool_value, invalid.err);

    const valid_bool_plus_extra =
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" ++
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01" ++
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" ++
        "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    const oversize = try abi_comptime_decoder.decodeComptimeValue(
        testing.allocator,
        &heap,
        resolver,
        bool_layout,
        target_type,
        valid_bool_plus_extra,
    );
    try testing.expectEqual(abi_comptime_decoder.DecodeError.oversize_buffer, oversize.err);
}

fn expectSerializedAbiLayoutParsesToStaticSir(
    ty: compiler.sema.Type,
    expected_layout: []const u8,
    args: []const u8,
    operands: []const u8,
    operand_types: []const u8,
    expected_store_count: usize,
    expected_needles: []const ExpectedSirNeedleCount,
) !void {
    const layout = try serializeAbiLayoutForType(ty);
    defer testing.allocator.free(layout);
    try testing.expectEqualStrings(expected_layout, layout);

    const text = try std.fmt.allocPrint(testing.allocator,
        \\module {{
        \\  ora.contract @C {{
        \\    func.func @encode({s}) {{
        \\      %encoded = "ora.abi_encode"({s}) {{layout = "{s}"}} : ({s}) -> !ora.int<256, false>
        \\      ora.return
        \\    }}
        \\  }}
        \\}}
    , .{ args, operands, layout, operand_types });
    defer testing.allocator.free(text);

    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraConvertToSIR(ctx, module, false));

    const rendered = try renderSirTextForModule(ctx, module);
    defer testing.allocator.free(rendered);

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_encode"));
    try testing.expectEqual(expected_store_count, std.mem.count(u8, rendered, "mstore256"));
    for (expected_needles) |expected| {
        try testing.expectEqual(expected.count, std.mem.count(u8, rendered, expected.needle));
    }
}

fn expectSerializedAbiLayoutParsesToDynamicSir(
    ty: compiler.sema.Type,
    expected_layout: []const u8,
    args: []const u8,
    operands: []const u8,
    operand_types: []const u8,
    expected_needles: []const []const u8,
) !void {
    const layout = try serializeAbiLayoutForType(ty);
    defer testing.allocator.free(layout);
    try testing.expectEqualStrings(expected_layout, layout);

    const text = try std.fmt.allocPrint(testing.allocator,
        \\module {{
        \\  ora.contract @C {{
        \\    func.func @encode({s}) {{
        \\      %encoded = "ora.abi_encode"({s}) {{layout = "{s}"}} : ({s}) -> !ora.int<256, false>
        \\      ora.return
        \\    }}
        \\  }}
        \\}}
    , .{ args, operands, layout, operand_types });
    defer testing.allocator.free(text);

    const ctx = createOraMlirContext();
    defer mlir.oraContextDestroy(ctx);

    const module = try parseOraModule(ctx, text);
    defer mlir.oraModuleDestroy(module);

    try testing.expect(mlir.mlirOperationVerify(mlir.oraModuleGetOperation(module)));
    try testing.expect(mlir.oraConvertToSIR(ctx, module, false));

    const rendered = try renderSirTextForModule(ctx, module);
    defer testing.allocator.free(rendered);

    try testing.expect(!std.mem.containsAtLeast(u8, rendered, 1, "ora.abi_encode"));
    for (expected_needles) |needle| {
        try testing.expect(std.mem.containsAtLeast(u8, rendered, 1, needle));
    }
}

fn findCallable(contract_abi: *const abi.ContractAbi, kind: abi.CallableKind, name: []const u8) ?*const abi.AbiCallable {
    for (contract_abi.callables) |*callable| {
        if (callable.kind == kind and std.mem.eql(u8, callable.name, name)) {
            return callable;
        }
    }
    return null;
}

test "abi layout serializer is consumed by OraToSIR parser with matching static decisions" {
    const u256_ty: compiler.sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const u8_ty: compiler.sema.Type = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };
    const i16_ty: compiler.sema.Type = .{ .integer = .{ .bits = 16, .signed = true, .spelling = "i16" } };
    const bytes4_ty: compiler.sema.Type = .{ .fixed_bytes = .{ .len = 4, .spelling = "bytes4" } };

    const flat_types = [_]compiler.sema.Type{ u8_ty, i16_ty, .bool, .address, bytes4_ty };
    const flat_ty: compiler.sema.Type = .{ .tuple = flat_types[0..] };
    try expectSerializedAbiLayoutParsesToStaticSir(
        flat_ty,
        "tuple(static(uint8),static(int16),static(bool),static(address),static(bytes4))",
        "%a0: !ora.int<256, false>, %a1: !ora.int<256, false>, %a2: !ora.int<256, false>, %a3: !ora.int<256, false>, %a4: !ora.int<256, false>",
        "%a0, %a1, %a2, %a3, %a4",
        "!ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>, !ora.int<256, false>",
        5,
        &.{
            .{ .needle = "signextend", .count = 1 },
            .{ .needle = "shl", .count = 1 },
            .{ .needle = "and", .count = 4 },
        },
    );

    const array_element_ty = u256_ty;
    const fixed_array_ty: compiler.sema.Type = .{ .array = .{ .element_type = &array_element_ty, .len = 2 } };
    const array_param_types = [_]compiler.sema.Type{fixed_array_ty};
    const array_param_list_ty: compiler.sema.Type = .{ .tuple = array_param_types[0..] };
    try expectSerializedAbiLayoutParsesToStaticSir(
        array_param_list_ty,
        "tuple(array(2,static(uint256)))",
        "%values: memref<2xi256>",
        "%values",
        "memref<2xi256>",
        2,
        &.{.{ .needle = "mload256", .count = 2 }},
    );

    const empty_ty: compiler.sema.Type = .{ .tuple = &.{} };
    try expectSerializedAbiLayoutParsesToStaticSir(
        empty_ty,
        "tuple()",
        "",
        "",
        "",
        0,
        &.{},
    );
}

test "abi layout serializer is consumed by OraToSIR parser for dynamic array layouts" {
    const u256_ty: compiler.sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const bool_ty: compiler.sema.Type = .bool;
    const bytes_ty: compiler.sema.Type = .bytes;
    const string_ty: compiler.sema.Type = .string;

    // Dynamic-element array layouts are fed through static pointer-slot
    // memrefs here so this isolated test stays focused on parser and
    // materializer support. End-to-end runtime-count behavior is covered by
    // the cross-layer ABI parity tests.
    const string_slice_ty: compiler.sema.Type = .{ .slice = .{ .element_type = &string_ty } };
    const string_slice_params = [_]compiler.sema.Type{string_slice_ty};
    const string_slice_param_list_ty: compiler.sema.Type = .{ .tuple = string_slice_params[0..] };
    try expectSerializedAbiLayoutParsesToDynamicSir(
        string_slice_param_list_ty,
        "tuple(array(dynamic,dynamic(string)))",
        "%values: memref<2xi256>",
        "%values",
        "memref<2xi256>",
        &.{ "mload256", "mcopy" },
    );

    const u256_slice_ty: compiler.sema.Type = .{ .slice = .{ .element_type = &u256_ty } };
    const nested_u256_slice_ty: compiler.sema.Type = .{ .slice = .{ .element_type = &u256_slice_ty } };
    const nested_u256_slice_params = [_]compiler.sema.Type{nested_u256_slice_ty};
    const nested_u256_slice_param_list_ty: compiler.sema.Type = .{ .tuple = nested_u256_slice_params[0..] };
    try expectSerializedAbiLayoutParsesToDynamicSir(
        nested_u256_slice_param_list_ty,
        "tuple(array(dynamic,array(dynamic,static(uint256))))",
        "%values: memref<2xi256>",
        "%values",
        "memref<2xi256>",
        &.{ "mload256", "mstore256" },
    );

    const tuple_elems = [_]compiler.sema.Type{ u256_ty, string_ty };
    const dynamic_tuple_ty: compiler.sema.Type = .{ .tuple = tuple_elems[0..] };
    const dynamic_tuple_slice_ty: compiler.sema.Type = .{ .slice = .{ .element_type = &dynamic_tuple_ty } };
    const dynamic_tuple_slice_params = [_]compiler.sema.Type{dynamic_tuple_slice_ty};
    const dynamic_tuple_slice_param_list_ty: compiler.sema.Type = .{ .tuple = dynamic_tuple_slice_params[0..] };
    try expectSerializedAbiLayoutParsesToDynamicSir(
        dynamic_tuple_slice_param_list_ty,
        "tuple(array(dynamic,tuple(static(uint256),dynamic(string))))",
        "%values: memref<2xi256>",
        "%values",
        "memref<2xi256>",
        &.{ "mload256", "mcopy" },
    );

    const struct_fields = [_]compiler.sema.AnonymousStructField{
        .{ .name = "id", .ty = u256_ty },
        .{ .name = "name", .ty = string_ty },
    };
    const dynamic_struct_ty: compiler.sema.Type = .{ .anonymous_struct = .{ .fields = struct_fields[0..] } };
    const dynamic_struct_params = [_]compiler.sema.Type{dynamic_struct_ty};
    const dynamic_struct_param_list_ty: compiler.sema.Type = .{ .tuple = dynamic_struct_params[0..] };
    try expectSerializedAbiLayoutParsesToDynamicSir(
        dynamic_struct_param_list_ty,
        "tuple(tuple(static(uint256),dynamic(string)))",
        "%value: i256",
        "%value",
        "i256",
        &.{ "mload256", "mcopy" },
    );

    const inner_tuple_elems = [_]compiler.sema.Type{ string_ty, u256_ty };
    const inner_tuple_ty: compiler.sema.Type = .{ .tuple = inner_tuple_elems[0..] };
    const middle_tuple_elems = [_]compiler.sema.Type{ inner_tuple_ty, bytes_ty };
    const middle_tuple_ty: compiler.sema.Type = .{ .tuple = middle_tuple_elems[0..] };
    const deep_tuple_elems = [_]compiler.sema.Type{ middle_tuple_ty, bool_ty };
    const deep_tuple_ty: compiler.sema.Type = .{ .tuple = deep_tuple_elems[0..] };
    const deep_tuple_params = [_]compiler.sema.Type{deep_tuple_ty};
    const deep_tuple_param_list_ty: compiler.sema.Type = .{ .tuple = deep_tuple_params[0..] };
    try expectSerializedAbiLayoutParsesToDynamicSir(
        deep_tuple_param_list_ty,
        "tuple(tuple(tuple(tuple(dynamic(string),static(uint256)),dynamic(bytes)),static(bool)))",
        "%value: i256",
        "%value",
        "i256",
        &.{ "mload256", "mcopy" },
    );
}

const FunctionRef = struct {
    item_id: compiler.ItemId,
    function: compiler.ast.FunctionItem,
};

fn findContractFunction(ast_file: *const compiler.AstFile, contract: compiler.ast.ContractItem, name: []const u8) ?FunctionRef {
    for (contract.members) |member_id| {
        const item = ast_file.item(member_id).*;
        if (item != .Function or !std.mem.eql(u8, item.Function.name, name)) continue;
        return .{ .item_id = member_id, .function = item.Function };
    }
    return null;
}

const AbiLayoutTestLowerer = struct {
    allocator: std.mem.Allocator,
    file: *const compiler.AstFile,
    item_index: *const compiler.sema.ItemIndexResult,
    typecheck: *const compiler.sema.TypeCheckResult,
};

const AbiLayoutTestModuleLowering = hir_module_lowering.mixin(
    AbiLayoutTestLowerer,
    // Only ABI layout/count helpers are safe to call on this mixin instance;
    // ContractLowerer and FunctionLowerer are deliberately opaque.
    opaque {},
    opaque {},
    compiler.hir.HirSymbolKind,
);

fn moduleLoweringAbiLayoutForType(
    allocator: std.mem.Allocator,
    ast_file: *const compiler.AstFile,
    item_index: *const compiler.sema.ItemIndexResult,
    typecheck: *const compiler.sema.TypeCheckResult,
    ty: compiler.sema.Type,
) ![]const u8 {
    var lowerer = AbiLayoutTestLowerer{
        .allocator = allocator,
        .file = ast_file,
        .item_index = item_index,
        .typecheck = typecheck,
    };
    return AbiLayoutTestModuleLowering.abiLayoutForType(&lowerer, ty);
}

fn moduleLoweringStaticWordCountForType(
    allocator: std.mem.Allocator,
    ast_file: *const compiler.AstFile,
    item_index: *const compiler.sema.ItemIndexResult,
    typecheck: *const compiler.sema.TypeCheckResult,
    ty: compiler.sema.Type,
) ?usize {
    var lowerer = AbiLayoutTestLowerer{
        .allocator = allocator,
        .file = ast_file,
        .item_index = item_index,
        .typecheck = typecheck,
    };
    return AbiLayoutTestModuleLowering.staticAbiWordCountForType(&lowerer, ty);
}

fn moduleLoweringAbiLayoutForTypeExpr(
    allocator: std.mem.Allocator,
    ast_file: *const compiler.AstFile,
    item_index: *const compiler.sema.ItemIndexResult,
    typecheck: *const compiler.sema.TypeCheckResult,
    type_expr_id: compiler.ast.TypeExprId,
) ![]const u8 {
    var lowerer = AbiLayoutTestLowerer{
        .allocator = allocator,
        .file = ast_file,
        .item_index = item_index,
        .typecheck = typecheck,
    };
    return AbiLayoutTestModuleLowering.abiLayoutForTypeExpr(&lowerer, type_expr_id);
}

fn moduleLoweringStaticWordCountForTypeExpr(
    allocator: std.mem.Allocator,
    ast_file: *const compiler.AstFile,
    item_index: *const compiler.sema.ItemIndexResult,
    typecheck: *const compiler.sema.TypeCheckResult,
    type_expr_id: compiler.ast.TypeExprId,
) ?usize {
    var lowerer = AbiLayoutTestLowerer{
        .allocator = allocator,
        .file = ast_file,
        .item_index = item_index,
        .typecheck = typecheck,
    };
    return AbiLayoutTestModuleLowering.staticAbiWordCountForTypeExpr(&lowerer, type_expr_id);
}

fn testLayoutContext(
    allocator: std.mem.Allocator,
    ast_file: *const compiler.AstFile,
    item_index: *const compiler.sema.ItemIndexResult,
    typecheck: *const compiler.sema.TypeCheckResult,
) abi_layout_context.LayoutContext {
    return .{
        .allocator = allocator,
        .provider = compiler.sema.abiLayoutProvider(ast_file, item_index, typecheck),
    };
}

fn expectOptionalUsizeEqual(expected: ?usize, actual: ?usize) !void {
    try testing.expectEqual(expected != null, actual != null);
    if (expected) |expected_value| {
        try testing.expectEqual(expected_value, actual.?);
    }
}

fn expectManifestResultInputMatchesPlan(
    contract_abi: *const abi.ContractAbi,
    ctx: *const abi_layout_context.LayoutContext,
    result_ty: compiler.sema.Type,
    input_type_id: []const u8,
) !void {
    const plan = ctx.planResultCarrier(result_ty) orelse return error.TestUnexpectedResult;
    const bool_ty: compiler.sema.Type = .bool;
    if (plan.err) |err| {
        const elements = [_]compiler.sema.Type{ bool_ty, plan.payload, err };
        return expectManifestTupleTypeMatchesTypes(contract_abi, ctx, input_type_id, &elements);
    }
    const elements = [_]compiler.sema.Type{ bool_ty, plan.payload };
    return expectManifestTupleTypeMatchesTypes(contract_abi, ctx, input_type_id, &elements);
}

fn expectManifestTupleTypeMatchesTypes(
    contract_abi: *const abi.ContractAbi,
    ctx: *const abi_layout_context.LayoutContext,
    type_id: []const u8,
    elements: []const compiler.sema.Type,
) !void {
    const tuple_ty: compiler.sema.Type = .{ .tuple = elements };
    const expected_wire = try ctx.canonicalAbiTypeForType(tuple_ty);
    defer ctx.allocator.free(expected_wire);

    const manifest_type = contract_abi.findType(type_id) orelse return error.TestUnexpectedResult;
    try testing.expect(manifest_type.wire_type != null);
    try testing.expectEqualStrings(expected_wire, manifest_type.wire_type.?);
    try testing.expectEqual(elements.len, manifest_type.components.len);

    for (elements, manifest_type.components) |element, component_type_id| {
        const expected_component_wire = try ctx.canonicalAbiTypeForType(element);
        defer ctx.allocator.free(expected_component_wire);

        const component_type = contract_abi.findType(component_type_id) orelse return error.TestUnexpectedResult;
        try testing.expect(component_type.wire_type != null);
        try testing.expectEqualStrings(expected_component_wire, component_type.wire_type.?);
    }
}

fn countCallables(contract_abi: *const abi.ContractAbi, kind: abi.CallableKind, name: []const u8) usize {
    var count: usize = 0;
    for (contract_abi.callables) |callable| {
        if (callable.kind == kind and std.mem.eql(u8, callable.name, name)) {
            count += 1;
        }
    }
    return count;
}

fn hasEffectKind(callable: *const abi.AbiCallable, kind: abi.AbiEffectKind) bool {
    for (callable.effects) |effect| {
        if (effect.kind == kind) return true;
    }
    return false;
}

test "abi layout canonical strings match expected static type spellings" {
    const allocator = testing.allocator;
    const sema = compiler.sema;

    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const i128_ty: sema.Type = .{ .integer = .{ .bits = 128, .signed = true, .spelling = "i128" } };
    const bytes4_ty: sema.Type = .{ .fixed_bytes = .{ .len = 4, .spelling = "bytes4" } };
    const tuple_elems = [_]sema.Type{ u256_ty, .address, .bool, bytes4_ty };
    const tuple_ty: sema.Type = .{ .tuple = &tuple_elems };

    const cases = [_]struct {
        ty: sema.Type,
        wire: []const u8,
    }{
        .{ .ty = u256_ty, .wire = "uint256" },
        .{ .ty = i128_ty, .wire = "int128" },
        .{ .ty = .bool, .wire = "bool" },
        .{ .ty = .address, .wire = "address" },
        .{ .ty = bytes4_ty, .wire = "bytes4" },
    };
    for (cases) |case| {
        const layout_text = try abi_layout.canonicalAbiTypeFromType(allocator, case.ty);
        defer allocator.free(layout_text);
        try testing.expectEqualStrings(case.wire, layout_text);
    }

    const tuple_text = try abi_layout.canonicalAbiTypeFromType(allocator, tuple_ty);
    defer allocator.free(tuple_text);
    try testing.expectEqualStrings("(uint256,address,bool,bytes4)", tuple_text);
}

test "abi layout derives primitive family checks from builtin types" {
    const allocator = testing.allocator;
    const sema = compiler.sema;

    const u160_ty: sema.Type = .{ .integer = .{ .bits = 160, .signed = false, .spelling = "u160" } };
    const u160_text = try abi_layout.canonicalAbiTypeFromType(allocator, u160_ty);
    defer allocator.free(u160_text);
    try testing.expectEqualStrings("uint160", u160_text);
    try testing.expectEqual(@as(?usize, 1), abi_layout.staticWordCountForType(u160_ty));

    const u24_ty: sema.Type = .{ .integer = .{ .bits = 24, .signed = false, .spelling = "u24" } };
    try testing.expectError(error.InvalidIntegerWidth, abi_layout.canonicalAbiTypeFromType(allocator, u24_ty));
    try testing.expectEqual(@as(?usize, null), abi_layout.staticWordCountForType(u24_ty));

    const i96_ty: sema.Type = .{ .integer = .{ .bits = 96, .signed = true, .spelling = "i96" } };
    try testing.expectError(error.InvalidIntegerWidth, abi_layout.canonicalAbiTypeFromType(allocator, i96_ty));
    try testing.expectEqual(@as(?usize, null), abi_layout.staticWordCountForType(i96_ty));

    const unresolved_integer_ty: sema.Type = .{ .comptime_integer = .{ .spelling = "u256" } };
    try testing.expectError(error.UnsupportedAbiType, abi_layout.canonicalAbiTypeFromType(allocator, unresolved_integer_ty));
    try testing.expectEqual(@as(?usize, null), abi_layout.staticWordCountForType(unresolved_integer_ty));

    const bytes33_ty: sema.Type = .{ .fixed_bytes = .{ .len = 33, .spelling = "bytes33" } };
    try testing.expectError(error.InvalidFixedBytesWidth, abi_layout.canonicalAbiTypeFromType(allocator, bytes33_ty));
    try testing.expectEqual(@as(?usize, null), abi_layout.staticWordCountForType(bytes33_ty));

    const invalid_integer_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .uint = 24 } } };
    try testing.expectError(error.InvalidIntegerWidth, abi_layout.canonicalAbiType(allocator, invalid_integer_layout));
    try testing.expectError(error.InvalidIntegerWidth, abi_layout.serializeForMlirAttr(allocator, invalid_integer_layout));

    const invalid_fixed_bytes_layout: abi_layout.LayoutNode = .{ .static_word = .{ .path = .{}, .encoding = .{ .fixed_bytes = 33 } } };
    try testing.expectError(error.InvalidFixedBytesWidth, abi_layout.canonicalAbiType(allocator, invalid_fixed_bytes_layout));
    try testing.expectError(error.InvalidFixedBytesWidth, abi_layout.serializeForMlirAttr(allocator, invalid_fixed_bytes_layout));

    try testing.expectEqual(@as(?u8, 20), abi_layout.parseFixedBytesSpelling("bytes20"));
    try testing.expect(abi_layout.parseFixedBytesSpelling("bytes01") == null);
    try testing.expect(abi_layout.parseFixedBytesSpelling("bytes+5") == null);
    try testing.expect(abi_layout.parseFixedBytesSpelling("bytes1_6") == null);
}

test "abi layout classifies static and dynamic aggregate shapes" {
    const allocator = testing.allocator;
    const sema = compiler.sema;

    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const bool_ty: sema.Type = .bool;
    const static_tuple_elems = [_]sema.Type{ u256_ty, .address, bool_ty };
    const static_tuple_ty: sema.Type = .{ .tuple = &static_tuple_elems };

    var static_layout = try abi_layout.fromType(allocator, static_tuple_ty);
    defer static_layout.deinit(allocator);
    try testing.expect(!static_layout.isDynamic());
    try testing.expectEqual(@as(?usize, 3), static_layout.staticWordCount());
    try testing.expectEqual(@as(usize, 3), static_layout.headSlotWordCount());

    const string_ty: sema.Type = .string;
    const dynamic_tuple_elems = [_]sema.Type{ u256_ty, string_ty };
    const dynamic_tuple_ty: sema.Type = .{ .tuple = &dynamic_tuple_elems };

    var dynamic_layout = try abi_layout.fromType(allocator, dynamic_tuple_ty);
    defer dynamic_layout.deinit(allocator);
    try testing.expect(dynamic_layout.isDynamic());
    try testing.expectEqual(@as(?usize, null), dynamic_layout.staticWordCount());
    try testing.expectEqual(@as(usize, 1), dynamic_layout.headSlotWordCount());

    const array_element_ty: sema.Type = .bool;
    const array_ty: sema.Type = .{ .array = .{ .element_type = &array_element_ty, .len = 3 } };
    var array_layout = try abi_layout.fromType(allocator, array_ty);
    defer array_layout.deinit(allocator);
    try testing.expect(!array_layout.isDynamic());
    try testing.expectEqual(@as(?usize, 3), array_layout.staticWordCount());
}

test "abi layout records value paths including array element marker" {
    const allocator = testing.allocator;
    const sema = compiler.sema;

    const string_ty: sema.Type = .string;
    const array_ty: sema.Type = .{ .slice = .{ .element_type = &string_ty } };
    const tuple_elems = [_]sema.Type{ .address, array_ty };
    const tuple_ty: sema.Type = .{ .tuple = &tuple_elems };

    var layout = try abi_layout.fromType(allocator, tuple_ty);
    defer layout.deinit(allocator);

    const tuple = layout.tuple;
    try testing.expectEqual(@as(usize, 0), tuple.path.segments.len);
    try testing.expectEqual(@as(usize, 2), tuple.elements.len);

    const array = tuple.elements[1].dynamic_array;
    try testing.expectEqual(@as(usize, 1), array.path.segments.len);
    try testing.expectEqual(@as(u32, 1), array.path.segments[0].tuple_index);

    const element = array.element.dynamic_bytes;
    try testing.expectEqual(@as(usize, 2), element.path.segments.len);
    try testing.expectEqual(@as(u32, 1), element.path.segments[0].tuple_index);
    try testing.expect(element.path.segments[1] == .each_element);
}

test "abi layout renders dynamic arrays and fixed arrays canonically" {
    const allocator = testing.allocator;
    const sema = compiler.sema;

    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const fixed_array_ty: sema.Type = .{ .array = .{ .element_type = &u256_ty, .len = 2 } };
    const dynamic_array_ty: sema.Type = .{ .slice = .{ .element_type = &fixed_array_ty } };

    const fixed_text = try abi_layout.canonicalAbiTypeFromType(allocator, fixed_array_ty);
    defer allocator.free(fixed_text);
    try testing.expectEqualStrings("uint256[2]", fixed_text);

    const dynamic_text = try abi_layout.canonicalAbiTypeFromType(allocator, dynamic_array_ty);
    defer allocator.free(dynamic_text);
    try testing.expectEqualStrings("uint256[2][]", dynamic_text);
}

test "abi layout preserves legacy context-free bitfield spelling" {
    const allocator = testing.allocator;

    const bitfield_text = try abi_layout.canonicalAbiTypeFromType(allocator, .{ .bitfield = .{ .name = "Flags" } });
    defer allocator.free(bitfield_text);
    try testing.expectEqualStrings("uint256", bitfield_text);
}

test "abi layout rejects Result carrier shapes without item-index context" {
    const allocator = testing.allocator;
    const sema = compiler.sema;

    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const no_payload_error_ty: sema.Type = .{ .named = .{ .name = "Failure" } };
    const no_payload_errors = [_]sema.Type{no_payload_error_ty};
    const no_payload_result: sema.Type = .{ .error_union = .{ .payload_type = &u256_ty, .error_types = &no_payload_errors } };
    try testing.expectError(error.UnsupportedAbiType, abi_layout.canonicalAbiTypeFromType(allocator, no_payload_result));
    try testing.expectEqual(@as(?usize, null), abi_layout.staticWordCountForType(no_payload_result));

    const payload_error_ty: sema.Type = .{ .integer = .{ .bits = 8, .signed = false, .spelling = "u8" } };
    const payload_errors = [_]sema.Type{payload_error_ty};
    const payload_result: sema.Type = .{ .error_union = .{ .payload_type = &u256_ty, .error_types = &payload_errors } };
    try testing.expectError(error.UnsupportedAbiType, abi_layout.canonicalAbiTypeFromType(allocator, payload_result));
    try testing.expectEqual(@as(?usize, null), abi_layout.staticWordCountForType(payload_result));
}

test "abi layout matches generated ABI wire types for context-free public function boundary" {
    const allocator = testing.allocator;
    const source =
        \\contract C {
        \\    pub fn accept(
        \\        a: u256,
        \\        b: i128,
        \\        c: bool,
        \\        d: address,
        \\        e: string,
        \\        f: bytes,
        \\        g: bytes4,
        \\        h: slice[u256],
        \\        i: slice[string]
        \\    ) -> string {
        \\        return e;
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();

    const module = fixture.compilation.db.sources.module(fixture.compilation.root_module_id);
    const ast_file = try fixture.compilation.db.astFile(module.file_id);
    const typecheck = try fixture.compilation.db.moduleTypeCheck(fixture.compilation.root_module_id);
    const contract = ast_file.item(ast_file.root_items[0]).Contract;

    const callable = findCallable(&fixture.contract_abi, .function, "accept") orelse return error.TestUnexpectedResult;
    const function_ref = findContractFunction(ast_file, contract, "accept") orelse return error.TestUnexpectedResult;
    const function_type = switch (typecheck.item_types[function_ref.item_id.index()]) {
        .function => |function| function,
        else => return error.TestUnexpectedResult,
    };

    try testing.expectEqual(@as(usize, 9), callable.inputs.len);
    try testing.expectEqual(@as(usize, 1), callable.outputs.len);
    try testing.expectEqual(function_ref.function.parameters.len, callable.inputs.len);

    for (function_ref.function.parameters, callable.inputs) |parameter, input| {
        const ty = typecheck.pattern_types[parameter.pattern.index()].type;
        const layout_wire = try abi_layout.canonicalAbiTypeFromType(allocator, ty);
        defer allocator.free(layout_wire);

        const abi_type = fixture.contract_abi.findType(input.type_id) orelse return error.TestUnexpectedResult;
        try testing.expectEqualStrings(abi_type.wire_type.?, layout_wire);
    }

    const return_wire = try abi_layout.canonicalAbiTypeFromType(allocator, function_type.return_types[0]);
    defer allocator.free(return_wire);
    const output_type = fixture.contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings(output_type.wire_type.?, return_wire);
}

test "abi layout converges for context-bound module lowering types" {
    const allocator = testing.allocator;
    const source =
        \\enum Status : u8 {
        \\    Pending,
        \\    Done,
        \\}
        \\
        \\bitfield Flags : u8 {
        \\    enabled: bool @bits(0..1);
        \\    code: u8 @bits(1..8);
        \\}
        \\
        \\error Empty();
        \\error WithCode(code: u16);
        \\
        \\type SmallInt = u16;
        \\
        \\contract C {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    struct Outer {
        \\        pair: Pair,
        \\        status: Status,
        \\        flags: Flags,
        \\        label: string,
        \\        notes: slice[string],
        \\    }
        \\
        \\    pub fn converge(
        \\        plain: u256,
        \\        outer: Outer,
        \\        status: Status,
        \\        flags: Flags,
        \\        alias_value: SmallInt,
        \\        nested: Pair,
        \\        dynamic_nested: slice[Pair],
        \\        result_empty: Result<u256, Empty>,
        \\        result_payload: Result<u256, WithCode>,
        \\        result_struct_payload: Result<Pair, WithCode>
        \\    ) -> Pair {
        \\        return nested;
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(allocator, "abi-layout-convergence-test.ora", source);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const contract = ast_file.item(item_index.lookup("C").?).Contract;
    const function_ref = findContractFunction(ast_file, contract, "converge") orelse return error.TestUnexpectedResult;
    const function_type = switch (typecheck.item_types[function_ref.item_id.index()]) {
        .function => |function| function,
        else => return error.TestUnexpectedResult,
    };
    const ctx = testLayoutContext(allocator, ast_file, item_index, typecheck);

    const expected_names = [_][]const u8{
        "plain",
        "outer",
        "status",
        "flags",
        "alias_value",
        "nested",
        "dynamic_nested",
        "result_empty",
        "result_payload",
        "result_struct_payload",
    };
    try testing.expectEqual(expected_names.len, function_ref.function.parameters.len);
    try testing.expectEqual(expected_names.len, function_type.param_types.len);

    for (function_type.param_types, 0..) |ty, index| {
        const via_module_lowering = try moduleLoweringAbiLayoutForType(
            allocator,
            ast_file,
            item_index,
            typecheck,
            ty,
        );
        defer allocator.free(via_module_lowering);

        const via_context = try ctx.canonicalAbiTypeForType(ty);
        defer allocator.free(via_context);
        try testing.expectEqualStrings(via_context, via_module_lowering);

        const via_module_type_expr = try moduleLoweringAbiLayoutForTypeExpr(
            allocator,
            ast_file,
            item_index,
            typecheck,
            function_ref.function.parameters[index].type_expr,
        );
        defer allocator.free(via_module_type_expr);

        const via_context_type_expr = try ctx.canonicalAbiTypeForTypeExpr(function_ref.function.parameters[index].type_expr);
        defer allocator.free(via_context_type_expr);
        try testing.expectEqualStrings(via_context_type_expr, via_module_type_expr);
        try testing.expectEqualStrings(via_context, via_context_type_expr);
    }

    const return_layout = try moduleLoweringAbiLayoutForType(
        allocator,
        ast_file,
        item_index,
        typecheck,
        function_type.return_types[0],
    );
    defer allocator.free(return_layout);
    const return_context = try ctx.canonicalAbiTypeForType(function_type.return_types[0]);
    defer allocator.free(return_context);
    try testing.expectEqualStrings(return_context, return_layout);
}

test "abi static word counts converge for context-bound module lowering types" {
    const allocator = testing.allocator;
    const source =
        \\enum Status : u8 {
        \\    Pending,
        \\    Done,
        \\}
        \\
        \\bitfield Flags : u8 {
        \\    enabled: bool @bits(0..1);
        \\    code: u8 @bits(1..8);
        \\}
        \\
        \\error Empty();
        \\error WithCode(code: u16);
        \\
        \\type SmallInt = u16;
        \\
        \\contract C {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\
        \\    struct Outer {
        \\        pair: Pair,
        \\        status: Status,
        \\        flags: Flags,
        \\        label: string,
        \\        notes: slice[string],
        \\    }
        \\
        \\    pub fn converge(
        \\        plain: u256,
        \\        outer: Outer,
        \\        status: Status,
        \\        flags: Flags,
        \\        alias_value: SmallInt,
        \\        nested: Pair,
        \\        dynamic_nested: slice[Pair],
        \\        result_empty: Result<u256, Empty>,
        \\        result_payload: Result<u256, WithCode>,
        \\        result_struct_payload: Result<Pair, WithCode>
        \\    ) -> Pair {
        \\        return nested;
        \\    }
        \\}
    ;

    var compilation = try compiler.compileSource(allocator, "abi-word-count-convergence-test.ora", source);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const contract = ast_file.item(item_index.lookup("C").?).Contract;
    const function_ref = findContractFunction(ast_file, contract, "converge") orelse return error.TestUnexpectedResult;
    const function_type = switch (typecheck.item_types[function_ref.item_id.index()]) {
        .function => |function| function,
        else => return error.TestUnexpectedResult,
    };
    const ctx = testLayoutContext(allocator, ast_file, item_index, typecheck);

    const expected_names = [_][]const u8{
        "plain",
        "outer",
        "status",
        "flags",
        "alias_value",
        "nested",
        "dynamic_nested",
        "result_empty",
        "result_payload",
        "result_struct_payload",
    };
    try testing.expectEqual(expected_names.len, function_type.param_types.len);

    for (function_type.param_types, 0..) |ty, index| {
        const via_module_lowering = moduleLoweringStaticWordCountForType(allocator, ast_file, item_index, typecheck, ty);
        const via_context = ctx.staticWordCountForType(ty);
        try expectOptionalUsizeEqual(via_context, via_module_lowering);

        const via_type_expr = moduleLoweringStaticWordCountForTypeExpr(allocator, ast_file, item_index, typecheck, function_ref.function.parameters[index].type_expr);
        const via_context_type_expr = ctx.staticWordCountForTypeExpr(function_ref.function.parameters[index].type_expr);
        try expectOptionalUsizeEqual(via_context_type_expr, via_type_expr);
        try expectOptionalUsizeEqual(via_context, via_context_type_expr);
    }

    const return_via_module = moduleLoweringStaticWordCountForType(allocator, ast_file, item_index, typecheck, function_type.return_types[0]);
    const return_via_context = ctx.staticWordCountForType(function_type.return_types[0]);
    try expectOptionalUsizeEqual(return_via_context, return_via_module);
}

test "abi synthetic layout cases converge for struct error payloads and contract refs" {
    const allocator = testing.allocator;
    const sema = compiler.sema;
    const source =
        \\contract C {
        \\    struct Pair {
        \\        left: u256,
        \\        right: bool,
        \\    }
        \\}
        \\
        \\bitfield Loose {
        \\    enabled: bool @bits(0..1);
        \\}
        \\
        \\enum LooseStatus {
        \\    Active,
        \\    Paused,
        \\}
    ;

    var compilation = try compiler.compileSource(allocator, "abi-synthetic-convergence-test.ora", source);
    defer compilation.deinit();

    const module = compilation.db.sources.module(compilation.root_module_id);
    const ast_file = try compilation.db.astFile(module.file_id);
    const item_index = try compilation.db.itemIndex(compilation.root_module_id);
    const typecheck = try compilation.db.moduleTypeCheck(compilation.root_module_id);
    const ctx = testLayoutContext(allocator, ast_file, item_index, typecheck);
    const loose_bitfield = ast_file.item(item_index.lookup("Loose").?).Bitfield;
    try testing.expect(loose_bitfield.base_type == null);
    const loose_enum = ast_file.item(item_index.lookup("LooseStatus").?).Enum;
    try testing.expect(loose_enum.base_type == null);

    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const u16_ty: sema.Type = .{ .integer = .{ .bits = 16, .signed = false, .spelling = "u16" } };

    const anonymous_error_fields = [_]sema.AnonymousStructField{
        .{ .name = "code", .ty = u16_ty },
    };
    const anonymous_error_ty: sema.Type = .{ .anonymous_struct = .{ .fields = &anonymous_error_fields } };
    const anonymous_error_types = [_]sema.Type{anonymous_error_ty};
    const anonymous_error_result: sema.Type = .{ .error_union = .{ .payload_type = &u256_ty, .error_types = &anonymous_error_types } };

    const struct_error_ty: sema.Type = .{ .struct_ = .{ .name = "Pair" } };
    const struct_error_types = [_]sema.Type{struct_error_ty};
    const struct_error_result: sema.Type = .{ .error_union = .{ .payload_type = &u256_ty, .error_types = &struct_error_types } };

    const contract_ref_ty: sema.Type = .{ .contract = .{ .name = "C" } };
    const contract_error_types = [_]sema.Type{contract_ref_ty};
    const contract_error_result: sema.Type = .{ .error_union = .{ .payload_type = &u256_ty, .error_types = &contract_error_types } };

    const multi_error_types = [_]sema.Type{
        .{ .named = .{ .name = "First" } },
        .{ .named = .{ .name = "Second" } },
    };
    const multi_error_result: sema.Type = .{ .error_union = .{ .payload_type = &u256_ty, .error_types = &multi_error_types } };

    const no_base_bitfield_ty: sema.Type = .{ .bitfield = .{ .name = "Loose" } };
    const no_base_enum_ty: sema.Type = .{ .enum_ = .{ .name = "LooseStatus" } };

    const cases = [_]struct {
        name: []const u8,
        ty: sema.Type,
        expected_layout: ?[]const u8,
        expected_words: ?usize,
    }{
        .{
            .name = "anonymous_struct_error_payload",
            .ty = anonymous_error_result,
            .expected_layout = "(bool,uint256,(uint16))",
            .expected_words = 3,
        },
        .{
            .name = "struct_error_payload",
            .ty = struct_error_result,
            .expected_layout = null,
            .expected_words = null,
        },
        .{
            .name = "contract_ref",
            .ty = contract_ref_ty,
            .expected_layout = null,
            .expected_words = null,
        },
        .{
            .name = "contract_error_payload",
            .ty = contract_error_result,
            .expected_layout = null,
            .expected_words = null,
        },
        .{
            .name = "multi_error_result",
            .ty = multi_error_result,
            .expected_layout = null,
            .expected_words = null,
        },
        .{
            .name = "no_base_bitfield",
            .ty = no_base_bitfield_ty,
            .expected_layout = null,
            .expected_words = null,
        },
        .{
            .name = "no_base_enum",
            .ty = no_base_enum_ty,
            .expected_layout = "uint256",
            .expected_words = 1,
        },
    };

    for (cases) |case| {
        if (case.expected_layout) |expected| {
            const via_module = try moduleLoweringAbiLayoutForType(allocator, ast_file, item_index, typecheck, case.ty);
            defer allocator.free(via_module);
            try testing.expectEqualStrings(expected, via_module);

            const via_context = try ctx.canonicalAbiTypeForType(case.ty);
            defer allocator.free(via_context);
            try testing.expectEqualStrings(expected, via_context);
        } else {
            try testing.expectError(error.UnsupportedAbiType, moduleLoweringAbiLayoutForType(allocator, ast_file, item_index, typecheck, case.ty));
            try testing.expectError(error.UnsupportedAbiType, ctx.canonicalAbiTypeForType(case.ty));
        }

        try expectOptionalUsizeEqual(case.expected_words, moduleLoweringStaticWordCountForType(allocator, ast_file, item_index, typecheck, case.ty));
        try expectOptionalUsizeEqual(case.expected_words, ctx.staticWordCountForType(case.ty));
    }
}

test "abi generator resolves bitfield base type width" {
    const allocator = testing.allocator;
    const source =
        \\bitfield Flags : u8 {
        \\    enabled: bool @bits(0..1);
        \\    code: u8 @bits(1..8);
        \\}
        \\
        \\contract C {
        \\    pub fn set(flags: Flags) -> Flags {
        \\        return flags;
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();

    const callable = findCallable(&fixture.contract_abi, .function, "set") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), callable.inputs.len);
    try testing.expectEqual(@as(usize, 1), callable.outputs.len);

    const input_type = fixture.contract_abi.findType(callable.inputs[0].type_id) orelse return error.TestUnexpectedResult;
    const output_type = fixture.contract_abi.findType(callable.outputs[0].type_id) orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("uint8", input_type.wire_type.?);
    try testing.expectEqualStrings("uint8", output_type.wire_type.?);
}

test "abi layout serializes to MLIR attribute DSL" {
    const allocator = testing.allocator;
    const sema = compiler.sema;

    const u256_ty: sema.Type = .{ .integer = .{ .bits = 256, .signed = false, .spelling = "u256" } };
    const string_ty: sema.Type = .string;
    const string_array_ty: sema.Type = .{ .slice = .{ .element_type = &string_ty } };
    const tuple_elems = [_]sema.Type{ u256_ty, string_array_ty };
    const tuple_ty: sema.Type = .{ .tuple = &tuple_elems };

    var layout = try abi_layout.fromType(allocator, tuple_ty);
    defer layout.deinit(allocator);

    const serialized = try abi_layout.serializeForMlirAttr(allocator, layout);
    defer allocator.free(serialized);
    try testing.expectEqualStrings("tuple(static(uint256),array(dynamic,dynamic(string)))", serialized);
}

test "abi layout rejects non-wire Ora types" {
    const allocator = testing.allocator;

    try testing.expectError(error.UnsupportedAbiType, abi_layout.canonicalAbiTypeFromType(allocator, .{ .enum_ = .{ .name = "Status" } }));
    try testing.expectError(error.UnsupportedAbiType, abi_layout.canonicalAbiTypeFromType(allocator, .{ .function = .{ .name = "f" } }));
    try testing.expectError(error.UnsupportedAbiType, abi_layout.canonicalAbiTypeFromType(allocator, .{ .map = .{} }));
    try testing.expectError(error.UnsupportedAbiType, abi_layout.canonicalAbiTypeFromType(allocator, .{ .external_proxy = .{ .trait_name = "ERC20" } }));
}

test "abi manifest includes functions errors events effects and hashed type ids" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    storage var counter: u256;
        \\    error InvalidAmount(amount: u256);
        \\    log Transfer(indexed from: address, amount: u256);
        \\    pub fn pureFn() -> u256 { let x: u256 = 1; return x; }
        \\    pub fn viewFn() -> u256 { return counter; }
        \\    pub fn writeFn() { counter = 1; }
        \\    pub fn transferLike(from: address, amount: u256) { counter = amount; }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    try testing.expectEqual(@as(usize, 1), contract_abi.contract_count);
    try testing.expectEqualStrings("Test", contract_abi.contract_name);

    const pure_fn = findCallable(contract_abi, .function, "pureFn") orelse return error.TestUnexpectedResult;
    const view_fn = findCallable(contract_abi, .function, "viewFn") orelse return error.TestUnexpectedResult;
    const write_fn = findCallable(contract_abi, .function, "writeFn") orelse return error.TestUnexpectedResult;
    const invalid_amount = findCallable(contract_abi, .@"error", "InvalidAmount") orelse return error.TestUnexpectedResult;
    const transfer = findCallable(contract_abi, .event, "Transfer") orelse return error.TestUnexpectedResult;

    try testing.expect(pure_fn.selector != null);
    try testing.expectEqual(@as(usize, 1), pure_fn.outputs.len);
    try testing.expectEqual(@as(usize, 0), pure_fn.effects.len);

    try testing.expect(view_fn.selector != null);
    try testing.expect(hasEffectKind(view_fn, .reads));

    try testing.expect(write_fn.selector != null);
    try testing.expect(hasEffectKind(write_fn, .writes));

    try testing.expect(invalid_amount.selector != null);
    try testing.expectEqual(@as(usize, 1), invalid_amount.inputs.len);

    try testing.expect(transfer.selector == null);
    try testing.expectEqual(@as(usize, 2), transfer.inputs.len);
    try testing.expect(transfer.inputs[0].indexed orelse false);
    try testing.expect(!(transfer.inputs[1].indexed orelse false));

    for (contract_abi.types) |typ| {
        const type_id = typ.type_id orelse return error.TestUnexpectedResult;
        try testing.expect(std.mem.startsWith(u8, type_id, "t:"));
    }

    var u256_count: usize = 0;
    for (contract_abi.types) |typ| {
        if (typ.name) |name| {
            if (std.mem.eql(u8, name, "u256")) u256_count += 1;
        }
    }
    try testing.expectEqual(@as(usize, 1), u256_count);

    const manifest_json = try contract_abi.toJson(allocator);
    defer allocator.free(manifest_json);

    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"schemaVersion\":\"ora-abi-0.1\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"contract\":{\"name\":\"Test\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"wireProfiles\":[{\"id\":\"evm-default\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"kind\":\"error\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"kind\":\"event\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"kind\":\"reads\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"kind\":\"writes\"") != null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"group\":\"Read\"") == null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"dangerLevel\"") == null);
    try testing.expect(std.mem.indexOf(u8, manifest_json, "\"messageTemplate\"") == null);

    const extras_json = try contract_abi.toExtrasJson(allocator);
    defer allocator.free(extras_json);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"schemaVersion\":\"ora-abi-extras-0.1\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"baseSchemaVersion\":\"ora-abi-0.1\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"ui\":{\"group\":\"Read\",\"dangerLevel\":\"info\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"ui\":{\"group\":\"Write\",\"dangerLevel\":\"normal\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"forms\":{\"from\":{\"widget\":\"address\"},\"amount\":{\"widget\":\"number\"}}") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"group\":\"Errors\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"messageTemplate\":\"InvalidAmount: amount={amount}\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"group\":\"Events\"") != null);
    try testing.expect(std.mem.indexOf(u8, extras_json, "\"ui\":{\"widget\":\"number\"}") != null);

    const solidity_json = try contract_abi.toSolidityJson(allocator);
    defer allocator.free(solidity_json);

    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"stateMutability\":\"pure\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"stateMutability\":\"view\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"stateMutability\":\"nonpayable\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"type\":\"error\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"type\":\"event\"") != null);
}

test "abi manifest uses pinned event_name metadata for event wire identity" {
    const allocator = testing.allocator;
    const source =
        \\contract Test {
        \\    log TransferV2(indexed from: address, amount: u256) {
        \\        pub const event_name = "Transfer";
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    try testing.expect(findCallable(contract_abi, .event, "TransferV2") == null);
    const transfer = findCallable(contract_abi, .event, "Transfer") orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings("Transfer(address,uint256)", transfer.signature);
}

test "abi emits one bundle for multiple contracts and disambiguates callable ids" {
    const allocator = testing.allocator;
    const source =
        \\contract A {
        \\    error Boom(code: u256);
        \\    pub fn ping() {}
        \\}
        \\contract B {
        \\    error Boom(code: u256);
        \\    pub fn ping() {}
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    try testing.expectEqual(@as(usize, 2), contract_abi.contract_count);
    try testing.expectEqualStrings("bundle", contract_abi.contract_name);

    try testing.expectEqual(@as(usize, 2), countCallables(contract_abi, .function, "ping"));
    try testing.expectEqual(@as(usize, 2), countCallables(contract_abi, .@"error", "Boom"));

    var found_a_fn = false;
    var found_b_fn = false;
    var found_a_error = false;
    var found_b_error = false;

    for (contract_abi.callables) |callable| {
        if (std.mem.eql(u8, callable.id, "c:A.ping()")) found_a_fn = true;
        if (std.mem.eql(u8, callable.id, "c:B.ping()")) found_b_fn = true;
        if (std.mem.eql(u8, callable.id, "c:A.Boom(uint256)")) found_a_error = true;
        if (std.mem.eql(u8, callable.id, "c:B.Boom(uint256)")) found_b_error = true;
    }

    try testing.expect(found_a_fn);
    try testing.expect(found_b_fn);
    try testing.expect(found_a_error);
    try testing.expect(found_b_error);
}

test "abi models init as constructor instead of runtime function" {
    const allocator = testing.allocator;
    const source =
        \\contract Token {
        \\    storage var owner: address;
        \\
        \\    pub fn init(owner_arg: address) {
        \\        owner = owner_arg;
        \\    }
        \\
        \\    pub fn run() -> address { return owner; }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    const ctor = findCallable(contract_abi, .constructor, "init") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), ctor.inputs.len);
    try testing.expectEqual(@as(usize, 0), ctor.outputs.len);
    try testing.expect(ctor.selector == null);
    try testing.expectEqual(@as(usize, 0), countCallables(contract_abi, .function, "init"));

    const solidity_json = try contract_abi.toSolidityJson(allocator);
    defer allocator.free(solidity_json);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"type\":\"constructor\"") != null);
    try testing.expect(std.mem.indexOf(u8, solidity_json, "\"name\":\"init\"") == null);
}

test "abi type ids are stable across repeated generation for same source" {
    const allocator = testing.allocator;
    const source =
        \\contract Stable {
        \\    pub fn f(a: u256, b: address) -> u256 { return a; }
        \\}
    ;

    var fixture_a = try generateAbiForSource(allocator, source);
    defer fixture_a.deinit();
    const abi_a = &fixture_a.contract_abi;

    var fixture_b = try generateAbiForSource(allocator, source);
    defer fixture_b.deinit();
    const abi_b = &fixture_b.contract_abi;

    try testing.expectEqual(abi_a.types.len, abi_b.types.len);

    for (abi_a.types) |left| {
        const left_id = left.type_id orelse return error.TestUnexpectedResult;
        var found = false;
        for (abi_b.types) |right| {
            const right_id = right.type_id orelse return error.TestUnexpectedResult;
            if (std.mem.eql(u8, left_id, right_id)) {
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }

    const json_a = try abi_a.toJson(allocator);
    defer allocator.free(json_a);
    const json_b = try abi_b.toJson(allocator);
    defer allocator.free(json_b);

    try testing.expectEqualStrings(json_a, json_b);
}

test "abi projects public Result return as payload output" {
    const allocator = testing.allocator;
    const source =
        \\error Failure();
        \\contract MatchContract {
        \\    pub fn quote(flag: bool, amount: u256) -> Result<u256, Failure> {
        \\        if (flag) {
        \\            return Ok(amount);
        \\        }
        \\        return Err(Failure());
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    const quote = findCallable(contract_abi, .function, "quote") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 2), quote.inputs.len);
    try testing.expectEqual(@as(usize, 1), quote.outputs.len);

    const output_type = contract_abi.findType(quote.outputs[0].type_id) orelse return error.TestUnexpectedResult;
    try testing.expect(output_type.wire_type != null);
    try testing.expectEqualStrings("uint256", output_type.wire_type.?);
}

test "abi exposes supported public Result input as tagged tuple" {
    const allocator = testing.allocator;
    const source =
        \\error Failure();
        \\contract MatchContract {
        \\    pub fn run(value: Result<u256, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    const run = findCallable(contract_abi, .function, "run") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), run.inputs.len);

    const input_type = contract_abi.findType(run.inputs[0].type_id) orelse return error.TestUnexpectedResult;
    try testing.expect(input_type.wire_type != null);
    try testing.expectEqualStrings("(bool,uint256)", input_type.wire_type.?);
}

test "abi exposes payload-carrying public Result input as tagged triple" {
    const allocator = testing.allocator;
    const source =
        \\error Failure(code: u256);
        \\contract MatchContract {
        \\    pub fn run(value: Result<u256, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    const run = findCallable(contract_abi, .function, "run") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), run.inputs.len);

    const input_type = contract_abi.findType(run.inputs[0].type_id) orelse return error.TestUnexpectedResult;
    try testing.expect(input_type.wire_type != null);
    try testing.expectEqualStrings("(bool,uint256,uint256)", input_type.wire_type.?);
}

test "abi exposes multi-word public Result input with static payload layouts" {
    const allocator = testing.allocator;
    const source =
        \\struct Pair {
        \\    left: u256,
        \\    right: u256,
        \\}
        \\error Failure(code: u256, owner: address);
        \\contract MatchContract {
        \\    pub fn run(value: Result<Pair, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => inner.left,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    const run = findCallable(contract_abi, .function, "run") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), run.inputs.len);

    const input_type = contract_abi.findType(run.inputs[0].type_id) orelse return error.TestUnexpectedResult;
    try testing.expect(input_type.wire_type != null);
    try testing.expectEqualStrings("(bool,(uint256,uint256),(uint256,address))", input_type.wire_type.?);
}

test "abi exposes dynamic bytes public Result input as tagged tuple" {
    const allocator = testing.allocator;
    const source =
        \\error Failure();
        \\contract MatchContract {
        \\    pub fn run(value: Result<bytes, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => 1,
        \\            Err(err) => 0,
        \\        };
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    const run = findCallable(contract_abi, .function, "run") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), run.inputs.len);

    const input_type = contract_abi.findType(run.inputs[0].type_id) orelse return error.TestUnexpectedResult;
    try testing.expect(input_type.wire_type != null);
    try testing.expectEqualStrings("(bool,bytes)", input_type.wire_type.?);
}

test "abi exposes dynamic bytes public Result input with payload error as tagged triple" {
    const allocator = testing.allocator;
    const source =
        \\error Failure(code: u256);
        \\contract MatchContract {
        \\    pub fn run(value: Result<bytes, Failure>) -> u256 {
        \\        return match (value) {
        \\            Ok(inner) => 1,
        \\            Err(err) => err.code,
        \\        };
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    const run = findCallable(contract_abi, .function, "run") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), run.inputs.len);

    const input_type = contract_abi.findType(run.inputs[0].type_id) orelse return error.TestUnexpectedResult;
    try testing.expect(input_type.wire_type != null);
    try testing.expectEqualStrings("(bool,bytes,uint256)", input_type.wire_type.?);
}

test "abi exposes dynamic slice public Result input with payload error as tagged triple" {
    const allocator = testing.allocator;
    const source =
        \\error Failure(code: u256);
        \\contract MatchContract {
        \\    pub fn run(value: Result<slice[u256], Failure>) -> u256 {
        \\        let total = 0;
        \\        match (value) {
        \\            Ok(inner) => {
        \\                for (inner) |item| {
        \\                    total = total + item;
        \\                }
        \\            },
        \\            Err(err) => {
        \\                total = err.code;
        \\            }
        \\        }
        \\        return total;
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();
    const contract_abi = &fixture.contract_abi;

    const run = findCallable(contract_abi, .function, "run") orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 1), run.inputs.len);

    const input_type = contract_abi.findType(run.inputs[0].type_id) orelse return error.TestUnexpectedResult;
    try testing.expect(input_type.wire_type != null);
    try testing.expectEqualStrings("(bool,uint256[],uint256)", input_type.wire_type.?);
}

test "abi manifest Result input carrier projection follows layout context plan" {
    const allocator = testing.allocator;
    const source =
        \\error Empty();
        \\error WithCode(code: u16);
        \\
        \\struct Pair {
        \\    left: u256,
        \\    right: bool,
        \\}
        \\
        \\contract Manifest {
        \\    pub fn converge(
        \\        no_payload: Result<u256, Empty>,
        \\        payload: Result<u256, WithCode>,
        \\        struct_payload: Result<Pair, WithCode>,
        \\        dynamic_payload: Result<bytes, WithCode>
        \\    ) -> u256 {
        \\        return 0;
        \\    }
        \\}
    ;

    var fixture = try generateAbiForSource(allocator, source);
    defer fixture.deinit();

    const module = fixture.compilation.db.sources.module(fixture.compilation.root_module_id);
    const ast_file = try fixture.compilation.db.astFile(module.file_id);
    const item_index = try fixture.compilation.db.itemIndex(fixture.compilation.root_module_id);
    const typecheck = try fixture.compilation.db.moduleTypeCheck(fixture.compilation.root_module_id);
    const contract = ast_file.item(item_index.lookup("Manifest").?).Contract;
    const function_ref = findContractFunction(ast_file, contract, "converge") orelse return error.TestUnexpectedResult;
    const function_type = switch (typecheck.item_types[function_ref.item_id.index()]) {
        .function => |function| function,
        else => return error.TestUnexpectedResult,
    };
    const callable = findCallable(&fixture.contract_abi, .function, "converge") orelse return error.TestUnexpectedResult;
    const ctx = testLayoutContext(allocator, ast_file, item_index, typecheck);

    try testing.expectEqual(function_type.param_types.len, callable.inputs.len);
    for (function_type.param_types, callable.inputs) |param_ty, input| {
        try expectManifestResultInputMatchesPlan(&fixture.contract_abi, &ctx, param_ty, input.type_id);
    }
}
