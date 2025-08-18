const std = @import("std");
const testing = std.testing;
const ast = @import("ast.zig");
const typer = @import("typer.zig");
const type_utilities = @import("type_utilities.zig");

test "TypeUtilities - exact type matching" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test primitive type matching
    const u256_type = ast.TypeRef.U256;
    const u128_type = ast.TypeRef.U128;
    const u256_type2 = ast.TypeRef.U256;

    try testing.expect(utils.isTypeRefExactMatch(&u256_type, &u256_type2));
    try testing.expect(!utils.isTypeRefExactMatch(&u256_type, &u128_type));

    // Test identifier matching
    const id1 = ast.TypeRef{ .Identifier = "MyStruct" };
    const id2 = ast.TypeRef{ .Identifier = "MyStruct" };
    const id3 = ast.TypeRef{ .Identifier = "OtherStruct" };

    try testing.expect(utils.isTypeRefExactMatch(&id1, &id2));
    try testing.expect(!utils.isTypeRefExactMatch(&id1, &id3));
}

test "TypeUtilities - type size calculation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test primitive type sizes
    try testing.expect(utils.getSize(&ast.TypeRef.Bool) == 1);
    try testing.expect(utils.getSize(&ast.TypeRef.U8) == 1);
    try testing.expect(utils.getSize(&ast.TypeRef.U16) == 2);
    try testing.expect(utils.getSize(&ast.TypeRef.U32) == 4);
    try testing.expect(utils.getSize(&ast.TypeRef.U64) == 8);
    try testing.expect(utils.getSize(&ast.TypeRef.U128) == 16);
    try testing.expect(utils.getSize(&ast.TypeRef.U256) == 32);
    try testing.expect(utils.getSize(&ast.TypeRef.Address) == 20);
}

test "TypeUtilities - type properties" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test primitive type identification
    try testing.expect(utils.isPrimitive(&ast.TypeRef.U256));
    try testing.expect(utils.isPrimitive(&ast.TypeRef.Bool));
    try testing.expect(utils.isPrimitive(&ast.TypeRef.Address));
    try testing.expect(!utils.isPrimitive(&ast.TypeRef.Unknown));

    // Test numeric type identification
    try testing.expect(utils.isNumeric(&ast.TypeRef.U256));
    try testing.expect(utils.isNumeric(&ast.TypeRef.I128));
    try testing.expect(!utils.isNumeric(&ast.TypeRef.Bool));
    try testing.expect(!utils.isNumeric(&ast.TypeRef.Address));

    // Test integer type identification
    try testing.expect(utils.isInteger(&ast.TypeRef.U256));
    try testing.expect(utils.isInteger(&ast.TypeRef.I128));
    try testing.expect(!utils.isInteger(&ast.TypeRef.Bool));

    // Test signed/unsigned identification
    try testing.expect(utils.isUnsignedInteger(&ast.TypeRef.U256));
    try testing.expect(!utils.isUnsignedInteger(&ast.TypeRef.I128));
    try testing.expect(utils.isSignedInteger(&ast.TypeRef.I128));
    try testing.expect(!utils.isSignedInteger(&ast.TypeRef.U256));
}

test "TypeUtilities - nullable types" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // In Ora, primitive types are not nullable
    try testing.expect(!utils.isNullable(&ast.TypeRef.U256));
    try testing.expect(!utils.isNullable(&ast.TypeRef.Bool));
    try testing.expect(!utils.isNullable(&ast.TypeRef.Address));
    try testing.expect(!utils.isNullable(&ast.TypeRef.String));
}

test "TypeUtilities - explicit annotation validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test explicit annotation requirements
    try testing.expect(utils.requiresExplicitAnnotation(&ast.TypeRef.Unknown));
    try testing.expect(!utils.requiresExplicitAnnotation(&ast.TypeRef.U256));
    try testing.expect(!utils.requiresExplicitAnnotation(&ast.TypeRef.Bool));

    // Test explicit annotation validation
    try testing.expect(try utils.validateExplicitAnnotation(&ast.TypeRef.U256));
    try testing.expect(try utils.validateExplicitAnnotation(&ast.TypeRef.Bool));
    try testing.expect(!try utils.validateExplicitAnnotation(&ast.TypeRef.Unknown));
}

test "TypeUtilities - alignment calculation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test alignment requirements
    try testing.expect(utils.getAlignment(&ast.TypeRef.Bool) == 1);
    try testing.expect(utils.getAlignment(&ast.TypeRef.U8) == 1);
    try testing.expect(utils.getAlignment(&ast.TypeRef.U16) == 2);
    try testing.expect(utils.getAlignment(&ast.TypeRef.U32) == 4);
    try testing.expect(utils.getAlignment(&ast.TypeRef.U64) == 8);
    try testing.expect(utils.getAlignment(&ast.TypeRef.U128) == 16);
    try testing.expect(utils.getAlignment(&ast.TypeRef.U256) == 32);
    try testing.expect(utils.getAlignment(&ast.TypeRef.Address) == 32);
}

test "TypeUtilities - mapping key validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test valid mapping keys
    try testing.expect(utils.isValidMappingKey(&ast.TypeRef.U256));
    try testing.expect(utils.isValidMappingKey(&ast.TypeRef.Address));
    try testing.expect(utils.isValidMappingKey(&ast.TypeRef.String));
    try testing.expect(utils.isValidMappingKey(&ast.TypeRef.Bool));

    // Test invalid mapping keys
    try testing.expect(!utils.isValidMappingKey(&ast.TypeRef.Unknown));
}

test "TypeUtilities - operation support" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test arithmetic operation support
    try testing.expect(utils.supportsArithmetic(&ast.TypeRef.U256));
    try testing.expect(utils.supportsArithmetic(&ast.TypeRef.I128));
    try testing.expect(!utils.supportsArithmetic(&ast.TypeRef.Bool));
    try testing.expect(!utils.supportsArithmetic(&ast.TypeRef.Address));

    // Test bitwise operation support
    try testing.expect(utils.supportsBitwise(&ast.TypeRef.U256));
    try testing.expect(utils.supportsBitwise(&ast.TypeRef.I128));
    try testing.expect(!utils.supportsBitwise(&ast.TypeRef.Bool));

    // Test comparison operation support
    try testing.expect(utils.supportsComparison(&ast.TypeRef.U256));
    try testing.expect(utils.supportsComparison(&ast.TypeRef.Bool));
    try testing.expect(utils.supportsComparison(&ast.TypeRef.Address));
    try testing.expect(utils.supportsComparison(&ast.TypeRef.String));
}

test "TypeUtilities - type string representation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test primitive type string representations
    try testing.expectEqualStrings("bool", try utils.toString(&ast.TypeRef.Bool));
    try testing.expectEqualStrings("u256", try utils.toString(&ast.TypeRef.U256));
    try testing.expectEqualStrings("i128", try utils.toString(&ast.TypeRef.I128));
    try testing.expectEqualStrings("address", try utils.toString(&ast.TypeRef.Address));
    try testing.expectEqualStrings("string", try utils.toString(&ast.TypeRef.String));

    // Test identifier type
    const id_type = ast.TypeRef{ .Identifier = "MyStruct" };
    try testing.expectEqualStrings("MyStruct", try utils.toString(&id_type));
}

test "TypeUtilities - implicit conversion (should always be false in Ora)" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // In Ora, no implicit conversions are allowed
    try testing.expect(!utils.isImplicitlyConvertible(&ast.TypeRef.U32, &ast.TypeRef.U64));
    try testing.expect(!utils.isImplicitlyConvertible(&ast.TypeRef.U8, &ast.TypeRef.U256));
    try testing.expect(!utils.isImplicitlyConvertible(&ast.TypeRef.I32, &ast.TypeRef.I64));

    // Only identical types are "implicitly convertible"
    try testing.expect(utils.isImplicitlyConvertible(&ast.TypeRef.U256, &ast.TypeRef.U256));
    try testing.expect(utils.isImplicitlyConvertible(&ast.TypeRef.Bool, &ast.TypeRef.Bool));
}

test "TypeUtilities - explicit conversion" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Most explicit conversions are allowed in Ora
    try testing.expect(utils.isExplicitlyConvertible(&ast.TypeRef.U32, &ast.TypeRef.U64));
    try testing.expect(utils.isExplicitlyConvertible(&ast.TypeRef.U64, &ast.TypeRef.U32));
    try testing.expect(utils.isExplicitlyConvertible(&ast.TypeRef.Bool, &ast.TypeRef.U8));

    // Cannot convert from/to unknown types
    try testing.expect(!utils.isExplicitlyConvertible(&ast.TypeRef.Unknown, &ast.TypeRef.U256));
    try testing.expect(!utils.isExplicitlyConvertible(&ast.TypeRef.U256, &ast.TypeRef.Unknown));
}

test "TypeUtilities - default values" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test default values for primitive types
    try testing.expectEqualStrings("false", try utils.getDefaultValue(&ast.TypeRef.Bool));
    try testing.expectEqualStrings("0", try utils.getDefaultValue(&ast.TypeRef.U256));
    try testing.expectEqualStrings("0", try utils.getDefaultValue(&ast.TypeRef.I128));
    try testing.expectEqualStrings("\"\"", try utils.getDefaultValue(&ast.TypeRef.String));
    try testing.expectEqualStrings("0x0000000000000000000000000000000000000000", try utils.getDefaultValue(&ast.TypeRef.Address));
}

test "TypeUtilities - container type identification" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test container type identification
    try testing.expect(!utils.isContainer(&ast.TypeRef.U256));
    try testing.expect(!utils.isContainer(&ast.TypeRef.Bool));
    try testing.expect(!utils.isContainer(&ast.TypeRef.Address));

    // Note: We can't easily test slice, mapping, etc. without creating the complex types
    // This would require allocating the inner types, which is more complex for this test
}

test "TypeUtilities - OraType exact matching" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var utils = type_utilities.TypeUtilities.init(allocator);

    // Test OraType exact matching
    try testing.expect(utils.isExactMatch(typer.OraType.U256, typer.OraType.U256));
    try testing.expect(!utils.isExactMatch(typer.OraType.U256, typer.OraType.U128));
    try testing.expect(utils.isExactMatch(typer.OraType.Bool, typer.OraType.Bool));
    try testing.expect(!utils.isExactMatch(typer.OraType.Bool, typer.OraType.Address));
};