const std = @import("std");
const lib = @import("ora");
const tio = lib.ast.type_info;

fn renderType(allocator: std.mem.Allocator, t: tio.OraType) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    try t.render(buf.writer());
    return buf.toOwnedSlice();
}

test "OraType.render anonymous struct and unions" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    // struct { a: u32, b: bool }
    const field_type0 = try a.create(tio.OraType);
    defer a.destroy(field_type0);
    field_type0.* = tio.OraType.u32;
    const field_type1 = try a.create(tio.OraType);
    defer a.destroy(field_type1);
    field_type1.* = tio.OraType.bool;

    var anon_fields = try a.alloc(tio.AnonymousStructFieldType, 2);
    defer a.free(anon_fields);
    anon_fields[0] = .{ .name = "a", .typ = field_type0 };
    anon_fields[1] = .{ .name = "b", .typ = field_type1 };

    const anon = tio.OraType{ .anonymous_struct = anon_fields };
    const s1 = try renderType(a, anon);
    defer a.free(s1);
    try std.testing.expect(std.mem.indexOf(u8, s1, "struct {") != null);

    // !u256 | ErrorType
    const succ_ptr = try a.create(tio.OraType);
    defer a.destroy(succ_ptr);
    succ_ptr.* = tio.OraType.u256;
    var union_members = try a.alloc(tio.OraType, 2);
    defer a.free(union_members);
    union_members[0] = tio.OraType{ .error_union = succ_ptr };
    union_members[1] = tio.OraType{ .struct_type = "ErrorType" };
    const eu = tio.OraType{ ._union = union_members };
    const s2 = try renderType(a, eu);
    defer a.free(s2);
    try std.testing.expect(std.mem.indexOf(u8, s2, "!") != null);
    try std.testing.expect(std.mem.indexOf(u8, s2, "|") != null);

    // map[u8, slice[bool]]
    const map_key = try a.create(tio.OraType);
    defer a.destroy(map_key);
    map_key.* = tio.OraType.u8;
    const map_val_elem = try a.create(tio.OraType);
    defer a.destroy(map_val_elem);
    map_val_elem.* = tio.OraType.bool;
    const map_val = tio.OraType{ .slice = map_val_elem };
    const mapping = tio.MapType{ .key = map_key, .value = &map_val };
    const m = tio.OraType{ .map = mapping };
    const s3 = try renderType(a, m);
    defer a.free(s3);
    try std.testing.expect(std.mem.indexOf(u8, s3, "map[") != null);
}

// Simple structural equality helper for tests
fn typesEqual(a: tio.OraType, b: tio.OraType) bool {
    return tio.OraType.equals(a, b);
}

fn typeHash(a: tio.OraType) u64 {
    return tio.OraType.hash(a);
}

test "OraType equals/hash basic" {
    try std.testing.expect(typesEqual(tio.OraType.u32, tio.OraType.u32));
    try std.testing.expect(!typesEqual(tio.OraType.u32, tio.OraType.u64));
    try std.testing.expect(typeHash(tio.OraType.u32) == typeHash(tio.OraType.u32));
}
