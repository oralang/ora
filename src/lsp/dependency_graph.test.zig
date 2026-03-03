const std = @import("std");
const testing = std.testing;
const ora_root = @import("ora_root");

const graph_mod = ora_root.lsp.dependency_graph;

fn contains(values: []const []const u8, candidate: []const u8) bool {
    for (values) |value| {
        if (std.mem.eql(u8, value, candidate)) return true;
    }
    return false;
}

test "dependency graph: collect transitive dependents" {
    const allocator = testing.allocator;
    var graph = graph_mod.Graph.init(allocator);
    defer graph.deinit();

    const c_imports = [_][]const u8{};
    try graph.upsert("file:///tmp/c.ora", "/tmp/c.ora", c_imports[0..]);

    const b_imports = [_][]const u8{"/tmp/c.ora"};
    try graph.upsert("file:///tmp/b.ora", "/tmp/b.ora", b_imports[0..]);

    const a_imports = [_][]const u8{"/tmp/b.ora"};
    try graph.upsert("file:///tmp/a.ora", "/tmp/a.ora", a_imports[0..]);

    const dependents = try graph.collectDependents(allocator, "/tmp/c.ora");
    defer allocator.free(dependents);

    try testing.expectEqual(@as(usize, 2), dependents.len);
    try testing.expect(contains(dependents, "file:///tmp/b.ora"));
    try testing.expect(contains(dependents, "file:///tmp/a.ora"));
}

test "dependency graph: remove document updates reverse index" {
    const allocator = testing.allocator;
    var graph = graph_mod.Graph.init(allocator);
    defer graph.deinit();

    const b_imports = [_][]const u8{"/tmp/c.ora"};
    try graph.upsert("file:///tmp/b.ora", "/tmp/b.ora", b_imports[0..]);

    const removed_path = try graph.remove("file:///tmp/b.ora");
    try testing.expect(removed_path != null);
    defer allocator.free(removed_path.?);

    const dependents = try graph.collectDependents(allocator, "/tmp/c.ora");
    defer allocator.free(dependents);

    try testing.expectEqual(@as(usize, 0), dependents.len);
}

test "dependency graph: upsert replaces previous imports" {
    const allocator = testing.allocator;
    var graph = graph_mod.Graph.init(allocator);
    defer graph.deinit();

    const initial_imports = [_][]const u8{"/tmp/old.ora"};
    try graph.upsert("file:///tmp/a.ora", "/tmp/a.ora", initial_imports[0..]);

    const updated_imports = [_][]const u8{"/tmp/new.ora"};
    try graph.upsert("file:///tmp/a.ora", "/tmp/a.ora", updated_imports[0..]);

    const old_dependents = try graph.collectDependents(allocator, "/tmp/old.ora");
    defer allocator.free(old_dependents);
    try testing.expectEqual(@as(usize, 0), old_dependents.len);

    const new_dependents = try graph.collectDependents(allocator, "/tmp/new.ora");
    defer allocator.free(new_dependents);
    try testing.expectEqual(@as(usize, 1), new_dependents.len);
    try testing.expectEqualStrings("file:///tmp/a.ora", new_dependents[0]);
}
