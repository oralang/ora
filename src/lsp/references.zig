const std = @import("std");
const definition = @import("definition.zig");
const frontend = @import("frontend.zig");
const token_cache = @import("token_cache.zig");

const Allocator = std.mem.Allocator;

pub const Occurrence = struct {
    name: []const u8,
    range: frontend.Range,
    definition_range: frontend.Range,
};

pub const OccurrenceIndex = struct {
    occurrences: []const Occurrence,
    builder_capacity_requested: usize = 0,
    builder_items_built: usize = 0,
    builder_unused_capacity: usize = 0,
    builder_growth_events: usize = 0,

    pub fn init(
        allocator: Allocator,
        source: []const u8,
        tokens: []const token_cache.Token,
        analysis: *definition.Analysis,
    ) !OccurrenceIndex {
        var occurrences = std.ArrayList(Occurrence){};
        var builder_growth_events: usize = 0;
        errdefer {
            deinitOccurrences(allocator, occurrences.items);
            occurrences.deinit(allocator);
        }

        for (tokens) |token| {
            if (token.type != .Identifier) continue;

            const token_range = tokenSelectionRange(token);
            const maybe_def = definition.definitionAtCached(analysis, source, token_range.start);
            if (maybe_def == null) continue;

            const name_copy = try allocator.dupe(u8, token.lexeme);
            errdefer allocator.free(name_copy);

            const capacity_before = occurrences.capacity;
            try occurrences.append(allocator, .{
                .name = name_copy,
                .range = token_range,
                .definition_range = maybe_def.?.range,
            });
            if (occurrences.capacity > capacity_before) {
                builder_growth_events += 1;
            }
        }

        const builder_capacity_requested = occurrences.capacity;
        const builder_items_built = occurrences.items.len;
        return .{
            .occurrences = try occurrences.toOwnedSlice(allocator),
            .builder_capacity_requested = builder_capacity_requested,
            .builder_items_built = builder_items_built,
            .builder_unused_capacity = if (builder_capacity_requested > builder_items_built) builder_capacity_requested - builder_items_built else 0,
            .builder_growth_events = builder_growth_events,
        };
    }

    pub fn deinit(self: *OccurrenceIndex, allocator: Allocator) void {
        deinitOccurrences(allocator, self.occurrences);
        allocator.free(self.occurrences);
        self.* = undefined;
    }

    pub fn occurrenceAt(self: *const OccurrenceIndex, position: frontend.Position) ?Occurrence {
        for (self.occurrences) |occurrence| {
            if (positionInRange(position, occurrence.range)) return occurrence;
        }
        return null;
    }

    pub fn estimatedByteSize(self: *const OccurrenceIndex) usize {
        var total: usize = self.occurrences.len * @sizeOf(Occurrence);
        for (self.occurrences) |occurrence| {
            total = addSat(total, occurrence.name.len);
        }
        return total;
    }

    pub fn builderCapacityRequested(self: *const OccurrenceIndex) usize {
        return self.builder_capacity_requested;
    }

    pub fn builderItemsBuilt(self: *const OccurrenceIndex) usize {
        return self.builder_items_built;
    }

    pub fn builderUnusedCapacity(self: *const OccurrenceIndex) usize {
        return self.builder_unused_capacity;
    }

    pub fn builderGrowthEvents(self: *const OccurrenceIndex) usize {
        return self.builder_growth_events;
    }
};

pub const ImportedMemberOccurrence = struct {
    imported_path: []const u8,
    alias: []const u8,
    member_name: []const u8,
    range: frontend.Range,
};

pub const ImportedMemberIndex = struct {
    occurrences: []const ImportedMemberOccurrence,
    builder_capacity_requested: usize = 0,
    builder_items_built: usize = 0,
    builder_unused_capacity: usize = 0,
    builder_growth_events: usize = 0,

    pub fn estimatedByteSize(self: *const ImportedMemberIndex) usize {
        var total: usize = self.occurrences.len * @sizeOf(ImportedMemberOccurrence);
        for (self.occurrences) |occurrence| {
            total = addSat(total, occurrence.imported_path.len);
            total = addSat(total, occurrence.alias.len);
            total = addSat(total, occurrence.member_name.len);
        }
        return total;
    }

    pub fn builderCapacityRequested(self: *const ImportedMemberIndex) usize {
        return self.builder_capacity_requested;
    }

    pub fn builderItemsBuilt(self: *const ImportedMemberIndex) usize {
        return self.builder_items_built;
    }

    pub fn builderUnusedCapacity(self: *const ImportedMemberIndex) usize {
        return self.builder_unused_capacity;
    }

    pub fn builderGrowthEvents(self: *const ImportedMemberIndex) usize {
        return self.builder_growth_events;
    }
};

pub const ImportBinding = struct {
    alias: []const u8,
    resolved_path: []const u8,
};

pub fn referencesAt(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
    include_declaration: bool,
) ![]frontend.Range {
    var analysis = (try definition.Analysis.init(allocator, source)) orelse
        return try allocator.alloc(frontend.Range, 0);
    defer analysis.deinit();

    var tokens = try token_cache.Cache.init(allocator, source);
    defer tokens.deinit(allocator);

    return referencesAtCached(allocator, source, position, include_declaration, tokens.tokens, &analysis);
}

pub fn referencesAtCached(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
    include_declaration: bool,
    tokens: []const token_cache.Token,
    analysis: *definition.Analysis,
) ![]frontend.Range {
    const target_range = blk: {
        const def = definition.definitionAtCached(analysis, source, position) orelse
            return try allocator.alloc(frontend.Range, 0);
        break :blk def.range;
    };

    var ranges = std.ArrayList(frontend.Range){};
    errdefer ranges.deinit(allocator);

    for (tokens) |token| {
        if (token.type != .Identifier) continue;

        const token_range = tokenSelectionRange(token);
        const maybe_def = definition.definitionAtCached(analysis, source, token_range.start);
        if (maybe_def == null) continue;

        if (!rangesEqual(maybe_def.?.range, target_range)) continue;
        if (!include_declaration and rangesEqual(token_range, target_range)) continue;

        try appendUniqueRange(allocator, &ranges, token_range);
    }

    if (include_declaration) {
        try appendUniqueRange(allocator, &ranges, target_range);
    }

    return ranges.toOwnedSlice(allocator);
}

pub fn referencesAtOccurrenceIndex(
    allocator: Allocator,
    index: *const OccurrenceIndex,
    target_name: []const u8,
    target_definition_range: frontend.Range,
    include_declaration: bool,
) ![]frontend.Range {
    var ranges = std.ArrayList(frontend.Range){};
    errdefer ranges.deinit(allocator);

    for (index.occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.name, target_name)) continue;
        if (!rangesEqual(occurrence.definition_range, target_definition_range)) continue;
        if (!include_declaration and rangesEqual(occurrence.range, target_definition_range)) continue;
        try appendUniqueRange(allocator, &ranges, occurrence.range);
    }

    if (include_declaration) {
        try appendUniqueRange(allocator, &ranges, target_definition_range);
    }

    return ranges.toOwnedSlice(allocator);
}

pub fn buildImportedMemberIndex(
    allocator: Allocator,
    source: []const u8,
    imports: []const ImportBinding,
) !ImportedMemberIndex {
    return buildImportedMemberIndexWithScratch(allocator, allocator, source, imports);
}

pub fn buildImportedMemberIndexWithScratch(
    result_allocator: Allocator,
    scratch_allocator: Allocator,
    source: []const u8,
    imports: []const ImportBinding,
) !ImportedMemberIndex {
    var cache = try token_cache.Cache.initWithScratch(scratch_allocator, scratch_allocator, source);
    defer cache.deinit(scratch_allocator);

    return buildImportedMemberIndexFromTokens(result_allocator, cache.tokens, imports);
}

pub fn buildImportedMemberIndexFromTokens(
    allocator: Allocator,
    tokens: []const token_cache.Token,
    imports: []const ImportBinding,
) !ImportedMemberIndex {
    if (imports.len == 0) {
        return emptyImportedMemberIndex(allocator);
    }

    var occurrences = std.ArrayList(ImportedMemberOccurrence){};
    var builder_growth_events: usize = 0;
    errdefer {
        deinitImportedMemberOccurrences(allocator, occurrences.items);
        occurrences.deinit(allocator);
    }

    var i: usize = 0;
    while (i + 2 < tokens.len) : (i += 1) {
        const alias_token = tokens[i];
        if (alias_token.type != .Identifier) continue;
        if (tokens[i + 1].type != .Dot) continue;

        const member_token = tokens[i + 2];
        if (member_token.type != .Identifier) continue;

        const import_binding = findImportBinding(imports, alias_token.lexeme) orelse continue;
        const capacity_before = occurrences.capacity;
        try appendImportedMemberOccurrence(
            allocator,
            &occurrences,
            import_binding,
            member_token.lexeme,
            tokenSelectionRange(member_token),
        );
        if (occurrences.capacity > capacity_before) {
            builder_growth_events += 1;
        }
        i += 2;
    }

    const builder_capacity_requested = occurrences.capacity;
    const builder_items_built = occurrences.items.len;
    return .{
        .occurrences = try occurrences.toOwnedSlice(allocator),
        .builder_capacity_requested = builder_capacity_requested,
        .builder_items_built = builder_items_built,
        .builder_unused_capacity = if (builder_capacity_requested > builder_items_built) builder_capacity_requested - builder_items_built else 0,
        .builder_growth_events = builder_growth_events,
    };
}

pub fn deinitImportedMemberIndex(allocator: Allocator, index: *ImportedMemberIndex) void {
    deinitImportedMemberOccurrences(allocator, index.occurrences);
    allocator.free(index.occurrences);
    index.* = undefined;
}

pub fn importedMemberReferencesTo(
    allocator: Allocator,
    index: *const ImportedMemberIndex,
    target_path: []const u8,
    target_name: []const u8,
) ![]frontend.Range {
    var ranges = std.ArrayList(frontend.Range){};
    errdefer ranges.deinit(allocator);

    for (index.occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
        if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
        try appendUniqueRange(allocator, &ranges, occurrence.range);
    }

    return ranges.toOwnedSlice(allocator);
}

fn tokenSelectionRange(token: anytype) frontend.Range {
    const start_line = if (token.line > 0) token.line - 1 else 0;
    const start_char = if (token.column > 0) token.column - 1 else 0;

    const lexeme_len = std.math.cast(u32, token.lexeme.len) orelse std.math.maxInt(u32);
    const end_char = std.math.add(u32, start_char, lexeme_len) catch std.math.maxInt(u32);

    return .{
        .start = .{ .line = start_line, .character = start_char },
        .end = .{ .line = start_line, .character = end_char },
    };
}

fn appendUniqueRange(
    allocator: Allocator,
    ranges: *std.ArrayList(frontend.Range),
    range: frontend.Range,
) !void {
    for (ranges.items) |existing| {
        if (rangesEqual(existing, range)) return;
    }
    try ranges.append(allocator, range);
}

fn rangesEqual(a: frontend.Range, b: frontend.Range) bool {
    return a.start.line == b.start.line and
        a.start.character == b.start.character and
        a.end.line == b.end.line and
        a.end.character == b.end.character;
}

fn findImportBinding(imports: []const ImportBinding, alias: []const u8) ?ImportBinding {
    for (imports) |import_binding| {
        if (std.mem.eql(u8, import_binding.alias, alias)) return import_binding;
    }
    return null;
}

fn emptyImportedMemberIndex(allocator: Allocator) !ImportedMemberIndex {
    return .{ .occurrences = try allocator.alloc(ImportedMemberOccurrence, 0) };
}

fn appendImportedMemberOccurrence(
    allocator: Allocator,
    occurrences: *std.ArrayList(ImportedMemberOccurrence),
    import_binding: ImportBinding,
    member_name: []const u8,
    range: frontend.Range,
) !void {
    const imported_path_copy = try allocator.dupe(u8, import_binding.resolved_path);
    errdefer allocator.free(imported_path_copy);

    const alias_copy = try allocator.dupe(u8, import_binding.alias);
    errdefer allocator.free(alias_copy);

    const member_name_copy = try allocator.dupe(u8, member_name);
    errdefer allocator.free(member_name_copy);

    try occurrences.append(allocator, .{
        .imported_path = imported_path_copy,
        .alias = alias_copy,
        .member_name = member_name_copy,
        .range = range,
    });
}

fn deinitImportedMemberOccurrences(allocator: Allocator, occurrences: []const ImportedMemberOccurrence) void {
    for (occurrences) |occurrence| {
        allocator.free(occurrence.imported_path);
        allocator.free(occurrence.alias);
        allocator.free(occurrence.member_name);
    }
}

fn deinitOccurrences(allocator: Allocator, occurrences: []const Occurrence) void {
    for (occurrences) |occurrence| {
        allocator.free(occurrence.name);
    }
}

fn positionInRange(pos: frontend.Position, range: frontend.Range) bool {
    if (pos.line < range.start.line) return false;
    if (pos.line > range.end.line) return false;
    if (pos.line == range.start.line and pos.character < range.start.character) return false;
    if (pos.line == range.end.line and pos.character > range.end.character) return false;
    return true;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
