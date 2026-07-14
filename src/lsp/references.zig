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

const RangeIndexEntry = struct {
    range: frontend.Range,
    item_index: u32,
};

const NameIndexEntry = struct {
    name: []const u8,
    item_index: u32,
};

pub const OccurrenceIndex = struct {
    occurrences: []const Occurrence,
    range_indexes: []const RangeIndexEntry = &.{},
    name_indexes: []const NameIndexEntry = &.{},
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
        _ = source;
        _ = tokens;

        const resolved_names = try definition.collectDefinitionsCached(allocator, analysis);
        defer allocator.free(resolved_names);

        var occurrences = std.ArrayList(Occurrence).empty;
        var builder_growth_events: usize = 0;
        errdefer {
            deinitOccurrences(allocator, occurrences.items);
            occurrences.deinit(allocator);
        }

        const occurrence_capacity_before = occurrences.capacity;
        try occurrences.ensureTotalCapacity(allocator, resolved_names.len);
        if (occurrences.capacity > occurrence_capacity_before) {
            builder_growth_events += 1;
        }

        for (resolved_names) |resolved| {
            const name_copy = try allocator.dupe(u8, resolved.name);
            errdefer allocator.free(name_copy);

            occurrences.appendAssumeCapacity(.{
                .name = name_copy,
                .range = resolved.range,
                .definition_range = resolved.definition_range,
            });
        }

        const range_index_builder = try buildRangeIndexes(allocator, occurrences.items);
        errdefer allocator.free(range_index_builder.items);
        const name_index_builder = try buildNameIndexes(allocator, occurrences.items);
        errdefer allocator.free(name_index_builder.items);

        const occurrences_capacity_requested = occurrences.capacity;
        const occurrence_items_built = occurrences.items.len;
        const occurrence_slice = try occurrences.toOwnedSlice(allocator);
        errdefer {
            deinitOccurrences(allocator, occurrence_slice);
            allocator.free(occurrence_slice);
        }

        const builder_capacity_requested = addSat(
            addSat(occurrences_capacity_requested, range_index_builder.capacity_requested),
            name_index_builder.capacity_requested,
        );
        const builder_items_built = addSat(
            addSat(occurrence_items_built, range_index_builder.items.len),
            name_index_builder.items.len,
        );
        return .{
            .occurrences = occurrence_slice,
            .range_indexes = range_index_builder.items,
            .name_indexes = name_index_builder.items,
            .builder_capacity_requested = builder_capacity_requested,
            .builder_items_built = builder_items_built,
            .builder_unused_capacity = if (builder_capacity_requested > builder_items_built) builder_capacity_requested - builder_items_built else 0,
            .builder_growth_events = builder_growth_events + range_index_builder.growth_events + name_index_builder.growth_events,
        };
    }

    pub fn deinit(self: *OccurrenceIndex, allocator: Allocator) void {
        deinitOccurrences(allocator, self.occurrences);
        allocator.free(self.occurrences);
        allocator.free(self.range_indexes);
        allocator.free(self.name_indexes);
        self.* = undefined;
    }

    pub fn occurrenceAt(self: *const OccurrenceIndex, position: frontend.Position) ?Occurrence {
        if (self.range_indexes.len == self.occurrences.len) {
            const index = occurrenceIndexAtPosition(self.occurrences, self.range_indexes, position) orelse return null;
            return self.occurrences[index];
        }
        for (self.occurrences) |occurrence| {
            if (positionInRange(position, occurrence.range)) return occurrence;
        }
        return null;
    }

    fn matchingNameIndexes(self: *const OccurrenceIndex, name: []const u8) []const NameIndexEntry {
        if (self.name_indexes.len != self.occurrences.len) return &.{};
        const start = lowerBoundName(self.name_indexes, name);
        const end = upperBoundName(self.name_indexes, name, start);
        return self.name_indexes[start..end];
    }

    pub fn estimatedByteSize(self: *const OccurrenceIndex) usize {
        var total: usize = addSat(
            self.occurrences.len * @sizeOf(Occurrence),
            self.range_indexes.len * @sizeOf(RangeIndexEntry),
        );
        total = addSat(total, self.name_indexes.len * @sizeOf(NameIndexEntry));
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

pub fn referencesAtCached(
    allocator: Allocator,
    source: []const u8,
    position: frontend.Position,
    include_declaration: bool,
    tokens: []const token_cache.Token,
    analysis: *definition.Analysis,
) ![]frontend.Range {
    var index = try OccurrenceIndex.init(allocator, source, tokens, analysis);
    defer index.deinit(allocator);

    const target = index.occurrenceAt(position) orelse return try allocator.alloc(frontend.Range, 0);
    return referencesAtOccurrenceIndex(
        allocator,
        &index,
        target.name,
        target.definition_range,
        include_declaration,
    );
}

pub fn referencesAtOccurrenceIndex(
    allocator: Allocator,
    index: *const OccurrenceIndex,
    target_name: []const u8,
    target_definition_range: frontend.Range,
    include_declaration: bool,
) ![]frontend.Range {
    var ranges = std.ArrayList(frontend.Range).empty;
    errdefer ranges.deinit(allocator);

    if (index.name_indexes.len == index.occurrences.len) {
        const matching_indexes = index.matchingNameIndexes(target_name);
        for (matching_indexes) |name_index| {
            const occurrence = index.occurrences[name_index.item_index];
            if (!rangesEqual(occurrence.definition_range, target_definition_range)) continue;
            if (!include_declaration and rangesEqual(occurrence.range, target_definition_range)) continue;
            try appendUniqueRange(allocator, &ranges, occurrence.range);
        }
    } else {
        for (index.occurrences) |occurrence| {
            if (!std.mem.eql(u8, occurrence.name, target_name)) continue;
            if (!rangesEqual(occurrence.definition_range, target_definition_range)) continue;
            if (!include_declaration and rangesEqual(occurrence.range, target_definition_range)) continue;
            try appendUniqueRange(allocator, &ranges, occurrence.range);
        }
    }

    if (include_declaration) {
        try appendUniqueRange(allocator, &ranges, target_definition_range);
    }

    return ranges.toOwnedSlice(allocator);
}

pub fn referenceRangeCountAtOccurrenceIndex(
    index: *const OccurrenceIndex,
    target_name: []const u8,
    target_definition_range: frontend.Range,
    include_declaration: bool,
) usize {
    var count: usize = 0;
    var declaration_seen = false;
    if (index.name_indexes.len == index.occurrences.len) {
        const matching_indexes = index.matchingNameIndexes(target_name);
        for (matching_indexes) |name_index| {
            const occurrence = index.occurrences[name_index.item_index];
            if (!rangesEqual(occurrence.definition_range, target_definition_range)) continue;
            if (rangesEqual(occurrence.range, target_definition_range)) {
                declaration_seen = true;
                if (!include_declaration) continue;
            }
            count += 1;
        }
    } else {
        for (index.occurrences) |occurrence| {
            if (!std.mem.eql(u8, occurrence.name, target_name)) continue;
            if (!rangesEqual(occurrence.definition_range, target_definition_range)) continue;
            if (rangesEqual(occurrence.range, target_definition_range)) {
                declaration_seen = true;
                if (!include_declaration) continue;
            }
            count += 1;
        }
    }
    if (include_declaration and !declaration_seen) count += 1;
    return count;
}

pub fn referenceRangeCapacityHintAtOccurrenceIndex(
    index: *const OccurrenceIndex,
    target_name: []const u8,
    include_declaration: bool,
) usize {
    const base = if (index.name_indexes.len == index.occurrences.len)
        index.matchingNameIndexes(target_name).len
    else
        index.occurrences.len;
    return base + @intFromBool(include_declaration);
}

pub fn appendReferenceRangesAtOccurrenceIndex(
    index: *const OccurrenceIndex,
    target_name: []const u8,
    target_definition_range: frontend.Range,
    include_declaration: bool,
    context: anytype,
    comptime appendRange: fn (@TypeOf(context), frontend.Range) anyerror!void,
) !void {
    var declaration_seen = false;
    if (index.name_indexes.len == index.occurrences.len) {
        const matching_indexes = index.matchingNameIndexes(target_name);
        for (matching_indexes) |name_index| {
            const occurrence = index.occurrences[name_index.item_index];
            if (!rangesEqual(occurrence.definition_range, target_definition_range)) continue;
            if (rangesEqual(occurrence.range, target_definition_range)) {
                declaration_seen = true;
                if (!include_declaration) continue;
            }
            try appendRange(context, occurrence.range);
        }
    } else {
        for (index.occurrences) |occurrence| {
            if (!std.mem.eql(u8, occurrence.name, target_name)) continue;
            if (!rangesEqual(occurrence.definition_range, target_definition_range)) continue;
            if (rangesEqual(occurrence.range, target_definition_range)) {
                declaration_seen = true;
                if (!include_declaration) continue;
            }
            try appendRange(context, occurrence.range);
        }
    }
    if (include_declaration and !declaration_seen) {
        try appendRange(context, target_definition_range);
    }
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

    var occurrences = std.ArrayList(ImportedMemberOccurrence).empty;
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
    var ranges = std.ArrayList(frontend.Range).empty;
    errdefer ranges.deinit(allocator);

    for (index.occurrences) |occurrence| {
        if (!std.mem.eql(u8, occurrence.imported_path, target_path)) continue;
        if (!std.mem.eql(u8, occurrence.member_name, target_name)) continue;
        try appendUniqueRange(allocator, &ranges, occurrence.range);
    }

    return ranges.toOwnedSlice(allocator);
}

const RangeIndexBuilder = struct {
    items: []RangeIndexEntry,
    capacity_requested: usize,
    growth_events: usize,
};

const NameIndexBuilder = struct {
    items: []NameIndexEntry,
    capacity_requested: usize,
    growth_events: usize,
};

fn buildRangeIndexes(allocator: Allocator, occurrences: []const Occurrence) !RangeIndexBuilder {
    var indexes = std.ArrayList(RangeIndexEntry).empty;
    var growth_events: usize = 0;
    errdefer indexes.deinit(allocator);

    const capacity_before = indexes.capacity;
    try indexes.ensureTotalCapacity(allocator, occurrences.len);
    if (indexes.capacity > capacity_before) growth_events += 1;

    for (occurrences, 0..) |occurrence, index| {
        const item_index = std.math.cast(u32, index) orelse return error.IndexTooLarge;
        indexes.appendAssumeCapacity(.{
            .range = occurrence.range,
            .item_index = item_index,
        });
    }

    std.mem.sort(RangeIndexEntry, indexes.items, {}, lessRangeIndexStart);
    const capacity_requested = indexes.capacity;
    return .{
        .items = try indexes.toOwnedSlice(allocator),
        .capacity_requested = capacity_requested,
        .growth_events = growth_events,
    };
}

fn buildNameIndexes(allocator: Allocator, occurrences: []const Occurrence) !NameIndexBuilder {
    var indexes = std.ArrayList(NameIndexEntry).empty;
    var growth_events: usize = 0;
    errdefer indexes.deinit(allocator);

    const capacity_before = indexes.capacity;
    try indexes.ensureTotalCapacity(allocator, occurrences.len);
    if (indexes.capacity > capacity_before) growth_events += 1;

    for (occurrences, 0..) |occurrence, index| {
        const item_index = std.math.cast(u32, index) orelse return error.IndexTooLarge;
        indexes.appendAssumeCapacity(.{
            .name = occurrence.name,
            .item_index = item_index,
        });
    }

    std.mem.sort(NameIndexEntry, indexes.items, {}, lessNameIndex);
    const capacity_requested = indexes.capacity;
    return .{
        .items = try indexes.toOwnedSlice(allocator),
        .capacity_requested = capacity_requested,
        .growth_events = growth_events,
    };
}

fn lessNameIndex(_: void, a: NameIndexEntry, b: NameIndexEntry) bool {
    const order = std.mem.order(u8, a.name, b.name);
    if (order != .eq) return order == .lt;
    return a.item_index < b.item_index;
}

fn lowerBoundName(indexes: []const NameIndexEntry, name: []const u8) usize {
    var low: usize = 0;
    var high: usize = indexes.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (std.mem.order(u8, indexes[mid].name, name) == .lt) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

fn upperBoundName(indexes: []const NameIndexEntry, name: []const u8, start: usize) usize {
    var low = start;
    var high: usize = indexes.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (std.mem.order(u8, indexes[mid].name, name) != .gt) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

fn occurrenceIndexAtPosition(
    occurrences: []const Occurrence,
    indexes: []const RangeIndexEntry,
    position: frontend.Position,
) ?usize {
    if (indexes.len == 0) return null;
    var cursor = upperBoundRangeStart(indexes, position);
    while (cursor > 0) {
        cursor -= 1;
        const entry = indexes[cursor];
        const item_index: usize = entry.item_index;
        if (item_index >= occurrences.len) continue;
        const occurrence = occurrences[item_index];
        if (positionInRange(position, occurrence.range)) return item_index;
        if (positionLessThan(occurrence.range.end, position)) break;
    }
    return null;
}

fn upperBoundRangeStart(indexes: []const RangeIndexEntry, position: frontend.Position) usize {
    var low: usize = 0;
    var high: usize = indexes.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (!positionLessThan(position, indexes[mid].range.start)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

fn lessRangeIndexStart(_: void, lhs: RangeIndexEntry, rhs: RangeIndexEntry) bool {
    if (positionLessThan(lhs.range.start, rhs.range.start)) return true;
    if (positionLessThan(rhs.range.start, lhs.range.start)) return false;
    return lhs.item_index < rhs.item_index;
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

fn positionLessThan(lhs: frontend.Position, rhs: frontend.Position) bool {
    if (lhs.line < rhs.line) return true;
    if (lhs.line > rhs.line) return false;
    return lhs.character < rhs.character;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}
