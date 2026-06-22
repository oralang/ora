const std = @import("std");
const frontend = @import("frontend.zig");
const semantic_index = @import("semantic_index.zig");
const references = @import("references.zig");
const call_hierarchy = @import("call_hierarchy.zig");
const workspace = @import("workspace.zig");
const line_index = @import("line_index.zig");

const Allocator = std.mem.Allocator;
/// Default cold workspace-entry cache budget. The JSON-RPC benchmark currently peaks near 35 MiB RSS on
/// the large-contract/stress fixtures, so 64 MiB gives headroom while keeping cold overlays bounded.
pub const default_max_bytes: usize = 64 * 1024 * 1024;

pub const FeatureSet = struct {
    symbols: bool = true,
    occurrences: bool = false,
    imported_members: bool = false,
    call_edges: bool = false,

    pub fn covers(self: FeatureSet, required: FeatureSet) bool {
        return (!required.symbols or self.symbols) and
            (!required.occurrences or self.occurrences) and
            (!required.imported_members or self.imported_members) and
            (!required.call_edges or self.call_edges);
    }

    pub fn merged(self: FeatureSet, other: FeatureSet) FeatureSet {
        return .{
            .symbols = self.symbols or other.symbols,
            .occurrences = self.occurrences or other.occurrences,
            .imported_members = self.imported_members or other.imported_members,
            .call_edges = self.call_edges or other.call_edges,
        };
    }

    pub const symbols_only: FeatureSet = .{};
    pub const references: FeatureSet = .{ .occurrences = true, .imported_members = true };
    pub const calls: FeatureSet = .{ .call_edges = true };
};

pub const Symbol = struct {
    name: []const u8,
    detail: ?[]const u8,
    kind: semantic_index.SymbolKind,
    range: frontend.Range,
    selection_range: frontend.Range,
    parent: ?usize,
};

pub const Import = struct {
    specifier: []const u8,
    alias: ?[]const u8,
    resolved_path: []const u8,
};

const SymbolNameIndexEntry = struct {
    name: []const u8,
    symbol_index: u32,
};

const RangeIndexEntry = struct {
    range: frontend.Range,
    item_index: u32,
};

pub const FileEntry = struct {
    arena: std.heap.ArenaAllocator,
    uri: []const u8,
    version: i32,
    generation: u64,
    is_cold: bool,
    features: FeatureSet,
    line_index: line_index.LineIndex,
    symbols: []Symbol,
    root_symbol_indexes: []u32,
    callable_symbol_indexes: []u32,
    root_symbol_name_indexes: []SymbolNameIndexEntry = &.{},
    callable_symbol_name_indexes: []SymbolNameIndexEntry = &.{},
    callable_symbol_range_indexes: []RangeIndexEntry = &.{},
    imports: []Import,
    occurrences: []references.Occurrence,
    occurrence_range_indexes: []RangeIndexEntry = &.{},
    imported_members: []references.ImportedMemberOccurrence,
    call_edges: []call_hierarchy.CallEdge,
    interned_string_bytes: usize = 0,
    interned_string_count: usize = 0,
    duplicate_string_bytes_saved: usize = 0,
    interned_string_capacity_requested: usize = 0,
    interned_string_items_built: usize = 0,
    interned_string_unused_capacity: usize = 0,
    interned_string_growth_events: usize = 0,
    builder_capacity_requested: usize = 0,
    builder_items_built: usize = 0,
    builder_unused_capacity: usize = 0,
    builder_growth_events: usize = 0,
    side_map_capacity_requested: usize = 0,
    side_map_items_built: usize = 0,
    side_map_unused_capacity: usize = 0,
    side_map_growth_events: usize = 0,
    byte_size: usize,
    last_access: u64 = 0,

    pub fn init(
        allocator: Allocator,
        uri: []const u8,
        version: i32,
        generation: u64,
        is_cold: bool,
        features: FeatureSet,
        source_line_index: *const line_index.LineIndex,
        semantic_symbols: []const semantic_index.Symbol,
        resolved_imports: []const workspace.ResolvedImport,
        occurrence_index: ?*const references.OccurrenceIndex,
        imported_member_index: ?*const references.ImportedMemberIndex,
        call_edge_index: ?*const call_hierarchy.CallEdgeIndex,
    ) !FileEntry {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const entry_allocator = arena.allocator();

        var interner = StringInterner.init(allocator, entry_allocator);
        defer interner.deinit();

        const uri_copy = try interner.intern(uri);
        const entry_line_index = try copyLineIndex(entry_allocator, source_line_index);
        const symbols = try copySymbols(entry_allocator, &interner, semantic_symbols);
        const root_symbol_index_builder = try buildRootSymbolIndexes(entry_allocator, symbols);
        const root_symbol_indexes = root_symbol_index_builder.items;
        const callable_symbol_index_builder = try buildCallableSymbolIndexes(entry_allocator, symbols);
        const callable_symbol_indexes = callable_symbol_index_builder.items;
        const root_symbol_name_index_builder = try buildSymbolNameIndexes(entry_allocator, symbols, root_symbol_indexes);
        const root_symbol_name_indexes = root_symbol_name_index_builder.items;
        const callable_symbol_name_index_builder = try buildSymbolNameIndexes(entry_allocator, symbols, callable_symbol_indexes);
        const callable_symbol_name_indexes = callable_symbol_name_index_builder.items;
        const callable_symbol_range_index_builder = try buildSymbolRangeIndexes(entry_allocator, symbols, callable_symbol_indexes);
        const callable_symbol_range_indexes = callable_symbol_range_index_builder.items;
        const imports = try copyImports(entry_allocator, &interner, resolved_imports);
        const occurrences = if (features.occurrences)
            try copyOccurrences(entry_allocator, &interner, occurrence_index.?.occurrences)
        else
            try entry_allocator.alloc(references.Occurrence, 0);
        const occurrence_range_index_builder = try buildOccurrenceRangeIndexes(entry_allocator, occurrences);
        const occurrence_range_indexes = occurrence_range_index_builder.items;
        const imported_members = if (features.imported_members)
            try copyImportedMembers(entry_allocator, &interner, imported_member_index.?.occurrences)
        else
            try entry_allocator.alloc(references.ImportedMemberOccurrence, 0);
        const call_edges = if (features.call_edges)
            try copyCallEdges(entry_allocator, &interner, call_edge_index.?.edges)
        else
            try entry_allocator.alloc(call_hierarchy.CallEdge, 0);
        const byte_size = estimateByteSize(
            &entry_line_index,
            symbols,
            root_symbol_indexes,
            callable_symbol_indexes,
            root_symbol_name_indexes,
            callable_symbol_name_indexes,
            callable_symbol_range_indexes,
            imports,
            occurrences,
            occurrence_range_indexes,
            imported_members,
            call_edges,
            interner.unique_bytes,
        );
        const builder_capacity_requested = workspaceEntryBuilderCapacityRequested(
            &entry_line_index,
            symbols,
            root_symbol_index_builder,
            callable_symbol_index_builder,
            root_symbol_name_index_builder,
            callable_symbol_name_index_builder,
            callable_symbol_range_index_builder,
            imports,
            occurrences,
            occurrence_range_index_builder,
            imported_members,
            call_edges,
        );
        const builder_items_built = workspaceEntryBuilderItemsBuilt(
            &entry_line_index,
            symbols,
            root_symbol_indexes,
            callable_symbol_indexes,
            root_symbol_name_indexes,
            callable_symbol_name_indexes,
            callable_symbol_range_indexes,
            imports,
            occurrences,
            occurrence_range_indexes,
            imported_members,
            call_edges,
        );
        var side_map_capacity_requested = addSat(root_symbol_index_builder.capacity_requested, callable_symbol_index_builder.capacity_requested);
        side_map_capacity_requested = addSat(side_map_capacity_requested, root_symbol_name_index_builder.capacity_requested);
        side_map_capacity_requested = addSat(side_map_capacity_requested, callable_symbol_name_index_builder.capacity_requested);
        side_map_capacity_requested = addSat(side_map_capacity_requested, callable_symbol_range_index_builder.capacity_requested);
        side_map_capacity_requested = addSat(side_map_capacity_requested, occurrence_range_index_builder.capacity_requested);
        var side_map_items_built = addSat(root_symbol_indexes.len, callable_symbol_indexes.len);
        side_map_items_built = addSat(side_map_items_built, root_symbol_name_indexes.len);
        side_map_items_built = addSat(side_map_items_built, callable_symbol_name_indexes.len);
        side_map_items_built = addSat(side_map_items_built, callable_symbol_range_indexes.len);
        side_map_items_built = addSat(side_map_items_built, occurrence_range_indexes.len);

        return .{
            .arena = arena,
            .uri = uri_copy,
            .version = version,
            .generation = generation,
            .is_cold = is_cold,
            .features = features,
            .line_index = entry_line_index,
            .symbols = symbols,
            .root_symbol_indexes = root_symbol_indexes,
            .callable_symbol_indexes = callable_symbol_indexes,
            .root_symbol_name_indexes = root_symbol_name_indexes,
            .callable_symbol_name_indexes = callable_symbol_name_indexes,
            .callable_symbol_range_indexes = callable_symbol_range_indexes,
            .imports = imports,
            .occurrences = occurrences,
            .occurrence_range_indexes = occurrence_range_indexes,
            .imported_members = imported_members,
            .call_edges = call_edges,
            .interned_string_bytes = interner.unique_bytes,
            .interned_string_count = interner.unique_count,
            .duplicate_string_bytes_saved = interner.duplicateBytesSaved(),
            .interned_string_capacity_requested = interner.capacity_requested,
            .interned_string_items_built = interner.unique_count,
            .interned_string_unused_capacity = if (interner.capacity_requested > interner.unique_count) interner.capacity_requested - interner.unique_count else 0,
            .interned_string_growth_events = interner.growth_events,
            .builder_capacity_requested = builder_capacity_requested,
            .builder_items_built = builder_items_built,
            .builder_unused_capacity = if (builder_capacity_requested > builder_items_built) builder_capacity_requested - builder_items_built else 0,
            .builder_growth_events = sumBuilderGrowthEvents(&.{
                root_symbol_index_builder.growth_events,
                callable_symbol_index_builder.growth_events,
                root_symbol_name_index_builder.growth_events,
                callable_symbol_name_index_builder.growth_events,
                callable_symbol_range_index_builder.growth_events,
                occurrence_range_index_builder.growth_events,
            }),
            .side_map_capacity_requested = side_map_capacity_requested,
            .side_map_items_built = side_map_items_built,
            .side_map_unused_capacity = if (side_map_capacity_requested > side_map_items_built) side_map_capacity_requested - side_map_items_built else 0,
            .side_map_growth_events = sumBuilderGrowthEvents(&.{
                root_symbol_index_builder.growth_events,
                callable_symbol_index_builder.growth_events,
                root_symbol_name_index_builder.growth_events,
                callable_symbol_name_index_builder.growth_events,
                callable_symbol_range_index_builder.growth_events,
                occurrence_range_index_builder.growth_events,
            }),
            .byte_size = byte_size,
        };
    }

    pub fn deinit(self: *FileEntry) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn occurrenceAt(self: *const FileEntry, position: frontend.Position) ?references.Occurrence {
        if (self.occurrence_range_indexes.len == self.occurrences.len) {
            const index = occurrenceIndexAtPosition(self.occurrences, self.occurrence_range_indexes, position) orelse return null;
            return self.occurrences[index];
        }
        for (self.occurrences) |occurrence| {
            if (positionInRange(position, occurrence.range)) return occurrence;
        }
        return null;
    }

    pub fn rootSymbolIndexes(self: *const FileEntry) []const u32 {
        return self.root_symbol_indexes;
    }

    pub fn rootSymbolNamed(self: *const FileEntry, name: []const u8) ?*const Symbol {
        if (symbolNamed(self.symbols, self.root_symbol_name_indexes, name)) |symbol| return symbol;
        for (self.root_symbol_indexes) |raw_index| {
            const symbol_index: usize = raw_index;
            const symbol = &self.symbols[symbol_index];
            if (std.mem.eql(u8, symbol.name, name)) return symbol;
        }
        return null;
    }

    pub fn callableSymbolIndexes(self: *const FileEntry) []const u32 {
        return self.callable_symbol_indexes;
    }

    pub fn callableSymbolIndexByRange(self: *const FileEntry, range: frontend.Range) ?usize {
        if (symbolIndexByRange(self.symbols, self.callable_symbol_range_indexes, range)) |symbol_index| return symbol_index;
        for (self.callable_symbol_indexes) |raw_index| {
            const symbol_index: usize = raw_index;
            const symbol = self.symbols[symbol_index];
            if (frontendRangesEqual(symbol.range, range)) return symbol_index;
        }
        return null;
    }

    pub fn callableSymbolNamed(self: *const FileEntry, name: []const u8) ?*const Symbol {
        if (symbolNamed(self.symbols, self.callable_symbol_name_indexes, name)) |symbol| return symbol;
        for (self.callable_symbol_indexes) |raw_index| {
            const symbol_index: usize = raw_index;
            const symbol = &self.symbols[symbol_index];
            if (std.mem.eql(u8, symbol.name, name)) return symbol;
        }
        return null;
    }

    pub fn occurrenceIndex(self: *const FileEntry) references.OccurrenceIndex {
        return .{ .occurrences = self.occurrences };
    }

    pub fn importedMemberIndex(self: *const FileEntry) references.ImportedMemberIndex {
        return .{ .occurrences = self.imported_members };
    }

    pub fn callEdgeIndex(self: *const FileEntry) call_hierarchy.CallEdgeIndex {
        return .{ .edges = self.call_edges };
    }

    pub fn builderCapacityRequested(self: *const FileEntry) usize {
        return self.builder_capacity_requested;
    }

    pub fn builderItemsBuilt(self: *const FileEntry) usize {
        return self.builder_items_built;
    }

    pub fn builderUnusedCapacity(self: *const FileEntry) usize {
        return self.builder_unused_capacity;
    }

    pub fn builderGrowthEvents(self: *const FileEntry) usize {
        return self.builder_growth_events;
    }

    pub fn sideMapCapacityRequested(self: *const FileEntry) usize {
        return self.side_map_capacity_requested;
    }

    pub fn sideMapItemsBuilt(self: *const FileEntry) usize {
        return self.side_map_items_built;
    }

    pub fn sideMapUnusedCapacity(self: *const FileEntry) usize {
        return self.side_map_unused_capacity;
    }

    pub fn sideMapGrowthEvents(self: *const FileEntry) usize {
        return self.side_map_growth_events;
    }

    pub fn internedStringCapacityRequested(self: *const FileEntry) usize {
        return self.interned_string_capacity_requested;
    }

    pub fn internedStringItemsBuilt(self: *const FileEntry) usize {
        return self.interned_string_items_built;
    }

    pub fn internedStringUnusedCapacity(self: *const FileEntry) usize {
        return self.interned_string_unused_capacity;
    }

    pub fn internedStringGrowthEvents(self: *const FileEntry) usize {
        return self.interned_string_growth_events;
    }
};

pub const Index = struct {
    allocator: Allocator,
    entries: std.StringHashMap(*FileEntry),
    max_bytes: usize,
    current_bytes: usize = 0,
    access_clock: u64 = 0,
    evictions: usize = 0,

    pub fn init(allocator: Allocator) Index {
        return initWithBudget(allocator, default_max_bytes);
    }

    pub fn initWithBudget(allocator: Allocator, max_bytes: usize) Index {
        return .{
            .allocator = allocator,
            .entries = std.StringHashMap(*FileEntry).init(allocator),
            .max_bytes = max_bytes,
        };
    }

    pub fn deinit(self: *Index) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.entries.deinit();
        self.current_bytes = 0;
        self.access_clock = 0;
        self.evictions = 0;
    }

    pub fn invalidate(self: *Index, uri: []const u8) void {
        if (self.entries.fetchRemove(uri)) |removed| {
            const entry = removed.value;
            self.subtractBytes(entry.byte_size);
            entry.deinit();
            self.allocator.destroy(entry);
        }
    }

    pub fn getFresh(self: *Index, uri: []const u8, version: i32, generation: u64, required: FeatureSet) ?*const FileEntry {
        const entry = self.entries.get(uri) orelse return null;
        if (entry.version != version or entry.generation != generation) return null;
        if (!entry.features.covers(required)) return null;
        return self.touch(entry);
    }

    pub fn getFreshAny(self: *Index, uri: []const u8, version: i32, generation: u64) ?*const FileEntry {
        const entry = self.entries.get(uri) orelse return null;
        if (entry.version != version or entry.generation != generation) return null;
        return self.touch(entry);
    }

    pub fn upsert(self: *Index, entry: FileEntry) !void {
        var incoming = entry;
        self.access_clock +%= 1;
        incoming.last_access = self.access_clock;

        const entry_ptr = try self.allocator.create(FileEntry);
        errdefer self.allocator.destroy(entry_ptr);
        entry_ptr.* = incoming;
        errdefer entry_ptr.deinit();

        try self.entries.ensureUnusedCapacity(1);
        if (self.entries.fetchRemove(entry_ptr.uri)) |removed| {
            const old_entry = removed.value;
            self.subtractBytes(old_entry.byte_size);
            old_entry.deinit();
            self.allocator.destroy(old_entry);
        }
        self.current_bytes = addSat(self.current_bytes, entry_ptr.byte_size);
        self.entries.putAssumeCapacity(entry_ptr.uri, entry_ptr);
        self.evictToBudget(entry_ptr.uri);
    }

    fn touch(self: *Index, entry: *FileEntry) *const FileEntry {
        self.access_clock +%= 1;
        entry.last_access = self.access_clock;
        return entry;
    }

    fn evictToBudget(self: *Index, protected_uri: []const u8) void {
        while (self.current_bytes > self.max_bytes and self.entries.count() > 1) {
            const oldest_uri = self.oldestEvictableUri(protected_uri) orelse return;
            self.evict(oldest_uri);
        }
    }

    fn oldestEvictableUri(self: *Index, protected_uri: []const u8) ?[]const u8 {
        var oldest_uri: ?[]const u8 = null;
        var oldest_access: u64 = std.math.maxInt(u64);

        var it = self.entries.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, entry.key_ptr.*, protected_uri)) continue;
            const value = entry.value_ptr.*;
            if (!value.is_cold) continue;
            if (value.last_access < oldest_access) {
                oldest_access = value.last_access;
                oldest_uri = entry.key_ptr.*;
            }
        }

        return oldest_uri;
    }

    fn evict(self: *Index, uri: []const u8) void {
        if (self.entries.fetchRemove(uri)) |removed| {
            const entry = removed.value;
            self.subtractBytes(entry.byte_size);
            entry.deinit();
            self.allocator.destroy(entry);
            self.evictions += 1;
        }
    }

    fn subtractBytes(self: *Index, byte_size: usize) void {
        self.current_bytes = if (byte_size > self.current_bytes) 0 else self.current_bytes - byte_size;
    }

    pub fn builderCapacityRequested(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.builderCapacityRequested());
        }
        return total;
    }

    pub fn builderItemsBuilt(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.builderItemsBuilt());
        }
        return total;
    }

    pub fn builderUnusedCapacity(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.builderUnusedCapacity());
        }
        return total;
    }

    pub fn builderGrowthEvents(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.builderGrowthEvents());
        }
        return total;
    }

    pub fn sideMapCapacityRequested(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.sideMapCapacityRequested());
        }
        return total;
    }

    pub fn sideMapItemsBuilt(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.sideMapItemsBuilt());
        }
        return total;
    }

    pub fn sideMapUnusedCapacity(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.sideMapUnusedCapacity());
        }
        return total;
    }

    pub fn sideMapGrowthEvents(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.sideMapGrowthEvents());
        }
        return total;
    }

    pub fn coldEntryCount(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            if (entry.*.is_cold) total = addSat(total, 1);
        }
        return total;
    }

    pub fn coldBytes(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            if (entry.*.is_cold) total = addSat(total, entry.*.byte_size);
        }
        return total;
    }

    pub fn coldInternedStringBytes(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            if (entry.*.is_cold) total = addSat(total, entry.*.interned_string_bytes);
        }
        return total;
    }

    pub fn coldInternedStringCount(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            if (entry.*.is_cold) total = addSat(total, entry.*.interned_string_count);
        }
        return total;
    }

    pub fn coldDuplicateStringBytesSaved(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            if (entry.*.is_cold) total = addSat(total, entry.*.duplicate_string_bytes_saved);
        }
        return total;
    }

    pub fn internedStringCapacityRequested(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.internedStringCapacityRequested());
        }
        return total;
    }

    pub fn internedStringItemsBuilt(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.internedStringItemsBuilt());
        }
        return total;
    }

    pub fn internedStringUnusedCapacity(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.internedStringUnusedCapacity());
        }
        return total;
    }

    pub fn internedStringGrowthEvents(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            total = addSat(total, entry.*.internedStringGrowthEvents());
        }
        return total;
    }

    pub fn coldInternedStringCapacityRequested(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            if (entry.*.is_cold) total = addSat(total, entry.*.internedStringCapacityRequested());
        }
        return total;
    }

    pub fn coldInternedStringItemsBuilt(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            if (entry.*.is_cold) total = addSat(total, entry.*.internedStringItemsBuilt());
        }
        return total;
    }

    pub fn coldInternedStringUnusedCapacity(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            if (entry.*.is_cold) total = addSat(total, entry.*.internedStringUnusedCapacity());
        }
        return total;
    }

    pub fn coldInternedStringGrowthEvents(self: *const Index) usize {
        var total: usize = 0;
        var iterator = self.entries.valueIterator();
        while (iterator.next()) |entry| {
            if (entry.*.is_cold) total = addSat(total, entry.*.internedStringGrowthEvents());
        }
        return total;
    }
};

fn copyLineIndex(allocator: Allocator, source: *const line_index.LineIndex) !line_index.LineIndex {
    const starts = try allocator.dupe(u32, source.line_starts);
    errdefer allocator.free(starts);

    return .{
        .line_starts = starts,
        .line_ascii = try allocator.dupe(bool, source.line_ascii),
        .source_len = source.source_len,
    };
}

const StringInterner = struct {
    map: std.StringHashMap([]const u8),
    string_allocator: Allocator,
    unique_bytes: usize = 0,
    requested_bytes: usize = 0,
    unique_count: usize = 0,
    capacity_requested: usize = 0,
    growth_events: usize = 0,

    fn init(map_allocator: Allocator, string_allocator: Allocator) StringInterner {
        return .{
            .map = std.StringHashMap([]const u8).init(map_allocator),
            .string_allocator = string_allocator,
        };
    }

    fn deinit(self: *StringInterner) void {
        self.map.deinit();
    }

    fn intern(self: *StringInterner, value: []const u8) ![]const u8 {
        self.requested_bytes = addSat(self.requested_bytes, value.len);
        if (self.map.get(value)) |interned| return interned;

        const owned = try self.string_allocator.dupe(u8, value);
        errdefer self.string_allocator.free(owned);

        const before_capacity = self.map.capacity();
        try self.map.put(owned, owned);
        const after_capacity = self.map.capacity();
        self.capacity_requested = after_capacity;
        if (after_capacity > before_capacity) self.growth_events = addSat(self.growth_events, 1);
        self.unique_bytes = addSat(self.unique_bytes, owned.len);
        self.unique_count = addSat(self.unique_count, 1);
        return owned;
    }

    fn duplicateBytesSaved(self: *const StringInterner) usize {
        return if (self.unique_bytes > self.requested_bytes) 0 else self.requested_bytes - self.unique_bytes;
    }
};

fn copySymbols(allocator: Allocator, interner: *StringInterner, symbols: []const semantic_index.Symbol) ![]Symbol {
    const result = try allocator.alloc(Symbol, symbols.len);
    for (symbols, 0..) |symbol, i| {
        result[i] = .{
            .name = try interner.intern(symbol.name),
            .detail = if (symbol.detail) |detail| try interner.intern(detail) else null,
            .kind = symbol.kind,
            .range = symbol.range,
            .selection_range = symbol.selection_range,
            .parent = symbol.parent,
        };
    }
    return result;
}

const BuiltIndexSlice = struct {
    items: []u32,
    capacity_requested: usize,
    growth_events: usize = 0,
};

const BuiltSymbolNameIndex = struct {
    items: []SymbolNameIndexEntry,
    capacity_requested: usize,
    growth_events: usize = 0,
};

const BuiltRangeIndex = struct {
    items: []RangeIndexEntry,
    capacity_requested: usize,
    growth_events: usize = 0,
};

fn buildRootSymbolIndexes(allocator: Allocator, symbols: []const Symbol) !BuiltIndexSlice {
    var indexes = std.ArrayList(u32).empty;
    errdefer indexes.deinit(allocator);
    try indexes.ensureTotalCapacity(allocator, symbols.len);

    for (symbols, 0..) |symbol, index| {
        if (symbol.parent != null) continue;
        indexes.appendAssumeCapacity(try symbolIndexU32(index));
    }

    const capacity_requested = indexes.capacity;
    return .{
        .items = try indexes.toOwnedSlice(allocator),
        .capacity_requested = capacity_requested,
    };
}

fn buildCallableSymbolIndexes(allocator: Allocator, symbols: []const Symbol) !BuiltIndexSlice {
    var indexes = std.ArrayList(u32).empty;
    errdefer indexes.deinit(allocator);
    try indexes.ensureTotalCapacity(allocator, symbols.len);

    for (symbols, 0..) |symbol, index| {
        if (!isCallableSymbol(symbol.kind)) continue;
        indexes.appendAssumeCapacity(try symbolIndexU32(index));
    }

    const capacity_requested = indexes.capacity;
    return .{
        .items = try indexes.toOwnedSlice(allocator),
        .capacity_requested = capacity_requested,
    };
}

fn buildSymbolNameIndexes(
    allocator: Allocator,
    symbols: []const Symbol,
    symbol_indexes: []const u32,
) !BuiltSymbolNameIndex {
    var indexes = std.ArrayList(SymbolNameIndexEntry).empty;
    errdefer indexes.deinit(allocator);
    try indexes.ensureTotalCapacity(allocator, symbol_indexes.len);

    for (symbol_indexes) |symbol_index| {
        indexes.appendAssumeCapacity(.{
            .name = symbols[symbol_index].name,
            .symbol_index = symbol_index,
        });
    }
    std.mem.sort(SymbolNameIndexEntry, indexes.items, {}, lessSymbolNameIndex);

    const capacity_requested = indexes.capacity;
    return .{
        .items = try indexes.toOwnedSlice(allocator),
        .capacity_requested = capacity_requested,
    };
}

fn buildSymbolRangeIndexes(
    allocator: Allocator,
    symbols: []const Symbol,
    symbol_indexes: []const u32,
) !BuiltRangeIndex {
    var indexes = std.ArrayList(RangeIndexEntry).empty;
    errdefer indexes.deinit(allocator);
    try indexes.ensureTotalCapacity(allocator, symbol_indexes.len);

    for (symbol_indexes) |symbol_index| {
        indexes.appendAssumeCapacity(.{
            .range = symbols[symbol_index].range,
            .item_index = symbol_index,
        });
    }
    std.mem.sort(RangeIndexEntry, indexes.items, {}, lessRangeIndex);

    const capacity_requested = indexes.capacity;
    return .{
        .items = try indexes.toOwnedSlice(allocator),
        .capacity_requested = capacity_requested,
    };
}

fn buildOccurrenceRangeIndexes(allocator: Allocator, occurrences: []const references.Occurrence) !BuiltRangeIndex {
    var indexes = std.ArrayList(RangeIndexEntry).empty;
    errdefer indexes.deinit(allocator);
    try indexes.ensureTotalCapacity(allocator, occurrences.len);

    for (occurrences, 0..) |occurrence, index| {
        indexes.appendAssumeCapacity(.{
            .range = occurrence.range,
            .item_index = try symbolIndexU32(index),
        });
    }
    std.mem.sort(RangeIndexEntry, indexes.items, {}, lessRangeIndex);

    const capacity_requested = indexes.capacity;
    return .{
        .items = try indexes.toOwnedSlice(allocator),
        .capacity_requested = capacity_requested,
    };
}

fn symbolIndexU32(index: usize) !u32 {
    return std.math.cast(u32, index) orelse error.SymbolIndexOverflow;
}

fn copyImports(allocator: Allocator, interner: *StringInterner, imports: []const workspace.ResolvedImport) ![]Import {
    const result = try allocator.alloc(Import, imports.len);
    for (imports, 0..) |item, i| {
        result[i] = .{
            .specifier = try interner.intern(item.specifier),
            .alias = if (item.alias) |alias_text| try interner.intern(alias_text) else null,
            .resolved_path = try interner.intern(item.resolved_path),
        };
    }
    return result;
}

fn copyOccurrences(allocator: Allocator, interner: *StringInterner, occurrences: []const references.Occurrence) ![]references.Occurrence {
    const result = try allocator.alloc(references.Occurrence, occurrences.len);
    for (occurrences, 0..) |occurrence, i| {
        result[i] = .{
            .name = try interner.intern(occurrence.name),
            .range = occurrence.range,
            .definition_range = occurrence.definition_range,
        };
    }
    return result;
}

fn copyImportedMembers(
    allocator: Allocator,
    interner: *StringInterner,
    occurrences: []const references.ImportedMemberOccurrence,
) ![]references.ImportedMemberOccurrence {
    const result = try allocator.alloc(references.ImportedMemberOccurrence, occurrences.len);
    for (occurrences, 0..) |occurrence, i| {
        result[i] = .{
            .imported_path = try interner.intern(occurrence.imported_path),
            .alias = try interner.intern(occurrence.alias),
            .member_name = try interner.intern(occurrence.member_name),
            .range = occurrence.range,
        };
    }
    return result;
}

fn copyCallEdges(allocator: Allocator, interner: *StringInterner, edges: []const call_hierarchy.CallEdge) ![]call_hierarchy.CallEdge {
    const result = try allocator.alloc(call_hierarchy.CallEdge, edges.len);
    for (edges, 0..) |edge, i| {
        result[i] = .{
            .caller_symbol_index = edge.caller_symbol_index,
            .callee_name = try interner.intern(edge.callee_name),
            .range = edge.range,
        };
    }
    return result;
}

fn estimateByteSize(
    entry_line_index: *const line_index.LineIndex,
    symbols: []const Symbol,
    root_symbol_indexes: []const u32,
    callable_symbol_indexes: []const u32,
    root_symbol_name_indexes: []const SymbolNameIndexEntry,
    callable_symbol_name_indexes: []const SymbolNameIndexEntry,
    callable_symbol_range_indexes: []const RangeIndexEntry,
    imports: []const Import,
    occurrences: []const references.Occurrence,
    occurrence_range_indexes: []const RangeIndexEntry,
    imported_members: []const references.ImportedMemberOccurrence,
    call_edges: []const call_hierarchy.CallEdge,
    interned_string_bytes: usize,
) usize {
    var total: usize = interned_string_bytes;
    total = addSat(total, entry_line_index.estimatedByteSize());
    total = addSat(total, symbols.len * @sizeOf(Symbol));
    total = addSat(total, root_symbol_indexes.len * @sizeOf(u32));
    total = addSat(total, callable_symbol_indexes.len * @sizeOf(u32));
    total = addSat(total, root_symbol_name_indexes.len * @sizeOf(SymbolNameIndexEntry));
    total = addSat(total, callable_symbol_name_indexes.len * @sizeOf(SymbolNameIndexEntry));
    total = addSat(total, callable_symbol_range_indexes.len * @sizeOf(RangeIndexEntry));
    total = addSat(total, imports.len * @sizeOf(Import));
    total = addSat(total, occurrences.len * @sizeOf(references.Occurrence));
    total = addSat(total, occurrence_range_indexes.len * @sizeOf(RangeIndexEntry));
    total = addSat(total, imported_members.len * @sizeOf(references.ImportedMemberOccurrence));
    total = addSat(total, call_edges.len * @sizeOf(call_hierarchy.CallEdge));

    return total;
}

fn workspaceEntryBuilderCapacityRequested(
    entry_line_index: *const line_index.LineIndex,
    symbols: []const Symbol,
    root_symbol_indexes: BuiltIndexSlice,
    callable_symbol_indexes: BuiltIndexSlice,
    root_symbol_name_indexes: BuiltSymbolNameIndex,
    callable_symbol_name_indexes: BuiltSymbolNameIndex,
    callable_symbol_range_indexes: BuiltRangeIndex,
    imports: []const Import,
    occurrences: []const references.Occurrence,
    occurrence_range_indexes: BuiltRangeIndex,
    imported_members: []const references.ImportedMemberOccurrence,
    call_edges: []const call_hierarchy.CallEdge,
) usize {
    var total: usize = entry_line_index.line_starts.len;
    total = addSat(total, symbols.len);
    total = addSat(total, root_symbol_indexes.capacity_requested);
    total = addSat(total, callable_symbol_indexes.capacity_requested);
    total = addSat(total, root_symbol_name_indexes.capacity_requested);
    total = addSat(total, callable_symbol_name_indexes.capacity_requested);
    total = addSat(total, callable_symbol_range_indexes.capacity_requested);
    total = addSat(total, imports.len);
    total = addSat(total, occurrences.len);
    total = addSat(total, occurrence_range_indexes.capacity_requested);
    total = addSat(total, imported_members.len);
    total = addSat(total, call_edges.len);
    return total;
}

fn workspaceEntryBuilderItemsBuilt(
    entry_line_index: *const line_index.LineIndex,
    symbols: []const Symbol,
    root_symbol_indexes: []const u32,
    callable_symbol_indexes: []const u32,
    root_symbol_name_indexes: []const SymbolNameIndexEntry,
    callable_symbol_name_indexes: []const SymbolNameIndexEntry,
    callable_symbol_range_indexes: []const RangeIndexEntry,
    imports: []const Import,
    occurrences: []const references.Occurrence,
    occurrence_range_indexes: []const RangeIndexEntry,
    imported_members: []const references.ImportedMemberOccurrence,
    call_edges: []const call_hierarchy.CallEdge,
) usize {
    var total: usize = entry_line_index.line_starts.len;
    total = addSat(total, symbols.len);
    total = addSat(total, root_symbol_indexes.len);
    total = addSat(total, callable_symbol_indexes.len);
    total = addSat(total, root_symbol_name_indexes.len);
    total = addSat(total, callable_symbol_name_indexes.len);
    total = addSat(total, callable_symbol_range_indexes.len);
    total = addSat(total, imports.len);
    total = addSat(total, occurrences.len);
    total = addSat(total, occurrence_range_indexes.len);
    total = addSat(total, imported_members.len);
    total = addSat(total, call_edges.len);
    return total;
}

fn addSat(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

fn positionInRange(pos: frontend.Position, range: frontend.Range) bool {
    if (pos.line < range.start.line) return false;
    if (pos.line > range.end.line) return false;
    if (pos.line == range.start.line and pos.character < range.start.character) return false;
    if (pos.line == range.end.line and pos.character > range.end.character) return false;
    return true;
}

fn symbolNamed(symbols: []const Symbol, indexes: []const SymbolNameIndexEntry, name: []const u8) ?*const Symbol {
    var index = lowerBoundSymbolName(indexes, name);
    while (index < indexes.len and std.mem.eql(u8, indexes[index].name, name)) : (index += 1) {
        return &symbols[indexes[index].symbol_index];
    }
    return null;
}

fn symbolIndexByRange(symbols: []const Symbol, indexes: []const RangeIndexEntry, range: frontend.Range) ?usize {
    var index = lowerBoundRangeStart(indexes, range.start);
    while (index < indexes.len and positionsEqual(indexes[index].range.start, range.start)) : (index += 1) {
        const symbol_index: usize = indexes[index].item_index;
        if (frontendRangesEqual(symbols[symbol_index].range, range)) return symbol_index;
    }
    return null;
}

fn occurrenceIndexAtPosition(
    occurrences: []const references.Occurrence,
    indexes: []const RangeIndexEntry,
    position: frontend.Position,
) ?usize {
    var index = upperBoundRangeStart(indexes, position);
    if (index == 0) return null;
    index -= 1;
    const occurrence_index: usize = indexes[index].item_index;
    if (positionInRange(position, occurrences[occurrence_index].range)) return occurrence_index;
    return null;
}

fn lowerBoundSymbolName(indexes: []const SymbolNameIndexEntry, name: []const u8) usize {
    var low: usize = 0;
    var high: usize = indexes.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (std.mem.lessThan(u8, indexes[mid].name, name)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

fn lowerBoundRangeStart(indexes: []const RangeIndexEntry, start: frontend.Position) usize {
    var low: usize = 0;
    var high: usize = indexes.len;
    while (low < high) {
        const mid = low + (high - low) / 2;
        if (positionLessThan(indexes[mid].range.start, start)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
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

fn lessSymbolNameIndex(_: void, lhs: SymbolNameIndexEntry, rhs: SymbolNameIndexEntry) bool {
    if (std.mem.lessThan(u8, lhs.name, rhs.name)) return true;
    if (std.mem.lessThan(u8, rhs.name, lhs.name)) return false;
    return lhs.symbol_index < rhs.symbol_index;
}

fn lessRangeIndex(_: void, lhs: RangeIndexEntry, rhs: RangeIndexEntry) bool {
    if (positionLessThan(lhs.range.start, rhs.range.start)) return true;
    if (positionLessThan(rhs.range.start, lhs.range.start)) return false;
    return lhs.item_index < rhs.item_index;
}

fn isCallableSymbol(kind: semantic_index.SymbolKind) bool {
    return kind == .function or kind == .method;
}

fn frontendRangesEqual(a: frontend.Range, b: frontend.Range) bool {
    return a.start.line == b.start.line and
        a.start.character == b.start.character and
        a.end.line == b.end.line and
        a.end.character == b.end.character;
}

fn positionsEqual(a: frontend.Position, b: frontend.Position) bool {
    return a.line == b.line and a.character == b.character;
}

fn positionLessThan(lhs: frontend.Position, rhs: frontend.Position) bool {
    if (lhs.line < rhs.line) return true;
    if (lhs.line > rhs.line) return false;
    return lhs.character < rhs.character;
}

fn sumBuilderGrowthEvents(events: []const usize) usize {
    var total: usize = 0;
    for (events) |event_count| total = addSat(total, event_count);
    return total;
}
