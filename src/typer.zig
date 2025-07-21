const std = @import("std");
const ast = @import("ast.zig");

/// Advanced struct semantics and lifecycle management
pub const StructSemantics = struct {
    /// Constructor function (optional)
    constructor: ?*const Function,
    /// Destructor function (optional)
    destructor: ?*const Function,
    /// Copy behavior
    copy_behavior: CopyBehavior,
    /// Move behavior
    move_behavior: MoveBehavior,
    /// Drop behavior
    drop_behavior: DropBehavior,
    /// Lifetime management
    lifetime_policy: LifetimePolicy,

    pub const CopyBehavior = enum {
        /// Shallow copy (copy pointers/references)
        shallow,
        /// Deep copy (copy all nested data)
        deep,
        /// Copy disabled (move-only type)
        disabled,
    };

    pub const MoveBehavior = enum {
        /// Move allowed (transfers ownership)
        allowed,
        /// Move disabled (copy-only type)
        disabled,
    };

    pub const DropBehavior = enum {
        /// Automatic cleanup when going out of scope
        automatic,
        /// Manual cleanup required
        manual,
        /// No cleanup needed
        none,
    };

    pub const LifetimePolicy = enum {
        /// Follows owner's lifetime
        owner,
        /// Independent lifetime
        independent,
        /// Static lifetime
        static,
    };

    pub fn init() StructSemantics {
        return StructSemantics{
            .constructor = null,
            .destructor = null,
            .copy_behavior = .shallow,
            .move_behavior = .allowed,
            .drop_behavior = .automatic,
            .lifetime_policy = .owner,
        };
    }

    pub fn canCopy(self: *const StructSemantics) bool {
        return self.copy_behavior != .disabled;
    }

    pub fn canMove(self: *const StructSemantics) bool {
        return self.move_behavior == .allowed;
    }

    pub fn needsDestructor(self: *const StructSemantics) bool {
        return self.destructor != null or self.drop_behavior == .manual;
    }
};

/// Enhanced struct type definition with advanced semantics
pub const StructType = struct {
    name: []const u8,
    fields: []StructField,
    allocator: std.mem.Allocator,
    layout: MemoryLayout,
    semantics: StructSemantics,

    pub const StructField = struct {
        name: []const u8,
        typ: OraType,
        offset: u32, // Byte offset within the struct
        slot: u32, // Storage slot number (for storage variables)
        slot_offset: u32, // Byte offset within the storage slot
        mutable: bool, // Field mutability
        access_level: AccessLevel, // Field access control

        pub const AccessLevel = enum {
            public,
            protected,
            private,
        };
    };

    /// Memory layout information for the struct
    pub const MemoryLayout = struct {
        total_size: u32, // Total size in bytes
        storage_slots: u32, // Number of storage slots needed
        alignment: u32, // Required alignment
        packed_efficiently: bool, // Whether packing was successful
    };

    pub fn init(allocator: std.mem.Allocator, name: []const u8, fields: []StructField) StructType {
        return StructType{
            .name = name,
            .fields = fields,
            .allocator = allocator,
            .layout = MemoryLayout{
                .total_size = 0,
                .storage_slots = 0,
                .alignment = 32,
                .packed_efficiently = false,
            },
            .semantics = StructSemantics.init(),
        };
    }

    pub fn deinit(self: *StructType) void {
        self.allocator.free(self.fields);
    }

    pub fn getField(self: *const StructType, field_name: []const u8) ?*const StructField {
        for (self.fields) |*field| {
            if (std.mem.eql(u8, field.name, field_name)) {
                return field;
            }
        }
        return null;
    }

    /// Set constructor function for the struct
    pub fn setConstructor(self: *StructType, constructor: *const Function) void {
        self.semantics.constructor = constructor;
    }

    /// Set destructor function for the struct
    pub fn setDestructor(self: *StructType, destructor: *const Function) void {
        self.semantics.destructor = destructor;
    }

    /// Configure copy behavior
    pub fn setCopyBehavior(self: *StructType, behavior: StructSemantics.CopyBehavior) void {
        self.semantics.copy_behavior = behavior;
    }

    /// Configure move behavior
    pub fn setMoveBehavior(self: *StructType, behavior: StructSemantics.MoveBehavior) void {
        self.semantics.move_behavior = behavior;
    }

    /// Configure drop behavior
    pub fn setDropBehavior(self: *StructType, behavior: StructSemantics.DropBehavior) void {
        self.semantics.drop_behavior = behavior;
    }

    /// Check if field is accessible in current context
    pub fn isFieldAccessible(self: *const StructType, field_name: []const u8, context: AccessContext) bool {
        if (self.getField(field_name)) |field| {
            return switch (field.access_level) {
                .public => true,
                .protected => context == .same_module or context == .derived_class,
                .private => context == .same_struct,
            };
        }
        return false;
    }

    /// Generate copy operation for the struct
    pub fn generateCopyOperation(self: *const StructType, dest: []const u8, src: []const u8) []const u8 {
        _ = dest;
        _ = src;
        // This would generate the appropriate copy logic based on copy_behavior
        // For now, return a placeholder
        return switch (self.semantics.copy_behavior) {
            .shallow => "shallow_copy",
            .deep => "deep_copy",
            .disabled => "copy_disabled",
        };
    }

    /// Generate move operation for the struct
    pub fn generateMoveOperation(self: *const StructType, dest: []const u8, src: []const u8) []const u8 {
        _ = dest;
        _ = src;
        // This would generate the appropriate move logic
        // For now, return a placeholder
        return switch (self.semantics.move_behavior) {
            .allowed => "move_operation",
            .disabled => "move_disabled",
        };
    }

    /// Generate drop operation for the struct
    pub fn generateDropOperation(self: *const StructType, instance: []const u8) []const u8 {
        _ = instance;
        // This would generate the appropriate drop logic
        // For now, return a placeholder
        return switch (self.semantics.drop_behavior) {
            .automatic => "auto_drop",
            .manual => "manual_drop",
            .none => "no_drop",
        };
    }

    /// Calculate optimal memory layout with advanced packing strategies
    pub fn optimizeLayout(self: *StructType, region: ast.MemoryRegion) void {
        switch (region) {
            .Storage => self.optimizeStorageLayout(),
            .Memory => self.optimizeMemoryLayout(),
            .Stack => self.optimizeStackLayout(),
            else => self.optimizeDefaultLayout(),
        }
    }

    /// Optimize layout for EVM storage (most critical for gas costs)
    fn optimizeStorageLayout(self: *StructType) void {
        // Sort fields by size for optimal packing (largest first, then pack smaller ones)
        var field_list = std.ArrayList(StructField).init(self.allocator);
        defer field_list.deinit();

        // Copy fields and sort by size (descending)
        for (self.fields) |field| {
            field_list.append(field) catch return;
        }

        // Sort fields: largest first, then by type alignment requirements
        std.sort.pdq(StructField, field_list.items, {}, compareFieldsForStorage);

        var current_slot: u32 = 0;
        var current_offset: u32 = 0;
        var total_offset: u32 = 0;

        for (field_list.items, 0..) |*field, i| {
            const field_size = getTypeSize(field.typ);
            const field_alignment = getTypeAlignment(field.typ);

            // Try to pack into current slot
            const aligned_offset = alignUp(current_offset, field_alignment);

            if (aligned_offset + field_size <= 32) {
                // Fits in current slot
                field.slot = current_slot;
                field.slot_offset = aligned_offset;
                field.offset = total_offset + aligned_offset;
                current_offset = aligned_offset + field_size;
            } else {
                // Need new slot
                current_slot += 1;
                total_offset += 32; // Move to next 32-byte slot
                current_offset = 0;

                const new_aligned_offset = alignUp(current_offset, field_alignment);
                field.slot = current_slot;
                field.slot_offset = new_aligned_offset;
                field.offset = total_offset + new_aligned_offset;
                current_offset = new_aligned_offset + field_size;
            }

            // Update the original field
            self.fields[i] = field.*;
        }

        // Update layout information
        self.layout = MemoryLayout{
            .total_size = (current_slot + 1) * 32,
            .storage_slots = current_slot + 1,
            .alignment = 32,
            .packed_efficiently = self.calculatePackingEfficiency() > 0.75,
        };
    }

    /// Optimize layout for EVM memory (transaction-scoped)
    fn optimizeMemoryLayout(self: *StructType) void {
        // For memory, we prioritize access patterns over storage efficiency
        var total_offset: u32 = 0;

        for (self.fields, 0..) |*field, i| {
            const field_size = getTypeSize(field.typ);
            const field_alignment = getTypeAlignment(field.typ);

            // Align to field requirements
            total_offset = alignUp(total_offset, field_alignment);

            field.offset = total_offset;
            field.slot = total_offset / 32;
            field.slot_offset = total_offset % 32;

            total_offset += field_size;

            self.fields[i] = field.*;
        }

        // Align total size to 32-byte boundary
        total_offset = alignUp(total_offset, 32);

        self.layout = MemoryLayout{
            .total_size = total_offset,
            .storage_slots = (total_offset + 31) / 32,
            .alignment = 32,
            .packed_efficiently = true, // Memory layout is always considered efficient
        };
    }

    /// Optimize layout for stack variables (local variables)
    fn optimizeStackLayout(self: *StructType) void {
        // Stack variables can use tighter packing
        var total_offset: u32 = 0;

        // Sort by alignment requirements for better packing
        var field_list = std.ArrayList(StructField).init(self.allocator);
        defer field_list.deinit();

        for (self.fields) |field| {
            field_list.append(field) catch return;
        }

        std.sort.pdq(StructField, field_list.items, {}, compareFieldsForStack);

        for (field_list.items, 0..) |*field, i| {
            const field_size = getTypeSize(field.typ);
            const field_alignment = getTypeAlignment(field.typ);

            total_offset = alignUp(total_offset, field_alignment);

            field.offset = total_offset;
            field.slot = total_offset / 32;
            field.slot_offset = total_offset % 32;

            total_offset += field_size;
            self.fields[i] = field.*;
        }

        self.layout = MemoryLayout{
            .total_size = total_offset,
            .storage_slots = (total_offset + 31) / 32,
            .alignment = 1, // Stack can use byte alignment
            .packed_efficiently = true,
        };
    }

    /// Default layout optimization
    fn optimizeDefaultLayout(self: *StructType) void {
        self.optimizeMemoryLayout(); // Default to memory layout
    }

    /// Calculate packing efficiency (ratio of used bytes to total bytes)
    fn calculatePackingEfficiency(self: *const StructType) f32 {
        var used_bytes: u32 = 0;
        for (self.fields) |field| {
            used_bytes += getTypeSize(field.typ);
        }

        if (self.layout.total_size == 0) return 0.0;
        return @as(f32, @floatFromInt(used_bytes)) / @as(f32, @floatFromInt(self.layout.total_size));
    }

    /// Get total size (for compatibility)
    pub fn calculateSize(self: *const StructType) u32 {
        return self.layout.total_size;
    }

    /// Get number of storage slots needed
    pub fn getStorageSlots(self: *const StructType) u32 {
        return self.layout.storage_slots;
    }

    /// Check if layout is efficiently packed
    pub fn isEfficientlyPacked(self: *const StructType) bool {
        return self.layout.packed_efficiently;
    }
};

/// Access context for field access control
pub const AccessContext = enum {
    same_struct,
    same_module,
    derived_class,
    external,
};

/// Function placeholder for constructor/destructor
pub const Function = struct {
    name: []const u8,
    // Additional function metadata would go here
};

/// Enum type definition with discriminant values
pub const EnumType = struct {
    name: []const u8,
    variants: []EnumVariant,
    base_type: OraType, // The underlying type (e.g., u32, u8, etc.)
    layout: EnumLayout,
    allocator: std.mem.Allocator,

    pub const EnumVariant = struct {
        name: []const u8,
        discriminant: u64, // The numeric value of this variant
        span: ast.SourceSpan,
    };

    pub const EnumLayout = struct {
        size: u32, // Size in bytes
        alignment: u32, // Alignment requirements
        discriminant_size: u32, // Size of discriminant field
    };

    pub fn init(allocator: std.mem.Allocator, name: []const u8, base_type: OraType) EnumType {
        return EnumType{
            .name = name,
            .variants = &[_]EnumVariant{},
            .base_type = base_type,
            .layout = EnumLayout{
                .size = getTypeSize(base_type),
                .alignment = getTypeAlignment(base_type),
                .discriminant_size = getTypeSize(base_type),
            },
            .allocator = allocator,
        };
    }

    pub fn addVariant(self: *EnumType, variant: EnumVariant) !void {
        const new_variants = try self.allocator.realloc(self.variants, self.variants.len + 1);
        new_variants[new_variants.len - 1] = variant;
        self.variants = new_variants;
    }

    pub fn findVariant(self: *const EnumType, name: []const u8) ?*const EnumVariant {
        for (self.variants) |*variant| {
            if (std.mem.eql(u8, variant.name, name)) {
                return variant;
            }
        }
        return null;
    }

    pub fn getVariantDiscriminant(self: *const EnumType, name: []const u8) ?u64 {
        if (self.findVariant(name)) |variant| {
            return variant.discriminant;
        }
        return null;
    }

    pub fn calculateSize(self: *const EnumType) u32 {
        return self.layout.size;
    }

    pub fn calculateAlignment(self: *const EnumType) u32 {
        return self.layout.alignment;
    }

    pub fn deinit(self: *EnumType) void {
        self.allocator.free(self.variants);
    }
};

/// Compare fields for storage optimization (largest first, then by alignment)
fn compareFieldsForStorage(context: void, a: StructType.StructField, b: StructType.StructField) bool {
    _ = context;
    const size_a = getTypeSize(a.typ);
    const size_b = getTypeSize(b.typ);

    if (size_a != size_b) {
        return size_a > size_b; // Larger fields first
    }

    // Same size, sort by alignment requirements
    const align_a = getTypeAlignment(a.typ);
    const align_b = getTypeAlignment(b.typ);
    return align_a > align_b;
}

/// Compare fields for stack optimization (by alignment, then size)
fn compareFieldsForStack(context: void, a: StructType.StructField, b: StructType.StructField) bool {
    _ = context;
    const align_a = getTypeAlignment(a.typ);
    const align_b = getTypeAlignment(b.typ);

    if (align_a != align_b) {
        return align_a > align_b; // Higher alignment first
    }

    const size_a = getTypeSize(a.typ);
    const size_b = getTypeSize(b.typ);
    return size_a > size_b;
}

/// Get alignment requirements for a type
pub fn getTypeAlignment(ora_type: OraType) u32 {
    return switch (ora_type) {
        .Bool => 1,
        .U8, .I8 => 1,
        .U16, .I16 => 2,
        .U32, .I32 => 4,
        .U64, .I64 => 8,
        .U128, .I128 => 16,
        .U256, .I256 => 32,
        .Address => 20, // Ethereum addresses are 20 bytes
        .String, .Bytes => 32, // Dynamic types require 32-byte alignment
        .Slice => 32,
        .Mapping, .DoubleMap => 32,
        .Struct => |struct_type| struct_type.layout.alignment,
        .Enum => |enum_type| enum_type.layout.alignment,
        .Function => 32,
        .Void => 1,
        .Unknown, .Error => 32,
        .Tuple => |tuple| blk: {
            var max_alignment: u32 = 1;
            for (tuple.types) |typ| {
                max_alignment = @max(max_alignment, getTypeAlignment(typ));
            }
            break :blk max_alignment;
        },
    };
}

/// Align a value up to the specified alignment
fn alignUp(value: u32, alignment: u32) u32 {
    return (value + alignment - 1) & ~(alignment - 1);
}

/// Type system for Ora
pub const OraType = union(enum) {
    // Primitive types
    Bool: void,
    Address: void,
    U8: void,
    U16: void,
    U32: void,
    U64: void,
    U128: void,
    U256: void,
    I8: void,
    I16: void,
    I32: void,
    I64: void,
    I128: void,
    I256: void,
    String: void,
    Bytes: void,

    // Complex types
    Slice: *OraType,
    Mapping: struct {
        key: *OraType,
        value: *OraType,
    },
    DoubleMap: struct {
        key1: *OraType,
        key2: *OraType,
        value: *OraType,
    },

    // Custom types
    Struct: *StructType,
    Enum: *EnumType,

    // Function type
    Function: struct {
        params: []OraType,
        return_type: ?*OraType,
    },

    // Special types
    Void: void,
    Unknown: void,
    Error: void,
    Tuple: struct {
        types: []OraType,
    },
};

/// Get the size of a type in bytes
pub fn getTypeSize(ora_type: OraType) u32 {
    return switch (ora_type) {
        .Bool => 1,
        .Address => 20,
        .U8, .I8 => 1,
        .U16, .I16 => 2,
        .U32, .I32 => 4,
        .U64, .I64 => 8,
        .U128, .I128 => 16,
        .U256, .I256 => 32,
        .String, .Bytes => 32, // Dynamic size, stored as pointer
        .Slice => 32, // Dynamic size, stored as pointer
        .Mapping, .DoubleMap => 32, // Storage slot reference
        .Struct => |struct_type| struct_type.calculateSize(),
        .Enum => |enum_type| enum_type.calculateSize(),
        .Function => 32, // Function pointer
        .Void => 0,
        .Unknown, .Error => 32, // Default to 32 bytes
        .Tuple => |tuple| blk: {
            var size: u32 = 0;
            for (tuple.types) |typ| {
                size += getTypeSize(typ);
            }
            break :blk size;
        },
    };
}

/// Type checking errors
pub const TyperError = error{
    UndeclaredVariable,
    TypeMismatch,
    InvalidOperation,
    UndeclaredFunction,
    ArgumentCountMismatch,
    InvalidMemoryRegion,
    OutOfMemory,
};

/// Symbol table entry
pub const Symbol = struct {
    name: []const u8,
    typ: OraType,
    region: ast.MemoryRegion,
    mutable: bool,
    span: ast.SourceSpan,
};

/// Symbol table for scope management (using ArrayList to avoid HashMap overflow in Zig 0.14.1)
pub const SymbolTable = struct {
    symbols: std.ArrayList(Symbol),
    parent: ?*SymbolTable,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, parent: ?*SymbolTable) SymbolTable {
        return SymbolTable{
            .symbols = std.ArrayList(Symbol).init(allocator),
            .parent = parent,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SymbolTable) void {
        self.symbols.deinit();
    }

    pub fn declare(self: *SymbolTable, symbol: Symbol) !void {
        try self.symbols.append(symbol);
    }

    pub fn lookup(self: *SymbolTable, name: []const u8) ?Symbol {
        // Linear search - O(n) but fine for small symbol tables
        for (self.symbols.items) |symbol| {
            if (std.mem.eql(u8, symbol.name, name)) {
                return symbol;
            }
        }

        if (self.parent) |parent| {
            return parent.lookup(name);
        }

        return null;
    }
};

/// Type checker for ZigOra
pub const Typer = struct {
    allocator: std.mem.Allocator,
    global_scope: SymbolTable,
    current_scope: *SymbolTable,
    current_function: ?[]const u8,
    type_arena: std.heap.ArenaAllocator,
    /// Track allocated function parameter arrays for cleanup
    function_params: std.ArrayList([]OraType),
    /// Registry for custom struct types
    struct_types: std.ArrayList(StructType),

    pub fn init(allocator: std.mem.Allocator) Typer {
        return Typer{
            .allocator = allocator,
            .global_scope = SymbolTable.init(allocator, null),
            .current_scope = undefined, // Will be fixed in fixSelfReferences
            .current_function = null,
            .type_arena = std.heap.ArenaAllocator.init(allocator),
            .function_params = std.ArrayList([]OraType).init(allocator),
            .struct_types = std.ArrayList(StructType).init(allocator),
        };
    }

    /// Fix self-references after struct initialization
    pub fn fixSelfReferences(self: *Typer) void {
        self.current_scope = &self.global_scope;
        // Initialize standard library symbols after self-references are fixed
        self.initStandardLibrary() catch {};
    }

    /// Register a custom struct type with memory layout optimization
    pub fn registerStructType(self: *Typer, struct_decl: *ast.StructDeclNode) TyperError!void {
        // Convert AST struct fields to StructType fields (initial layout)
        var fields = try self.allocator.alloc(StructType.StructField, struct_decl.fields.len);

        for (struct_decl.fields, 0..) |ast_field, i| {
            const field_type = try self.convertAstTypeToOraType(&ast_field.typ);
            fields[i] = StructType.StructField{
                .name = ast_field.name,
                .typ = field_type,
                .offset = 0, // Will be calculated during optimization
                .slot = 0, // Will be calculated during optimization
                .slot_offset = 0, // Will be calculated during optimization
                .mutable = false, // Default to immutable
                .access_level = .public, // Default to public
            };
        }

        // Create the struct type with initial fields
        var struct_type = StructType.init(self.allocator, struct_decl.name, fields);

        // Optimize layout for storage by default (most common use case)
        // The actual memory region will be determined when the struct variable is declared
        struct_type.optimizeLayout(.Storage);

        try self.struct_types.append(struct_type);
    }

    /// Look up a struct type by name
    pub fn getStructType(self: *const Typer, name: []const u8) ?*const StructType {
        for (self.struct_types.items) |*struct_type| {
            if (std.mem.eql(u8, struct_type.name, name)) {
                return struct_type;
            }
        }
        return null;
    }

    /// Initialize standard library symbols in global scope
    pub fn initStandardLibrary(self: *Typer) TyperError!void {
        // Add 'std' as a module identifier
        const std_symbol = Symbol{
            .name = "std",
            .typ = OraType.Unknown, // Module type - will be refined later
            .region = .Stack,
            .mutable = false,
            .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 3 },
        };
        try self.current_scope.declare(std_symbol);

        // Initialize standard library modules and their fields
        try self.initTransactionModule();
        try self.initBlockModule();
        try self.initConstantsModule();

        // Add built-in functions
        const require_params = try self.allocator.alloc(OraType, 1);
        require_params[0] = OraType.Bool;
        try self.function_params.append(require_params); // Track for cleanup

        const require_symbol = Symbol{
            .name = "require",
            .typ = OraType{ .Function = .{
                .params = require_params,
                .return_type = null,
            } },
            .region = .Stack,
            .mutable = false,
            .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 7 },
        };
        try self.current_scope.declare(require_symbol);
    }

    /// Initialize std.transaction module symbols
    fn initTransactionModule(self: *Typer) TyperError!void {
        // Add module identifier - silently ignore if already exists
        const transaction_symbol = Symbol{
            .name = "transaction",
            .typ = OraType.Unknown, // Transaction context module
            .region = .Stack,
            .mutable = false,
            .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 11 },
        };
        self.current_scope.declare(transaction_symbol) catch {
            // Ignore errors - symbol may already exist or OOM (non-critical)
        };
    }

    /// Initialize std.block module symbols
    fn initBlockModule(self: *Typer) TyperError!void {
        // Add module identifier - silently ignore if already exists
        const block_symbol = Symbol{
            .name = "block",
            .typ = OraType.Unknown, // Block context module
            .region = .Stack,
            .mutable = false,
            .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 5 },
        };
        self.current_scope.declare(block_symbol) catch {
            // Ignore errors - symbol may already exist or OOM (non-critical)
        };
    }

    /// Initialize std.constants module symbols
    fn initConstantsModule(self: *Typer) TyperError!void {
        // Add module identifier - silently ignore if already exists
        const constants_symbol = Symbol{
            .name = "constants",
            .typ = OraType.Unknown, // Constants module
            .region = .Stack,
            .mutable = false,
            .span = ast.SourceSpan{ .line = 0, .column = 0, .length = 9 },
        };
        self.current_scope.declare(constants_symbol) catch {
            // Ignore errors - symbol may already exist or OOM (non-critical)
        };
    }

    pub fn deinit(self: *Typer) void {
        // Free all types at once with arena
        self.type_arena.deinit();
        self.global_scope.deinit();
        // Clean up function parameter arrays
        for (self.function_params.items) |params| {
            self.allocator.free(params);
        }
        self.function_params.deinit();
        // Clean up struct types
        for (self.struct_types.items) |*struct_type| {
            struct_type.deinit();
        }
        self.struct_types.deinit();
    }

    /// Type check a list of top-level AST nodes
    pub fn typeCheck(self: *Typer, nodes: []ast.AstNode) TyperError!void {
        // First pass: collect all declarations
        for (nodes) |*node| {
            try self.collectDeclarations(node);
        }

        // Second pass: type check implementations
        for (nodes) |*node| {
            try self.typeCheckNode(node);
        }
    }

    /// Collect all declarations for forward references
    fn collectDeclarations(self: *Typer, node: *ast.AstNode) TyperError!void {
        switch (node.*) {
            .Contract => |*contract| {
                // Create contract scope and collect members
                for (contract.body) |*member| {
                    try self.collectDeclarations(member);
                }
            },
            .Function => |*function| {
                // Add function to symbol table (simplified to avoid createFunctionType issues)
                const symbol = Symbol{
                    .name = function.name,
                    .typ = OraType.Unknown, // Simplified for now
                    .region = .Stack, // Functions don't have memory regions
                    .mutable = false,
                    .span = function.span,
                };
                try self.current_scope.declare(symbol);
            },
            .VariableDecl => |*var_decl| {
                // Add variable to symbol table
                const var_type = try self.convertAstTypeToOraType(&var_decl.typ);
                const symbol = Symbol{
                    .name = var_decl.name,
                    .typ = var_type,
                    .region = var_decl.region,
                    .mutable = var_decl.mutable,
                    .span = var_decl.span,
                };
                try self.current_scope.declare(symbol);
            },
            .StructDecl => |*struct_decl| {
                // Register struct type in first pass to allow forward references
                try self.registerStructType(struct_decl);
            },
            .EnumDecl => |*enum_decl| {
                // Register enum type in first pass
                const enum_symbol = Symbol{
                    .name = enum_decl.name,
                    .typ = OraType.Unknown, // Will be filled later
                    .region = .Stack,
                    .mutable = false,
                    .span = enum_decl.span,
                };
                try self.current_scope.declare(enum_symbol);
            },
            else => {
                // Skip other node types in declaration phase
            },
        }
    }

    /// Type check a single AST node
    fn typeCheckNode(self: *Typer, node: *ast.AstNode) TyperError!void {
        switch (node.*) {
            .Contract => |*contract| {
                for (contract.body) |*member| {
                    try self.typeCheckNode(member);
                }
            },
            .Function => |*function| {
                try self.typeCheckFunction(function);
            },
            .VariableDecl => |*var_decl| {
                try self.typeCheckVariableDecl(var_decl);
            },
            .StructDecl => {
                // Struct already registered in first pass
            },
            else => {
                // TODO: Add type checking for: EnumDecl, LogDecl, Import, ErrorDecl (top-level), Block, Expression, Statement, TryBlock
            },
        }
    }

    /// Type check a function
    fn typeCheckFunction(self: *Typer, function: *ast.FunctionNode) TyperError!void {
        // Create function scope
        var func_scope = SymbolTable.init(self.allocator, self.current_scope);
        defer func_scope.deinit();

        const prev_scope = self.current_scope;
        const prev_function = self.current_function;
        self.current_scope = &func_scope;
        self.current_function = function.name;
        defer {
            self.current_scope = prev_scope;
            self.current_function = prev_function;
        }

        // Add parameters to function scope
        for (function.parameters) |*param| {
            const param_type = try self.convertAstTypeToOraType(&param.typ);
            const symbol = Symbol{
                .name = param.name,
                .typ = param_type,
                .region = .Stack,
                .mutable = false, // Parameters are immutable by default
                .span = param.span,
            };
            try self.current_scope.declare(symbol);
        }

        // Type check function body
        try self.typeCheckBlock(&function.body);

        // Verify return type consistency
        if (function.return_type) |*return_type| {
            const expected_return = try self.convertAstTypeToOraType(return_type);
            // TODO: Verify all return statements match this type
            _ = expected_return;
        }
    }

    /// Type check a variable declaration
    fn typeCheckVariableDecl(self: *Typer, var_decl: *ast.VariableDeclNode) TyperError!void {
        // Handle tuple unpacking
        if (var_decl.tuple_names) |tuple_names| {
            // Tuple unpacking: let (a, b) = expr
            if (var_decl.value) |*init_expr| {
                const init_type = try self.typeCheckExpression(init_expr);

                // Ensure initializer is a tuple type
                if (init_type != .Tuple) {
                    return TyperError.TypeMismatch;
                }

                const tuple_type = init_type.Tuple;

                // Ensure tuple arity matches
                if (tuple_names.len != tuple_type.types.len) {
                    return TyperError.TypeMismatch;
                }

                // Declare each tuple variable
                for (tuple_names, tuple_type.types) |name, typ| {
                    const symbol = Symbol{
                        .name = name,
                        .typ = typ,
                        .region = var_decl.region,
                        .mutable = var_decl.mutable,
                        .span = var_decl.span,
                    };

                    try self.current_scope.declare(symbol);
                }
            } else {
                return TyperError.TypeMismatch; // Tuple unpacking requires initializer
            }
        } else {
            // Regular variable declaration
            const var_type = try self.convertAstTypeToOraType(&var_decl.typ);

            // Type check initializer if present
            if (var_decl.value) |*init_expr| {
                const init_type = try self.typeCheckExpression(init_expr);
                if (!self.typesCompatible(var_type, init_type)) {
                    return TyperError.TypeMismatch;
                }
            }

            // Validate memory region constraints
            try self.validateMemoryRegion(var_decl.region, var_type);

            // Add the variable to the symbol table
            const symbol = Symbol{
                .name = var_decl.name,
                .typ = var_type,
                .region = var_decl.region,
                .mutable = var_decl.mutable,
                .span = var_decl.span,
            };

            try self.current_scope.declare(symbol);
        }
    }

    /// Type check a block of statements
    fn typeCheckBlock(self: *Typer, block: *ast.BlockNode) TyperError!void {
        for (block.statements) |*stmt| {
            try self.typeCheckStatement(stmt);
        }
    }

    /// Type check a statement
    fn typeCheckStatement(self: *Typer, stmt: *ast.StmtNode) TyperError!void {
        switch (stmt.*) {
            .Expr => |*expr| {
                _ = try self.typeCheckExpression(expr);
            },
            .VariableDecl => |*var_decl| {
                try self.typeCheckVariableDecl(var_decl);
            },
            .Return => |*ret| {
                if (ret.value) |*value| {
                    _ = try self.typeCheckExpression(value);
                    // TODO: Verify return type matches function signature
                }
            },
            .Log => |*log| {
                // Type check log arguments
                for (log.args) |*arg| {
                    _ = try self.typeCheckExpression(arg);
                }
            },
            .Lock => |*lock| {
                // Type check lock path
                _ = try self.typeCheckExpression(&lock.path);
            },
            .ErrorDecl => |*error_decl| {
                // Error declarations don't need type checking
                _ = error_decl;
            },
            .TryBlock => |*try_block| {
                try self.typeCheckBlock(&try_block.try_block);
                if (try_block.catch_block) |*catch_block| {
                    try self.typeCheckBlock(&catch_block.block);
                }
            },
            .If => |*if_stmt| {
                // Type check condition
                const condition_type = try self.typeCheckExpression(&if_stmt.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }

                // Type check then branch
                try self.typeCheckBlock(&if_stmt.then_branch);

                // Type check else branch if present
                if (if_stmt.else_branch) |*else_branch| {
                    try self.typeCheckBlock(else_branch);
                }
            },
            .While => |*while_stmt| {
                // Type check condition
                const condition_type = try self.typeCheckExpression(&while_stmt.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }

                // Type check body
                try self.typeCheckBlock(&while_stmt.body);
            },
            .Break => |*break_stmt| {
                // Break statements are always valid (context validation happens elsewhere)
                _ = break_stmt;
            },
            .Continue => |*continue_stmt| {
                // Continue statements are always valid (context validation happens elsewhere)
                _ = continue_stmt;
            },
            .Invariant => |*invariant| {
                // Invariant condition must be boolean
                const condition_type = try self.typeCheckExpression(&invariant.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }
            },
            .Requires => |*requires| {
                // Requires condition must be boolean
                const condition_type = try self.typeCheckExpression(&requires.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }
            },
            .Ensures => |*ensures| {
                // Ensures condition must be boolean
                const condition_type = try self.typeCheckExpression(&ensures.condition);
                if (!std.meta.eql(condition_type, OraType.Bool)) {
                    return TyperError.TypeMismatch;
                }
            },
        }
    }

    /// Type check an expression and return its type
    pub fn typeCheckExpression(self: *Typer, expr: *ast.ExprNode) TyperError!OraType {
        switch (expr.*) {
            .Identifier => |*ident| {
                if (self.current_scope.lookup(ident.name)) |symbol| {
                    return symbol.typ;
                } else {
                    return TyperError.UndeclaredVariable;
                }
            },
            .Literal => |*literal| {
                return try self.getLiteralType(literal);
            },
            .Binary => |*binary| {
                const lhs_type = try self.typeCheckExpression(binary.lhs);
                const rhs_type = try self.typeCheckExpression(binary.rhs);

                return try self.typeCheckBinaryOp(binary.operator, lhs_type, rhs_type);
            },
            .Assignment => |*assign| {
                const target_type = try self.typeCheckExpression(assign.target);
                const value_type = try self.typeCheckExpression(assign.value);

                if (!self.typesCompatible(target_type, value_type)) {
                    return TyperError.TypeMismatch;
                }

                return target_type;
            },
            .Call => |*call| {
                return try self.typeCheckFunctionCall(call);
            },
            .Try => |*try_expr| {
                // Try expressions return the success type of the error union
                return try self.typeCheckExpression(try_expr.expr);
            },
            .ErrorReturn => |*error_return| {
                // Error returns should be validated elsewhere
                _ = error_return;
                return OraType.Error;
            },
            .ErrorCast => |*error_cast| {
                // Error casts convert to error union type
                _ = try self.typeCheckExpression(error_cast.operand);
                return try self.convertAstTypeToOraType(&error_cast.target_type);
            },
            .Shift => |*shift| {
                // Type check shift expression components
                _ = try self.typeCheckExpression(shift.mapping);
                _ = try self.typeCheckExpression(shift.source);
                _ = try self.typeCheckExpression(shift.dest);
                _ = try self.typeCheckExpression(shift.amount);
                // Shift operations return void
                return OraType.Void;
            },
            .Tuple => |*tuple| {
                // Type check tuple expressions
                var tuple_types = std.ArrayList(OraType).init(self.allocator);
                defer tuple_types.deinit();

                for (tuple.elements) |*element| {
                    const element_type = try self.typeCheckExpression(element);
                    try tuple_types.append(element_type);
                }

                return OraType{ .Tuple = .{
                    .types = try tuple_types.toOwnedSlice(),
                } };
            },
            .Unary => |*unary| {
                const operand_type = try self.typeCheckExpression(unary.operand);
                return try self.typeCheckUnaryOp(unary.operator, operand_type);
            },
            .CompoundAssignment => |*compound| {
                const target_type = try self.typeCheckExpression(compound.target);
                const value_type = try self.typeCheckExpression(compound.value);

                // Validate compound operation (e.g., += requires numeric types)
                const result_type = try self.typeCheckCompoundAssignmentOp(compound.operator, target_type, value_type);
                if (!self.typesCompatible(target_type, result_type)) {
                    return TyperError.TypeMismatch;
                }
                return target_type;
            },
            .Index => |*index| {
                const target_type = try self.typeCheckExpression(index.target);
                const index_type = try self.typeCheckExpression(index.index);

                return try self.typeCheckIndexAccess(target_type, index_type);
            },
            .FieldAccess => |*field| {
                const target_type = try self.typeCheckExpression(field.target);
                return try self.typeCheckFieldAccess(target_type, field.field);
            },
            .Cast => |*cast| {
                const operand_type = try self.typeCheckExpression(cast.operand);
                const target_type = try self.convertAstTypeToOraType(&cast.target_type);

                // Validate cast safety
                if (!self.isCastValid(operand_type, target_type)) {
                    return TyperError.TypeMismatch;
                }
                return target_type;
            },
            .Old => |*old| {
                // old() expressions have the same type as their inner expression
                return try self.typeCheckExpression(old.expr);
            },
            .Comptime => |*comptime_block| {
                // Comptime blocks return void (they're evaluated at compile time)
                try self.typeCheckBlock(&comptime_block.block);
                return OraType.Void;
            },
            .StructInstantiation => |*struct_inst| {
                // The struct name should be an identifier
                if (struct_inst.struct_name.* != .Identifier) {
                    return TyperError.TypeMismatch;
                }

                const struct_name = struct_inst.struct_name.Identifier.name;
                if (self.getStructType(struct_name)) |struct_type| {
                    // Type check all field initializers
                    for (struct_inst.fields) |*field| {
                        _ = try self.typeCheckExpression(field.value);

                        // TODO: Validate field exists and type matches
                    }

                    // Return the struct type
                    const struct_type_ptr = try self.type_arena.allocator().create(StructType);
                    struct_type_ptr.* = struct_type.*;
                    return OraType{ .Struct = struct_type_ptr };
                } else {
                    return TyperError.UndeclaredVariable; // Struct type not found
                }
            },
            .EnumLiteral => |*enum_literal| {
                // Look up the enum type
                if (self.current_scope.lookup(enum_literal.enum_name)) |enum_symbol| {
                    if (enum_symbol.typ == .Enum) {
                        const enum_type = enum_symbol.typ.Enum;

                        // Validate that the variant exists
                        if (enum_type.findVariant(enum_literal.variant_name) != null) {
                            return enum_symbol.typ;
                        } else {
                            return TyperError.UndeclaredVariable; // Variant not found
                        }
                    } else {
                        return TyperError.TypeMismatch; // Not an enum type
                    }
                } else {
                    return TyperError.UndeclaredVariable; // Enum type not found
                }
            },
        }
    }

    /// Get the type of a literal
    fn getLiteralType(self: *Typer, literal: *ast.LiteralNode) TyperError!OraType {
        _ = self;
        return switch (literal.*) {
            .Integer => |*int_lit| {
                // Infer the smallest suitable integer type
                const value_str = int_lit.value;

                // Check if it's a negative number
                const is_negative = value_str.len > 0 and value_str[0] == '-';
                const abs_str = if (is_negative) value_str[1..] else value_str;

                if (is_negative) {
                    // Try parsing as different signed integer types
                    if (std.fmt.parseInt(i8, abs_str, 10)) |_| {
                        return OraType.I8;
                    } else |_| {}

                    if (std.fmt.parseInt(i16, abs_str, 10)) |_| {
                        return OraType.I16;
                    } else |_| {}

                    if (std.fmt.parseInt(i32, abs_str, 10)) |_| {
                        return OraType.I32;
                    } else |_| {}

                    if (std.fmt.parseInt(i64, abs_str, 10)) |_| {
                        return OraType.I64;
                    } else |_| {}

                    if (std.fmt.parseInt(i128, abs_str, 10)) |_| {
                        return OraType.I128;
                    } else |_| {}

                    // Default to i256 for very large negative numbers
                    return OraType.I256;
                } else {
                    // Try parsing as different unsigned integer types
                    if (std.fmt.parseInt(u8, value_str, 10)) |_| {
                        return OraType.U8;
                    } else |_| {}

                    if (std.fmt.parseInt(u16, value_str, 10)) |_| {
                        return OraType.U16;
                    } else |_| {}

                    if (std.fmt.parseInt(u32, value_str, 10)) |_| {
                        return OraType.U32;
                    } else |_| {}

                    if (std.fmt.parseInt(u64, value_str, 10)) |_| {
                        return OraType.U64;
                    } else |_| {}

                    if (std.fmt.parseInt(u128, value_str, 10)) |_| {
                        return OraType.U128;
                    } else |_| {}

                    // Default to u256 for very large numbers
                    return OraType.U256;
                }
            },
            .String => OraType.String,
            .Bool => OraType.Bool,
            .Address => OraType.Address,
            .Hex => OraType.U256, // Hex literals default to U256
        };
    }

    /// Type check a binary operation
    fn typeCheckBinaryOp(self: *Typer, op: ast.BinaryOp, lhs: OraType, rhs: OraType) TyperError!OraType {
        switch (op) {
            // Arithmetic operators
            .Plus, .Minus, .Star, .Slash, .Percent => {
                if (self.isNumericType(lhs) and self.isNumericType(rhs)) {
                    return self.commonNumericType(lhs, rhs);
                }
                return TyperError.TypeMismatch;
            },
            // Equality comparison operators
            .EqualEqual, .BangEqual => {
                if (self.typesCompatible(lhs, rhs)) {
                    return OraType.Bool;
                }
                // Special case: enum types can be compared for equality
                if (lhs == .Enum and rhs == .Enum) {
                    if (std.mem.eql(u8, lhs.Enum.name, rhs.Enum.name)) {
                        return OraType.Bool;
                    }
                }
                return TyperError.TypeMismatch;
            },
            // Ordered comparison operators
            .Less, .LessEqual, .Greater, .GreaterEqual => {
                if (self.isNumericType(lhs) and self.isNumericType(rhs)) {
                    return OraType.Bool;
                }
                // Special case: enum types can be compared for ordering based on discriminant values
                if (lhs == .Enum and rhs == .Enum) {
                    if (std.mem.eql(u8, lhs.Enum.name, rhs.Enum.name)) {
                        return OraType.Bool;
                    }
                }
                return TyperError.TypeMismatch;
            },
            // Logical operators
            .And, .Or => {
                if (std.meta.eql(lhs, OraType.Bool) and std.meta.eql(rhs, OraType.Bool)) {
                    return OraType.Bool;
                }
                return TyperError.TypeMismatch;
            },
            // Bitwise operators (for flag enums)
            .BitAnd, .BitOr, .BitXor => {
                if (self.isNumericType(lhs) and self.isNumericType(rhs)) {
                    return self.commonNumericType(lhs, rhs);
                }
                // Special case: enum types can use bitwise operators for flag enums
                if (lhs == .Enum and rhs == .Enum) {
                    if (std.mem.eql(u8, lhs.Enum.name, rhs.Enum.name)) {
                        return lhs; // Return the enum type
                    }
                }
                return TyperError.TypeMismatch;
            },
            else => {
                return TyperError.InvalidOperation;
            },
        }
    }

    /// Type check a compound assignment operation
    fn typeCheckCompoundAssignmentOp(self: *Typer, op: ast.CompoundAssignmentOp, lhs: OraType, rhs: OraType) TyperError!OraType {
        switch (op) {
            .PlusEqual, .MinusEqual, .StarEqual, .SlashEqual, .PercentEqual => {
                if (self.isNumericType(lhs) and self.isNumericType(rhs)) {
                    return self.commonNumericType(lhs, rhs);
                }
                return TyperError.TypeMismatch;
            },
        }
    }

    /// Type check a function call
    fn typeCheckFunctionCall(self: *Typer, call: *ast.CallExpr) TyperError!OraType {
        // Extract function name from callee (assuming it's an identifier)
        const function_name = switch (call.callee.*) {
            .Identifier => |*ident| ident.name,
            else => return TyperError.InvalidOperation, // Complex callees not supported yet
        };

        // Check if function exists in symbol table
        if (self.current_scope.lookup(function_name)) |symbol| {
            switch (symbol.typ) {
                .Function => |func_type| {
                    // Validate argument count
                    if (call.arguments.len != func_type.params.len) {
                        return TyperError.ArgumentCountMismatch;
                    }

                    // Type check each argument
                    for (call.arguments, func_type.params) |*arg, expected_param| {
                        const arg_type = try self.typeCheckExpression(arg);
                        if (!self.typesCompatible(arg_type, expected_param)) {
                            return TyperError.TypeMismatch;
                        }
                    }

                    // Return function's return type
                    if (func_type.return_type) |return_type| {
                        return return_type.*;
                    } else {
                        return OraType.Void;
                    }
                },
                else => {
                    // Not a function - trying to call a variable
                    return TyperError.InvalidOperation;
                },
            }
        }

        // Check for built-in functions
        if (self.isBuiltinFunction(function_name)) {
            return try self.typeCheckBuiltinCall(call);
        }

        return TyperError.UndeclaredFunction;
    }

    /// Check if a function is a built-in function
    fn isBuiltinFunction(self: *Typer, name: []const u8) bool {
        _ = self;
        // Actual built-in functions in Ora language
        return std.mem.eql(u8, name, "requires") or
            std.mem.eql(u8, name, "ensures") or
            std.mem.eql(u8, name, "invariant") or
            std.mem.eql(u8, name, "old") or
            std.mem.eql(u8, name, "log") or
            // Division functions (with @ prefix)
            std.mem.eql(u8, name, "@divmod") or
            std.mem.eql(u8, name, "@divTrunc") or
            std.mem.eql(u8, name, "@divFloor") or
            std.mem.eql(u8, name, "@divCeil") or
            std.mem.eql(u8, name, "@divExact");
    }

    /// Type check built-in function calls
    fn typeCheckBuiltinCall(self: *Typer, call: *ast.CallExpr) TyperError!OraType {
        // Extract function name from callee
        const function_name = switch (call.callee.*) {
            .Identifier => |*ident| ident.name,
            else => return TyperError.InvalidOperation,
        };

        if (std.mem.eql(u8, function_name, "requires")) {
            // requires(condition, [message]) -> void
            if (call.arguments.len < 1 or call.arguments.len > 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const condition_type = try self.typeCheckExpression(&call.arguments[0]);
            if (!std.meta.eql(condition_type, OraType.Bool)) {
                return TyperError.TypeMismatch;
            }

            if (call.arguments.len == 2) {
                const message_type = try self.typeCheckExpression(&call.arguments[1]);
                if (!std.meta.eql(message_type, OraType.String)) {
                    return TyperError.TypeMismatch;
                }
            }

            return OraType.Void;
        }

        if (std.mem.eql(u8, function_name, "ensures")) {
            // ensures(condition, [message]) -> void
            if (call.arguments.len < 1 or call.arguments.len > 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const condition_type = try self.typeCheckExpression(&call.arguments[0]);
            if (!std.meta.eql(condition_type, OraType.Bool)) {
                return TyperError.TypeMismatch;
            }

            if (call.arguments.len == 2) {
                const message_type = try self.typeCheckExpression(&call.arguments[1]);
                if (!std.meta.eql(message_type, OraType.String)) {
                    return TyperError.TypeMismatch;
                }
            }

            return OraType.Void;
        }

        if (std.mem.eql(u8, function_name, "invariant")) {
            // invariant(condition, [message]) -> void
            if (call.arguments.len < 1 or call.arguments.len > 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const condition_type = try self.typeCheckExpression(&call.arguments[0]);
            if (!std.meta.eql(condition_type, OraType.Bool)) {
                return TyperError.TypeMismatch;
            }

            if (call.arguments.len == 2) {
                const message_type = try self.typeCheckExpression(&call.arguments[1]);
                if (!std.meta.eql(message_type, OraType.String)) {
                    return TyperError.TypeMismatch;
                }
            }

            return OraType.Void;
        }

        if (std.mem.eql(u8, function_name, "old")) {
            // old(expression) -> same type as expression
            if (call.arguments.len != 1) {
                return TyperError.ArgumentCountMismatch;
            }

            // Return the same type as the argument
            return try self.typeCheckExpression(&call.arguments[0]);
        }

        if (std.mem.eql(u8, function_name, "log")) {
            // log is handled differently as it's a statement, not a function call
            // But if it appears in expression context, it returns void
            return OraType.Void;
        }

        // Division functions (Zig-inspired, with @ prefix)
        if (std.mem.eql(u8, function_name, "@divTrunc") or
            std.mem.eql(u8, function_name, "@divFloor") or
            std.mem.eql(u8, function_name, "@divCeil") or
            std.mem.eql(u8, function_name, "@divExact"))
        {
            // @divTrunc(a, b) -> same type as a and b (must be compatible)
            if (call.arguments.len != 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const lhs_type = try self.typeCheckExpression(&call.arguments[0]);
            const rhs_type = try self.typeCheckExpression(&call.arguments[1]);

            if (!self.isNumericType(lhs_type) or !self.isNumericType(rhs_type)) {
                return TyperError.TypeMismatch;
            }

            return self.commonNumericType(lhs_type, rhs_type);
        }

        if (std.mem.eql(u8, function_name, "@divmod")) {
            // @divmod(a, b) -> (quotient, remainder) tuple
            if (call.arguments.len != 2) {
                return TyperError.ArgumentCountMismatch;
            }

            const lhs_type = try self.typeCheckExpression(&call.arguments[0]);
            const rhs_type = try self.typeCheckExpression(&call.arguments[1]);

            if (!self.isNumericType(lhs_type) or !self.isNumericType(rhs_type)) {
                return TyperError.TypeMismatch;
            }

            const common_type = self.commonNumericType(lhs_type, rhs_type);

            // Return tuple type (quotient, remainder) both same type
            var tuple_types = std.ArrayList(OraType).init(self.allocator);
            defer tuple_types.deinit();

            try tuple_types.append(common_type); // quotient
            try tuple_types.append(common_type); // remainder

            return OraType{ .Tuple = .{
                .types = try tuple_types.toOwnedSlice(),
            } };
        }

        // Default for other built-ins
        return OraType.Unknown;
    }

    /// Convert AST type reference to ZigOra type
    pub fn convertAstTypeToOraType(self: *Typer, ast_type: *const ast.TypeRef) TyperError!OraType {
        return switch (ast_type.*) {
            .Bool => OraType.Bool,
            .Address => OraType.Address,
            .U8 => OraType.U8,
            .U16 => OraType.U16,
            .U32 => OraType.U32,
            .U64 => OraType.U64,
            .U128 => OraType.U128,
            .U256 => OraType.U256,
            .I8 => OraType.I8,
            .I16 => OraType.I16,
            .I32 => OraType.I32,
            .I64 => OraType.I64,
            .I128 => OraType.I128,
            .I256 => OraType.I256,
            .String => OraType.String,
            .Bytes => OraType.Bytes,
            .Slice => |slice_element_type| {
                // Use arena allocator for type lifetime management
                const element_type = try self.type_arena.allocator().create(OraType);
                element_type.* = try self.convertAstTypeToOraType(slice_element_type);
                return OraType{ .Slice = element_type };
            },
            .Mapping => |mapping_info| {
                // Use arena allocator for type lifetime management
                const key_type = try self.type_arena.allocator().create(OraType);
                key_type.* = try self.convertAstTypeToOraType(mapping_info.key);
                const value_type = try self.type_arena.allocator().create(OraType);
                value_type.* = try self.convertAstTypeToOraType(mapping_info.value);
                return OraType{ .Mapping = .{
                    .key = key_type,
                    .value = value_type,
                } };
            },
            .DoubleMap => |double_map_info| {
                // Use arena allocator for type lifetime management
                const key1_type = try self.type_arena.allocator().create(OraType);
                key1_type.* = try self.convertAstTypeToOraType(double_map_info.key1);
                const key2_type = try self.type_arena.allocator().create(OraType);
                key2_type.* = try self.convertAstTypeToOraType(double_map_info.key2);
                const value_type = try self.type_arena.allocator().create(OraType);
                value_type.* = try self.convertAstTypeToOraType(double_map_info.value);
                return OraType{ .DoubleMap = .{
                    .key1 = key1_type,
                    .key2 = key2_type,
                    .value = value_type,
                } };
            },
            .Identifier => |identifier| {
                // Look up custom types (structs, enums)
                if (self.getStructType(identifier)) |struct_type| {
                    // Return a const pointer directly to avoid copying issues
                    const struct_type_const_ptr = @as(*const StructType, struct_type);
                    // Cast to mutable pointer as required by OraType (but we won't mutate it)
                    const struct_type_ptr = @as(*StructType, @constCast(struct_type_const_ptr));
                    return OraType{ .Struct = struct_type_ptr };
                }

                // Look up enum types
                if (self.current_scope.lookup(identifier)) |symbol| {
                    if (symbol.typ == .Enum) {
                        return symbol.typ;
                    }
                }

                return OraType.Unknown;
            },
            .ErrorUnion => |error_union| {
                // Error unions use the success type as the primary type
                return try self.convertAstTypeToOraType(error_union.success_type);
            },
            .Result => |result| {
                // Result types use the ok type as the primary type
                return try self.convertAstTypeToOraType(result.ok_type);
            },
            .Tuple => |tuple| {
                // Convert tuple elements to OraType
                const element_types = try self.type_arena.allocator().alloc(OraType, tuple.types.len);
                for (tuple.types, 0..) |*element_type, i| {
                    element_types[i] = try self.convertAstTypeToOraType(element_type);
                }
                return OraType{ .Tuple = .{
                    .types = element_types,
                } };
            },
            .Unknown => OraType.Unknown,
        };
    }

    /// Create function type from function node
    fn createFunctionType(self: *Typer, function: *ast.FunctionNode) TyperError!OraType {
        _ = self;
        _ = function;
        // TODO: Implement proper function type creation without pointer issues
        // For now, return a simple type to avoid the segmentation fault
        return OraType.Unknown;
    }

    /// Check if two types are compatible
    fn typesCompatible(self: *Typer, lhs: OraType, rhs: OraType) bool {
        // Exact type match
        if (self.typeEquals(lhs, rhs)) {
            return true;
        }

        // Special case: enum types are compatible if they have the same name
        if (lhs == .Enum and rhs == .Enum) {
            return std.mem.eql(u8, lhs.Enum.name, rhs.Enum.name);
        }

        // Allow compatible numeric conversions
        return self.isNumericConversionValid(rhs, lhs);
    }

    /// Check if a numeric conversion is valid (from -> to)
    fn isNumericConversionValid(self: *Typer, from: OraType, to: OraType) bool {
        // Allow promotion within unsigned types
        const unsigned_hierarchy = [_]OraType{ .U8, .U16, .U32, .U64, .U128, .U256 };
        if (self.isTypeInHierarchy(from, &unsigned_hierarchy) and self.isTypeInHierarchy(to, &unsigned_hierarchy)) {
            return self.getTypeHierarchyIndex(from, &unsigned_hierarchy) <= self.getTypeHierarchyIndex(to, &unsigned_hierarchy);
        }

        // Allow promotion within signed types
        const signed_hierarchy = [_]OraType{ .I8, .I16, .I32, .I64, .I128, .I256 };
        if (self.isTypeInHierarchy(from, &signed_hierarchy) and self.isTypeInHierarchy(to, &signed_hierarchy)) {
            return self.getTypeHierarchyIndex(from, &signed_hierarchy) <= self.getTypeHierarchyIndex(to, &signed_hierarchy);
        }

        // Allow unsigned to signed conversion if the signed type is larger or equal
        switch (from) {
            .U8 => switch (to) {
                .I8, .I16, .I32, .I64, .I128, .I256 => return true,
                else => return false,
            },
            .U16 => switch (to) {
                .I16, .I32, .I64, .I128, .I256 => return true,
                else => return false,
            },
            .U32 => switch (to) {
                .I32, .I64, .I128, .I256 => return true,
                else => return false,
            },
            .U64 => switch (to) {
                .I64, .I128, .I256 => return true,
                else => return false,
            },
            .U128 => switch (to) {
                .I128, .I256 => return true,
                else => return false,
            },
            .U256 => switch (to) {
                .I256 => return true,
                else => return false,
            },
            else => return false,
        }
    }

    /// Check if a type is in a hierarchy
    fn isTypeInHierarchy(self: *Typer, typ: OraType, hierarchy: []const OraType) bool {
        for (hierarchy) |h_type| {
            if (self.typeEquals(typ, h_type)) {
                return true;
            }
        }
        return false;
    }

    /// Get the index of a type in a hierarchy
    fn getTypeHierarchyIndex(self: *Typer, typ: OraType, hierarchy: []const OraType) usize {
        for (hierarchy, 0..) |h_type, i| {
            if (self.typeEquals(typ, h_type)) {
                return i;
            }
        }
        return hierarchy.len; // Not found
    }

    /// Check if two types are structurally equal
    fn typeEquals(self: *Typer, lhs: OraType, rhs: OraType) bool {
        return switch (lhs) {
            .Bool => switch (rhs) {
                .Bool => true,
                else => false,
            },
            .Address => switch (rhs) {
                .Address => true,
                else => false,
            },
            .U8 => switch (rhs) {
                .U8 => true,
                else => false,
            },
            .U16 => switch (rhs) {
                .U16 => true,
                else => false,
            },
            .U32 => switch (rhs) {
                .U32 => true,
                else => false,
            },
            .U64 => switch (rhs) {
                .U64 => true,
                else => false,
            },
            .U128 => switch (rhs) {
                .U128 => true,
                else => false,
            },
            .U256 => switch (rhs) {
                .U256 => true,
                else => false,
            },
            .I8 => switch (rhs) {
                .I8 => true,
                else => false,
            },
            .I16 => switch (rhs) {
                .I16 => true,
                else => false,
            },
            .I32 => switch (rhs) {
                .I32 => true,
                else => false,
            },
            .I64 => switch (rhs) {
                .I64 => true,
                else => false,
            },
            .I128 => switch (rhs) {
                .I128 => true,
                else => false,
            },
            .I256 => switch (rhs) {
                .I256 => true,
                else => false,
            },
            .String => switch (rhs) {
                .String => true,
                else => false,
            },
            .Bytes => switch (rhs) {
                .Bytes => true,
                else => false,
            },
            .Void => switch (rhs) {
                .Void => true,
                else => false,
            },
            .Unknown => true, // Unknown types are compatible with everything
            .Error => switch (rhs) {
                .Error => true,
                else => false,
            },
            .Slice => |lhs_elem| switch (rhs) {
                .Slice => |rhs_elem| self.typeEquals(lhs_elem.*, rhs_elem.*),
                else => false,
            },
            .Mapping => |lhs_map| switch (rhs) {
                .Mapping => |rhs_map| self.typeEquals(lhs_map.key.*, rhs_map.key.*) and
                    self.typeEquals(lhs_map.value.*, rhs_map.value.*),
                else => false,
            },
            .DoubleMap => |lhs_dmap| switch (rhs) {
                .DoubleMap => |rhs_dmap| self.typeEquals(lhs_dmap.key1.*, rhs_dmap.key1.*) and
                    self.typeEquals(lhs_dmap.key2.*, rhs_dmap.key2.*) and
                    self.typeEquals(lhs_dmap.value.*, rhs_dmap.value.*),
                else => false,
            },
            .Function => |lhs_func| switch (rhs) {
                .Function => |rhs_func| {
                    // Compare parameter count
                    if (lhs_func.params.len != rhs_func.params.len) return false;
                    // Compare each parameter type
                    for (lhs_func.params, rhs_func.params) |lhs_param, rhs_param| {
                        if (!self.typeEquals(lhs_param, rhs_param)) return false;
                    }
                    // Compare return types
                    if (lhs_func.return_type) |lhs_ret| {
                        if (rhs_func.return_type) |rhs_ret| {
                            return self.typeEquals(lhs_ret.*, rhs_ret.*);
                        } else {
                            return false; // lhs has return type, rhs doesn't
                        }
                    } else {
                        return rhs_func.return_type == null; // both should have no return type
                    }
                },
                else => false,
            },
            .Tuple => |lhs_tuple| switch (rhs) {
                .Tuple => |rhs_tuple| {
                    // Compare element count
                    if (lhs_tuple.types.len != rhs_tuple.types.len) return false;
                    // Compare each element type
                    for (lhs_tuple.types, rhs_tuple.types) |lhs_elem, rhs_elem| {
                        if (!self.typeEquals(lhs_elem, rhs_elem)) return false;
                    }
                    return true;
                },
                else => false,
            },
            .Struct => |lhs_struct| switch (rhs) {
                .Struct => |rhs_struct| {
                    // Compare struct names (structs are equal if they have the same name)
                    return std.mem.eql(u8, lhs_struct.name, rhs_struct.name);
                },
                else => false,
            },
            .Enum => |lhs_enum| switch (rhs) {
                .Enum => |rhs_enum| {
                    // Compare enum names (enums are equal if they have the same name)
                    return std.mem.eql(u8, lhs_enum.name, rhs_enum.name);
                },
                else => false,
            },
        };
    }

    /// Check if a type is numeric
    fn isNumericType(self: *Typer, typ: OraType) bool {
        _ = self;
        return switch (typ) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }

    /// Get common numeric type for operations
    fn commonNumericType(self: *Typer, lhs: OraType, rhs: OraType) OraType {
        // If both types are the same, return that type
        if (self.typeEquals(lhs, rhs)) {
            return lhs;
        }

        // Mixed signed/unsigned arithmetic: promote to the larger signed type
        const signed_hierarchy = [_]OraType{ .I8, .I16, .I32, .I64, .I128, .I256 };
        const unsigned_hierarchy = [_]OraType{ .U8, .U16, .U32, .U64, .U128, .U256 };

        const lhs_is_signed = self.isTypeInHierarchy(lhs, &signed_hierarchy);
        const rhs_is_signed = self.isTypeInHierarchy(rhs, &signed_hierarchy);

        // If both are signed, promote to the larger one
        if (lhs_is_signed and rhs_is_signed) {
            const lhs_idx = self.getTypeHierarchyIndex(lhs, &signed_hierarchy);
            const rhs_idx = self.getTypeHierarchyIndex(rhs, &signed_hierarchy);
            return signed_hierarchy[@max(lhs_idx, rhs_idx)];
        }

        // If both are unsigned, promote to the larger one
        if (!lhs_is_signed and !rhs_is_signed) {
            const lhs_idx = self.getTypeHierarchyIndex(lhs, &unsigned_hierarchy);
            const rhs_idx = self.getTypeHierarchyIndex(rhs, &unsigned_hierarchy);
            return unsigned_hierarchy[@max(lhs_idx, rhs_idx)];
        }

        // Mixed signed/unsigned: promote to a signed type that can hold both
        const signed_type = if (lhs_is_signed) lhs else rhs;
        const unsigned_type = if (lhs_is_signed) rhs else lhs;

        const signed_idx = self.getTypeHierarchyIndex(signed_type, &signed_hierarchy);
        const unsigned_idx = self.getTypeHierarchyIndex(unsigned_type, &unsigned_hierarchy);

        // Use the signed type if it's large enough, otherwise promote to a larger signed type
        const min_signed_idx = @max(signed_idx, unsigned_idx);
        return signed_hierarchy[@min(min_signed_idx, signed_hierarchy.len - 1)];
    }

    /// Validate memory region constraints
    fn validateMemoryRegion(self: *Typer, region: ast.MemoryRegion, typ: OraType) TyperError!void {
        _ = self;

        switch (region) {
            .Storage => {
                // Only certain types can be stored in storage
                switch (typ) {
                    .Mapping, .DoubleMap => {}, // OK
                    .Bool, .Address, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .String => {}, // OK
                    .Struct, .Enum => {}, // Custom types are OK in storage
                    else => return TyperError.InvalidMemoryRegion,
                }
            },
            else => {
                // Other regions are more permissive
            },
        }
    }

    /// Type check unary operations
    fn typeCheckUnaryOp(self: *Typer, op: ast.UnaryOp, operand_type: OraType) TyperError!OraType {
        return switch (op) {
            .Minus => {
                if (self.isNumericType(operand_type)) {
                    return operand_type;
                }
                return TyperError.TypeMismatch;
            },
            .Bang => {
                if (std.meta.eql(operand_type, OraType.Bool)) {
                    return OraType.Bool;
                }
                return TyperError.TypeMismatch;
            },
            .BitNot => {
                if (self.isIntegerType(operand_type)) {
                    return operand_type;
                }
                return TyperError.TypeMismatch;
            },
        };
    }

    /// Type check index access (arrays, mappings)
    fn typeCheckIndexAccess(self: *Typer, target_type: OraType, index_type: OraType) TyperError!OraType {
        return switch (target_type) {
            .Slice => |elem_type| {
                // Array/slice indexing requires integer index
                if (self.isIntegerType(index_type)) {
                    return elem_type.*;
                }
                return TyperError.TypeMismatch;
            },
            .Mapping => |mapping| {
                // Mapping access requires compatible key type
                if (self.typesCompatible(index_type, mapping.key.*)) {
                    return mapping.value.*;
                }
                return TyperError.TypeMismatch;
            },
            .DoubleMap => |_| {
                // DoubleMap requires special syntax - shouldn't reach here with single index
                return TyperError.InvalidOperation;
            },
            else => return TyperError.InvalidOperation,
        };
    }

    /// Type check field access
    fn typeCheckFieldAccess(self: *Typer, target_type: OraType, field_name: []const u8) TyperError!OraType {
        _ = self;

        // Handle struct field access
        if (target_type == .Struct) {
            const struct_type = target_type.Struct;
            if (struct_type.getField(field_name)) |field| {
                return field.typ;
            }
            return TyperError.UndeclaredVariable; // Field doesn't exist
        }

        // Handle standard library module field access
        if (target_type == .Unknown) {
            // This could be std library access
            if (std.mem.eql(u8, field_name, "transaction") or
                std.mem.eql(u8, field_name, "block") or
                std.mem.eql(u8, field_name, "constants"))
            {
                // std.transaction, std.block, std.constants return module context types
                return OraType.Unknown; // Module type for further field access
            }

            // Handle nested field access for std library modules
            if (std.mem.eql(u8, field_name, "sender") or std.mem.eql(u8, field_name, "origin")) {
                return OraType.Address;
            }
            if (std.mem.eql(u8, field_name, "value") or std.mem.eql(u8, field_name, "gasprice")) {
                return OraType.U256;
            }
            if (std.mem.eql(u8, field_name, "ZERO_ADDRESS")) {
                return OraType.Address;
            }
            if (std.mem.eql(u8, field_name, "MAX_UINT256") or std.mem.eql(u8, field_name, "MIN_UINT256")) {
                return OraType.U256;
            }
            if (std.mem.eql(u8, field_name, "timestamp") or std.mem.eql(u8, field_name, "number") or
                std.mem.eql(u8, field_name, "difficulty") or std.mem.eql(u8, field_name, "gaslimit"))
            {
                return OraType.U256;
            }
            if (std.mem.eql(u8, field_name, "coinbase")) {
                return OraType.Address;
            }
        }

        // TODO: Implement enum field access when enums are added
        return OraType.Unknown;
    }

    /// Check if cast is valid
    fn isCastValid(self: *Typer, from: OraType, to: OraType) bool {
        // Same type casts are always valid
        if (self.typeEquals(from, to)) {
            return true;
        }

        // Numeric type conversions
        if (self.isNumericType(from) and self.isNumericType(to)) {
            return true; // Allow all numeric conversions (with potential warnings)
        }

        // Address <-> U256 conversions
        if ((std.meta.eql(from, OraType.Address) and std.meta.eql(to, OraType.U256)) or
            (std.meta.eql(from, OraType.U256) and std.meta.eql(to, OraType.Address)))
        {
            return true;
        }

        // Unknown types can be cast to anything (for incomplete code)
        if (std.meta.eql(from, OraType.Unknown) or std.meta.eql(to, OraType.Unknown)) {
            return true;
        }

        return false;
    }

    /// Check if type is an integer type
    fn isIntegerType(self: *Typer, typ: OraType) bool {
        _ = self;
        return switch (typ) {
            .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256 => true,
            else => false,
        };
    }
};
