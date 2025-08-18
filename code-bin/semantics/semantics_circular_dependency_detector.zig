const std = @import("std");
pub const ast = @import("../ast.zig");
const semantics_errors = @import("semantics_errors.zig");
const semantics_memory_safety = @import("semantics_memory_safety.zig");

// Forward declaration for SemanticAnalyzer
const SemanticAnalyzer = @import("semantics_core.zig").SemanticAnalyzer;

/// Dependency node in the dependency graph
pub const DependencyNode = struct {
    name: []const u8,
    node_type: DependencyType,
    dependencies: std.ArrayList([]const u8),
    span: ast.SourceSpan,
    allocator: std.mem.Allocator,

    pub const DependencyType = enum {
        Contract,
        Struct,
        Function,
        Variable,
        Type,
        Import,
    };

    pub fn init(allocator: std.mem.Allocator, name: []const u8, node_type: DependencyType, span: ast.SourceSpan) DependencyNode {
        return DependencyNode{
            .name = name,
            .node_type = node_type,
            .dependencies = std.ArrayList([]const u8).init(allocator),
            .span = span,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DependencyNode) void {
        self.dependencies.deinit();
    }

    pub fn addDependency(self: *DependencyNode, dependency: []const u8) !void {
        // Avoid duplicate dependencies
        for (self.dependencies.items) |existing| {
            if (std.mem.eql(u8, existing, dependency)) {
                return;
            }
        }
        try self.dependencies.append(dependency);
    }

    pub fn hasDependency(self: *DependencyNode, dependency: []const u8) bool {
        for (self.dependencies.items) |existing| {
            if (std.mem.eql(u8, existing, dependency)) {
                return true;
            }
        }
        return false;
    }
};

/// Circular dependency cycle information
pub const DependencyCycle = struct {
    cycle_nodes: [][]const u8,
    cycle_type: CycleType,
    severity: CycleSeverity,
    error_message: []const u8,
    spans: []ast.SourceSpan,

    pub const CycleType = enum {
        DirectCycle, // A -> B -> A
        IndirectCycle, // A -> B -> C -> A
        SelfReference, // A -> A
    };

    pub const CycleSeverity = enum {
        Error, // Must be fixed
        Warning, // Should be reviewed
        Info, // Informational only
    };

    pub fn deinit(self: *DependencyCycle, allocator: std.mem.Allocator) void {
        allocator.free(self.cycle_nodes);
        allocator.free(self.error_message);
        allocator.free(self.spans);
    }
};

/// Circular dependency detection result
pub const CircularDependencyResult = struct {
    has_cycles: bool,
    cycles: []DependencyCycle,
    total_nodes: u32,
    total_dependencies: u32,
    analysis_successful: bool,

    pub fn deinit(self: *CircularDependencyResult, allocator: std.mem.Allocator) void {
        for (self.cycles) |*cycle| {
            cycle.deinit(allocator);
        }
        allocator.free(self.cycles);
    }
};

/// Circular dependency detector for comprehensive dependency analysis
pub const CircularDependencyDetector = struct {
    allocator: std.mem.Allocator,
    dependency_graph: std.HashMap([]const u8, DependencyNode, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    visited_nodes: std.HashMap([]const u8, VisitState, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    current_path: std.ArrayList([]const u8),
    detected_cycles: std.ArrayList(DependencyCycle),

    const VisitState = enum {
        Unvisited,
        Visiting,
        Visited,
    };

    pub fn init(allocator: std.mem.Allocator) CircularDependencyDetector {
        return CircularDependencyDetector{
            .allocator = allocator,
            .dependency_graph = std.HashMap([]const u8, DependencyNode, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .visited_nodes = std.HashMap([]const u8, VisitState, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .current_path = std.ArrayList([]const u8).init(allocator),
            .detected_cycles = std.ArrayList(DependencyCycle).init(allocator),
        };
    }

    pub fn deinit(self: *CircularDependencyDetector) void {
        // Clean up dependency nodes
        var iterator = self.dependency_graph.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.dependency_graph.deinit();

        self.visited_nodes.deinit();
        self.current_path.deinit();

        // Clean up detected cycles
        for (self.detected_cycles.items) |*cycle| {
            cycle.deinit(self.allocator);
        }
        self.detected_cycles.deinit();
    }

    /// Add a node to the dependency graph
    pub fn addNode(self: *CircularDependencyDetector, name: []const u8, node_type: DependencyNode.DependencyType, span: ast.SourceSpan) !void {
        if (self.dependency_graph.contains(name)) {
            return; // Node already exists
        }

        const owned_name = try self.allocator.dupe(u8, name);
        const node = DependencyNode.init(self.allocator, owned_name, node_type, span);
        try self.dependency_graph.put(owned_name, node);
    }

    /// Add a dependency between two nodes
    pub fn addDependency(self: *CircularDependencyDetector, from: []const u8, to: []const u8) !void {
        // Ensure both nodes exist
        if (!self.dependency_graph.contains(from) or !self.dependency_graph.contains(to)) {
            return; // Skip if nodes don't exist
        }

        const from_node = self.dependency_graph.getPtr(from).?;
        try from_node.addDependency(to);
    }

    /// Detect circular dependencies in the graph
    pub fn detectCircularDependencies(self: *CircularDependencyDetector) !CircularDependencyResult {
        // Clear previous detection state
        self.visited_nodes.clearRetainingCapacity();
        self.current_path.clearRetainingCapacity();
        self.detected_cycles.clearRetainingCapacity();

        // Initialize all nodes as unvisited
        var iterator = self.dependency_graph.iterator();
        while (iterator.next()) |entry| {
            try self.visited_nodes.put(entry.key_ptr.*, .Unvisited);
        }

        // Perform DFS from each unvisited node
        iterator = self.dependency_graph.iterator();
        while (iterator.next()) |entry| {
            const node_name = entry.key_ptr.*;
            if (self.visited_nodes.get(node_name) == .Unvisited) {
                try self.dfsDetectCycles(node_name);
            }
        }

        // Calculate statistics
        const total_nodes = @as(u32, @intCast(self.dependency_graph.count()));
        var total_dependencies: u32 = 0;
        iterator = self.dependency_graph.iterator();
        while (iterator.next()) |entry| {
            total_dependencies += @as(u32, @intCast(entry.value_ptr.dependencies.items.len));
        }

        return CircularDependencyResult{
            .has_cycles = self.detected_cycles.items.len > 0,
            .cycles = try self.detected_cycles.toOwnedSlice(),
            .total_nodes = total_nodes,
            .total_dependencies = total_dependencies,
            .analysis_successful = true,
        };
    }

    /// Analyze contract dependencies
    pub fn analyzeContract(self: *CircularDependencyDetector, analyzer: *SemanticAnalyzer, contract: *ast.ContractNode) !void {
        // Add contract node
        try self.addNode(contract.name, .Contract, contract.span);

        // Analyze contract members for dependencies
        for (contract.body) |*member| {
            switch (member.*) {
                .VariableDecl => |*var_decl| {
                    try self.analyzeVariableDependencies(analyzer, contract.name, var_decl);
                },
                .Function => |*function| {
                    try self.analyzeFunctionDependencies(analyzer, contract.name, function);
                },
                .StructDecl => |*struct_decl| {
                    try self.analyzeStructDependencies(analyzer, contract.name, struct_decl);
                },
                else => {},
            }
        }
    }

    /// Analyze struct dependencies
    pub fn analyzeStruct(self: *CircularDependencyDetector, analyzer: *SemanticAnalyzer, struct_decl: *ast.StructNode) !void {
        // Add struct node
        try self.addNode(struct_decl.name, .Struct, struct_decl.span);

        // Analyze struct field dependencies
        for (struct_decl.fields) |*field| {
            try self.analyzeTypeDependencies(analyzer, struct_decl.name, &field.field_type);
        }
    }

    /// Analyze function dependencies
    pub fn analyzeFunction(self: *CircularDependencyDetector, analyzer: *SemanticAnalyzer, function: *ast.FunctionNode) !void {
        // Add function node
        try self.addNode(function.name, .Function, function.span);

        // Analyze parameter dependencies
        for (function.params) |*param| {
            try self.analyzeTypeDependencies(analyzer, function.name, &param.param_type);
        }

        // Analyze return type dependencies
        if (function.return_type) |*return_type| {
            try self.analyzeTypeDependencies(analyzer, function.name, return_type);
        }

        // Analyze function body dependencies (simplified)
        try self.analyzeBlockDependencies(analyzer, function.name, &function.body);
    }

    /// Private helper methods
    fn dfsDetectCycles(self: *CircularDependencyDetector, node_name: []const u8) !void {
        // Mark as visiting
        try self.visited_nodes.put(node_name, .Visiting);
        try self.current_path.append(node_name);

        // Get node dependencies
        const node = self.dependency_graph.get(node_name) orelse return;

        for (node.dependencies.items) |dependency| {
            const dep_state = self.visited_nodes.get(dependency) orelse .Unvisited;

            switch (dep_state) {
                .Visiting => {
                    // Found a cycle
                    try self.recordCycle(dependency);
                },
                .Unvisited => {
                    // Continue DFS
                    try self.dfsDetectCycles(dependency);
                },
                .Visited => {
                    // Already processed, skip
                },
            }
        }

        // Mark as visited and remove from current path
        try self.visited_nodes.put(node_name, .Visited);
        _ = self.current_path.pop();
    }

    fn recordCycle(self: *CircularDependencyDetector, cycle_start: []const u8) !void {
        // Find the start of the cycle in current path
        var cycle_start_index: ?usize = null;
        for (self.current_path.items, 0..) |path_node, i| {
            if (std.mem.eql(u8, path_node, cycle_start)) {
                cycle_start_index = i;
                break;
            }
        }

        const start_index = cycle_start_index orelse return;

        // Extract cycle nodes
        const cycle_nodes = try self.allocator.dupe([]const u8, self.current_path.items[start_index..]);

        // Determine cycle type
        const cycle_type = if (cycle_nodes.len == 1)
            DependencyCycle.CycleType.SelfReference
        else if (cycle_nodes.len == 2)
            DependencyCycle.CycleType.DirectCycle
        else
            DependencyCycle.CycleType.IndirectCycle;

        // Create error message
        const error_message = try self.createCycleErrorMessage(cycle_nodes, cycle_type);

        // Get spans for cycle nodes
        var spans = std.ArrayList(ast.SourceSpan).init(self.allocator);
        defer spans.deinit();

        for (cycle_nodes) |node_name| {
            if (self.dependency_graph.get(node_name)) |node| {
                try spans.append(node.span);
            }
        }

        // Create cycle record
        const cycle = DependencyCycle{
            .cycle_nodes = cycle_nodes,
            .cycle_type = cycle_type,
            .severity = self.determineCycleSeverity(cycle_type),
            .error_message = error_message,
            .spans = try spans.toOwnedSlice(),
        };

        try self.detected_cycles.append(cycle);
    }

    fn createCycleErrorMessage(self: *CircularDependencyDetector, cycle_nodes: [][]const u8, cycle_type: DependencyCycle.CycleType) ![]const u8 {
        switch (cycle_type) {
            .SelfReference => {
                return try std.fmt.allocPrint(self.allocator, "Self-reference detected: {s} depends on itself", .{cycle_nodes[0]});
            },
            .DirectCycle => {
                return try std.fmt.allocPrint(self.allocator, "Direct circular dependency: {s} -> {s} -> {s}", .{ cycle_nodes[0], cycle_nodes[1], cycle_nodes[0] });
            },
            .IndirectCycle => {
                var message = std.ArrayList(u8).init(self.allocator);
                defer message.deinit();

                try message.appendSlice("Circular dependency chain: ");
                for (cycle_nodes, 0..) |node, i| {
                    if (i > 0) try message.appendSlice(" -> ");
                    try message.appendSlice(node);
                }
                try message.appendSlice(" -> ");
                try message.appendSlice(cycle_nodes[0]);

                return message.toOwnedSlice();
            },
        }
    }

    fn determineCycleSeverity(self: *CircularDependencyDetector, cycle_type: DependencyCycle.CycleType) DependencyCycle.CycleSeverity {
        _ = self;
        return switch (cycle_type) {
            .SelfReference => .Warning,
            .DirectCycle => .Error,
            .IndirectCycle => .Error,
        };
    }

    fn analyzeVariableDependencies(self: *CircularDependencyDetector, analyzer: *SemanticAnalyzer, parent: []const u8, var_decl: *ast.VariableDeclNode) !void {
        try self.analyzeTypeDependencies(analyzer, parent, &var_decl.typ);
    }

    fn analyzeFunctionDependencies(self: *CircularDependencyDetector, analyzer: *SemanticAnalyzer, parent: []const u8, function: *ast.FunctionNode) !void {
        _ = self;
        _ = analyzer;
        _ = parent;
        _ = function;
        // TODO: Implement function dependency analysis
    }

    fn analyzeStructDependencies(self: *CircularDependencyDetector, analyzer: *SemanticAnalyzer, parent: []const u8, struct_decl: *ast.StructNode) !void {
        _ = analyzer;
        try self.addDependency(parent, struct_decl.name);
    }

    fn analyzeTypeDependencies(self: *CircularDependencyDetector, analyzer: *SemanticAnalyzer, parent: []const u8, type_node: *ast.TypeNode) !void {
        switch (type_node.*) {
            .Identifier => |*ident| {
                try self.addDependency(parent, ident.name);
            },
            .Array => |*array| {
                try self.analyzeTypeDependencies(analyzer, parent, array.element_type);
            },
            .Mapping => |*mapping| {
                try self.analyzeTypeDependencies(analyzer, parent, mapping.key_type);
                try self.analyzeTypeDependencies(analyzer, parent, mapping.value_type);
            },
            .ErrorUnion => |*error_union| {
                try self.analyzeTypeDependencies(analyzer, parent, error_union.value_type);
            },
            else => {},
        }
    }

    fn analyzeBlockDependencies(self: *CircularDependencyDetector, analyzer: *SemanticAnalyzer, parent: []const u8, block: *ast.BlockNode) !void {
        _ = self;
        _ = analyzer;
        _ = parent;
        _ = block;
        // TODO: Implement block dependency analysis
    }
};

/// Detect circular dependencies in contract
pub fn detectContractCircularDependencies(analyzer: *SemanticAnalyzer, contract: *ast.ContractNode) semantics_errors.SemanticError!void {
    var detector = CircularDependencyDetector.init(analyzer.allocator);
    defer detector.deinit();

    // Analyze contract dependencies
    detector.analyzeContract(analyzer, contract) catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Failed to analyze contract dependencies", contract.span);
                return;
            },
        }
    };

    // Detect cycles
    var result = detector.detectCircularDependencies() catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Failed to detect circular dependencies", contract.span);
                return;
            },
        }
    };
    defer result.deinit(analyzer.allocator);

    // Report detected cycles
    for (result.cycles) |cycle| {
        switch (cycle.severity) {
            .Error => {
                try semantics_errors.addErrorStatic(analyzer, cycle.error_message, cycle.spans[0]);
            },
            .Warning => {
                try semantics_errors.addWarningStatic(analyzer, cycle.error_message, cycle.spans[0]);
            },
            .Info => {
                try semantics_errors.addInfo(analyzer, try analyzer.allocator.dupe(u8, cycle.error_message), cycle.spans[0]);
            },
        }
    }

    if (result.has_cycles) {
        return semantics_errors.SemanticError.CircularDependency;
    }
}

/// Detect circular dependencies in struct
pub fn detectStructCircularDependencies(analyzer: *SemanticAnalyzer, struct_decl: *ast.StructNode) semantics_errors.SemanticError!void {
    var detector = CircularDependencyDetector.init(analyzer.allocator);
    defer detector.deinit();

    // Analyze struct dependencies
    detector.analyzeStruct(analyzer, struct_decl) catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Failed to analyze struct dependencies", struct_decl.span);
                return;
            },
        }
    };

    // Detect cycles
    var result = detector.detectCircularDependencies() catch |err| {
        switch (err) {
            error.OutOfMemory => return semantics_errors.SemanticError.OutOfMemory,
            else => {
                try semantics_errors.addErrorStatic(analyzer, "Failed to detect circular dependencies", struct_decl.span);
                return;
            },
        }
    };
    defer result.deinit(analyzer.allocator);

    // Report detected cycles
    for (result.cycles) |cycle| {
        switch (cycle.severity) {
            .Error => {
                try semantics_errors.addErrorStatic(analyzer, cycle.error_message, cycle.spans[0]);
            },
            .Warning => {
                try semantics_errors.addWarningStatic(analyzer, cycle.error_message, cycle.spans[0]);
            },
            .Info => {
                try semantics_errors.addInfo(analyzer, try analyzer.allocator.dupe(u8, cycle.error_message), cycle.spans[0]);
            },
        }
    }

    if (result.has_cycles) {
        return semantics_errors.SemanticError.CircularDependency;
    }
}
