const std = @import("std");
const source = @import("../source/mod.zig");
const syntax = @import("../syntax/mod.zig");
const support = @import("support.zig");
const ids = @import("ids.zig");
const nodes = @import("nodes.zig");

const SyntaxKind = syntax.SyntaxKind;
const SyntaxNode = syntax.SyntaxNode;
const SyntaxToken = syntax.SyntaxToken;
const SyntaxElement = syntax.SyntaxElement;
const ItemId = ids.ItemId;
const BodyId = ids.BodyId;
const StmtId = ids.StmtId;
const ExprId = ids.ExprId;
const TypeExprId = ids.TypeExprId;
const PatternId = ids.PatternId;
const Visibility = nodes.Visibility;
const BindingKind = nodes.BindingKind;
const StorageClass = nodes.StorageClass;
const SpecClauseKind = nodes.SpecClauseKind;
const UnaryOp = nodes.UnaryOp;
const BinaryOp = nodes.BinaryOp;
const AssignmentOp = nodes.AssignmentOp;
const TypeArg = nodes.TypeArg;
const Parameter = nodes.Parameter;
const StructField = nodes.StructField;
const BitfieldField = nodes.BitfieldField;
const EnumVariant = nodes.EnumVariant;
const SwitchPattern = nodes.SwitchPattern;
const CatchClause = nodes.CatchClause;
const TypeIntegerLiteral = nodes.TypeIntegerLiteral;
const stripQuotes = support.stripQuotes;

pub fn mixin(Builder: type) type {
    const Support = support.mixin(Builder);

    return struct {
        const Lowering = @This();

        pub fn lowerFileFromSyntax(self: *Builder) anyerror!void {
            const root = syntax.rootNode(self.tree);
            var it = root.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |node| {
                        const item_id = try Lowering.lowerTopLevelItemNode(self, node, null);
                        try self.root_items.append(self.allocator, item_id);
                    },
                }
            }
        }

        fn lowerTopLevelItemNode(self: *Builder, node: SyntaxNode, parent_contract: ?ItemId) anyerror!ItemId {
            return Lowering.lowerTopLevelItemNodeInner(self, node, parent_contract);
        }

        fn lowerTopLevelItemNodeInner(self: *Builder, node: SyntaxNode, parent_contract: ?ItemId) anyerror!ItemId {
            return switch (node.kind()) {
                .ContractItem => Lowering.lowerContractItemNode(self, node),
                .FunctionItem => Lowering.lowerFunctionItemNode(self, node, parent_contract),
                .StructItem => Lowering.lowerStructItemNode(self, node),
                .BitfieldItem => Lowering.lowerBitfieldItemNode(self, node),
                .EnumItem => Lowering.lowerEnumItemNode(self, node),
                .TypeAliasItem => Lowering.lowerTypeAliasItemNode(self, node),
                .ImportItem => Lowering.lowerImportItemNode(self, node),
                .GhostItem => Lowering.lowerGhostItemNode(self, node, parent_contract),
                .FieldItem => Lowering.lowerFieldItemNode(self, node),
                .ConstantItem => Lowering.lowerConstantItemNode(self, node),
                .LogDeclItem => Lowering.lowerLogDeclItemNode(self, node),
                .ErrorDeclItem => Lowering.lowerErrorDeclItemNode(self, node),
                .Error => Support.pushItem(self, .{ .Error = .{ .range = node.range() } }),
                else => Lowering.unsupportedItem(self, node),
            };
        }

        fn lowerContractItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const name = tokenText(firstDirectTokenOfKind(node, .Identifier) orelse return Lowering.malformedItem(self, node, "missing contract name"));
            const template_params_node = firstDirectChildOfKind(node, .ParameterList);
            const template_parameters = if (template_params_node) |params_node|
                try Lowering.lowerParameterListNode(self, params_node)
            else
                &.{};
            var members: std.ArrayList(ItemId) = .{};
            var invariants: std.ArrayList(ExprId) = .{};

            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |child_node| switch (child_node.kind()) {
                        .ContractInvariantItem => try invariants.append(self.allocator, try Lowering.lowerContractInvariantNode(self, child_node)),
                        .FunctionItem,
                        .StructItem,
                        .BitfieldItem,
                        .EnumItem,
                        .TypeAliasItem,
                        .ImportItem,
                        .GhostItem,
                        .FieldItem,
                        .ConstantItem,
                        .LogDeclItem,
                        .ErrorDeclItem,
                        => try members.append(self.allocator, try Lowering.lowerTopLevelItemNode(self, child_node, null)),
                        else => _ = try Lowering.unsupportedItem(self, child_node),
                    },
                }
            }

            const item_id = try Support.pushItem(self, .{ .Contract = .{
                .range = node.range(),
                .name = name,
                .is_generic = Lowering.hasGenericTemplateParameters(self, template_parameters),
                .template_parameters = template_parameters,
                .members = try members.toOwnedSlice(self.allocator),
                .invariants = try invariants.toOwnedSlice(self.allocator),
            } });

            for (self.items.items[item_id.index()].Contract.members) |member_id| {
                if (self.items.items[member_id.index()] == .Function) {
                    self.items.items[member_id.index()].Function.parent_contract = item_id;
                }
            }
            return item_id;
        }

        fn lowerGhostItemNode(self: *Builder, node: SyntaxNode, parent_contract: ?ItemId) !ItemId {
            const child_node = firstDirectNode(node) orelse return Lowering.malformedItem(self, node, "missing ghost target");
            return switch (child_node.kind()) {
                .FunctionItem, .FieldItem, .ConstantItem => blk: {
                    const item_id = try Lowering.lowerTopLevelItemNode(self, child_node, parent_contract);
                    Lowering.markItemGhost(self, item_id);
                    break :blk item_id;
                },
                .Body => Support.pushItem(self, .{ .GhostBlock = .{
                    .range = node.range(),
                    .body = try Lowering.lowerBodyNode(self, child_node),
                } }),
                else => Lowering.unsupportedItem(self, child_node),
            };
        }

        fn markItemGhost(self: *Builder, item_id: ItemId) void {
            switch (self.items.items[item_id.index()]) {
                .Function => |*function| function.is_ghost = true,
                .Field => |*field| field.is_ghost = true,
                .Constant => |*constant| constant.is_ghost = true,
                else => {},
            }
        }

        fn lowerFunctionItemNode(self: *Builder, node: SyntaxNode, parent_contract: ?ItemId) !ItemId {
            const name = tokenText(nthDirectIdentifierLikeToken(node, 0) orelse return Lowering.malformedItem(self, node, "missing function name"));
            const visibility: Visibility = if (firstDirectTokenOfKind(node, .Pub) != null) .public else .private;
            const params_node = firstDirectChildOfKind(node, .ParameterList) orelse return Lowering.malformedItem(self, node, "missing function parameter list");
            const parameters = try Lowering.lowerParameterListNode(self, params_node);
            const return_type = firstDirectTypeChild(node) orelse null;

            var clauses: std.ArrayList(nodes.SpecClause) = .{};
            var body: ?BodyId = null;

            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |child_node| switch (child_node.kind()) {
                        .SpecClause => try clauses.append(self.allocator, try Lowering.lowerSpecClauseNode(self, child_node)),
                        .Body => body = try Lowering.lowerBodyNode(self, child_node),
                        else => {},
                    },
                }
            }

            const body_id = body orelse return Lowering.malformedItem(self, node, "missing function body");
            var is_generic = false;
            for (parameters) |parameter| {
                if (parameter.is_comptime) {
                    is_generic = true;
                    break;
                }
            }
            return Support.pushItem(self, .{ .Function = .{
                .range = node.range(),
                .name = name,
                .is_generic = is_generic,
                .visibility = visibility,
                .parameters = parameters,
                .return_type = if (return_type) |type_node| try Lowering.lowerTypeNode(self, type_node) else null,
                .clauses = try clauses.toOwnedSlice(self.allocator),
                .body = body_id,
                .parent_contract = parent_contract,
            } });
        }

        fn lowerStructItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const name = tokenText(firstDirectTokenOfKind(node, .Identifier) orelse return Lowering.malformedItem(self, node, "missing struct name"));
            const template_params_node = firstDirectChildOfKind(node, .ParameterList);
            const template_parameters = if (template_params_node) |params_node|
                try Lowering.lowerParameterListNode(self, params_node)
            else
                &.{};
            var fields: std.ArrayList(StructField) = .{};

            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |field_node| {
                        if (field_node.kind() != .StructField) continue;
                        try fields.append(self.allocator, try Lowering.lowerStructFieldNode(self, field_node));
                    },
                }
            }

            return Support.pushItem(self, .{ .Struct = .{
                .range = node.range(),
                .name = name,
                .is_generic = Lowering.hasGenericTemplateParameters(self, template_parameters),
                .template_parameters = template_parameters,
                .fields = try fields.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerBitfieldItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const name = tokenText(firstDirectTokenOfKind(node, .Identifier) orelse return Lowering.malformedItem(self, node, "missing bitfield name"));
            const template_params_node = firstDirectChildOfKind(node, .ParameterList);
            const template_parameters = if (template_params_node) |params_node|
                try Lowering.lowerParameterListNode(self, params_node)
            else
                &.{};
            var fields: std.ArrayList(BitfieldField) = .{};
            const base_type = firstDirectTypeChild(node);

            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |field_node| {
                        if (field_node.kind() != .BitfieldField) continue;
                        try fields.append(self.allocator, try Lowering.lowerBitfieldFieldNode(self, field_node));
                    },
                }
            }

            return Support.pushItem(self, .{ .Bitfield = .{
                .range = node.range(),
                .name = name,
                .is_generic = Lowering.hasGenericTemplateParameters(self, template_parameters),
                .template_parameters = template_parameters,
                .base_type = if (base_type) |type_node| try Lowering.lowerTypeNode(self, type_node) else null,
                .fields = try fields.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerEnumItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const name = tokenText(firstDirectTokenOfKind(node, .Identifier) orelse return Lowering.malformedItem(self, node, "missing enum name"));
            const template_params_node = firstDirectChildOfKind(node, .ParameterList);
            const template_parameters = if (template_params_node) |params_node|
                try Lowering.lowerParameterListNode(self, params_node)
            else
                &.{};
            var variants: std.ArrayList(EnumVariant) = .{};

            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |variant_node| {
                        if (variant_node.kind() != .EnumVariant) continue;
                        const token = firstDirectTokenOfKind(variant_node, .Identifier) orelse blk: {
                            _ = try Lowering.malformedItem(self, variant_node, "missing enum variant name");
                            break :blk null;
                        };
                        try variants.append(self.allocator, .{
                            .range = variant_node.range(),
                            .name = if (token) |name_token| tokenText(name_token) else "",
                        });
                    },
                }
            }

            return Support.pushItem(self, .{ .Enum = .{
                .range = node.range(),
                .name = name,
                .is_generic = Lowering.hasGenericTemplateParameters(self, template_parameters),
                .template_parameters = template_parameters,
                .variants = try variants.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerTypeAliasItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const name = tokenText(nthDirectIdentifierLikeToken(node, 1) orelse return Lowering.malformedItem(self, node, "missing type alias name"));
            const template_params_node = firstDirectChildOfKind(node, .ParameterList);
            const template_parameters = if (template_params_node) |params_node|
                try Lowering.lowerParameterListNode(self, params_node)
            else
                &.{};
            const target_node = firstDirectTypeChild(node) orelse return Lowering.malformedItem(self, node, "missing type alias target");

            return Support.pushItem(self, .{ .TypeAlias = .{
                .range = node.range(),
                .name = name,
                .is_generic = Lowering.hasGenericTemplateParameters(self, template_parameters),
                .template_parameters = template_parameters,
                .target_type = try Lowering.lowerTypeNode(self, target_node),
            } });
        }

        fn lowerStructFieldNode(self: *Builder, node: SyntaxNode) !StructField {
            const name = if (firstDirectTokenOfKind(node, .Identifier)) |name_token|
                tokenText(name_token)
            else blk: {
                _ = try Lowering.malformedItem(self, node, "missing struct field name");
                break :blk "";
            };
            const type_expr = if (firstDirectTypeChild(node)) |type_node|
                try Lowering.lowerTypeNode(self, type_node)
            else
                try Lowering.malformedType(self, node, "missing struct field type");
            return .{
                .range = node.range(),
                .name = name,
                .type_expr = type_expr,
            };
        }

        fn lowerBitfieldFieldNode(self: *Builder, node: SyntaxNode) !BitfieldField {
            const name = if (firstDirectTokenOfKind(node, .Identifier)) |name_token|
                tokenText(name_token)
            else blk: {
                _ = try Lowering.malformedItem(self, node, "missing bitfield field name");
                break :blk "";
            };
            const type_expr = if (firstDirectTypeChild(node)) |type_node|
                try Lowering.lowerTypeNode(self, type_node)
            else
                try Lowering.malformedType(self, node, "missing bitfield field type");
            return .{
                .range = node.range(),
                .name = name,
                .type_expr = type_expr,
                .offset = parseBitfieldOffset(node),
                .width = parseBitfieldWidth(node),
            };
        }

        fn lowerParameterListNode(self: *Builder, node: SyntaxNode) ![]Parameter {
            var params: std.ArrayList(Parameter) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |param_node| {
                        if (param_node.kind() != .Parameter) continue;
                        try params.append(self.allocator, try Lowering.lowerParameterNode(self, param_node));
                    },
                }
            }
            return params.toOwnedSlice(self.allocator);
        }

        fn lowerParameterNode(self: *Builder, node: SyntaxNode) !Parameter {
            const name_token = nthDirectIdentifierLikeToken(node, 0);
            const pattern = try Support.pushPattern(self, .{ .Name = .{
                .range = if (name_token) |token| token.range() else node.range(),
                .name = if (name_token) |token| tokenText(token) else "",
            } });
            if (name_token == null) _ = try Lowering.malformedItem(self, node, "missing parameter name");
            const type_expr = if (firstDirectTypeChild(node)) |type_node|
                try Lowering.lowerTypeNode(self, type_node)
            else
                try Lowering.malformedType(self, node, "missing parameter type");
            return .{
                .range = node.range(),
                .is_comptime = firstDirectTokenOfKind(node, .Comptime) != null,
                .pattern = pattern,
                .type_expr = type_expr,
            };
        }

        fn isGenericTypeParameter(self: *Builder, parameter: Parameter) bool {
            if (!parameter.is_comptime) return false;
            return switch (self.type_exprs.items[parameter.type_expr.index()]) {
                .Path => |path| std.mem.eql(u8, path.name, "type"),
                else => false,
            };
        }

        fn isGenericTemplateParameter(self: *Builder, parameter: Parameter) bool {
            _ = self;
            return parameter.is_comptime;
        }

        fn hasGenericTemplateParameters(self: *Builder, parameters: []const Parameter) bool {
            for (parameters) |parameter| {
                if (Lowering.isGenericTemplateParameter(self, parameter)) return true;
            }
            return false;
        }

        fn lowerBodyNode(self: *Builder, node: SyntaxNode) !BodyId {
            var statements: std.ArrayList(StmtId) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |stmt_node| try statements.append(self.allocator, try Lowering.lowerStatementNode(self, stmt_node)),
                }
            }
            return Support.pushBody(self, .{
                .range = node.range(),
                .statements = try statements.toOwnedSlice(self.allocator),
            });
        }

        fn lowerStatementNode(self: *Builder, node: SyntaxNode) anyerror!StmtId {
            return switch (node.kind()) {
                .BlockStmt => Lowering.lowerBlockStmtNode(self, node),
                .ReturnStmt => Lowering.lowerReturnStmtNode(self, node),
                .IfStmt => Lowering.lowerIfStmtNode(self, node),
                .WhileStmt => Lowering.lowerWhileStmtNode(self, node),
                .ForStmt => Lowering.lowerForStmtNode(self, node),
                .SwitchStmt => Lowering.lowerSwitchStmtNode(self, node),
                .TryStmt => Lowering.lowerTryStmtNode(self, node),
                .LogStmt => Lowering.lowerLogStmtNode(self, node),
                .LockStmt => Lowering.lowerLockStmtNode(self, node),
                .UnlockStmt => Lowering.lowerUnlockStmtNode(self, node),
                .AssertStmt => Lowering.lowerAssertStmtNode(self, node),
                .AssumeStmt => Lowering.lowerAssumeStmtNode(self, node),
                .HavocStmt => Lowering.lowerHavocStmtNode(self, node),
                .BreakStmt => Support.pushStmt(self, .{ .Break = .{ .range = node.range() } }),
                .ContinueStmt => Support.pushStmt(self, .{ .Continue = .{ .range = node.range() } }),
                .VariableDeclStmt => Lowering.lowerVariableDeclStmtNode(self, node),
                .DestructuringAssignStmt => Lowering.lowerDestructuringAssignStmtNode(self, node),
                .AssignStmt => Lowering.lowerAssignStmtNode(self, node),
                .CompoundAssignStmt => Lowering.lowerAssignStmtNode(self, node),
                .ExprStmt => Lowering.lowerExprStmtNode(self, node),
                .LabeledBlockStmt => Lowering.lowerLabeledBlockStmtNode(self, node),
                .Error => Support.pushStmt(self, .{ .Error = .{ .range = node.range() } }),
                else => Lowering.unsupportedStmt(self, node),
            };
        }

        fn lowerBlockStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const body_node = firstDirectChildOfKind(node, .Body) orelse return Lowering.malformedStmt(self, node, "missing block body");
            const body = try Lowering.lowerBodyNode(self, body_node);
            return Support.pushStmt(self, .{ .Block = .{
                .range = node.range(),
                .body = body,
            } });
        }

        fn lowerReturnStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const expr_node = firstDirectExprChild(node);
            return Support.pushStmt(self, .{ .Return = .{
                .range = node.range(),
                .value = if (expr_node) |value_node| try Lowering.lowerExpressionNode(self, value_node) else null,
            } });
        }

        fn lowerIfStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const condition_node = firstDirectExprChild(node) orelse return Lowering.malformedStmt(self, node, "missing if condition");
            const then_body_node = nthDirectChildOfKind(node, .Body, 0) orelse return Lowering.malformedStmt(self, node, "missing then body");
            const else_body = if (nthDirectChildOfKind(node, .Body, 1)) |else_body_node|
                try Lowering.lowerBodyNode(self, else_body_node)
            else if (firstDirectChildOfKind(node, .IfStmt)) |nested_if|
                try Lowering.wrapStmtAsBody(self, try Lowering.lowerIfStmtNode(self, nested_if), nested_if.range())
            else
                null;

            return Support.pushStmt(self, .{ .If = .{
                .range = node.range(),
                .condition = try Lowering.lowerExpressionNode(self, condition_node),
                .then_body = try Lowering.lowerBodyNode(self, then_body_node),
                .else_body = else_body,
            } });
        }

        fn lowerWhileStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const condition_node = firstDirectExprChild(node) orelse return Lowering.malformedStmt(self, node, "missing while condition");
            const body_node = firstDirectChildOfKind(node, .Body) orelse return Lowering.malformedStmt(self, node, "missing while body");
            const invariants = try Lowering.lowerInvariantClauses(self, node);
            return Support.pushStmt(self, .{ .While = .{
                .range = node.range(),
                .condition = try Lowering.lowerExpressionNode(self, condition_node),
                .invariants = invariants,
                .body = try Lowering.lowerBodyNode(self, body_node),
            } });
        }

        fn lowerForStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const iterable_node = firstDirectExprChild(node) orelse return Lowering.malformedStmt(self, node, "missing for iterable");
            const loop_vars = parseForBindings(node) orelse return Lowering.malformedStmt(self, node, "missing for bindings");
            const item_pattern = try Support.pushPattern(self, .{ .Name = .{
                .range = loop_vars.item.range(),
                .name = tokenText(loop_vars.item),
            } });
            const index_pattern = if (loop_vars.index) |index_token|
                try Support.pushPattern(self, .{ .Name = .{
                    .range = index_token.range(),
                    .name = tokenText(index_token),
                } })
            else
                null;
            const body_node = firstDirectChildOfKind(node, .Body) orelse return Lowering.malformedStmt(self, node, "missing for body");
            const invariants = try Lowering.lowerInvariantClauses(self, node);
            return Support.pushStmt(self, .{ .For = .{
                .range = node.range(),
                .iterable = try Lowering.lowerExpressionNode(self, iterable_node),
                .item_pattern = item_pattern,
                .index_pattern = index_pattern,
                .invariants = invariants,
                .body = try Lowering.lowerBodyNode(self, body_node),
            } });
        }

        fn lowerInvariantClauses(self: *Builder, node: SyntaxNode) ![]ExprId {
            var invariants: std.ArrayList(ExprId) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |child_node| {
                        if (child_node.kind() != .InvariantClause) continue;
                        if (firstDirectExprChild(child_node)) |expr_node| {
                            try invariants.append(self.allocator, try Lowering.lowerExpressionNode(self, expr_node));
                        } else {
                            try invariants.append(self.allocator, try Lowering.malformedExpr(self, child_node, "missing invariant expression"));
                        }
                    },
                }
            }
            return invariants.toOwnedSlice(self.allocator);
        }

        fn lowerSwitchStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const condition_node = firstDirectExprChild(node) orelse return Lowering.malformedStmt(self, node, "missing switch condition");
            var arms: std.ArrayList(nodes.SwitchArm) = .{};
            var else_body: ?BodyId = null;

            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |arm_node| {
                        if (arm_node.kind() != .SwitchArm) continue;
                        const lowered = try Lowering.lowerSwitchArmNode(self, arm_node);
                        switch (lowered.pattern) {
                            .Else => else_body = lowered.body,
                            else => try arms.append(self.allocator, lowered),
                        }
                    },
                }
            }

            return Support.pushStmt(self, .{ .Switch = .{
                .range = node.range(),
                .condition = try Lowering.lowerExpressionNode(self, condition_node),
                .arms = try arms.toOwnedSlice(self.allocator),
                .else_body = else_body,
            } });
        }

        fn lowerSwitchArmNode(self: *Builder, node: SyntaxNode) !nodes.SwitchArm {
            const pattern = if (firstDirectNode(node)) |pattern_node|
                try Lowering.lowerSwitchPatternNode(self, pattern_node)
            else
                nodes.SwitchPattern{ .Expr = try Lowering.malformedExpr(self, node, "missing switch arm pattern") };
            const body = if (firstDirectChildOfKind(node, .Body)) |body_node|
                try Lowering.lowerBodyNode(self, body_node)
            else if (firstDirectChildOfKind(node, .ExprStmt)) |expr_stmt_node|
                try Lowering.wrapStmtAsBody(self, try Lowering.lowerExprStmtNode(self, expr_stmt_node), expr_stmt_node.range())
            else blk: {
                const error_stmt = try Lowering.malformedStmt(self, node, "missing switch arm body");
                break :blk try Lowering.wrapStmtAsBody(self, error_stmt, node.range());
            };

            return .{
                .range = node.range(),
                .pattern = pattern,
                .body = body,
            };
        }

        fn lowerSwitchPatternNode(self: *Builder, node: SyntaxNode) !SwitchPattern {
            return switch (node.kind()) {
                .RangeExpr => .{ .Range = try Lowering.lowerRangePatternNode(self, node) },
                .ErrorExpr => blk: {
                    const token = firstToken(node) orelse break :blk .{ .Expr = try Lowering.malformedExpr(self, node, "missing switch else token") };
                    if (token.kind() != .Else) break :blk .{ .Expr = try Lowering.malformedExpr(self, node, "invalid switch else pattern") };
                    break :blk .{ .Else = token.range() };
                },
                else => .{ .Expr = try Lowering.lowerExpressionNode(self, node) },
            };
        }

        fn lowerRangePatternNode(self: *Builder, node: SyntaxNode) !nodes.RangeSwitchPattern {
            const start = if (nthDirectNode(node, 0)) |start_node|
                try Lowering.lowerExpressionNode(self, start_node)
            else
                try Lowering.malformedExpr(self, node, "missing range start");
            const end = if (nthDirectNode(node, 1)) |end_node|
                try Lowering.lowerExpressionNode(self, end_node)
            else
                try Lowering.malformedExpr(self, node, "missing range end");
            const op_token = nthDirectToken(node, 0);
            if (op_token == null) _ = try Lowering.malformedExpr(self, node, "missing range operator");
            return .{
                .range = node.range(),
                .start = start,
                .end = end,
                .inclusive = if (op_token) |token| token.kind() == .DotDotDot else true,
            };
        }

        fn lowerTryStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const try_body_node = firstDirectChildOfKind(node, .Body) orelse return Lowering.malformedStmt(self, node, "missing try body");
            const catch_clause = if (firstDirectChildOfKind(node, .CatchClause)) |catch_node|
                try Lowering.lowerCatchClauseNode(self, catch_node)
            else
                null;
            return Support.pushStmt(self, .{ .Try = .{
                .range = node.range(),
                .try_body = try Lowering.lowerBodyNode(self, try_body_node),
                .catch_clause = catch_clause,
            } });
        }

        fn lowerCatchClauseNode(self: *Builder, node: SyntaxNode) !CatchClause {
            var error_pattern: ?PatternId = null;
            if (firstDirectChildOfKind(node, .GroupParen)) |group_node| {
                if (firstTokenOfKind(group_node, .Identifier)) |name_token| {
                    error_pattern = try Support.pushPattern(self, .{ .Name = .{
                        .range = name_token.range(),
                        .name = tokenText(name_token),
                    } });
                } else {
                    _ = try Lowering.malformedStmt(self, node, "missing catch binding");
                    error_pattern = try Support.pushPattern(self, .{ .Error = .{ .range = group_node.range() } });
                }
            }
            const body = if (firstDirectChildOfKind(node, .Body)) |body_node|
                try Lowering.lowerBodyNode(self, body_node)
            else blk: {
                const error_stmt = try Lowering.malformedStmt(self, node, "missing catch body");
                break :blk try Lowering.wrapStmtAsBody(self, error_stmt, node.range());
            };
            return .{
                .range = node.range(),
                .error_pattern = error_pattern,
                .body = body,
            };
        }

        fn lowerLogStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const name_token = nthDirectIdentifierLikeToken(node, 0) orelse return Lowering.malformedStmt(self, node, "missing log name");
            var args: std.ArrayList(ExprId) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |arg_node| if (isExprKind(arg_node.kind())) {
                        try args.append(self.allocator, try Lowering.lowerExpressionNode(self, arg_node));
                    },
                }
            }
            return Support.pushStmt(self, .{ .Log = .{
                .range = node.range(),
                .name = tokenText(name_token),
                .args = try args.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerLockStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const path = if (firstDirectExprChild(node)) |expr_node|
                try Lowering.lowerExpressionNode(self, expr_node)
            else
                try Lowering.malformedExpr(self, node, "missing @lock path");
            return Support.pushStmt(self, .{ .Lock = .{
                .range = node.range(),
                .path = path,
            } });
        }

        fn lowerUnlockStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const path = if (firstDirectExprChild(node)) |expr_node|
                try Lowering.lowerExpressionNode(self, expr_node)
            else
                try Lowering.malformedExpr(self, node, "missing @unlock path");
            return Support.pushStmt(self, .{ .Unlock = .{
                .range = node.range(),
                .path = path,
            } });
        }

        fn lowerAssertStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const condition_node = nthDirectExprChild(node, 0) orelse return Lowering.malformedStmt(self, node, "missing assert condition");
            const message = if (nthDirectExprChild(node, 1)) |message_node|
                try Lowering.lowerAssertMessage(self, message_node)
            else
                null;
            return Support.pushStmt(self, .{ .Assert = .{
                .range = node.range(),
                .condition = try Lowering.lowerExpressionNode(self, condition_node),
                .message = message,
            } });
        }

        fn lowerAssumeStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const condition_node = firstDirectExprChild(node) orelse return Lowering.malformedStmt(self, node, "missing assume condition");
            return Support.pushStmt(self, .{ .Assume = .{
                .range = node.range(),
                .condition = try Lowering.lowerExpressionNode(self, condition_node),
            } });
        }

        fn lowerHavocStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const name_token = nthDirectIdentifierLikeToken(node, 0) orelse return Lowering.malformedStmt(self, node, "missing havoc target");
            return Support.pushStmt(self, .{ .Havoc = .{
                .range = node.range(),
                .name = tokenText(name_token),
            } });
        }

        fn lowerVariableDeclStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const name_token = lastDirectIdentifierLikeToken(node) orelse return Lowering.malformedStmt(self, node, "missing variable name");
            const pattern = try Support.pushPattern(self, .{ .Name = .{
                .range = name_token.range(),
                .name = tokenText(name_token),
            } });
            const type_node = firstDirectTypeChild(node);
            const value_node = lastDirectExprChild(node);
            return Support.pushStmt(self, .{ .VariableDecl = .{
                .range = node.range(),
                .pattern = pattern,
                .binding_kind = parseBindingKind(node),
                .storage_class = parseStorageClass(node),
                .type_expr = if (type_node) |lowered_type| try Lowering.lowerTypeNode(self, lowered_type) else null,
                .value = if (value_node) |value_expr| try Lowering.lowerExpressionNode(self, value_expr) else null,
            } });
        }

        fn lowerDestructuringAssignStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const pattern_node = firstDirectChildOfKind(node, .DestructuringPattern) orelse return Lowering.malformedStmt(self, node, "missing destructuring pattern");
            const pattern = try Lowering.lowerDestructuringPatternNode(self, pattern_node);
            const value = if (firstDirectExprChild(node)) |value_node|
                try Lowering.lowerExpressionNode(self, value_node)
            else
                try Lowering.malformedExpr(self, node, "missing destructuring value");
            return Support.pushStmt(self, .{ .VariableDecl = .{
                .range = node.range(),
                .pattern = pattern,
                .binding_kind = parseBindingKind(node),
                .storage_class = parseStorageClass(node),
                .type_expr = null,
                .value = value,
            } });
        }

        fn lowerDestructuringPatternNode(self: *Builder, node: SyntaxNode) !PatternId {
            var fields: std.ArrayList(nodes.StructDestructureField) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |field_node| {
                        if (field_node.kind() != .DestructuringField) continue;
                        try fields.append(self.allocator, try Lowering.lowerDestructuringFieldNode(self, field_node));
                    },
                }
            }
            return Support.pushPattern(self, .{ .StructDestructure = .{
                .range = node.range(),
                .fields = try fields.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerDestructuringFieldNode(self: *Builder, node: SyntaxNode) !nodes.StructDestructureField {
            const field_token = nthDirectIdentifierLikeToken(node, 0);
            if (field_token == null) _ = try Lowering.malformedStmt(self, node, "missing destructuring field name");
            const binding_token = nthDirectIdentifierLikeToken(node, 1) orelse field_token;
            if (binding_token == null) _ = try Lowering.malformedStmt(self, node, "missing destructuring binding name");
            const binding = try Support.pushPattern(self, .{ .Name = .{
                .range = if (binding_token) |token| token.range() else node.range(),
                .name = if (binding_token) |token| tokenText(token) else "",
            } });
            return .{
                .range = node.range(),
                .name = if (field_token) |token| tokenText(token) else "",
                .binding = binding,
            };
        }

        fn lowerAssignStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const target_node = nthDirectExprChild(node, 0) orelse return Lowering.malformedStmt(self, node, "missing assignment target");
            const value_node = nthDirectExprChild(node, 1) orelse return Lowering.malformedStmt(self, node, "missing assignment value");
            const op_token = firstDirectAssignmentToken(node) orelse return Lowering.malformedStmt(self, node, "missing assignment operator");
            const target_expr = try Lowering.lowerExpressionNode(self, target_node);
            return Support.pushStmt(self, .{ .Assign = .{
                .range = node.range(),
                .op = mapAssignmentOp(op_token.kind()) orelse return Lowering.malformedStmt(self, node, "invalid assignment operator"),
                .target = try Lowering.patternFromExpr(self, target_expr),
                .value = try Lowering.lowerExpressionNode(self, value_node),
            } });
        }

        fn patternFromExpr(self: *Builder, expr_id: ExprId) !PatternId {
            return switch (Support.exprRef(self, expr_id).*) {
                .Name => |name| Support.pushPattern(self, .{ .Name = .{
                    .range = name.range,
                    .name = name.name,
                } }),
                .Field => |field| Support.pushPattern(self, .{ .Field = .{
                    .range = field.range,
                    .base = try Lowering.patternFromExpr(self, field.base),
                    .name = field.name,
                } }),
                .Index => |index| Support.pushPattern(self, .{ .Index = .{
                    .range = index.range,
                    .base = try Lowering.patternFromExpr(self, index.base),
                    .index = index.index,
                } }),
                .Group => |group| Lowering.patternFromExpr(self, group.expr),
                else => blk: {
                    try self.diagnostics.appendError("invalid assignment target", .{
                        .file_id = self.file.file_id,
                        .range = Support.exprRange(self, expr_id),
                    });
                    break :blk Support.pushPattern(self, .{ .Error = .{ .range = Support.exprRange(self, expr_id) } });
                },
            };
        }

        fn lowerExprStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const expr_node = firstDirectExprChild(node) orelse return Lowering.malformedStmt(self, node, "missing expression");
            return Support.pushStmt(self, .{ .Expr = .{
                .range = node.range(),
                .expr = try Lowering.lowerExpressionNode(self, expr_node),
            } });
        }

        fn lowerLabeledBlockStmtNode(self: *Builder, node: SyntaxNode) !StmtId {
            const label_token = firstDirectTokenOfKind(node, .Identifier) orelse return Lowering.malformedStmt(self, node, "missing label name");
            const body_node = firstDirectChildOfKind(node, .Body) orelse return Lowering.malformedStmt(self, node, "missing labeled block body");
            return Support.pushStmt(self, .{ .LabeledBlock = .{
                .range = node.range(),
                .label = tokenText(label_token),
                .body = try Lowering.lowerBodyNode(self, body_node),
            } });
        }

        fn lowerExpressionNode(self: *Builder, node: SyntaxNode) anyerror!ExprId {
            return switch (node.kind()) {
                .Literal => Lowering.lowerLiteralExprNode(self, node),
                .NameExpr => Lowering.lowerNameExprNode(self, node),
                .PathType, .GenericType, .TupleType, .ArrayType, .SliceType, .ErrorUnionType => Lowering.lowerTypeValueExprNode(self, node),
                .UnaryExpr => Lowering.lowerUnaryExprNode(self, node),
                .BinaryExpr => Lowering.lowerBinaryExprNode(self, node),
                .CallExpr => Lowering.lowerCallExprNode(self, node),
                .FieldExpr => Lowering.lowerFieldExprNode(self, node),
                .IndexExpr => Lowering.lowerIndexExprNode(self, node),
                .GroupExpr => Lowering.lowerGroupExprNode(self, node),
                .TupleExpr => Lowering.lowerTupleExprNode(self, node),
                .ArrayLiteral => Lowering.lowerArrayLiteralExprNode(self, node),
                .StructLiteral => Lowering.lowerStructLiteralExprNode(self, node),
                .SwitchExpr => Lowering.lowerSwitchExprNode(self, node),
                .QuantifiedExpr => Lowering.lowerQuantifiedExprNode(self, node),
                .OldExpr => Lowering.lowerOldExprNode(self, node),
                .ComptimeExpr => Lowering.lowerComptimeExprNode(self, node),
                .BuiltinExpr => Lowering.lowerBuiltinExprNode(self, node),
                .ErrorExpr => Support.pushExpr(self, .{ .Error = .{ .range = node.range() } }),
                else => Lowering.unsupportedExpr(self, node),
            };
        }

        fn lowerLiteralExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const token = firstToken(node) orelse return Lowering.malformedExpr(self, node, "missing literal token");
            return switch (token.kind()) {
                .IntegerLiteral, .BinaryLiteral, .HexLiteral => Support.pushExpr(self, .{ .IntegerLiteral = .{
                    .range = node.range(),
                    .text = tokenText(token),
                } }),
                .StringLiteral, .RawStringLiteral, .CharacterLiteral => Support.pushExpr(self, .{ .StringLiteral = .{
                    .range = node.range(),
                    .text = stripQuotes(tokenText(token)),
                } }),
                .True, .False => Support.pushExpr(self, .{ .BoolLiteral = .{
                    .range = node.range(),
                    .value = token.kind() == .True,
                } }),
                .AddressLiteral => Support.pushExpr(self, .{ .AddressLiteral = .{
                    .range = node.range(),
                    .text = tokenText(token),
                } }),
                .BytesLiteral => Support.pushExpr(self, .{ .BytesLiteral = .{
                    .range = node.range(),
                    .text = stripBytesLiteral(tokenText(token)),
                } }),
                else => Lowering.malformedExpr(self, node, "unsupported literal token"),
            };
        }

        fn lowerTypeValueExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            return Support.pushExpr(self, .{ .TypeValue = .{
                .range = node.range(),
                .type_expr = try Lowering.lowerTypeNode(self, node),
            } });
        }

        fn lowerNameExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const token = firstToken(node) orelse return Lowering.malformedExpr(self, node, "missing name token");
            if (token.kind() == .Result) {
                return Support.pushExpr(self, .{ .Result = .{ .range = node.range() } });
            }
            return Support.pushExpr(self, .{ .Name = .{
                .range = node.range(),
                .name = tokenText(token),
            } });
        }

        fn lowerUnaryExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const op_token = firstDirectToken(node) orelse return Lowering.malformedExpr(self, node, "missing unary operator");
            const operand_node = firstDirectNode(node) orelse return Lowering.malformedExpr(self, node, "missing unary operand");
            const operand = try Lowering.lowerExpressionNode(self, operand_node);
            const op: UnaryOp = switch (op_token.kind()) {
                .Minus => .neg,
                .Bang => .not_,
                .Tilde => .bit_not,
                .Try => .try_,
                .Plus => return operand,
                else => return Lowering.malformedExpr(self, node, "invalid unary operator"),
            };
            return Support.pushExpr(self, .{ .Unary = .{
                .range = node.range(),
                .op = op,
                .operand = operand,
            } });
        }

        fn lowerBinaryExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const lhs_node = nthDirectNode(node, 0) orelse return Lowering.malformedExpr(self, node, "missing binary lhs");
            const rhs_node = nthDirectNode(node, 1) orelse return Lowering.malformedExpr(self, node, "missing binary rhs");
            const op_token = firstDirectToken(node) orelse return Lowering.malformedExpr(self, node, "missing binary operator");
            return Support.pushExpr(self, .{ .Binary = .{
                .range = node.range(),
                .op = mapBinaryOp(op_token.kind()) orelse return Lowering.malformedExpr(self, node, "invalid binary operator"),
                .lhs = try Lowering.lowerExpressionNode(self, lhs_node),
                .rhs = try Lowering.lowerExpressionNode(self, rhs_node),
            } });
        }

        fn lowerCallExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const callee_node = nthDirectNode(node, 0) orelse return Lowering.malformedExpr(self, node, "missing call callee");
            var args: std.ArrayList(ExprId) = .{};
            var ordinal: usize = 1;
            while (nthDirectNode(node, ordinal)) |arg_node| : (ordinal += 1) {
                try args.append(self.allocator, try Lowering.lowerExpressionNode(self, arg_node));
            }

            const callee = try Lowering.lowerExpressionNode(self, callee_node);
            if (try Lowering.maybeLowerErrorReturn(self, callee, node, args.items)) |error_expr| return error_expr;

            return Support.pushExpr(self, .{ .Call = .{
                .range = node.range(),
                .callee = callee,
                .args = try args.toOwnedSlice(self.allocator),
            } });
        }

        fn maybeLowerErrorReturn(self: *Builder, callee: ExprId, node: SyntaxNode, args: []const ExprId) !?ExprId {
            const callee_expr = Support.exprRef(self, callee);
            if (callee_expr.* != .Field) return null;
            const field = callee_expr.Field;
            const base_expr = Support.exprRef(self, field.base);
            if (base_expr.* != .Name) return null;
            if (!std.mem.eql(u8, base_expr.Name.name, "error")) return null;
            return try Support.pushExpr(self, .{ .ErrorReturn = .{
                .range = node.range(),
                .name = field.name,
                .args = try self.allocator.dupe(ExprId, args),
            } });
        }

        fn lowerFieldExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const base_node = nthDirectNode(node, 0) orelse return Lowering.malformedExpr(self, node, "missing field base");
            const name_token = nthDirectIdentifierLikeToken(node, 0) orelse return Lowering.malformedExpr(self, node, "missing field name");
            return Support.pushExpr(self, .{ .Field = .{
                .range = node.range(),
                .base = try Lowering.lowerExpressionNode(self, base_node),
                .name = tokenText(name_token),
            } });
        }

        fn lowerIndexExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const base_node = nthDirectNode(node, 0) orelse return Lowering.malformedExpr(self, node, "missing index base");
            const index_node = nthDirectNode(node, 1) orelse return Lowering.malformedExpr(self, node, "missing index expression");
            return Support.pushExpr(self, .{ .Index = .{
                .range = node.range(),
                .base = try Lowering.lowerExpressionNode(self, base_node),
                .index = try Lowering.lowerExpressionNode(self, index_node),
            } });
        }

        fn lowerGroupExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const inner = firstDirectNode(node) orelse return Lowering.malformedExpr(self, node, "missing grouped expression");
            return Support.pushExpr(self, .{ .Group = .{
                .range = node.range(),
                .expr = try Lowering.lowerExpressionNode(self, inner),
            } });
        }

        fn lowerTupleExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            var elements: std.ArrayList(ExprId) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |element_node| if (isExprKind(element_node.kind())) {
                        try elements.append(self.allocator, try Lowering.lowerExpressionNode(self, element_node));
                    },
                }
            }
            return Support.pushExpr(self, .{ .Tuple = .{
                .range = node.range(),
                .elements = try elements.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerArrayLiteralExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            var elements: std.ArrayList(ExprId) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |element_node| if (isExprKind(element_node.kind())) {
                        try elements.append(self.allocator, try Lowering.lowerExpressionNode(self, element_node));
                    },
                }
            }
            return Support.pushExpr(self, .{ .ArrayLiteral = .{
                .range = node.range(),
                .elements = try elements.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerStructLiteralExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const base_node = nthDirectNode(node, 0) orelse return Lowering.malformedExpr(self, node, "missing struct literal base");
            const base_expr = try Lowering.lowerExpressionNode(self, base_node);
            const type_name = Lowering.structLiteralTypeName(self, base_expr) orelse return Lowering.malformedExpr(self, node, "invalid struct literal type");

            var fields: std.ArrayList(nodes.StructFieldInit) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |field_node| {
                        if (field_node.kind() != .AnonymousStructLiteralField) continue;
                        try fields.append(self.allocator, try Lowering.lowerStructLiteralFieldNode(self, field_node));
                    },
                }
            }

            return Support.pushExpr(self, .{ .StructLiteral = .{
                .range = node.range(),
                .type_name = type_name,
                .fields = try fields.toOwnedSlice(self.allocator),
            } });
        }

        fn structLiteralTypeName(self: *Builder, expr_id: ExprId) ?[]const u8 {
            return switch (Support.exprRef(self, expr_id).*) {
                .Name => |name| name.name,
                .Group => |group| Lowering.structLiteralTypeName(self, group.expr),
                else => null,
            };
        }

        fn lowerStructLiteralFieldNode(self: *Builder, node: SyntaxNode) !nodes.StructFieldInit {
            const name_token = firstDirectTokenOfKind(node, .Identifier) orelse blk: {
                _ = try Lowering.malformedExpr(self, node, "missing struct literal field name");
                break :blk null;
            };
            const value = if (firstDirectExprChild(node)) |value_node|
                try Lowering.lowerExpressionNode(self, value_node)
            else
                try Lowering.malformedExpr(self, node, "missing struct literal field value");
            return .{
                .range = node.range(),
                .name = if (name_token) |token| tokenText(token) else "",
                .value = value,
            };
        }

        fn lowerOldExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const expr_node = firstDirectExprChild(node) orelse return Lowering.malformedExpr(self, node, "missing old() expression");
            return Support.pushExpr(self, .{ .Old = .{
                .range = node.range(),
                .expr = try Lowering.lowerExpressionNode(self, expr_node),
            } });
        }

        fn lowerComptimeExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const body_node = firstDirectChildOfKind(node, .Body) orelse return Lowering.malformedExpr(self, node, "missing comptime body");
            return Support.pushExpr(self, .{ .Comptime = .{
                .range = node.range(),
                .body = try Lowering.lowerBodyNode(self, body_node),
            } });
        }

        fn lowerSwitchExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const condition_node = firstDirectExprChild(node) orelse return Lowering.malformedExpr(self, node, "missing switch expression condition");
            var arms: std.ArrayList(nodes.SwitchExprArm) = .{};
            var else_expr: ?ExprId = null;

            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |arm_node| {
                        if (arm_node.kind() != .SwitchExprArm) continue;
                        const lowered = try Lowering.lowerSwitchExprArmNode(self, arm_node);
                        switch (lowered.pattern) {
                            .Else => else_expr = lowered.value,
                            else => try arms.append(self.allocator, lowered),
                        }
                    },
                }
            }

            return Support.pushExpr(self, .{ .Switch = .{
                .range = node.range(),
                .condition = try Lowering.lowerExpressionNode(self, condition_node),
                .arms = try arms.toOwnedSlice(self.allocator),
                .else_expr = else_expr,
            } });
        }

        fn lowerSwitchExprArmNode(self: *Builder, node: SyntaxNode) !nodes.SwitchExprArm {
            const pattern = if (nthDirectNode(node, 0)) |pattern_node|
                try Lowering.lowerSwitchPatternNode(self, pattern_node)
            else
                nodes.SwitchPattern{ .Expr = try Lowering.malformedExpr(self, node, "missing switch expression pattern") };
            const value = if (nthDirectNode(node, 1)) |value_node|
                try Lowering.lowerExpressionNode(self, value_node)
            else
                try Lowering.malformedExpr(self, node, "missing switch expression value");
            return .{
                .range = node.range(),
                .pattern = pattern,
                .value = value,
            };
        }

        fn lowerQuantifiedExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const quant_token = firstDirectToken(node) orelse return Lowering.malformedExpr(self, node, "missing quantifier");
            const pattern_token = nthDirectIdentifierLikeToken(node, 0) orelse return Lowering.malformedExpr(self, node, "missing quantified binding");
            const type_node = firstDirectTypeChild(node) orelse return Lowering.malformedExpr(self, node, "missing quantified type");
            const body_node = lastDirectExprChild(node) orelse return Lowering.malformedExpr(self, node, "missing quantified body");
            const condition_node = if (firstDirectTokenOfKind(node, .Where) != null) nthDirectExprChild(node, 0) else null;
            const pattern = try Support.pushPattern(self, .{ .Name = .{
                .range = pattern_token.range(),
                .name = tokenText(pattern_token),
            } });

            return Support.pushExpr(self, .{ .Quantified = .{
                .range = node.range(),
                .quantifier = switch (quant_token.kind()) {
                    .Forall => .forall,
                    .Exists => .exists,
                    else => return Lowering.malformedExpr(self, node, "invalid quantifier"),
                },
                .pattern = pattern,
                .type_expr = try Lowering.lowerTypeNode(self, type_node),
                .condition = if (condition_node) |cond| try Lowering.lowerExpressionNode(self, cond) else null,
                .body = try Lowering.lowerExpressionNode(self, body_node),
            } });
        }

        fn lowerBuiltinExprNode(self: *Builder, node: SyntaxNode) !ExprId {
            const name_token = nthDirectIdentifierLikeToken(node, 0) orelse return Lowering.malformedExpr(self, node, "missing builtin name");
            const name = tokenText(name_token);
            const type_node = firstDirectTypeChild(node);
            var args: std.ArrayList(ExprId) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |arg_node| {
                        if (!isExprKind(arg_node.kind())) continue;
                        try args.append(self.allocator, try Lowering.lowerExpressionNode(self, arg_node));
                    },
                }
            }
            return Support.pushExpr(self, .{ .Builtin = .{
                .range = node.range(),
                .name = name,
                .type_arg = if (type_node) |arg_type| try Lowering.lowerTypeNode(self, arg_type) else null,
                .args = try args.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerTypeNode(self: *Builder, node: SyntaxNode) anyerror!TypeExprId {
            return switch (node.kind()) {
                .PathType => Lowering.lowerPathTypeNode(self, node),
                .GenericType => Lowering.lowerGenericTypeNode(self, node),
                .TupleType => Lowering.lowerTupleTypeNode(self, node),
                .ArrayType => Lowering.lowerArrayTypeNode(self, node),
                .SliceType => Lowering.lowerSliceTypeNode(self, node),
                .ErrorUnionType => Lowering.lowerErrorUnionTypeNode(self, node),
                .Error => Support.pushTypeExpr(self, .{ .Error = .{ .range = node.range() } }),
                else => Lowering.unsupportedType(self, node),
            };
        }

        fn lowerPathTypeNode(self: *Builder, node: SyntaxNode) !TypeExprId {
            const token = firstToken(node) orelse return Lowering.malformedType(self, node, "missing type name");
            return Support.pushTypeExpr(self, .{ .Path = .{
                .range = node.range(),
                .name = tokenText(token),
            } });
        }

        fn lowerGenericTypeNode(self: *Builder, node: SyntaxNode) !TypeExprId {
            const name_token = firstToken(node) orelse return Lowering.malformedType(self, node, "missing generic type name");
            var args: std.ArrayList(TypeArg) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => |token| switch (token.kind()) {
                        .IntegerLiteral, .BinaryLiteral, .HexLiteral => try args.append(self.allocator, .{ .Integer = .{
                            .range = token.range(),
                            .text = tokenText(token),
                        } }),
                        else => {},
                    },
                    .node => |arg_node| if (isTypeKind(arg_node.kind())) {
                        try args.append(self.allocator, .{ .Type = try Lowering.lowerTypeNode(self, arg_node) });
                    },
                }
            }
            return Support.pushTypeExpr(self, .{ .Generic = .{
                .range = node.range(),
                .name = tokenText(name_token),
                .args = try args.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerTupleTypeNode(self: *Builder, node: SyntaxNode) !TypeExprId {
            var elements: std.ArrayList(TypeExprId) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |element_node| if (isTypeKind(element_node.kind())) {
                        try elements.append(self.allocator, try Lowering.lowerTypeNode(self, element_node));
                    } else if (element_node.kind() == .AnonymousStructType) {
                        try elements.append(self.allocator, try Lowering.unsupportedType(self, element_node));
                    },
                }
            }
            if (elements.items.len == 1) return elements.items[0];
            return Support.pushTypeExpr(self, .{ .Tuple = .{
                .range = node.range(),
                .elements = try elements.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerArrayTypeNode(self: *Builder, node: SyntaxNode) !TypeExprId {
            const element_node = firstDirectTypeChild(node) orelse return Lowering.malformedType(self, node, "missing array element type");
            const size_token = firstDirectArraySizeToken(node) orelse return Lowering.malformedType(self, node, "missing array size");
            return Support.pushTypeExpr(self, .{ .Array = .{
                .range = node.range(),
                .element = try Lowering.lowerTypeNode(self, element_node),
                .size = switch (size_token.kind()) {
                    .IntegerLiteral, .BinaryLiteral, .HexLiteral => .{ .Integer = .{
                        .range = size_token.range(),
                        .text = tokenText(size_token),
                    } },
                    else => .{ .Name = .{
                        .range = size_token.range(),
                        .name = tokenText(size_token),
                    } },
                },
            } });
        }

        fn lowerSliceTypeNode(self: *Builder, node: SyntaxNode) !TypeExprId {
            const element_node = firstDirectTypeChild(node) orelse return Lowering.malformedType(self, node, "missing slice element type");
            return Support.pushTypeExpr(self, .{ .Slice = .{
                .range = node.range(),
                .element = try Lowering.lowerTypeNode(self, element_node),
            } });
        }

        fn lowerErrorUnionTypeNode(self: *Builder, node: SyntaxNode) !TypeExprId {
            const payload_node = nthDirectTypeChild(node, 0) orelse return Lowering.malformedType(self, node, "missing error union payload type");
            var errors: std.ArrayList(TypeExprId) = .{};
            var ordinal: usize = 1;
            while (nthDirectTypeChild(node, ordinal)) |error_node| : (ordinal += 1) {
                try errors.append(self.allocator, try Lowering.lowerTypeNode(self, error_node));
            }
            return Support.pushTypeExpr(self, .{ .ErrorUnion = .{
                .range = node.range(),
                .payload = try Lowering.lowerTypeNode(self, payload_node),
                .errors = try errors.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerAssertMessage(self: *Builder, node: SyntaxNode) !?[]const u8 {
            if (node.kind() == .Literal) {
                const token = firstToken(node) orelse {
                    _ = try Lowering.malformedExpr(self, node, "missing assert message literal");
                    return null;
                };
                return switch (token.kind()) {
                    .StringLiteral, .RawStringLiteral, .CharacterLiteral => stripQuotes(tokenText(token)),
                    else => blk: {
                        _ = try Lowering.malformedExpr(self, node, "assert message must be a string literal");
                        break :blk null;
                    },
                };
            }
            _ = try Lowering.malformedExpr(self, node, "assert message must be a string literal");
            return null;
        }

        fn lowerImportItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const path_token = firstDirectTokenOfKind(node, .StringLiteral) orelse return Lowering.malformedItem(self, node, "missing import path");
            const alias = if (firstDirectTokenOfKind(node, .Const) != null)
                tokenText(nthDirectIdentifierLikeToken(node, 0) orelse return Lowering.malformedItem(self, node, "missing import alias"))
            else
                null;
            return Support.pushItem(self, .{ .Import = .{
                .range = node.range(),
                .path = stripQuotes(tokenText(path_token)),
                .alias = alias,
                .is_comptime = firstDirectTokenOfKind(node, .Comptime) != null,
            } });
        }

        fn lowerFieldItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const name_token = lastDirectIdentifierLikeToken(node) orelse return Lowering.malformedItem(self, node, "missing field name");
            const type_node = firstDirectTypeChild(node);
            const value_node = firstDirectExprChild(node);
            return Support.pushItem(self, .{ .Field = .{
                .range = node.range(),
                .name = tokenText(name_token),
                .binding_kind = parseBindingKind(node),
                .storage_class = parseStorageClass(node),
                .type_expr = if (type_node) |lowered_type| try Lowering.lowerTypeNode(self, lowered_type) else null,
                .value = if (value_node) |value_expr| try Lowering.lowerExpressionNode(self, value_expr) else null,
            } });
        }

        fn lowerConstantItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const name_token = nthDirectIdentifierLikeToken(node, 0) orelse return Lowering.malformedItem(self, node, "missing constant name");
            const type_node = firstDirectTypeChild(node);
            const value_node = firstDirectExprChild(node) orelse return Lowering.malformedItem(self, node, "missing constant value");
            return Support.pushItem(self, .{ .Constant = .{
                .range = node.range(),
                .name = tokenText(name_token),
                .is_comptime = firstDirectTokenOfKind(node, .Comptime) != null,
                .type_expr = if (type_node) |lowered_type| try Lowering.lowerTypeNode(self, lowered_type) else null,
                .value = try Lowering.lowerExpressionNode(self, value_node),
            } });
        }

        fn lowerLogDeclItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const name_token = nthDirectIdentifierLikeToken(node, 0) orelse return Lowering.malformedItem(self, node, "missing log declaration name");
            var fields: std.ArrayList(nodes.LogField) = .{};
            var it = node.children();
            while (it.next()) |child| {
                switch (child) {
                    .token => {},
                    .node => |field_node| {
                        if (field_node.kind() != .LogField) continue;
                        try fields.append(self.allocator, try Lowering.lowerLogFieldNode(self, field_node));
                    },
                }
            }
            return Support.pushItem(self, .{ .LogDecl = .{
                .range = node.range(),
                .name = tokenText(name_token),
                .fields = try fields.toOwnedSlice(self.allocator),
            } });
        }

        fn lowerErrorDeclItemNode(self: *Builder, node: SyntaxNode) !ItemId {
            const name_token = firstDirectTokenOfKind(node, .Identifier) orelse return Lowering.malformedItem(self, node, "missing error declaration name");
            const params_node = firstDirectChildOfKind(node, .ParameterList);
            return Support.pushItem(self, .{ .ErrorDecl = .{
                .range = node.range(),
                .name = tokenText(name_token),
                .parameters = if (params_node) |list| try Lowering.lowerParameterListNode(self, list) else &.{},
            } });
        }

        fn lowerLogFieldNode(self: *Builder, node: SyntaxNode) !nodes.LogField {
            const indexed_token = nthDirectIdentifierLikeToken(node, 0);
            const indexed = if (indexed_token) |token| std.mem.eql(u8, tokenText(token), "indexed") else false;
            const name_token = nthDirectIdentifierLikeToken(node, if (indexed) 1 else 0);
            if (indexed_token == null) _ = try Lowering.malformedItem(self, node, "missing log field name");
            if (indexed and name_token == null) _ = try Lowering.malformedItem(self, node, "missing log field name");
            const type_expr = if (firstDirectTypeChild(node)) |type_node|
                try Lowering.lowerTypeNode(self, type_node)
            else
                try Lowering.malformedType(self, node, "missing log field type");
            return .{
                .range = node.range(),
                .name = if (name_token) |token| tokenText(token) else if (indexed_token) |token| tokenText(token) else "",
                .type_expr = type_expr,
                .indexed = indexed,
            };
        }

        fn lowerContractInvariantNode(self: *Builder, node: SyntaxNode) !ExprId {
            const expr_node = firstDirectExprChild(node) orelse return Lowering.malformedExpr(self, node, "missing contract invariant expression");
            return Lowering.lowerExpressionNode(self, expr_node);
        }

        fn lowerSpecClauseNode(self: *Builder, node: SyntaxNode) !nodes.SpecClause {
            const kind: nodes.SpecClauseKind = if (firstDirectToken(node)) |token| switch (token.kind()) {
                .Requires => .requires,
                .Ensures => .ensures,
                else => blk: {
                    _ = try Lowering.malformedExpr(self, node, "invalid specification keyword");
                    break :blk .requires;
                },
            } else blk: {
                _ = try Lowering.malformedExpr(self, node, "missing specification keyword");
                break :blk .requires;
            };
            const expr = if (firstDirectExprChild(node)) |expr_node|
                try Lowering.lowerExpressionNode(self, expr_node)
            else
                try Lowering.malformedExpr(self, node, "missing specification expression");
            return .{
                .range = node.range(),
                .kind = kind,
                .expr = expr,
            };
        }

        fn wrapStmtAsBody(self: *Builder, stmt: StmtId, range: source.TextRange) !BodyId {
            var statements: std.ArrayList(StmtId) = .{};
            try statements.append(self.allocator, stmt);
            return Support.pushBody(self, .{
                .range = range,
                .statements = try statements.toOwnedSlice(self.allocator),
            });
        }

        fn unsupportedItem(self: *Builder, node: SyntaxNode) !ItemId {
            try Lowering.recordUnsupportedSyntax(self, node, "item");
            return Support.pushItem(self, .{ .Error = .{ .range = node.range() } });
        }

        fn malformedItem(self: *Builder, node: SyntaxNode, detail: []const u8) !ItemId {
            try Lowering.recordMalformedSyntax(self, node, "item", detail);
            return Support.pushItem(self, .{ .Error = .{ .range = node.range() } });
        }

        fn malformedStmt(self: *Builder, node: SyntaxNode, detail: []const u8) !StmtId {
            try Lowering.recordMalformedSyntax(self, node, "statement", detail);
            return Support.pushStmt(self, .{ .Error = .{ .range = node.range() } });
        }

        fn unsupportedStmt(self: *Builder, node: SyntaxNode) !StmtId {
            try Lowering.recordUnsupportedSyntax(self, node, "statement");
            return Support.pushStmt(self, .{ .Error = .{ .range = node.range() } });
        }

        fn malformedExpr(self: *Builder, node: SyntaxNode, detail: []const u8) !ExprId {
            try Lowering.recordMalformedSyntax(self, node, "expression", detail);
            return Support.pushExpr(self, .{ .Error = .{ .range = node.range() } });
        }

        fn unsupportedExpr(self: *Builder, node: SyntaxNode) !ExprId {
            try Lowering.recordUnsupportedSyntax(self, node, "expression");
            return Support.pushExpr(self, .{ .Error = .{ .range = node.range() } });
        }

        fn malformedType(self: *Builder, node: SyntaxNode, detail: []const u8) !TypeExprId {
            try Lowering.recordMalformedSyntax(self, node, "type", detail);
            return Support.pushTypeExpr(self, .{ .Error = .{ .range = node.range() } });
        }

        fn unsupportedType(self: *Builder, node: SyntaxNode) !TypeExprId {
            try Lowering.recordUnsupportedSyntax(self, node, "type");
            return Support.pushTypeExpr(self, .{ .Error = .{ .range = node.range() } });
        }

        fn recordUnsupportedSyntax(self: *Builder, node: SyntaxNode, comptime category: []const u8) !void {
            var buffer: [160]u8 = undefined;
            const message = std.fmt.bufPrint(&buffer, "unsupported syntax lowering for {s} node {s}", .{
                category,
                @tagName(node.kind()),
            }) catch "unsupported syntax lowering";
            try self.diagnostics.appendError(message, .{
                .file_id = self.file.file_id,
                .range = node.range(),
            });
        }

        fn recordMalformedSyntax(self: *Builder, node: SyntaxNode, comptime category: []const u8, detail: []const u8) !void {
            var buffer: [192]u8 = undefined;
            const message = std.fmt.bufPrint(&buffer, "malformed {s}: {s}", .{
                @tagName(node.kind()),
                detail,
            }) catch "malformed syntax";
            try self.diagnostics.appendError(message, .{
                .file_id = self.file.file_id,
                .range = node.range(),
            });
            _ = category;
        }

    };
}

fn isExprKind(kind: SyntaxKind) bool {
    return switch (kind) {
        .BinaryExpr,
        .RangeExpr,
        .UnaryExpr,
        .CallExpr,
        .FieldExpr,
        .IndexExpr,
        .GroupExpr,
        .TupleExpr,
        .ArrayLiteral,
        .StructLiteral,
        .SwitchExpr,
        .QuantifiedExpr,
        .OldExpr,
        .BuiltinExpr,
        .ErrorReturnExpr,
        .Literal,
        .NameExpr,
        .TryExpr,
        .ErrorExpr,
        .ComptimeExpr,
        => true,
        else => false,
    };
}

fn isTypeKind(kind: SyntaxKind) bool {
    return switch (kind) {
        .PathType,
        .GenericType,
        .TupleType,
        .ArrayType,
        .SliceType,
        .ErrorUnionType,
        .AnonymousStructType,
        => true,
        else => false,
    };
}

fn firstDirectChildOfKind(node: SyntaxNode, kind: SyntaxKind) ?SyntaxNode {
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .token => {},
            .node => |child_node| if (child_node.kind() == kind) return child_node,
        }
    }
    return null;
}

fn nthDirectChildOfKind(node: SyntaxNode, kind: SyntaxKind, ordinal: usize) ?SyntaxNode {
    var remaining = ordinal;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .token => {},
            .node => |child_node| {
                if (child_node.kind() != kind) continue;
                if (remaining == 0) return child_node;
                remaining -= 1;
            },
        }
    }
    return null;
}

fn firstDirectNode(node: SyntaxNode) ?SyntaxNode {
    return nthDirectNode(node, 0);
}

fn nthDirectNode(node: SyntaxNode, ordinal: usize) ?SyntaxNode {
    var remaining = ordinal;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .token => {},
            .node => |child_node| {
                if (remaining == 0) return child_node;
                remaining -= 1;
            },
        }
    }
    return null;
}

fn firstDirectExprChild(node: SyntaxNode) ?SyntaxNode {
    return nthDirectExprChild(node, 0);
}

fn nthDirectExprChild(node: SyntaxNode, ordinal: usize) ?SyntaxNode {
    var remaining = ordinal;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .token => {},
            .node => |child_node| {
                if (!isExprKind(child_node.kind())) continue;
                if (remaining == 0) return child_node;
                remaining -= 1;
            },
        }
    }
    return null;
}

fn lastDirectExprChild(node: SyntaxNode) ?SyntaxNode {
    var result: ?SyntaxNode = null;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .token => {},
            .node => |child_node| {
                if (isExprKind(child_node.kind())) result = child_node;
            },
        }
    }
    return result;
}

fn firstDirectTypeChild(node: SyntaxNode) ?SyntaxNode {
    return nthDirectTypeChild(node, 0);
}

fn nthDirectTypeChild(node: SyntaxNode, ordinal: usize) ?SyntaxNode {
    var remaining = ordinal;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .token => {},
            .node => |child_node| {
                if (!isTypeKind(child_node.kind()) and child_node.kind() != .Error) continue;
                if (remaining == 0) return child_node;
                remaining -= 1;
            },
        }
    }
    return null;
}

fn firstDirectToken(node: SyntaxNode) ?SyntaxToken {
    return nthDirectToken(node, 0);
}

fn nthDirectToken(node: SyntaxNode, ordinal: usize) ?SyntaxToken {
    var remaining = ordinal;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => {},
            .token => |token| {
                if (remaining == 0) return token;
                remaining -= 1;
            },
        }
    }
    return null;
}

fn firstDirectTokenOfKind(node: SyntaxNode, kind: syntax.TokenKind) ?SyntaxToken {
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => {},
            .token => |token| if (token.kind() == kind) return token,
        }
    }
    return null;
}

fn firstToken(node: SyntaxNode) ?SyntaxToken {
    return node.firstToken();
}

fn firstTokenOfKind(node: SyntaxNode, kind: syntax.TokenKind) ?SyntaxToken {
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .token => |token| if (token.kind() == kind) return token,
            .node => |child_node| if (firstTokenOfKind(child_node, kind)) |token| return token,
        }
    }
    return null;
}

fn nthDirectIdentifierLikeToken(node: SyntaxNode, ordinal: usize) ?SyntaxToken {
    var remaining = ordinal;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => {},
            .token => |token| if (isIdentifierLike(token.kind())) {
                if (remaining == 0) return token;
                remaining -= 1;
            },
        }
    }
    return null;
}

fn lastDirectIdentifierLikeToken(node: SyntaxNode) ?SyntaxToken {
    var result: ?SyntaxToken = null;
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => {},
            .token => |token| {
                if (isIdentifierLike(token.kind())) result = token;
            },
        }
    }
    return result;
}

fn firstDirectIntegerToken(node: SyntaxNode) ?SyntaxToken {
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => {},
            .token => |token| switch (token.kind()) {
                .IntegerLiteral, .BinaryLiteral, .HexLiteral => return token,
                else => {},
            },
        }
    }
    return null;
}

fn firstDirectArraySizeToken(node: SyntaxNode) ?SyntaxToken {
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .token => |token| switch (token.kind()) {
                .IntegerLiteral, .BinaryLiteral, .HexLiteral => return token,
                else => if (isIdentifierLike(token.kind())) return token,
            },
            .node => {},
        }
    }
    return null;
}

fn firstDirectAssignmentToken(node: SyntaxNode) ?SyntaxToken {
    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => {},
            .token => |token| if (mapAssignmentOp(token.kind()) != null) return token,
        }
    }
    return null;
}

fn tokenText(token: SyntaxToken) []const u8 {
    return token.text();
}

fn isIdentifierLike(kind: syntax.TokenKind) bool {
    return switch (kind) {
        .Identifier, .From, .To, .Error, .Result, .U8, .U16, .U32, .U64, .U128, .U256, .I8, .I16, .I32, .I64, .I128, .I256, .Bool, .Address, .String, .Bytes, .Void => true,
        else => false,
    };
}

fn parseBindingKind(node: SyntaxNode) BindingKind {
    if (firstDirectTokenOfKind(node, .Let) != null) return .let_;
    if (firstDirectTokenOfKind(node, .Var) != null) return .var_;
    if (firstDirectTokenOfKind(node, .Const) != null) return .constant;
    if (firstDirectTokenOfKind(node, .Immutable) != null) return .immutable;
    return .var_;
}

fn parseStorageClass(node: SyntaxNode) StorageClass {
    if (firstDirectTokenOfKind(node, .Storage) != null) return .storage;
    if (firstDirectTokenOfKind(node, .Memory) != null) return .memory;
    if (firstDirectTokenOfKind(node, .Tstore) != null) return .tstore;
    return .none;
}

fn mapAssignmentOp(kind: syntax.TokenKind) ?AssignmentOp {
    return switch (kind) {
        .Equal => .assign,
        .PlusEqual => .add_assign,
        .PlusPercentEqual => .wrapping_add_assign,
        .MinusEqual => .sub_assign,
        .MinusPercentEqual => .wrapping_sub_assign,
        .StarEqual => .mul_assign,
        .StarPercentEqual => .wrapping_mul_assign,
        .StarStarEqual => .pow_assign,
        .SlashEqual => .div_assign,
        .PercentEqual => .mod_assign,
        .AmpersandEqual => .bit_and_assign,
        .PipeEqual => .bit_or_assign,
        .CaretEqual => .bit_xor_assign,
        .LessLessEqual => .shl_assign,
        .GreaterGreaterEqual => .shr_assign,
        else => null,
    };
}

fn mapBinaryOp(kind: syntax.TokenKind) ?BinaryOp {
    return switch (kind) {
        .Plus => .add,
        .PlusPercent => .wrapping_add,
        .Minus => .sub,
        .MinusPercent => .wrapping_sub,
        .Star => .mul,
        .StarPercent => .wrapping_mul,
        .Slash => .div,
        .Percent => .mod,
        .StarStar => .pow,
        .StarStarPercent => .wrapping_pow,
        .EqualEqual => .eq,
        .BangEqual => .ne,
        .Less => .lt,
        .LessEqual => .le,
        .Greater => .gt,
        .GreaterEqual => .ge,
        .AmpersandAmpersand => .and_and,
        .PipePipe => .or_or,
        .Ampersand => .bit_and,
        .Pipe => .bit_or,
        .Caret => .bit_xor,
        .LessLess => .shl,
        .LessLessPercent => .wrapping_shl,
        .GreaterGreater => .shr,
        .GreaterGreaterPercent => .wrapping_shr,
        else => null,
    };
}

fn stripBytesLiteral(text: []const u8) []const u8 {
    if (std.mem.startsWith(u8, text, "hex\"") and text.len >= 5 and text[text.len - 1] == '"') {
        return text[4 .. text.len - 1];
    }
    return stripQuotes(text);
}

const ForBindings = struct {
    item: SyntaxToken,
    index: ?SyntaxToken,
};

fn parseForBindings(node: SyntaxNode) ?ForBindings {
    var seen_first_pipe = false;
    var seen_second_pipe = false;
    var item: ?SyntaxToken = null;
    var index: ?SyntaxToken = null;

    var it = node.children();
    while (it.next()) |child| {
        switch (child) {
            .node => {},
            .token => |token| {
                if (token.kind() == .Pipe) {
                    if (!seen_first_pipe) {
                        seen_first_pipe = true;
                    } else {
                        seen_second_pipe = true;
                        break;
                    }
                    continue;
                }
                if (!seen_first_pipe or seen_second_pipe) continue;
                if (!isIdentifierLike(token.kind())) continue;
                if (item == null) {
                    item = token;
                } else if (index == null) {
                    index = token;
                }
            },
        }
    }

    if (item) |item_token| {
        return .{ .item = item_token, .index = index };
    }
    return null;
}

fn parseBitfieldOffset(node: SyntaxNode) ?u32 {
    const at_token = firstDirectTokenOfKind(node, .At) orelse return null;
    const start = at_token.range().start;
    const text = node.tree.sourceSlice(.{ .start = start, .end = node.range().end });
    if (std.mem.indexOf(u8, text, "@at(")) |idx| {
        return parseFirstU32(text[idx + 4 ..]);
    }
    if (std.mem.indexOf(u8, text, "@bits(")) |idx| {
        return parseFirstU32(text[idx + 6 ..]);
    }
    return null;
}

fn parseBitfieldWidth(node: SyntaxNode) ?u32 {
    const at_token = firstDirectTokenOfKind(node, .At) orelse {
        if (firstDirectChildOfKind(node, .GroupParen) != null or firstDirectTokenOfKind(node, .LeftParen) != null) {
            return parseFirstU32(node.tree.sourceSlice(node.range()));
        }
        return null;
    };
    const start = at_token.range().start;
    const text = node.tree.sourceSlice(.{ .start = start, .end = node.range().end });
    if (std.mem.indexOf(u8, text, "@at(")) |idx| {
        const tail = text[idx + 4 ..];
        if (std.mem.indexOfScalar(u8, tail, ',')) |comma| {
            return parseFirstU32(tail[comma + 1 ..]);
        }
        return null;
    }
    if (std.mem.indexOf(u8, text, "@bits(")) |idx| {
        const tail = text[idx + 6 ..];
        const start_bit = parseFirstU32(tail) orelse return null;
        if (std.mem.indexOf(u8, tail, "..")) |dots| {
            const end_bit = parseFirstU32(tail[dots + 2 ..]) orelse return null;
            return end_bit - start_bit;
        }
    }
    return null;
}

fn parseFirstU32(text: []const u8) ?u32 {
    var start: ?usize = null;
    var end: usize = 0;
    for (text, 0..) |c, idx| {
        if (std.ascii.isDigit(c)) {
            if (start == null) start = idx;
            end = idx + 1;
        } else if (start != null) {
            break;
        }
    }
    if (start) |s| {
        return std.fmt.parseInt(u32, text[s..end], 10) catch null;
    }
    return null;
}
