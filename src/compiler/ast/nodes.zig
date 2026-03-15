const source = @import("../source/mod.zig");
const ids = @import("ids.zig");

const ItemId = ids.ItemId;
const BodyId = ids.BodyId;
const StmtId = ids.StmtId;
const ExprId = ids.ExprId;
const TypeExprId = ids.TypeExprId;
const PatternId = ids.PatternId;

pub const Visibility = enum { public, private };
pub const BindingKind = enum { let_, var_, constant, immutable };
pub const StorageClass = enum { none, storage, memory, tstore };
pub const SpecClauseKind = enum { requires, ensures, invariant };
pub const UnaryOp = enum { neg, not_, bit_not, try_ };
pub const BinaryOp = enum {
    add,
    wrapping_add,
    sub,
    wrapping_sub,
    mul,
    wrapping_mul,
    div,
    mod,
    pow,
    wrapping_pow,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    and_and,
    or_or,
    bit_and,
    bit_or,
    bit_xor,
    shl,
    wrapping_shl,
    shr,
    wrapping_shr,
};
pub const AssignmentOp = enum { assign, add_assign, sub_assign, mul_assign, div_assign, mod_assign, bit_and_assign, bit_or_assign, bit_xor_assign, shl_assign, shr_assign, pow_assign, wrapping_add_assign, wrapping_sub_assign, wrapping_mul_assign };
pub const Quantifier = enum { forall, exists };

pub const NodeError = struct {
    range: source.TextRange,
};

pub const TypeExpr = union(enum) {
    Path: PathTypeExpr,
    Generic: GenericTypeExpr,
    Tuple: TupleTypeExpr,
    Array: ArrayTypeExpr,
    Slice: SliceTypeExpr,
    ErrorUnion: ErrorUnionTypeExpr,
    Error: NodeError,
};

pub const PathTypeExpr = struct {
    range: source.TextRange,
    name: []const u8,
};

pub const TypeArg = union(enum) {
    Type: TypeExprId,
    Integer: TypeIntegerLiteral,
};

pub const TypeIntegerLiteral = struct {
    range: source.TextRange,
    text: []const u8,
};

pub const TypeArraySize = union(enum) {
    Integer: TypeIntegerLiteral,
    Name: PathTypeExpr,
};

pub const GenericTypeExpr = struct {
    range: source.TextRange,
    name: []const u8,
    args: []TypeArg,
};

pub const TupleTypeExpr = struct {
    range: source.TextRange,
    elements: []TypeExprId,
};

pub const ArrayTypeExpr = struct {
    range: source.TextRange,
    element: TypeExprId,
    size: TypeArraySize,
};

pub const SliceTypeExpr = struct {
    range: source.TextRange,
    element: TypeExprId,
};

pub const ErrorUnionTypeExpr = struct {
    range: source.TextRange,
    payload: TypeExprId,
    errors: []TypeExprId,
};

pub const Pattern = union(enum) {
    Name: NamePattern,
    Field: FieldPattern,
    Index: IndexPattern,
    StructDestructure: StructDestructurePattern,
    Error: NodeError,
};

pub const NamePattern = struct {
    range: source.TextRange,
    name: []const u8,
};

pub const FieldPattern = struct {
    range: source.TextRange,
    base: PatternId,
    name: []const u8,
};

pub const IndexPattern = struct {
    range: source.TextRange,
    base: PatternId,
    index: ExprId,
};

pub const StructDestructureField = struct {
    range: source.TextRange,
    name: []const u8,
    binding: PatternId,
};

pub const StructDestructurePattern = struct {
    range: source.TextRange,
    fields: []StructDestructureField,
};

pub const Expr = union(enum) {
    IntegerLiteral: IntegerLiteralExpr,
    StringLiteral: StringLiteralExpr,
    BoolLiteral: BoolLiteralExpr,
    AddressLiteral: AddressLiteralExpr,
    BytesLiteral: BytesLiteralExpr,
    TypeValue: TypeValueExpr,
    Tuple: TupleExpr,
    ArrayLiteral: ArrayLiteralExpr,
    StructLiteral: StructLiteralExpr,
    Switch: SwitchExpr,
    Comptime: ComptimeExpr,
    ErrorReturn: ErrorReturnExpr,
    Name: NameExpr,
    Result: ResultExpr,
    Unary: UnaryExpr,
    Binary: BinaryExpr,
    Call: CallExpr,
    Builtin: BuiltinExpr,
    Field: FieldExpr,
    Index: IndexExpr,
    Group: GroupExpr,
    Old: OldExpr,
    Quantified: QuantifiedExpr,
    Error: NodeError,
};

pub const IntegerLiteralExpr = struct {
    range: source.TextRange,
    text: []const u8,
};

pub const StringLiteralExpr = struct {
    range: source.TextRange,
    text: []const u8,
};

pub const BoolLiteralExpr = struct {
    range: source.TextRange,
    value: bool,
};

pub const AddressLiteralExpr = struct {
    range: source.TextRange,
    text: []const u8,
};

pub const BytesLiteralExpr = struct {
    range: source.TextRange,
    text: []const u8,
};

pub const TypeValueExpr = struct {
    range: source.TextRange,
    type_expr: TypeExprId,
};

pub const TupleExpr = struct {
    range: source.TextRange,
    elements: []ExprId,
};

pub const ArrayLiteralExpr = struct {
    range: source.TextRange,
    elements: []ExprId,
};

pub const StructFieldInit = struct {
    range: source.TextRange,
    name: []const u8,
    value: ExprId,
};

pub const StructLiteralExpr = struct {
    range: source.TextRange,
    type_name: []const u8,
    fields: []StructFieldInit,
};

pub const SwitchExprArm = struct {
    range: source.TextRange,
    pattern: SwitchPattern,
    value: ExprId,
};

pub const SwitchExpr = struct {
    range: source.TextRange,
    condition: ExprId,
    arms: []SwitchExprArm,
    else_expr: ?ExprId,
};

pub const ComptimeExpr = struct {
    range: source.TextRange,
    body: BodyId,
};

pub const ErrorReturnExpr = struct {
    range: source.TextRange,
    name: []const u8,
    args: []ExprId,
};

pub const NameExpr = struct {
    range: source.TextRange,
    name: []const u8,
};

pub const ResultExpr = struct {
    range: source.TextRange,
};

pub const UnaryExpr = struct {
    range: source.TextRange,
    op: UnaryOp,
    operand: ExprId,
};

pub const BinaryExpr = struct {
    range: source.TextRange,
    op: BinaryOp,
    lhs: ExprId,
    rhs: ExprId,
};

pub const CallExpr = struct {
    range: source.TextRange,
    callee: ExprId,
    args: []ExprId,
};

pub const BuiltinExpr = struct {
    range: source.TextRange,
    name: []const u8,
    type_arg: ?TypeExprId,
    args: []ExprId,
};

pub const FieldExpr = struct {
    range: source.TextRange,
    base: ExprId,
    name: []const u8,
};

pub const IndexExpr = struct {
    range: source.TextRange,
    base: ExprId,
    index: ExprId,
};

pub const GroupExpr = struct {
    range: source.TextRange,
    expr: ExprId,
};

pub const OldExpr = struct {
    range: source.TextRange,
    expr: ExprId,
};

pub const QuantifiedExpr = struct {
    range: source.TextRange,
    quantifier: Quantifier,
    pattern: PatternId,
    type_expr: TypeExprId,
    condition: ?ExprId,
    body: ExprId,
};

pub const Stmt = union(enum) {
    VariableDecl: VariableDeclStmt,
    Return: ReturnStmt,
    If: IfStmt,
    While: WhileStmt,
    For: ForStmt,
    Switch: SwitchStmt,
    Try: TryStmt,
    Log: LogStmt,
    Lock: LockStmt,
    Unlock: UnlockStmt,
    Assert: AssertStmt,
    Assume: AssumeStmt,
    Havoc: HavocStmt,
    Break: JumpStmt,
    Continue: JumpStmt,
    Assign: AssignStmt,
    Expr: ExprStmt,
    Block: BlockStmt,
    LabeledBlock: LabeledBlockStmt,
    Error: NodeError,
};

pub const VariableDeclStmt = struct {
    range: source.TextRange,
    pattern: PatternId,
    binding_kind: BindingKind,
    storage_class: StorageClass,
    type_expr: ?TypeExprId,
    value: ?ExprId,
};

pub const ReturnStmt = struct {
    range: source.TextRange,
    value: ?ExprId,
};

pub const IfStmt = struct {
    range: source.TextRange,
    condition: ExprId,
    then_body: BodyId,
    else_body: ?BodyId,
};

pub const WhileStmt = struct {
    range: source.TextRange,
    condition: ExprId,
    invariants: []ExprId,
    body: BodyId,
};

pub const ForStmt = struct {
    range: source.TextRange,
    iterable: ExprId,
    item_pattern: PatternId,
    index_pattern: ?PatternId,
    invariants: []ExprId,
    body: BodyId,
};

pub const SwitchPattern = union(enum) {
    Expr: ExprId,
    Range: RangeSwitchPattern,
    Else: source.TextRange,
};

pub const RangeSwitchPattern = struct {
    range: source.TextRange,
    start: ExprId,
    end: ExprId,
    inclusive: bool,
};

pub const SwitchArm = struct {
    range: source.TextRange,
    pattern: SwitchPattern,
    body: BodyId,
};

pub const SwitchStmt = struct {
    range: source.TextRange,
    condition: ExprId,
    arms: []SwitchArm,
    else_body: ?BodyId,
};

pub const CatchClause = struct {
    range: source.TextRange,
    error_pattern: ?PatternId,
    body: BodyId,
};

pub const TryStmt = struct {
    range: source.TextRange,
    try_body: BodyId,
    catch_clause: ?CatchClause,
};

pub const LogStmt = struct {
    range: source.TextRange,
    name: []const u8,
    args: []ExprId,
};

pub const LockStmt = struct {
    range: source.TextRange,
    path: ExprId,
};

pub const UnlockStmt = struct {
    range: source.TextRange,
    path: ExprId,
};

pub const AssertStmt = struct {
    range: source.TextRange,
    condition: ExprId,
    message: ?[]const u8,
};

pub const AssumeStmt = struct {
    range: source.TextRange,
    condition: ExprId,
};

pub const HavocStmt = struct {
    range: source.TextRange,
    name: []const u8,
};

pub const JumpStmt = struct {
    range: source.TextRange,
};

pub const AssignStmt = struct {
    range: source.TextRange,
    op: AssignmentOp,
    target: PatternId,
    value: ExprId,
};

pub const ExprStmt = struct {
    range: source.TextRange,
    expr: ExprId,
};

pub const BlockStmt = struct {
    range: source.TextRange,
    body: BodyId,
};

pub const LabeledBlockStmt = struct {
    range: source.TextRange,
    label: []const u8,
    body: BodyId,
};

pub const Body = struct {
    range: source.TextRange,
    statements: []StmtId,
};

pub const Parameter = struct {
    range: source.TextRange,
    is_comptime: bool = false,
    pattern: PatternId,
    type_expr: TypeExprId,
};

pub const SpecClause = struct {
    range: source.TextRange,
    kind: SpecClauseKind,
    expr: ExprId,
};

pub const StructField = struct {
    range: source.TextRange,
    name: []const u8,
    type_expr: TypeExprId,
};

pub const BitfieldField = struct {
    range: source.TextRange,
    name: []const u8,
    type_expr: TypeExprId,
    offset: ?u32,
    width: ?u32,
};

pub const LogField = struct {
    range: source.TextRange,
    name: []const u8,
    type_expr: TypeExprId,
    indexed: bool,
};

pub const EnumVariant = struct {
    range: source.TextRange,
    name: []const u8,
};

pub const TraitMethod = struct {
    range: source.TextRange,
    name: []const u8,
    has_self: bool,
    parameters: []Parameter,
    return_type: ?TypeExprId,
    clauses: []SpecClause,
    is_comptime: bool,
};

pub const Item = union(enum) {
    Import: ImportItem,
    Contract: ContractItem,
    Function: FunctionItem,
    Struct: StructItem,
    Bitfield: BitfieldItem,
    Enum: EnumItem,
    Trait: TraitItem,
    Impl: ImplItem,
    TypeAlias: TypeAliasItem,
    LogDecl: LogDeclItem,
    ErrorDecl: ErrorDeclItem,
    GhostBlock: GhostBlockItem,
    Field: FieldItem,
    Constant: ConstantItem,
    Error: NodeError,
};

pub const ImportItem = struct {
    range: source.TextRange,
    path: []const u8,
    alias: ?[]const u8,
    is_comptime: bool,
};

pub const ContractItem = struct {
    range: source.TextRange,
    name: []const u8,
    is_generic: bool = false,
    template_parameters: []const Parameter,
    members: []ItemId,
    invariants: []ExprId,
};

pub const FunctionItem = struct {
    range: source.TextRange,
    name: []const u8,
    is_ghost: bool = false,
    is_generic: bool = false,
    visibility: Visibility,
    parameters: []Parameter,
    return_type: ?TypeExprId,
    clauses: []SpecClause,
    body: BodyId,
    parent_contract: ?ItemId,
};

pub const StructItem = struct {
    range: source.TextRange,
    name: []const u8,
    is_generic: bool = false,
    template_parameters: []const Parameter,
    fields: []StructField,
};

pub const BitfieldItem = struct {
    range: source.TextRange,
    name: []const u8,
    is_generic: bool = false,
    template_parameters: []const Parameter,
    base_type: ?TypeExprId,
    fields: []BitfieldField,
};

pub const EnumItem = struct {
    range: source.TextRange,
    name: []const u8,
    is_generic: bool = false,
    template_parameters: []const Parameter,
    variants: []EnumVariant,
};

pub const TraitItem = struct {
    range: source.TextRange,
    name: []const u8,
    methods: []TraitMethod,
};

pub const ImplItem = struct {
    range: source.TextRange,
    trait_name: []const u8,
    target_name: []const u8,
    methods: []ItemId,
};

pub const TypeAliasItem = struct {
    range: source.TextRange,
    name: []const u8,
    is_generic: bool = false,
    template_parameters: []const Parameter,
    target_type: TypeExprId,
};

pub const LogDeclItem = struct {
    range: source.TextRange,
    name: []const u8,
    fields: []LogField,
};

pub const ErrorDeclItem = struct {
    range: source.TextRange,
    name: []const u8,
    parameters: []Parameter,
};

pub const GhostBlockItem = struct {
    range: source.TextRange,
    body: BodyId,
};

pub const FieldItem = struct {
    range: source.TextRange,
    name: []const u8,
    is_ghost: bool = false,
    binding_kind: BindingKind,
    storage_class: StorageClass,
    type_expr: ?TypeExprId,
    value: ?ExprId,
};

pub const ConstantItem = struct {
    range: source.TextRange,
    name: []const u8,
    is_ghost: bool = false,
    is_comptime: bool,
    type_expr: ?TypeExprId,
    value: ExprId,
};
