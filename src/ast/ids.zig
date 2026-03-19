const source = @import("../source/mod.zig");
const defineId = source.defineId;

pub const AstFileId = source.FileId;

pub const ItemId = defineId("ItemId");
pub const BodyId = defineId("BodyId");
pub const StmtId = defineId("StmtId");
pub const ExprId = defineId("ExprId");
pub const TypeExprId = defineId("TypeExprId");
pub const PatternId = defineId("PatternId");
