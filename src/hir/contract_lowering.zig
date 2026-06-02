const sema = @import("../sema/mod.zig");

pub fn mixin(ContractLowerer: type, Lowerer: type, FunctionLowerer: type) type {
    _ = Lowerer;
    return struct {
        pub fn lowerInvariant(self: *ContractLowerer, fact: sema.VerificationFact) !void {
            var function_lowerer = FunctionLowerer.initContractContext(self.parent, self.block);
            try FunctionLowerer.lowerInvariantFact(&function_lowerer, fact, &function_lowerer.locals);
        }
    };
}
