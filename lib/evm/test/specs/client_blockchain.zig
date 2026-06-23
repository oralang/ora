const std = @import("std");
const evm_mod = @import("evm");
const primitives = evm_mod.primitives;

pub const ValidationError = error{
    InvalidDifficulty,
    InvalidNonce,
    InvalidOmmersHash,
    InvalidExtraDataLength,
    InvalidGasLimit,
    InvalidGasUsed,
    InvalidTimestamp,
    InvalidBaseFee,
    MissingBaseFee,
    InvalidParentHash,
    InvalidBlockNumber,
    MissingParentHeader,
    BaseFeeMath,
};

pub const HeaderValidationContext = struct {
    allocator: std.mem.Allocator,
    hardfork: primitives.Hardfork,
    parent_header: ?*const primitives.BlockHeader.BlockHeader,
};

pub fn merge_header_validator(comptime PreMergeValidator: type) type {
    return struct {
        pub fn validate(
            header: *const primitives.BlockHeader.BlockHeader,
            ctx: HeaderValidationContext,
        ) ValidationError!void {
            try PreMergeValidator.validate(header, ctx);
            _ = header;
            _ = ctx;
        }
    };
}
