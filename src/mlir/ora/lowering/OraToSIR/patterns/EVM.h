#ifndef ORA_LOWERING_ORA_TO_SIR_PATTERNS_EVM_H
#define ORA_LOWERING_ORA_TO_SIR_PATTERNS_EVM_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
    namespace ora
    {
        struct ConvertEvmOp : public mlir::ConversionPattern
        {
            ConvertEvmOp(mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx)
                : mlir::ConversionPattern(typeConverter, mlir::Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, ctx)
            {
            }

            mlir::LogicalResult matchAndRewrite(
                mlir::Operation *op,
                mlir::ArrayRef<mlir::Value> operands,
                mlir::ConversionPatternRewriter &rewriter) const override;
        };
    } // namespace ora
} // namespace mlir

#endif // ORA_LOWERING_ORA_TO_SIR_PATTERNS_EVM_H
