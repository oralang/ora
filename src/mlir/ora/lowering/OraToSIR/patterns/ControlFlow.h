#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "OraDialect.h"
#include "SIR/SIRDialect.h"

namespace mlir
{
    namespace ora
    {

        // Forward declarations
        class OraToSIRTypeConverter;

        // Control flow operation conversions
        class ConvertReturnOp : public OpConversionPattern<ora::ReturnOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ReturnOp op,
                typename ora::ReturnOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertFuncOp : public OpConversionPattern<mlir::func::FuncOp>
        {
        public:
            using OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::func::FuncOp op,
                typename mlir::func::FuncOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertContractOp : public OpConversionPattern<ora::ContractOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ContractOp op,
                typename ora::ContractOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

    } // namespace ora
} // namespace mlir
