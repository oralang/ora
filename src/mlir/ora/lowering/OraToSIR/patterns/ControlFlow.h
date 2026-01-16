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

        class ConvertCallOp : public OpConversionPattern<mlir::func::CallOp>
        {
        public:
            using OpConversionPattern<mlir::func::CallOp>::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::func::CallOp op,
                typename mlir::func::CallOp::Adaptor adaptor,
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

        class ConvertWhileOp : public OpConversionPattern<ora::WhileOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::WhileOp op,
                typename ora::WhileOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertIfOp : public OpConversionPattern<ora::IfOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::IfOp op,
                typename ora::IfOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertIsolatedIfOp : public OpConversionPattern<ora::IsolatedIfOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::IsolatedIfOp op,
                typename ora::IsolatedIfOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertBreakOp : public OpConversionPattern<ora::BreakOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::BreakOp op,
                typename ora::BreakOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertContinueOp : public OpConversionPattern<ora::ContinueOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ContinueOp op,
                typename ora::ContinueOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertSwitchExprOp : public OpConversionPattern<ora::SwitchExprOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::SwitchExprOp op,
                typename ora::SwitchExprOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertSwitchOp : public OpConversionPattern<ora::SwitchOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::SwitchOp op,
                typename ora::SwitchOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertRangeOp : public OpConversionPattern<ora::RangeOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::RangeOp op,
                typename ora::RangeOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertTryCatchOp : public OpConversionPattern<ora::TryOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::TryOp op,
                typename ora::TryOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

    } // namespace ora
} // namespace mlir
