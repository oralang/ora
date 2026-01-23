#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

            LogicalResult matchAndRewrite(
                ora::ReturnOp op,
                OneToNOpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertReturnOpRaw : public OpConversionPattern<ora::ReturnOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ReturnOp op,
                typename ora::ReturnOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertReturnOpFallback final : public ConversionPattern
        {
        public:
            ConvertReturnOpFallback(const TypeConverter *typeConverter, MLIRContext *ctx, PatternBenefit benefit = 1)
                : ConversionPattern(ora::ReturnOp::getOperationName(), benefit, ctx),
                  typeConverter(typeConverter)
            {
            }

            LogicalResult matchAndRewrite(
                Operation *op,
                ArrayRef<Value> operands,
                ConversionPatternRewriter &rewriter) const override;

        private:
            const TypeConverter *typeConverter;
        };

        class ConvertReturnOpPre : public OpRewritePattern<ora::ReturnOp>
        {
        public:
            ConvertReturnOpPre(const TypeConverter *typeConverter, MLIRContext *ctx)
                : OpRewritePattern<ora::ReturnOp>(ctx), typeConverter(typeConverter)
            {
            }

            LogicalResult matchAndRewrite(
                ora::ReturnOp op,
                PatternRewriter &rewriter) const override;

        private:
            const TypeConverter *typeConverter;
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

        class ConvertErrorDeclOp : public OpConversionPattern<ora::ErrorDeclOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ErrorDeclOp op,
                typename ora::ErrorDeclOp::Adaptor adaptor,
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

        class ConvertCfBrOp : public OpConversionPattern<mlir::cf::BranchOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::cf::BranchOp op,
                typename mlir::cf::BranchOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertCfCondBrOp : public OpConversionPattern<mlir::cf::CondBranchOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::cf::CondBranchOp op,
                typename mlir::cf::CondBranchOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertCfAssertOp : public OpConversionPattern<mlir::cf::AssertOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::cf::AssertOp op,
                typename mlir::cf::AssertOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertScfIfOp : public OpConversionPattern<mlir::scf::IfOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::scf::IfOp op,
                typename mlir::scf::IfOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertScfForOp : public OpConversionPattern<mlir::scf::ForOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::scf::ForOp op,
                typename mlir::scf::ForOp::Adaptor adaptor,
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

        class ConvertTryStmtOp : public OpConversionPattern<ora::TryStmtOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::TryStmtOp op,
                typename ora::TryStmtOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertErrorOkOp : public OpConversionPattern<ora::ErrorOkOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ErrorOkOp op,
                typename ora::ErrorOkOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertErrorErrOp : public OpConversionPattern<ora::ErrorErrOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ErrorErrOp op,
                typename ora::ErrorErrOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertErrorIsErrorOp : public OpConversionPattern<ora::ErrorIsErrorOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ErrorIsErrorOp op,
                typename ora::ErrorIsErrorOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertErrorUnwrapOp : public OpConversionPattern<ora::ErrorUnwrapOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ErrorUnwrapOp op,
                typename ora::ErrorUnwrapOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertErrorGetErrorOp : public OpConversionPattern<ora::ErrorGetErrorOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ErrorGetErrorOp op,
                typename ora::ErrorGetErrorOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class FoldCondBrSameDestOp : public OpRewritePattern<sir::CondBrOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                sir::CondBrOp op,
                PatternRewriter &rewriter) const override;
        };

        class FoldCondBrDoubleIsZeroOp : public OpRewritePattern<sir::CondBrOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                sir::CondBrOp op,
                PatternRewriter &rewriter) const override;
        };

        class FoldCondBrConstOp : public OpRewritePattern<sir::CondBrOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                sir::CondBrOp op,
                PatternRewriter &rewriter) const override;
        };

        class FoldBrToBrOp : public OpRewritePattern<sir::BrOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                sir::BrOp op,
                PatternRewriter &rewriter) const override;
        };

    } // namespace ora
} // namespace mlir
