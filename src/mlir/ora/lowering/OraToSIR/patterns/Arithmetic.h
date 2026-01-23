#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "OraDialect.h"
#include "SIR/SIRDialect.h"

namespace mlir
{
    namespace ora
    {

        // Forward declarations
        class OraToSIRTypeConverter;

        // Arithmetic operation conversions
        class ConvertAddOp : public OpConversionPattern<ora::AddOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::AddOp op,
                typename ora::AddOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertMulOp : public OpConversionPattern<ora::MulOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::MulOp op,
                typename ora::MulOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertSubOp : public OpConversionPattern<ora::SubOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::SubOp op,
                typename ora::SubOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertDivOp : public OpConversionPattern<ora::DivOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::DivOp op,
                typename ora::DivOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertRemOp : public OpConversionPattern<ora::RemOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::RemOp op,
                typename ora::RemOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertCmpOp : public OpConversionPattern<ora::CmpOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::CmpOp op,
                typename ora::CmpOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertConstOp : public OpConversionPattern<ora::ConstOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ConstOp op,
                typename ora::ConstOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertStringConstantOp : public OpConversionPattern<ora::StringConstantOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::StringConstantOp op,
                typename ora::StringConstantOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertBytesConstantOp : public OpConversionPattern<ora::BytesConstantOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::BytesConstantOp op,
                typename ora::BytesConstantOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertHexConstantOp : public OpConversionPattern<ora::HexConstantOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::HexConstantOp op,
                typename ora::HexConstantOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertAddrToI160Op : public OpConversionPattern<ora::AddrToI160Op>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::AddrToI160Op op,
                typename ora::AddrToI160Op::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertI160ToAddrOp : public OpConversionPattern<ora::I160ToAddrOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::I160ToAddrOp op,
                typename ora::I160ToAddrOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertOldOp : public OpConversionPattern<ora::OldOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::OldOp op,
                typename ora::OldOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertRefinementToBaseOp : public OpConversionPattern<ora::RefinementToBaseOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::RefinementToBaseOp op,
                typename ora::RefinementToBaseOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertBaseToRefinementOp : public OpConversionPattern<ora::BaseToRefinementOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::BaseToRefinementOp op,
                typename ora::BaseToRefinementOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertInvariantOp : public OpConversionPattern<ora::InvariantOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::InvariantOp op,
                typename ora::InvariantOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertRequiresOp : public OpConversionPattern<ora::RequiresOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::RequiresOp op,
                typename ora::RequiresOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertEnsuresOp : public OpConversionPattern<ora::EnsuresOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::EnsuresOp op,
                typename ora::EnsuresOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertAssertOp : public OpConversionPattern<ora::AssertOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::AssertOp op,
                typename ora::AssertOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertAssumeOp : public OpConversionPattern<ora::AssumeOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::AssumeOp op,
                typename ora::AssumeOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertDecreasesOp : public OpConversionPattern<ora::DecreasesOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::DecreasesOp op,
                typename ora::DecreasesOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertIncreasesOp : public OpConversionPattern<ora::IncreasesOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::IncreasesOp op,
                typename ora::IncreasesOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertHavocOp : public OpConversionPattern<ora::HavocOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::HavocOp op,
                typename ora::HavocOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertQuantifiedOp : public OpConversionPattern<ora::QuantifiedOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::QuantifiedOp op,
                typename ora::QuantifiedOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        // Convert arith.constant with Ora types to sir.const
        class ConvertArithConstantOp : public OpConversionPattern<mlir::arith::ConstantOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::ConstantOp op,
                typename mlir::arith::ConstantOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithCmpIOp : public OpConversionPattern<mlir::arith::CmpIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::CmpIOp op,
                typename mlir::arith::CmpIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithAddIOp : public OpConversionPattern<mlir::arith::AddIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::AddIOp op,
                typename mlir::arith::AddIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithSubIOp : public OpConversionPattern<mlir::arith::SubIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::SubIOp op,
                typename mlir::arith::SubIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithMulIOp : public OpConversionPattern<mlir::arith::MulIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::MulIOp op,
                typename mlir::arith::MulIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithDivUIOp : public OpConversionPattern<mlir::arith::DivUIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::DivUIOp op,
                typename mlir::arith::DivUIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithRemUIOp : public OpConversionPattern<mlir::arith::RemUIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::RemUIOp op,
                typename mlir::arith::RemUIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithDivSIOp : public OpConversionPattern<mlir::arith::DivSIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::DivSIOp op,
                typename mlir::arith::DivSIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithAndIOp : public OpConversionPattern<mlir::arith::AndIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::AndIOp op,
                typename mlir::arith::AndIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithOrIOp : public OpConversionPattern<mlir::arith::OrIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::OrIOp op,
                typename mlir::arith::OrIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithXOrIOp : public OpConversionPattern<mlir::arith::XOrIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::XOrIOp op,
                typename mlir::arith::XOrIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithSelectOp : public OpConversionPattern<mlir::arith::SelectOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::SelectOp op,
                typename mlir::arith::SelectOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithExtUIOp : public OpConversionPattern<mlir::arith::ExtUIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::ExtUIOp op,
                typename mlir::arith::ExtUIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithIndexCastUIOp : public OpConversionPattern<mlir::arith::IndexCastUIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::IndexCastUIOp op,
                typename mlir::arith::IndexCastUIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertArithTruncIOp : public OpConversionPattern<mlir::arith::TruncIOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::arith::TruncIOp op,
                typename mlir::arith::TruncIOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class FoldRedundantBitcastOp : public OpRewritePattern<sir::BitcastOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                sir::BitcastOp op,
                PatternRewriter &rewriter) const override;
        };

        class FoldAndOneOp : public OpRewritePattern<sir::AndOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                sir::AndOp op,
                PatternRewriter &rewriter) const override;
        };

        class FoldEqSameOp : public OpRewritePattern<sir::EqOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                sir::EqOp op,
                PatternRewriter &rewriter) const override;
        };

        class FoldIsZeroConstOp : public OpRewritePattern<sir::IsZeroOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                sir::IsZeroOp op,
                PatternRewriter &rewriter) const override;
        };

        class FoldEqConstOp : public OpRewritePattern<sir::EqOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                sir::EqOp op,
                PatternRewriter &rewriter) const override;
        };

    } // namespace ora
} // namespace mlir
