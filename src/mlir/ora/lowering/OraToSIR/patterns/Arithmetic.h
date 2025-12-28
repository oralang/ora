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

        class ConvertRefinementToBaseOp : public OpConversionPattern<ora::RefinementToBaseOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::RefinementToBaseOp op,
                typename ora::RefinementToBaseOp::Adaptor adaptor,
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

    } // namespace ora
} // namespace mlir
