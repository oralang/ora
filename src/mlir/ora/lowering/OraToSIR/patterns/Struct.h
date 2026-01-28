#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "OraDialect.h"
#include "SIR/SIRDialect.h"

namespace mlir
{
    namespace ora
    {

        class OraToSIRTypeConverter;

        class ConvertStructInstantiateOp : public OpConversionPattern<ora::StructInstantiateOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::StructInstantiateOp op,
                typename ora::StructInstantiateOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertStructInitOp : public OpConversionPattern<ora::StructInitOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::StructInitOp op,
                typename ora::StructInitOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertStructFieldExtractOp : public OpConversionPattern<ora::StructFieldExtractOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::StructFieldExtractOp op,
                typename ora::StructFieldExtractOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertStructFieldUpdateOp : public OpConversionPattern<ora::StructFieldUpdateOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::StructFieldUpdateOp op,
                typename ora::StructFieldUpdateOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertStructDeclOp : public OpConversionPattern<ora::StructDeclOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::StructDeclOp op,
                typename ora::StructDeclOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class StripStructMaterializeOp : public OpConversionPattern<mlir::UnrealizedConversionCastOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::UnrealizedConversionCastOp op,
                mlir::UnrealizedConversionCastOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class StripAddressMaterializeOp : public OpConversionPattern<mlir::UnrealizedConversionCastOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::UnrealizedConversionCastOp op,
                mlir::UnrealizedConversionCastOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class StripBytesMaterializeOp : public OpConversionPattern<mlir::UnrealizedConversionCastOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::UnrealizedConversionCastOp op,
                mlir::UnrealizedConversionCastOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

    } // namespace ora
} // namespace mlir
