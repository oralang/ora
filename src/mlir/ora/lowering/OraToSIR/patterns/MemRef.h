#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir
{
    namespace ora
    {

        // Forward declarations
        class OraToSIRTypeConverter;

        // MemRef elimination conversions
        class ConvertMemRefLoadOp : public OpConversionPattern<mlir::memref::LoadOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::memref::LoadOp op,
                typename mlir::memref::LoadOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertMemRefStoreOp : public OpConversionPattern<mlir::memref::StoreOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::memref::StoreOp op,
                typename mlir::memref::StoreOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertMemRefAllocOp : public OpConversionPattern<mlir::memref::AllocaOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::memref::AllocaOp op,
                typename mlir::memref::AllocaOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertMemRefDimOp : public OpConversionPattern<mlir::memref::DimOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::memref::DimOp op,
                typename mlir::memref::DimOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

    } // namespace ora
} // namespace mlir

/// Clear the static helper map between pass invocations.
void clearMemRefHelperMap();

