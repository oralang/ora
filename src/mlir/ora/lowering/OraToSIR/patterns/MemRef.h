#pragma once

#include "patterns/Naming.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <map>

namespace mlir
{
    namespace ora
    {

        // Forward declarations
        class OraToSIRTypeConverter;

        struct MemRefNamingCache
        {
            SIRNamingHelper &get(Operation *op)
            {
                Operation *parentFunc = op->getParentOfType<mlir::func::FuncOp>();
                if (!parentFunc)
                    return fallback;

                auto it = helpers.find(parentFunc);
                if (it == helpers.end())
                {
                    SIRNamingHelper newHelper;
                    newHelper.reset();
                    auto inserted = helpers.emplace(parentFunc, newHelper);
                    return inserted.first->second;
                }
                return it->second;
            }

            std::map<Operation *, SIRNamingHelper> helpers;
            SIRNamingHelper fallback;
        };

        // MemRef elimination conversions
        class ConvertMemRefLoadOp : public OpConversionPattern<mlir::memref::LoadOp>
        {
        public:
            ConvertMemRefLoadOp(const TypeConverter &typeConverter, MLIRContext *context,
                                MemRefNamingCache &namingCache,
                                PatternBenefit benefit = 1)
                : OpConversionPattern(typeConverter, context, benefit),
                  namingCache(&namingCache) {}

            LogicalResult matchAndRewrite(
                mlir::memref::LoadOp op,
                typename mlir::memref::LoadOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;

        private:
            MemRefNamingCache *namingCache;
        };

        class NormalizeNarrowErrorUnionMemRefLoadOp : public OpRewritePattern<mlir::memref::LoadOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                mlir::memref::LoadOp op,
                PatternRewriter &rewriter) const override;
        };

        class ConvertMemRefStoreOp : public OpConversionPattern<mlir::memref::StoreOp>
        {
        public:
            ConvertMemRefStoreOp(const TypeConverter &typeConverter, MLIRContext *context,
                                 MemRefNamingCache &namingCache,
                                 PatternBenefit benefit = 1)
                : OpConversionPattern(typeConverter, context, benefit),
                  namingCache(&namingCache) {}

            LogicalResult matchAndRewrite(
                mlir::memref::StoreOp op,
                typename mlir::memref::StoreOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;

        private:
            MemRefNamingCache *namingCache;
        };

        class NormalizeNarrowErrorUnionMemRefStoreOp : public OpRewritePattern<mlir::memref::StoreOp>
        {
        public:
            using OpRewritePattern::OpRewritePattern;

            LogicalResult matchAndRewrite(
                mlir::memref::StoreOp op,
                PatternRewriter &rewriter) const override;
        };

        class ConvertMemRefAllocOp : public OpConversionPattern<mlir::memref::AllocaOp>
        {
        public:
            ConvertMemRefAllocOp(const TypeConverter &typeConverter, MLIRContext *context,
                                 MemRefNamingCache &namingCache,
                                 PatternBenefit benefit = 1)
                : OpConversionPattern(typeConverter, context, benefit),
                  namingCache(&namingCache) {}

            LogicalResult matchAndRewrite(
                mlir::memref::AllocaOp op,
                typename mlir::memref::AllocaOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;

        private:
            MemRefNamingCache *namingCache;
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
