#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "OraDialect.h"
#include "SIR/SIRDialect.h"

namespace mlir
{
    namespace ora
    {

        // Forward declarations
        class OraToSIRTypeConverter;

        // -----------------------------------------------------------------------------
        // Utility: Lookup global slot index from ora.global operation
        // -----------------------------------------------------------------------------
        inline uint64_t computeGlobalSlot(StringRef name, Operation *op)
        {
            // Get the module to look up the ora.global operation
            ModuleOp module = op->getParentOfType<ModuleOp>();
            if (!module)
            {
                // Fallback to hash if we can't find the module
                return std::hash<std::string>{}(name.str());
            }

            // Look up the ora.global operation by name using SymbolTable
            SymbolTable symbolTable(module);
            auto globalOp = symbolTable.lookup<ora::GlobalOp>(name);
            if (!globalOp)
            {
                // Fallback to hash if global not found
                return std::hash<std::string>{}(name.str());
            }

            // Check if the global has a slot index attribute
            auto slotAttr = globalOp->getAttrOfType<IntegerAttr>("ora.slot_index");
            if (slotAttr)
            {
                return slotAttr.getUInt();
            }

            // If no slot index attribute, fall back to a stable hash of the name.
            return std::hash<std::string>{}(name.str());
        }

        // Storage operation conversions
        class ConvertSLoadOp : public OpConversionPattern<ora::SLoadOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::SLoadOp op,
                typename ora::SLoadOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertSStoreOp : public OpConversionPattern<ora::SStoreOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::SStoreOp op,
                typename ora::SStoreOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertTLoadOp : public OpConversionPattern<ora::TLoadOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::TLoadOp op,
                typename ora::TLoadOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertTStoreOp : public OpConversionPattern<ora::TStoreOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::TStoreOp op,
                typename ora::TStoreOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertGlobalOp : public OpConversionPattern<ora::GlobalOp>
        {
        public:
            using OpConversionPattern<ora::GlobalOp>::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::GlobalOp op,
                typename ora::GlobalOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        // Map operation conversions
        class ConvertMapGetOp : public OpConversionPattern<ora::MapGetOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::MapGetOp op,
                typename ora::MapGetOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertMapStoreOp : public OpConversionPattern<ora::MapStoreOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::MapStoreOp op,
                typename ora::MapStoreOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

    } // namespace ora
} // namespace mlir
