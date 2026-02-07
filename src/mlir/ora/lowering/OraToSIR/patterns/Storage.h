#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "OraDialect.h"
#include "SIR/SIRDialect.h"
#include <optional>

namespace mlir
{
    namespace ora
    {

        // Forward declarations
        class OraToSIRTypeConverter;

        // -----------------------------------------------------------------------------
        // Utility: Lookup global slot index from ora.global operation
        // -----------------------------------------------------------------------------
        inline std::optional<uint64_t> computeGlobalSlot(StringRef name, Operation *op)
        {
            // Get the module to look up the ora.global operation
            ModuleOp module = op->getParentOfType<ModuleOp>();
            if (!module)
            {
                return std::nullopt;
            }

            // Look up the ora.global operation by name using SymbolTable
            SymbolTable symbolTable(module);
            auto globalOp = symbolTable.lookup<ora::GlobalOp>(name);
            if (!globalOp)
            {
                auto ctx = module.getContext();
                auto slotsAttr = module->getAttrOfType<DictionaryAttr>("ora.global_slots");
                uint64_t nextSlot = 0;
                SmallVector<NamedAttribute, 8> entries;
                if (slotsAttr)
                {
                    for (auto it : slotsAttr)
                    {
                        entries.push_back(it);
                        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(it.getValue()))
                        {
                            uint64_t v = intAttr.getUInt();
                            if (v >= nextSlot)
                                nextSlot = v + 1;
                        }
                    }
                    if (auto slotAttr = slotsAttr.get(name))
                    {
                        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(slotAttr))
                            return intAttr.getUInt();
                        return std::nullopt;
                    }
                }

                // Fallback: synthesize a slot for undeclared storage names.
                auto ui64Ty = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
                auto nameAttr = StringAttr::get(ctx, name);
                auto slotAttr = IntegerAttr::get(ui64Ty, nextSlot);
                entries.push_back(NamedAttribute(nameAttr, slotAttr));
                module->setAttr("ora.global_slots", DictionaryAttr::get(ctx, entries));
                return nextSlot;
            }

            // Check if the global has a slot index attribute
            auto slotAttr = globalOp->getAttrOfType<IntegerAttr>("ora.slot_index");
            if (slotAttr)
            {
                return slotAttr.getUInt();
            }

            return std::nullopt;
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

        class ConvertTensorExtractOp : public OpConversionPattern<mlir::tensor::ExtractOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::tensor::ExtractOp op,
                typename mlir::tensor::ExtractOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertTensorDimOp : public OpConversionPattern<mlir::tensor::DimOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::tensor::DimOp op,
                typename mlir::tensor::DimOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

    } // namespace ora
} // namespace mlir

/// Clear the static map hash cache between pass invocations.
void clearMapHashCache();

