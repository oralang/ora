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
            auto fallbackSlotFromName = [](StringRef key) -> std::optional<uint64_t> {
                if (key.empty())
                    return std::nullopt;
                // Deterministic FNV-1a hash for function-scoped storage vars that
                // don't materialize as ora.global symbols/module slot attrs.
                uint64_t hash = 1469598103934665603ULL;
                for (unsigned char c : key.bytes())
                {
                    hash ^= static_cast<uint64_t>(c);
                    hash *= 1099511628211ULL;
                }
                if (hash == 0)
                    hash = 1;
                return hash;
            };

            // Get the module to look up the ora.global operation
            ModuleOp module = op->getParentOfType<ModuleOp>();
            if (!module)
            {
                return fallbackSlotFromName(name);
            }

            // Look up the ora.global operation by name using SymbolTable
            SymbolTable symbolTable(module);
            auto globalOp = symbolTable.lookup<ora::GlobalOp>(name);
            if (!globalOp)
            {
                auto slotsAttr = module->getAttrOfType<DictionaryAttr>("ora.global_slots");
                if (!slotsAttr)
                    return fallbackSlotFromName(name);
                if (auto slotAttr = slotsAttr.get(name))
                {
                    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(slotAttr))
                        return intAttr.getUInt();
                }
                return fallbackSlotFromName(name);
            }

            // Check if the global has a slot index attribute
            auto slotAttr = globalOp->getAttrOfType<IntegerAttr>("ora.slot_index");
            if (slotAttr)
            {
                return slotAttr.getUInt();
            }

            return fallbackSlotFromName(name);
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

        class ConvertTStoreGuardOp : public OpConversionPattern<ora::TStoreGuardOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(
                ora::TStoreGuardOp op,
                typename ora::TStoreGuardOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertLockOp : public OpConversionPattern<ora::LockOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(
                ora::LockOp op,
                typename ora::LockOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        class ConvertUnlockOp : public OpConversionPattern<ora::UnlockOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;
            LogicalResult matchAndRewrite(
                ora::UnlockOp op,
                typename ora::UnlockOp::Adaptor adaptor,
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

        class ConvertTensorInsertOp : public OpConversionPattern<mlir::tensor::InsertOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                mlir::tensor::InsertOp op,
                typename mlir::tensor::InsertOp::Adaptor adaptor,
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
