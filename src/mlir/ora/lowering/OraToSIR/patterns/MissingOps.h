#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "OraDialect.h"
#include "SIR/SIRDialect.h"

namespace mlir
{
    namespace ora
    {

        /// ora.refinement_guard → sir.cond_br to revert block.
        class ConvertRefinementGuardOp : public OpConversionPattern<ora::RefinementGuardOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::RefinementGuardOp op,
                typename ora::RefinementGuardOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        /// ora.power → sir.exp.
        class ConvertPowerOp : public OpConversionPattern<ora::PowerOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::PowerOp op,
                typename ora::PowerOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        /// ora.mload → sir.load (via malloc + store variable-name hash, then load).
        /// For named memory slots, we treat the variable name as a memory pointer.
        class ConvertMLoadOp : public OpConversionPattern<ora::MLoadOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::MLoadOp op,
                typename ora::MLoadOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        /// ora.mstore → sir.store.
        class ConvertMStoreOp : public OpConversionPattern<ora::MStoreOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::MStoreOp op,
                typename ora::MStoreOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        /// ora.mload8 → sir.load8.
        class ConvertMLoad8Op : public OpConversionPattern<ora::MLoad8Op>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::MLoad8Op op,
                typename ora::MLoad8Op::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        /// ora.mstore8 → sir.store8.
        class ConvertMStore8Op : public OpConversionPattern<ora::MStore8Op>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::MStore8Op op,
                typename ora::MStore8Op::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        /// ora.enum_constant → sir.const (variant ordinal).
        class ConvertEnumConstantOp : public OpConversionPattern<ora::EnumConstantOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::EnumConstantOp op,
                typename ora::EnumConstantOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        /// ora.struct_field_store → sir.addptr + sir.store.
        class ConvertStructFieldStoreOp : public OpConversionPattern<ora::StructFieldStoreOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::StructFieldStoreOp op,
                typename ora::StructFieldStoreOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        /// ora.destructure → pass through (identity or field extract).
        class ConvertDestructureOp : public OpConversionPattern<ora::DestructureOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::DestructureOp op,
                typename ora::DestructureOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

        /// ora.immutable → passthrough (result = value operand).
        class ConvertImmutableOp : public OpConversionPattern<ora::ImmutableOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::ImmutableOp op,
                typename ora::ImmutableOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };

    } // namespace ora
} // namespace mlir
