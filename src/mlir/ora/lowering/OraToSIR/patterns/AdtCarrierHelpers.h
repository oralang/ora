#pragma once

#include "OraDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <utility>

namespace mlir
{
    namespace ora
    {
        namespace adt_helpers
        {

            // The wide ADT carrier is two back-to-back u256 words in memory:
            // tag at offset 0, payload at offset 32, total 64 bytes.
            inline constexpr uint64_t kAdtCarrierSize = 64;
            inline constexpr uint64_t kAdtPayloadOffset = 32;

            // Allocate a 64-byte handle, store `tag` at offset 0 and `payload`
            // at offset 32, and return the base pointer (sir.ptr<1>). Both
            // operands are coerced to sir.u256 if needed.
            ::mlir::Value materializeAdtHandle(::mlir::OpBuilder &builder,
                                               ::mlir::Location loc,
                                               ::mlir::Value tag,
                                               ::mlir::Value payload);

            // View `handle` as a base pointer to a 64-byte ADT carrier and load
            // (tag, payload). Accepts `handle` as sir.ptr<1> or sir.u256
            // (a u256 is bitcast to ptr first). Loads are typed sir.u256.
            std::pair<::mlir::Value, ::mlir::Value>
            loadAdtPartsFromHandle(::mlir::OpBuilder &builder,
                                   ::mlir::Location loc,
                                   ::mlir::Value handle);

            // Aggregate ADT payloads (tuples, structs, strings, bytes, memrefs)
            // ride the wide ADT carrier as a compiler-managed handle pointer.
            bool usesAggregateAdtPayloadHandle(::mlir::Type type);

            // Linear scan; the variant table is small in practice.
            ::mlir::FailureOr<unsigned> getAdtVariantIndex(::mlir::ora::AdtType type, ::mlir::StringRef variantName);

            // Wide ADT carrier pulled out of the (already-adapted) operand pair.
            ::mlir::FailureOr<std::pair<::mlir::Value, ::mlir::Value>>
            getNormalizedAdtPartsFromOperands(::mlir::PatternRewriter &rewriter,
                                              ::mlir::Location loc,
                                              ::mlir::ArrayRef<::mlir::Value> operands);

            // Wide ADT carrier pulled out of a `normalized_adt` materialization cast
            // sitting on the value's defining op.
            ::mlir::FailureOr<std::pair<::mlir::Value, ::mlir::Value>>
            getNormalizedAdtPartsFromValue(::mlir::PatternRewriter &rewriter,
                                           ::mlir::Location loc,
                                           ::mlir::Value value);

            // Bridge the carrier's payload word to the lowered MLIR result type.
            // For aggregate payloads the lowered type must be a sir.ptr; the word
            // is reinterpreted via bitcast.
            ::mlir::LogicalResult
            decodeAdtPayloadFromCarrier(::mlir::Operation *op,
                                        ::mlir::ConversionPatternRewriter &rewriter,
                                        ::mlir::Location loc,
                                        ::mlir::Type payloadType,
                                        ::mlir::Type loweredType,
                                        ::mlir::Value &payload);

            // Shared lowering for ora.adt.tag — replaces the op with the tag word.
            ::mlir::LogicalResult
            convertAdtTagCommon(::mlir::ora::AdtTagOp op,
                                ::mlir::ArrayRef<::mlir::Value> operands,
                                ::mlir::ConversionPatternRewriter &rewriter);

            // Shared lowering for ora.adt.payload — replaces the op with the
            // decoded payload value, bridged to the lowered result type.
            ::mlir::LogicalResult
            convertAdtPayloadCommon(::mlir::ora::AdtPayloadOp op,
                                    ::mlir::ArrayRef<::mlir::Value> operands,
                                    const ::mlir::TypeConverter *typeConverter,
                                    ::mlir::ConversionPatternRewriter &rewriter);

        } // namespace adt_helpers
    } // namespace ora
} // namespace mlir
