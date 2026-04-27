#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir
{
    namespace ora
    {
        // Materialization-kind constants attached to UnrealizedConversionCastOp
        // by the OraToSIR pass. The kind tells the cast-cleanup phase how to
        // resolve a specific cast: forward an operand, decode a packed carrier,
        // or load through a compiler-managed handle.
        //
        // The attribute name itself is `kOraMaterializationKindAttr`; the
        // string values stored in that attribute live in `mat_kind`.
        namespace mat_kind
        {

            // MLIR attribute name on the UnrealizedConversionCastOp.
            inline constexpr ::llvm::StringLiteral kAttrName{"ora.materialization_kind"};

            // Type-erased forwards: cast operand and result share the same
            // underlying SIR representation; the cast just exists to bridge
            // dialect-level type names.
            inline constexpr ::llvm::StringLiteral kPtrView{"ptr_view"};
            inline constexpr ::llvm::StringLiteral kAddressForward{"address_forward"};
            inline constexpr ::llvm::StringLiteral kNoneForward{"none_forward"};
            inline constexpr ::llvm::StringLiteral kPayloadForward{"payload_forward"};
            inline constexpr ::llvm::StringLiteral kIntegerForward{"integer_forward"};

            // Error-union carriers: narrow form is packed (payload<<1)|tag,
            // wide form is split into (tag, payload-carrier).
            inline constexpr ::llvm::StringLiteral kNormalizedErrorUnion{"normalized_error_union"};
            inline constexpr ::llvm::StringLiteral kWideErrorUnionJoin{"wide_error_union_join"};
            inline constexpr ::llvm::StringLiteral kWideErrorUnionSplit{"wide_error_union_split"};

            // ADT carriers: `kNormalizedAdt` wraps the (tag, payload) split;
            // `kAdtHandleView` views a u256 word as a pointer to a 64-byte
            // (tag, payload) handle for ADTs stored in struct fields.
            inline constexpr ::llvm::StringLiteral kNormalizedAdt{"normalized_adt"};
            inline constexpr ::llvm::StringLiteral kAdtHandleView{"adt_handle_view"};

        } // namespace mat_kind

        // Re-export the attribute name in `ora::` for backward compatibility
        // with sites that still reference it via the historical identifier.
        inline constexpr ::llvm::StringLiteral kOraMaterializationKindAttr =
            mat_kind::kAttrName;

    } // namespace ora
} // namespace mlir
