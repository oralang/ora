#pragma once

#include <cstdint>

namespace mlir
{
    namespace ora
    {
        namespace adt_helpers
        {

            // The wide ADT/error-union carrier is two consecutive u256 words:
            // tag first, payload second. Addressing differs by carrier kind
            // (memory pointer, storage slot, or memref index), but the word
            // ordering is shared.
            inline constexpr uint64_t kAdtCarrierWordBytes = 32;
            inline constexpr uint64_t kAdtCarrierTagWordIndex = 0;
            inline constexpr uint64_t kAdtCarrierPayloadWordIndex = 1;
            inline constexpr uint64_t kAdtCarrierWordCount = 2;
            inline constexpr uint64_t kAdtCarrierSize = kAdtCarrierWordCount * kAdtCarrierWordBytes;
            inline constexpr uint64_t kAdtPayloadOffset = kAdtCarrierPayloadWordIndex * kAdtCarrierWordBytes;
            inline constexpr uint64_t kAdtStoragePayloadSlotOffset = kAdtCarrierPayloadWordIndex;

        } // namespace adt_helpers
    } // namespace ora
} // namespace mlir
