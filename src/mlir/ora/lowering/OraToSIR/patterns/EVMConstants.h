#pragma once

/// EVM / SIR ABI constants used throughout the Ora→SIR lowering patterns.
/// Centralising these avoids magic-number repetition and makes ABI changes
/// easy to grep for.

namespace mlir::ora::evm {

/// Bit-width of an EVM word.
constexpr unsigned kWordBits = 256;

/// Byte-size of an EVM word (256 / 8).
constexpr unsigned kWordBytes = 32;

/// Byte-size of two EVM words (used for wide error-union tag+payload).
constexpr unsigned kDoubleWordBytes = 64;

/// Bit-width of an EVM address.
constexpr unsigned kAddressBits = 160;

/// Bit-width of a uint64 helper type.
constexpr unsigned kU64Bits = 64;

} // namespace mlir::ora::evm
