//===- OraGasConstants.h - EVM Gas Cost Constants --------------------===//
//
// This file defines EVM gas cost constants for Ora operations.
// Based on Ethereum Yellow Paper and EIP specifications.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_GAS_CONSTANTS_H
#define ORA_GAS_CONSTANTS_H

#include <cstdint>

namespace mlir
{
    namespace ora
    {
        namespace gas
        {

            // Storage operations (EIP-2929 post-Berlin)
            constexpr int64_t SLOAD_COLD = 2100;    // First access to storage slot
            constexpr int64_t SLOAD_WARM = 100;     // Subsequent access to same slot
            constexpr int64_t SSTORE_NEW = 20000;   // Write to zero slot
            constexpr int64_t SSTORE_UPDATE = 5000; // Update existing slot
            constexpr int64_t SSTORE_REFUND = 4800; // Refund for clearing slot

            // Arithmetic operations
            constexpr int64_t ADD = 3;
            constexpr int64_t MUL = 5;
            constexpr int64_t SUB = 3;
            constexpr int64_t DIV = 5;
            constexpr int64_t MOD = 5;
            constexpr int64_t EXP_BASE = 10;
            constexpr int64_t EXP_BYTE = 50;

            // Comparison and bitwise
            constexpr int64_t LT = 3;
            constexpr int64_t GT = 3;
            constexpr int64_t EQ = 3;
            constexpr int64_t ISZERO = 3;
            constexpr int64_t AND = 3;
            constexpr int64_t OR = 3;
            constexpr int64_t XOR = 3;
            constexpr int64_t NOT = 3;

            // Memory operations
            constexpr int64_t MLOAD = 3;
            constexpr int64_t MSTORE = 3;
            constexpr int64_t MSTORE8 = 3;

            // Hashing
            constexpr int64_t KECCAK256_BASE = 30;
            constexpr int64_t KECCAK256_WORD = 6;

            // Control flow
            constexpr int64_t JUMP = 8;
            constexpr int64_t JUMPI = 10;
            constexpr int64_t JUMPDEST = 1;

            // Function calls
            constexpr int64_t CALL_BASE = 700;
            constexpr int64_t CALL_VALUE_TRANSFER = 9000;
            constexpr int64_t CALL_NEW_ACCOUNT = 25000;

            // For conservative analysis, use worst-case costs
            constexpr int64_t SLOAD_DEFAULT = SLOAD_COLD;
            constexpr int64_t SSTORE_DEFAULT = SSTORE_NEW;

            //===----------------------------------------------------------------------===//
            // EVM Opcode Gas Costs (for reference and future use)
            //===----------------------------------------------------------------------===//

            // Arithmetic opcodes
            constexpr int64_t OP_STOP = 0;
            constexpr int64_t OP_ADD = 3;
            constexpr int64_t OP_MUL = 5;
            constexpr int64_t OP_SUB = 3;
            constexpr int64_t OP_DIV = 5;
            constexpr int64_t OP_SDIV = 5;
            constexpr int64_t OP_MOD = 5;
            constexpr int64_t OP_SMOD = 5;
            constexpr int64_t OP_ADDMOD = 8;
            constexpr int64_t OP_MULMOD = 8;
            constexpr int64_t OP_EXP = 10; // Base cost, +50 per byte of exponent

            // Comparison opcodes
            constexpr int64_t OP_LT = 3;
            constexpr int64_t OP_GT = 3;
            constexpr int64_t OP_SLT = 3;
            constexpr int64_t OP_SGT = 3;
            constexpr int64_t OP_EQ = 3;
            constexpr int64_t OP_ISZERO = 3;

            // Bitwise opcodes
            constexpr int64_t OP_AND = 3;
            constexpr int64_t OP_OR = 3;
            constexpr int64_t OP_XOR = 3;
            constexpr int64_t OP_NOT = 3;
            constexpr int64_t OP_BYTE = 3;
            constexpr int64_t OP_SHL = 3;
            constexpr int64_t OP_SHR = 3;
            constexpr int64_t OP_SAR = 3;

            // Memory opcodes
            constexpr int64_t OP_MLOAD = 3;
            constexpr int64_t OP_MSTORE = 3;
            constexpr int64_t OP_MSTORE8 = 3;
            constexpr int64_t OP_MSIZE = 2;

            // Storage opcodes (already defined above, but included for completeness)
            constexpr int64_t OP_SLOAD = SLOAD_COLD;
            constexpr int64_t OP_SSTORE = SSTORE_NEW;

            // Control flow opcodes
            constexpr int64_t OP_JUMP = 8;
            constexpr int64_t OP_JUMPI = 10;
            constexpr int64_t OP_JUMPDEST = 1;
            constexpr int64_t OP_PC = 2;

            // Stack opcodes
            constexpr int64_t OP_POP = 2;
            constexpr int64_t OP_PUSH1 = 3; // Base cost for PUSH operations
            constexpr int64_t OP_PUSH32 = 3;
            constexpr int64_t OP_DUP1 = 3; // Base cost for DUP operations
            constexpr int64_t OP_DUP16 = 3;
            constexpr int64_t OP_SWAP1 = 3; // Base cost for SWAP operations
            constexpr int64_t OP_SWAP16 = 3;

            // Log opcodes
            constexpr int64_t OP_LOG0 = 375; // Base cost + 375 per topic + 8 per byte of data
            constexpr int64_t OP_LOG1 = 750;
            constexpr int64_t OP_LOG2 = 1125;
            constexpr int64_t OP_LOG3 = 1500;
            constexpr int64_t OP_LOG4 = 1875;
            constexpr int64_t OP_LOG_TOPIC = 375;
            constexpr int64_t OP_LOG_DATA_BYTE = 8;

            // Hashing opcodes
            constexpr int64_t OP_SHA3 = 30; // Base cost + 6 per word
            constexpr int64_t OP_SHA3_WORD = 6;

            // Environmental opcodes
            constexpr int64_t OP_ADDRESS = 2;
            constexpr int64_t OP_BALANCE = 100; // Cold access, 2600 if warm (EIP-2929)
            constexpr int64_t OP_ORIGIN = 2;
            constexpr int64_t OP_CALLER = 2;
            constexpr int64_t OP_CALLVALUE = 2;
            constexpr int64_t OP_CALLDATALOAD = 3;
            constexpr int64_t OP_CALLDATASIZE = 2;
            constexpr int64_t OP_CALLDATACOPY = 3; // Base cost + 3 per word copied
            constexpr int64_t OP_CODESIZE = 2;
            constexpr int64_t OP_CODECOPY = 3; // Base cost + 3 per word copied
            constexpr int64_t OP_GASPRICE = 2;
            constexpr int64_t OP_COINBASE = 2;
            constexpr int64_t OP_TIMESTAMP = 2;
            constexpr int64_t OP_NUMBER = 2;
            constexpr int64_t OP_DIFFICULTY = 2;
            constexpr int64_t OP_GASLIMIT = 2;
            constexpr int64_t OP_CHAINID = 2;
            constexpr int64_t OP_SELFBALANCE = 5;

            // Blockhash opcode
            constexpr int64_t OP_BLOCKHASH = 20;

            // Call opcodes
            constexpr int64_t OP_CALL = 700; // Base cost, additional costs for value transfer, new account, etc.
            constexpr int64_t OP_CALLCODE = 700;
            constexpr int64_t OP_DELEGATECALL = 700;
            constexpr int64_t OP_STATICCALL = 700;
            constexpr int64_t OP_CREATE = 32000;
            constexpr int64_t OP_CREATE2 = 32000;

            // Return opcodes
            constexpr int64_t OP_RETURN = 0;
            constexpr int64_t OP_REVERT = 0;

            // Other opcodes
            constexpr int64_t OP_INVALID = 0;         // Consumes all remaining gas
            constexpr int64_t OP_SELFDESTRUCT = 5000; // + 24000 if new account is created
            constexpr int64_t OP_GAS = 2;

            // Extended opcodes (EIP-1153: Transient Storage)
            constexpr int64_t OP_TLOAD = 100;  // Transient storage load
            constexpr int64_t OP_TSTORE = 100; // Transient storage store

        } // namespace gas
    } // namespace ora
} // namespace mlir

#endif // ORA_GAS_CONSTANTS_H
