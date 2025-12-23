#include "OraToSIRTypeConverter.h"

#include "Ora/OraDialect.h" // Includes OraTypes.h.inc
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace ora;
using namespace sir;

namespace mlir
{
    namespace ora
    {

        OraToSIRTypeConverter::OraToSIRTypeConverter()
        {

            // =========================================================================
            // 1. Explicit Ora → SIR type conversions
            // =========================================================================

            // ora.int<N> → sir.u256  (for now: ABI-level erasure)
            addConversion([](ora::IntegerType type) -> Type
                          {
                llvm::errs() << "[OraToSIRTypeConverter] Converting ora::IntegerType: " << type << "\n";
                auto *ctx = type.getDialect().getContext();
                if (!ctx) {
                    llvm::errs() << "[OraToSIRTypeConverter] ERROR: null context\n";
                    return Type();
                }
                auto result = sir::U256Type::get(ctx);
                llvm::errs() << "[OraToSIRTypeConverter] Result: " << result << "\n";
                return result; });

            // ora.bool → MLIR i1  (SIR has no custom boolean type)
            addConversion([&](ora::BoolType type) -> Type
                          { return mlir::IntegerType::get(type.getDialect().getContext(), 1, mlir::IntegerType::Unsigned); });

            // ora.address → sir.u256 (addresses stored as 256-bit on EVM stack)
            addConversion([&](ora::AddressType type) -> Type
                          { return sir::U256Type::get(type.getDialect().getContext()); });

            // =========================================================================
            // 2. Fallback conversion
            // =========================================================================

            addConversion([&](Type type) -> Type
                          {
        // ---------------------------------------------------------------------
        // Already a SIR type → keep it
        // ---------------------------------------------------------------------
        if (isa<sir::U256Type, sir::PtrType>(type))
            return type;

        // ---------------------------------------------------------------------
        // Handle Ora types explicitly (in case specific conversions didn't match)
        // ---------------------------------------------------------------------
        if (type.getDialect().getNamespace() == "ora")
        {
            // Explicitly handle ora::IntegerType here as fallback
            if (isa<ora::IntegerType>(type))
            {
                auto *ctx = type.getDialect().getContext();
                if (!ctx)
                    return Type();
                return sir::U256Type::get(ctx);
            }
            // Other Ora types not handled
            llvm::errs() << "[OraToSIRTypeConverter] Fallback: Unhandled Ora type: " << type << "\n";
            return Type();   // Unhandled Ora type
        }

        // ---------------------------------------------------------------------
        // Convert MLIR builtin i256 to sir.u256 (EVM uses 256-bit values)
        // ---------------------------------------------------------------------
        if (auto intType = dyn_cast<mlir::IntegerType>(type))
        {
            if (intType.getWidth() == 256)
            {
                auto *ctx = type.getContext();
                if (!ctx)
                    return Type();
                return sir::U256Type::get(ctx);
            }
        }

        // ---------------------------------------------------------------------
        // Allow a small whitelist of MLIR builtin types
        // ---------------------------------------------------------------------
        if (isa<NoneType>(type) || isa<IndexType>(type))
            return type;

        // ---------------------------------------------------------------------
        // Tensor types: convert element types (used for storage arrays)
        // Tensors are placeholders for storage arrays in Ora, but we need to
        // convert them so patterns can match operations that use them
        // ---------------------------------------------------------------------
        if (auto tensorType = dyn_cast<mlir::RankedTensorType>(type))
        {
            Type elementType = tensorType.getElementType();
            Type convertedElementType = this->convertType(elementType);
            if (!convertedElementType || convertedElementType == elementType)
                return type; // No conversion needed or failed
            // Create new tensor type with converted element type
            return mlir::RankedTensorType::get(
                tensorType.getShape(),
                convertedElementType);
        }

        // ---------------------------------------------------------------------
        // MemRef types: convert to pointer type (memref.alloca becomes sir.malloc)
        // MemRef operations will be converted to SIR pointer operations
        // ---------------------------------------------------------------------
        if (auto memrefType = dyn_cast<mlir::MemRefType>(type))
        {
            // Convert memref to pointer type (address space 1 = memory)
            auto *ctx = type.getContext();
            if (!ctx)
                return Type();
            return sir::PtrType::get(ctx, /*addrSpace*/ 1);
        }

        // ---------------------------------------------------------------------
        // Everything else is illegal in SIR
        // ---------------------------------------------------------------------
        return Type(); });

            // =========================================================================
            // 3. Function signature conversion
            // =========================================================================

            addConversion([&](mlir::FunctionType fn) -> Type
                          {
        SmallVector<Type> inputs;
        SmallVector<Type> results;

        for (Type arg : fn.getInputs()) {
            Type converted = this->convertType(arg);
            if (!converted)
                return Type();
            inputs.push_back(converted);
        }

        for (Type res : fn.getResults()) {
            Type converted = this->convertType(res);
            if (!converted)
                return Type();
            results.push_back(converted);
        }

        return mlir::FunctionType::get(fn.getContext(), inputs, results); });

            // =========================================================================
            // 4. Materialization hooks
            // =========================================================================

            // Target materialization: convert i256 to sir.u256 when needed
            addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value
                                     {
                                         if (inputs.size() != 1)
                                             return Value();

                                         Value input = inputs[0];

                                         // Convert i256 to sir.u256
                                         if (auto intType = dyn_cast<mlir::IntegerType>(input.getType()))
                                         {
                                             if (intType.getWidth() == 256 && llvm::isa<sir::U256Type>(type))
                                             {
                                                 // Use bitcast to convert i256 to sir.u256
                                                 return builder.create<sir::BitcastOp>(loc, type, input);
                                             }
                                         }

                                         return Value(); // Don't handle other cases
                                     });

            // We should NEVER materialize from SIR back to Ora - this indicates a bug
            // Return nullptr to indicate we can't materialize (will cause conversion to fail)
            addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value
                                     {
                                         // If trying to materialize to an Ora type, this is an error
                                         if (type.getDialect().getNamespace() == "ora")
                                         {
                                             llvm::errs() << "[OraToSIRTypeConverter] ERROR: Attempted to materialize from SIR to Ora type: " << type << "\n";
                                             llvm::errs() << "[OraToSIRTypeConverter] This indicates an unconverted operation still expects Ora types.\n";
                                             llvm::errs() << "[OraToSIRTypeConverter] Location: " << loc << "\n";
                                             llvm::errs().flush();
                                             return Value(); // Return null value to fail materialization
                                         }
                                         return Value(); // Return null - don't handle other cases
                                     });
        }

    } // namespace ora
} // namespace mlir