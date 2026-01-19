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

            // ora.bool → sir.u256 (boolean values represented as 0/1 on EVM stack)
            addConversion([&](ora::BoolType type) -> Type
                          { return sir::U256Type::get(type.getDialect().getContext()); });

            // ora.address → sir.u256 (addresses stored as 256-bit on EVM stack)
            addConversion([&](ora::AddressType type) -> Type
                          { return sir::U256Type::get(type.getDialect().getContext()); });

            // ora.string/ora.bytes → sir.ptr<1> (dynamic byte array)
            addConversion([&](ora::StringType type) -> Type
                          { return sir::PtrType::get(type.getDialect().getContext(), /*addrSpace*/ 1); });
            addConversion([&](ora::BytesType type) -> Type
                          { return sir::PtrType::get(type.getDialect().getContext(), /*addrSpace*/ 1); });

            // ora.enum → sir.u256 (use repr type width, but SIR is u256)
            addConversion([](ora::EnumType type) -> Type
                          {
                auto *ctx = type.getDialect().getContext();
                if (!ctx) {
                    return Type();
                }
                return sir::U256Type::get(ctx); });

            // ora.error_union<T> → sir.u256 (tagged payload encoding)
            addConversion([](ora::ErrorUnionType type) -> Type
                          {
                auto *ctx = type.getDialect().getContext();
                if (!ctx) {
                    return Type();
                }
                return sir::U256Type::get(ctx); });

            // ora.map → sir.u256 (map values live in storage, represent as u256 handles)
            addConversion([](ora::MapType type) -> Type
                          {
                auto *ctx = type.getDialect().getContext();
                if (!ctx) {
                    return Type();
                }
                return sir::U256Type::get(ctx); });

            // ora.struct → sir.ptr<1> (structs lowered to packed memory)
            addConversion([&](ora::StructType type) -> Type
                          { return sir::PtrType::get(type.getDialect().getContext(), /*addrSpace*/ 1); });

            // refinement types → base type (erased to underlying representation)
            addConversion([this](ora::MinValueType type) -> Type
                          { return this->convertType(type.getBaseType()); });
            addConversion([this](ora::MaxValueType type) -> Type
                          { return this->convertType(type.getBaseType()); });
            addConversion([this](ora::InRangeType type) -> Type
                          { return this->convertType(type.getBaseType()); });
            addConversion([this](ora::ScaledType type) -> Type
                          { return this->convertType(type.getBaseType()); });
            addConversion([this](ora::ExactType type) -> Type
                          { return this->convertType(type.getBaseType()); });
            addConversion([](ora::NonZeroAddressType type) -> Type
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
            auto *ctx = type.getDialect().getContext();
            if (!ctx)
                return Type();

            auto u256 = [&]() -> Type { return sir::U256Type::get(ctx); };

            // Explicitly handle ora scalar/refinement types here as fallback
            if (isa<ora::IntegerType>(type))
                return u256();
            if (isa<ora::EnumType>(type))
                return u256();
            if (isa<ora::AddressType>(type))
                return u256();
            if (isa<ora::MapType>(type))
                return u256();
            if (isa<ora::ErrorUnionType>(type))
                return u256();
            if (isa<ora::StructType>(type))
                return sir::PtrType::get(ctx, /*addrSpace*/ 1);
            if (isa<ora::StringType>(type))
                return sir::PtrType::get(ctx, /*addrSpace*/ 1);
            if (isa<ora::BytesType>(type))
                return sir::PtrType::get(ctx, /*addrSpace*/ 1);
            if (isa<ora::NonZeroAddressType>(type))
                return u256();
            if (isa<ora::BoolType>(type))
                return u256();

            if (auto minType = dyn_cast<ora::MinValueType>(type))
            {
                Type base = minType.getBaseType();
                if (base.getDialect().getNamespace() == "ora")
                {
                    if (isa<ora::BoolType>(base))
                        return u256();
                    return u256();
                }
                return this->convertType(base);
            }
            if (auto maxType = dyn_cast<ora::MaxValueType>(type))
            {
                Type base = maxType.getBaseType();
                if (base.getDialect().getNamespace() == "ora")
                {
                    if (isa<ora::BoolType>(base))
                        return u256();
                    return u256();
                }
                return this->convertType(base);
            }
            if (auto rangeType = dyn_cast<ora::InRangeType>(type))
            {
                Type base = rangeType.getBaseType();
                if (base.getDialect().getNamespace() == "ora")
                {
                    if (isa<ora::BoolType>(base))
                        return u256();
                    return u256();
                }
                return this->convertType(base);
            }
            if (auto scaledType = dyn_cast<ora::ScaledType>(type))
            {
                Type base = scaledType.getBaseType();
                if (base.getDialect().getNamespace() == "ora")
                {
                    if (isa<ora::BoolType>(base))
                        return u256();
                    return u256();
                }
                return this->convertType(base);
            }
            if (auto exactType = dyn_cast<ora::ExactType>(type))
            {
                Type base = exactType.getBaseType();
                if (base.getDialect().getNamespace() == "ora")
                {
                    if (isa<ora::BoolType>(base))
                        return u256();
                    return u256();
                }
                return this->convertType(base);
            }

            // Other Ora types not handled
            llvm::errs() << "[OraToSIRTypeConverter] Fallback: Unhandled Ora type: " << type << "\n";
            return Type();   // Unhandled Ora type
        }

        // ---------------------------------------------------------------------
        // Keep MLIR builtin integer types as-is (used for indexing/arithmetic)
        // ---------------------------------------------------------------------
        if (auto intType = dyn_cast<mlir::IntegerType>(type))
        {
            if (intType.getWidth() <= 256)
            {
                return type;
            }
        }

        // ---------------------------------------------------------------------
        // Allow a small whitelist of MLIR builtin types
        // ---------------------------------------------------------------------
        if (isa<NoneType>(type) || isa<IndexType>(type))
            return type;

        // ---------------------------------------------------------------------
        // Tensor types: keep as-is for now (tensors are storage array placeholders)
        // ---------------------------------------------------------------------
        if (auto tensorType = dyn_cast<mlir::RankedTensorType>(type))
        {
            return tensorType;
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

                                         auto makeMask = [&](unsigned width) -> Value {
                                             if (width >= 256)
                                                 return Value();
                                             if (width == 64)
                                             {
                                                 auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                                 auto attr = mlir::IntegerAttr::get(ui64Type, std::numeric_limits<uint64_t>::max());
                                                 return builder.create<sir::ConstOp>(loc, sir::U256Type::get(builder.getContext()), attr);
                                             }
                                             if (width < 64)
                                             {
                                                 auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                                 uint64_t mask = (width == 0) ? 0ULL : ((1ULL << width) - 1ULL);
                                                 auto attr = mlir::IntegerAttr::get(ui64Type, mask);
                                                 return builder.create<sir::ConstOp>(loc, sir::U256Type::get(builder.getContext()), attr);
                                             }
                                             return Value();
                                         };

                                        // Convert integer types to sir.u256
                                        if (auto intType = dyn_cast<mlir::IntegerType>(input.getType()))
                                        {
                                            if (intType.getWidth() <= 256 && llvm::isa<sir::U256Type>(type))
                                            {
                                                Value value = builder.create<sir::BitcastOp>(loc, type, input);
                                                if (intType.getWidth() < 256)
                                                {
                                                    if (intType.isSigned())
                                                    {
                                                        auto u256 = sir::U256Type::get(builder.getContext());
                                                        auto ui64 = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                                        auto bAttr = mlir::IntegerAttr::get(ui64, intType.getWidth() / 8);
                                                        Value bConst = builder.create<sir::ConstOp>(loc, u256, bAttr);
                                                        return builder.create<sir::SignExtendOp>(loc, u256, bConst, value);
                                                    }
                                                    if (Value mask = makeMask(intType.getWidth()))
                                                    {
                                                        return builder.create<sir::AndOp>(loc, type, value, mask);
                                                    }
                                                }
                                                return value;
                                            }
                                        }

                                        // Convert sir.u256 to sir.ptr<1> when required by target types
                                        if (llvm::isa<sir::PtrType>(type) && llvm::isa<sir::U256Type>(input.getType()))
                                        {
                                            return builder.create<sir::BitcastOp>(loc, type, input);
                                        }

                                         auto maskForWidth = [&](unsigned width, bool is_signed) -> Value {
                                             if (width >= 256)
                                                 return Value();
                                             if (is_signed)
                                             {
                                                 auto u256 = sir::U256Type::get(builder.getContext());
                                                 auto ui64 = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                                 auto bAttr = mlir::IntegerAttr::get(ui64, width / 8);
                                                 Value bConst = builder.create<sir::ConstOp>(loc, u256, bAttr);
                                                 return builder.create<sir::SignExtendOp>(loc, u256, bConst, input);
                                             }
                                             if (Value mask = makeMask(width))
                                             {
                                                 return builder.create<sir::AndOp>(loc, type, input, mask);
                                             }
                                             return Value();
                                         };

                                         auto getBaseIntInfo = [&](Type t, unsigned &width_out, bool &signed_out) -> bool {
                                             if (auto oraInt = dyn_cast<ora::IntegerType>(t))
                                             {
                                                 width_out = oraInt.getWidth();
                                                 signed_out = oraInt.getIsSigned();
                                                 return true;
                                             }
                                             if (auto builtinInt = dyn_cast<mlir::IntegerType>(t))
                                             {
                                                 width_out = builtinInt.getWidth();
                                                 signed_out = builtinInt.isSigned();
                                                 return true;
                                             }
                                             return false;
                                         };

                                         // Convert ora.int / ora.enum / ora refinement types to sir.u256
                                         if (llvm::isa<sir::U256Type>(type))
                                         {
                                            if (llvm::isa<ora::IntegerType, ora::EnumType, ora::AddressType, ora::MinValueType, ora::MaxValueType,
                                                          ora::InRangeType, ora::ScaledType, ora::ExactType, ora::NonZeroAddressType>(input.getType()))
                                            {
                                                Value value = builder.create<sir::BitcastOp>(loc, type, input);
                                                if (auto enumTy = dyn_cast<ora::EnumType>(input.getType()))
                                                 {
                                                     if (auto reprInt = dyn_cast<mlir::IntegerType>(enumTy.getReprType()))
                                                     {
                                                         if (reprInt.getWidth() < 256)
                                                         {
                                                            if (Value mask = makeMask(reprInt.getWidth()))
                                                            {
                                                                return builder.create<sir::AndOp>(loc, type, value, mask);
                                                            }
                                                         }
                                                     }
                                                 }
                                                 if (auto minTy = dyn_cast<ora::MinValueType>(input.getType()))
                                                 {
                                                     unsigned width = 0;
                                                     bool is_signed = false;
                                                     if (getBaseIntInfo(minTy.getBaseType(), width, is_signed))
                                                     {
                                                         if (Value adjusted = maskForWidth(width, is_signed))
                                                             return adjusted;
                                                     }
                                                 }
                                                 if (auto maxTy = dyn_cast<ora::MaxValueType>(input.getType()))
                                                 {
                                                     unsigned width = 0;
                                                     bool is_signed = false;
                                                     if (getBaseIntInfo(maxTy.getBaseType(), width, is_signed))
                                                     {
                                                         if (Value adjusted = maskForWidth(width, is_signed))
                                                             return adjusted;
                                                     }
                                                 }
                                                 if (auto rangeTy = dyn_cast<ora::InRangeType>(input.getType()))
                                                 {
                                                     unsigned width = 0;
                                                     bool is_signed = false;
                                                     if (getBaseIntInfo(rangeTy.getBaseType(), width, is_signed))
                                                     {
                                                         if (Value adjusted = maskForWidth(width, is_signed))
                                                             return adjusted;
                                                     }
                                                 }
                                                 if (auto scaledTy = dyn_cast<ora::ScaledType>(input.getType()))
                                                 {
                                                     unsigned width = 0;
                                                     bool is_signed = false;
                                                     if (getBaseIntInfo(scaledTy.getBaseType(), width, is_signed))
                                                     {
                                                         if (Value adjusted = maskForWidth(width, is_signed))
                                                             return adjusted;
                                                     }
                                                 }
                                                 if (auto exactTy = dyn_cast<ora::ExactType>(input.getType()))
                                                 {
                                                     unsigned width = 0;
                                                     bool is_signed = false;
                                                     if (getBaseIntInfo(exactTy.getBaseType(), width, is_signed))
                                                     {
                                                         if (Value adjusted = maskForWidth(width, is_signed))
                                                             return adjusted;
                                                     }
                                                 }
                                                 if (auto oraInt = dyn_cast<ora::IntegerType>(input.getType()))
                                                 {
                                                     const bool is_signed = oraInt.getIsSigned();
                                                     const unsigned width = oraInt.getWidth();
                                                     if (width < 256)
                                                     {
                                                         if (is_signed)
                                                         {
                                                             Value adjusted = maskForWidth(width, is_signed);
                                                             if (adjusted)
                                                                 return adjusted;
                                                         }
                                                         if (Value mask = makeMask(width))
                                                         {
                                                             return builder.create<sir::AndOp>(loc, type, value, mask);
                                                         }
                                                     }
                                                 }
                                                 return value;
                                             }
                                         }

                                         // Convert sir.u256 to integer types
                                         if (auto intType = dyn_cast<mlir::IntegerType>(type))
                                         {
                                             if (intType.getWidth() <= 256 && llvm::isa<sir::U256Type>(input.getType()))
                                             {
                                                 if (intType.isSigned())
                                                 {
                                                     auto u256 = sir::U256Type::get(builder.getContext());
                                                     auto ui64 = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                                     auto bAttr = mlir::IntegerAttr::get(ui64, intType.getWidth() / 8);
                                                     Value bConst = builder.create<sir::ConstOp>(loc, u256, bAttr);
                                                     Value extended = builder.create<sir::SignExtendOp>(loc, u256, bConst, input);
                                                     return builder.create<sir::BitcastOp>(loc, type, extended);
                                                 }
                                                 Value value = builder.create<sir::BitcastOp>(loc, type, input);
                                                 if (intType.getWidth() < 256)
                                                 {
                                                     if (Value mask = makeMask(intType.getWidth()))
                                                     {
                                                         auto u256 = sir::U256Type::get(builder.getContext());
                                                         Value masked = builder.create<sir::AndOp>(loc, u256, input, mask);
                                                         return builder.create<sir::BitcastOp>(loc, type, masked);
                                                     }
                                                 }
                                                 return value;
                                             }
                                         }

                                         return Value(); // Don't handle other cases
                                     });

            // We should NEVER materialize from SIR back to Ora - this indicates a bug
            // Return nullptr to indicate we can't materialize (will cause conversion to fail)
            addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value
                                     {
                                         auto makeMask = [&](unsigned width) -> Value {
                                             if (width >= 256)
                                                 return Value();
                                             if (width == 64)
                                             {
                                                 auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                                 auto attr = mlir::IntegerAttr::get(ui64Type, std::numeric_limits<uint64_t>::max());
                                                 return builder.create<sir::ConstOp>(loc, sir::U256Type::get(builder.getContext()), attr);
                                             }
                                             if (width < 64)
                                             {
                                                 auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                                 uint64_t mask = (width == 0) ? 0ULL : ((1ULL << width) - 1ULL);
                                                 auto attr = mlir::IntegerAttr::get(ui64Type, mask);
                                                 return builder.create<sir::ConstOp>(loc, sir::U256Type::get(builder.getContext()), attr);
                                             }
                                             return Value();
                                         };

                                         if (inputs.size() != 1)
                                             return Value();
                                         Value input = inputs[0];

                                         if (auto intType = dyn_cast<mlir::IntegerType>(type))
                                         {
                                             if (intType.getWidth() <= 256 && llvm::isa<sir::U256Type>(input.getType()))
                                             {
                                                 if (intType.isSigned() && intType.getWidth() < 256)
                                                 {
                                                     auto u256 = sir::U256Type::get(builder.getContext());
                                                     auto ui64 = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                                     auto bAttr = mlir::IntegerAttr::get(ui64, intType.getWidth() / 8);
                                                     Value bConst = builder.create<sir::ConstOp>(loc, u256, bAttr);
                                                     Value extended = builder.create<sir::SignExtendOp>(loc, u256, bConst, input);
                                                     return builder.create<sir::BitcastOp>(loc, type, extended);
                                                 }
                                                 if (intType.getWidth() < 256)
                                                 {
                                                     if (Value mask = makeMask(intType.getWidth()))
                                                     {
                                                         auto u256 = sir::U256Type::get(builder.getContext());
                                                         Value masked = builder.create<sir::AndOp>(loc, u256, input, mask);
                                                         return builder.create<sir::BitcastOp>(loc, type, masked);
                                                     }
                                                 }
                                                 return builder.create<sir::BitcastOp>(loc, type, input);
                                             }
                                         }

                                        // If trying to materialize to an Ora type, this is an error
                                        if (type.getDialect().getNamespace() == "ora")
                                        {
                                            if (llvm::isa<ora::AddressType, ora::NonZeroAddressType, ora::MinValueType, ora::MaxValueType,
                                                          ora::InRangeType, ora::ScaledType, ora::ExactType>(type) &&
                                                llvm::isa<sir::U256Type>(input.getType()))
                                            {
                                                llvm::errs() << "[OraToSIRTypeConverter] NOTE: Materializing sir.u256 -> " << type
                                                             << " via UnrealizedConversionCast (refinement bridge)\n";
                                                auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc, type, input);
                                                return cast.getResult(0);
                                            }
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
