#include "OraToSIRTypeConverter.h"

#include "Ora/OraDialect.h" // Includes OraTypes.h.inc
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace ora;
using namespace sir;

namespace
{
    static std::optional<unsigned> getOraBitWidth(Type type)
    {
        if (!type)
            return std::nullopt;

        if (auto builtinInt = llvm::dyn_cast<mlir::IntegerType>(type))
            return builtinInt.getWidth();
        if (auto intType = llvm::dyn_cast<ora::IntegerType>(type))
            return intType.getWidth();
        if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
            return intType.getWidth();
        if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
            return intType.getWidth();
        if (llvm::isa<ora::BoolType>(type))
            return 1u;
        if (llvm::isa<ora::AddressType>(type))
            return 160u;
        if (auto enumType = llvm::dyn_cast<ora::EnumType>(type))
            return getOraBitWidth(enumType.getReprType());
        if (auto errType = llvm::dyn_cast<ora::ErrorUnionType>(type))
            return getOraBitWidth(errType.getSuccessType());
        if (auto minType = llvm::dyn_cast<ora::MinValueType>(type))
            return getOraBitWidth(minType.getBaseType());
        if (auto maxType = llvm::dyn_cast<ora::MaxValueType>(type))
            return getOraBitWidth(maxType.getBaseType());
        if (auto rangeType = llvm::dyn_cast<ora::InRangeType>(type))
            return getOraBitWidth(rangeType.getBaseType());
        if (auto scaledType = llvm::dyn_cast<ora::ScaledType>(type))
            return getOraBitWidth(scaledType.getBaseType());
        if (auto exactType = llvm::dyn_cast<ora::ExactType>(type))
            return getOraBitWidth(exactType.getBaseType());

        if (llvm::isa<ora::StringType, ora::BytesType, ora::StructType, ora::MapType>(type))
            return 256u;

        return std::nullopt;
    }

    static bool isNarrowErrorUnion(ora::ErrorUnionType type)
    {
        auto widthOpt = getOraBitWidth(type.getSuccessType());
        if (!widthOpt)
            return false;
        return *widthOpt <= 255;
    }
}

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

            // ora.error_union<T> → sir.u256 or (sir.u256, sir.u256) depending on payload width
            addConversion([](ora::ErrorUnionType type, SmallVectorImpl<Type> &results) -> LogicalResult
                          {
                auto *ctx = type.getDialect().getContext();
                if (!ctx) {
                    return failure();
                }

                auto u256 = sir::U256Type::get(ctx);
                if (isNarrowErrorUnion(type)) {
                    results.push_back(u256);
                } else {
                    results.push_back(u256); // tag
                    results.push_back(u256); // payload
                }
                return success(); });
            addConversion([](ora::ErrorUnionType type) -> Type
                          {
                auto *ctx = type.getDialect().getContext();
                if (!ctx)
                    return Type();
                if (isNarrowErrorUnion(type))
                    return sir::U256Type::get(ctx);
                return Type(); });

            // Preserve structs until we explicitly enable lowering.
            addConversion([this](ora::StructType type) -> Type
                          {
                if (enableStructLowering)
                    return sir::PtrType::get(type.getContext(), /*addrSpace*/ 1);
                return type;
            });

            // Preserve tensor types until we explicitly enable lowering.
            addConversion([this](RankedTensorType type) -> Type
                          {
                if (enableTensorLowering)
                    return sir::U256Type::get(type.getContext());
                return type; });
            addConversion([this](UnrankedTensorType type) -> Type
                          {
                if (enableTensorLowering)
                    return sir::U256Type::get(type.getContext());
                return type; });
            addConversion([this](MemRefType type) -> Type
                          {
                if (enableMemRefLowering)
                    return sir::PtrType::get(type.getContext(), /*addrSpace*/ 1);
                return type;
            });
            addConversion([this](UnrankedMemRefType type) -> Type
                          {
                if (enableMemRefLowering)
                    return sir::PtrType::get(type.getContext(), /*addrSpace*/ 1);
                return type;
            });

            // ora.map → sir.u256 (map values live in storage, represent as u256 handles)
            addConversion([](ora::MapType type) -> Type
                          {
                auto *ctx = type.getDialect().getContext();
                if (!ctx) {
                    return Type();
                }
                return sir::U256Type::get(ctx); });

            // ora.struct passes through until we enable lowering.
            addConversion([this](ora::StructType type) -> Type
                          {
                if (enableStructLowering)
                    return sir::PtrType::get(type.getContext(), /*addrSpace*/ 1);
                return type;
            });

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
            // 2. Explicit non-Ora conversions (no fallback)
            // =========================================================================

            // Keep SIR types as-is.
            addConversion([](sir::U256Type type) -> Type { return type; });
            addConversion([](sir::PtrType type) -> Type { return type; });

            // Builtin integer types (used for indices/booleans in lowering).
            addConversion([](mlir::IntegerType type) -> Type
                          {
                if (type.getWidth() == 1)
                    return sir::U256Type::get(type.getContext());
                if (type.getWidth() <= 256)
                    return type;
                return Type(); });

            // Builtin types allowed to pass through.
            addConversion([](mlir::NoneType type) -> Type { return type; });
            addConversion([](mlir::IndexType type) -> Type { return type; });

            // Tensor types pass through unless tensor lowering is enabled.
            addConversion([this](mlir::RankedTensorType type) -> Type
                          {
                if (enableTensorLowering)
                    return sir::U256Type::get(type.getContext());
                return type; });
            addConversion([this](mlir::MemRefType type) -> Type
                          {
                if (enableMemRefLowering)
                    return sir::PtrType::get(type.getContext(), /*addrSpace*/ 1);
                return type;
            });

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

                                         // Materialize refinement -> base via ora.refinement_to_base.
                                         if (llvm::isa<mlir::IntegerType>(type))
                                         {
                                             if (llvm::isa<ora::MinValueType, ora::MaxValueType, ora::InRangeType,
                                                           ora::ScaledType, ora::ExactType, ora::NonZeroAddressType>(input.getType()))
                                             {
                                                 auto cast = builder.create<ora::RefinementToBaseOp>(loc, type, input);
                                                 return cast.getResult();
                                             }
                                         }

                                         if (llvm::isa<sir::U256Type>(type))
                                         {
                                             if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType()))
                                             {
                                                 (void)tensorType;
                                                 return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, input).getResult(0);
                                             }
                                            if (llvm::isa<ora::AddressType, ora::NonZeroAddressType>(input.getType()))
                                            {
                                                return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, input).getResult(0);
                                            }
                                         }

                                         if (llvm::isa<sir::PtrType>(type))
                                         {
                                             if (llvm::isa<ora::StringType, ora::BytesType>(input.getType()))
                                             {
                                                 return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, input).getResult(0);
                                             }
                                         }

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

                                        // Convert ora.error_union<T> -> sir.u256 (narrow-only) when defined by ok/err ops.
                                        if (llvm::isa<sir::U256Type>(type))
                                        {
                                            if (auto errUnion = dyn_cast<ora::ErrorUnionType>(input.getType()))
                                            {
                                                if (!isNarrowErrorUnion(errUnion))
                                                    return Value();

                                                auto u256Type = sir::U256Type::get(builder.getContext());
                                                auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                                auto oneAttr = mlir::IntegerAttr::get(ui64Type, 1);
                                                auto zeroAttr = mlir::IntegerAttr::get(ui64Type, 0);
                                                Value one = builder.create<sir::ConstOp>(loc, u256Type, oneAttr);
                                                Value zero = builder.create<sir::ConstOp>(loc, u256Type, zeroAttr);

                                                auto toU256 = [&](Value v) -> Value {
                                                    if (llvm::isa<sir::U256Type>(v.getType()))
                                                        return v;
                                                    if (auto intTy = dyn_cast<mlir::IntegerType>(v.getType()))
                                                    {
                                                    if (intTy.getWidth() == 1)
                                                    {
                                                            return builder.create<sir::BitcastOp>(loc, u256Type, v);
                                                    }
                                                    }
                                                    return builder.create<sir::BitcastOp>(loc, u256Type, v);
                                                };

                                                if (auto okOp = input.getDefiningOp<ora::ErrorOkOp>())
                                                {
                                                    Value payload = toU256(okOp.getValue());
                                                    Value shifted = builder.create<sir::ShlOp>(loc, u256Type, one, payload);
                                                    return builder.create<sir::OrOp>(loc, u256Type, shifted, zero);
                                                }
                                                if (auto errOp = input.getDefiningOp<ora::ErrorErrOp>())
                                                {
                                                    Value payload = toU256(errOp.getValue());
                                                    Value shifted = builder.create<sir::ShlOp>(loc, u256Type, one, payload);
                                                    return builder.create<sir::OrOp>(loc, u256Type, shifted, one);
                                                }
                                                // If the error_union value is already materialized (e.g., block arg),
                                                // treat it as the packed u256 representation.
                                                return builder.create<sir::BitcastOp>(loc, u256Type, input);
                                            }
                                        }

                                        // Convert integer types to sir.u256
                                        if (auto intType = dyn_cast<mlir::IntegerType>(input.getType()))
                                        {
                                            if (intType.getWidth() <= 256 && llvm::isa<sir::U256Type>(type))
                                            {
                                                if (intType.getWidth() == 1)
                                                {
                                                    return builder.create<sir::BitcastOp>(loc, type, input);
                                                }
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

                                        // Convert sir.u256 to index when needed.
                                        if (llvm::isa<mlir::IndexType>(type) && llvm::isa<sir::U256Type>(input.getType()))
                                        {
                                            return builder.create<sir::BitcastOp>(loc, type, input);
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

            // Allow memref -> sir.ptr materialization during memref lowering.
            addSourceMaterialization([this](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value
                                     {
                                         if (!enableMemRefLowering)
                                             return Value();
                                         if (inputs.size() != 1)
                                             return Value();
                                         Value input = inputs[0];
                                         if (!llvm::isa<sir::PtrType>(type))
                                             return Value();
                                         if (!llvm::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(input.getType()))
                                             return Value();
                                         return builder.create<sir::BitcastOp>(loc, type, input);
                                     });

            // We should NEVER materialize from SIR back to Ora - this indicates a bug
            // Return nullptr to indicate we can't materialize (will cause conversion to fail)
            addSourceMaterialization([this](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value
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

                                        if (llvm::isa<mlir::IndexType>(type) && llvm::isa<sir::U256Type>(input.getType()))
                                        {
                                            return builder.create<sir::BitcastOp>(loc, type, input);
                                        }

                                        if (auto errType = dyn_cast<ora::ErrorUnionType>(type))
                                        {
                                            if (isNarrowErrorUnion(errType))
                                            {
                                                Value packed = input;
                                                if (!llvm::isa<sir::U256Type>(packed.getType()))
                                                {
                                                    if (llvm::isa<mlir::IntegerType>(packed.getType()))
                                                    {
                                                        packed = builder.create<sir::BitcastOp>(loc, sir::U256Type::get(builder.getContext()), packed);
                                                    }
                                                    else
                                                    {
                                                        return Value();
                                                    }
                                                }
                                                auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc, type, packed);
                                                cast->setAttr("ora.normalized_error_union", builder.getUnitAttr());
                                                return cast.getResult(0);
                                            }
                                        }

                                        if (enableStructLowering && llvm::isa<ora::StructType>(type))
                                        {
                                            if (llvm::isa<sir::PtrType>(input.getType()))
                                            {
                                                auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc, type, input);
                                                return cast.getResult(0);
                                            }
                                        }

                                        if (enableTensorLowering && llvm::isa<ora::AddressType, ora::NonZeroAddressType>(type))
                                        {
                                            if (llvm::isa<sir::U256Type>(input.getType()))
                                            {
                                                auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc, type, input);
                                                return cast.getResult(0);
                                            }
                                        }

                                        if (llvm::isa<ora::StringType, ora::BytesType>(type))
                                        {
                                            auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc, type, input);
                                            return cast.getResult(0);
                                        }

                                        // Allow materializing into refinement types via ora.base_to_refinement.
                                        if (llvm::isa<ora::MinValueType, ora::MaxValueType, ora::InRangeType,
                                                     ora::ScaledType, ora::ExactType, ora::NonZeroAddressType>(type))
                                        {
                                            // The refinement op is type-level; it can wrap an already-converted value.
                                            auto cast = builder.create<ora::BaseToRefinementOp>(loc, type, input);
                                            return cast.getResult();
                                        }

                                        // If trying to materialize to an Ora type, this is an error
                                        if (type.getDialect().getNamespace() == "ora")
                                        {
                                            llvm::errs() << "[OraToSIRTypeConverter] ERROR: Attempted to materialize from SIR to Ora type: " << type << "\n";
                                            llvm::errs() << "[OraToSIRTypeConverter] This indicates an unconverted operation still expects Ora types.\n";
                                            llvm::errs() << "[OraToSIRTypeConverter] Location: " << loc << "\n";
                                            llvm::errs().flush();
                                            return Value(); // fail materialization (no fallback)
                                        }
                                         return Value(); // Return null - don't handle other cases
                                     });

        }

    } // namespace ora
} // namespace mlir
