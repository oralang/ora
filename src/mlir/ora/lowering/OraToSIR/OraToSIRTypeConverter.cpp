#include "OraToSIRTypeConverter.h"
#include "OraMaterializationKinds.h"
#include "patterns/AdtCarrierHelpers.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ora-to-sir-types"

using namespace mlir;
using namespace ora;
using namespace sir;

namespace
{
    static Value createTaggedCast(OpBuilder &builder,
                                  Location loc,
                                  Type type,
                                  Value input,
                                  StringRef kind)
    {
        auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc, type, input);
        cast->setAttr(kOraMaterializationKindAttr, builder.getStringAttr(kind));
        return cast.getResult(0);
    }

    static Value createTaggedCast(OpBuilder &builder,
                                  Location loc,
                                  Type type,
                                  ValueRange inputs,
                                  StringRef kind)
    {
        auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc, TypeRange{type}, inputs);
        cast->setAttr(kOraMaterializationKindAttr, builder.getStringAttr(kind));
        return cast.getResult(0);
    }

    static std::optional<unsigned> getOraBitWidth(Type type)
    {
        if (!type)
            return std::nullopt;
        if (llvm::isa<mlir::NoneType>(type))
            return 0u;

        if (auto builtinInt = llvm::dyn_cast<mlir::IntegerType>(type))
            return builtinInt.getWidth();
        if (auto intType = llvm::dyn_cast<ora::IntegerType>(type))
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

        if (llvm::isa<ora::StringType, ora::BytesType, ora::StructType, ora::AnonymousStructType, ora::MapType>(type))
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

    static Type getWideErrorUnionCarrierType(MLIRContext *ctx, Type successType)
    {
        if (!ctx)
            return Type();
        if (llvm::isa<ora::TupleType, ora::StructType, ora::AnonymousStructType, ora::StringType, ora::BytesType,
                      mlir::MemRefType, mlir::UnrankedMemRefType>(successType))
            return sir::PtrType::get(ctx, /*addrSpace*/ 1);
        return sir::U256Type::get(ctx);
    }

    static Type getAdtPayloadCarrierType(MLIRContext *ctx)
    {
        if (!ctx)
            return Type();
        return sir::U256Type::get(ctx);
    }

    static bool isPayloadlessErrorStruct(Type type, Operation *contextOp)
    {
        auto structType = llvm::dyn_cast<ora::StructType>(type);
        if (!structType || !contextOp)
            return false;
        auto module = contextOp->getParentOfType<mlir::ModuleOp>();
        if (!module)
            return false;

        bool matched = false;
        module.walk([&](Operation *op) {
            if (matched)
                return;
            auto sym = op->getAttrOfType<mlir::StringAttr>("sym_name");
            if (!sym || sym.getValue() != structType.getName())
                return;
            if (!op->hasAttr("ora.error_decl") && !op->hasAttr("sir.error_decl"))
                return;
            auto params = op->getAttrOfType<mlir::ArrayAttr>("ora.param_types");
            if (!params || params.empty())
            {
                matched = true;
                return;
            }
        });
        return matched;
    }
}

static Value makeMaskValue(OpBuilder &builder, Location loc, unsigned width)
{
    if (width >= 256)
        return Value();
    auto u256 = sir::U256Type::get(builder.getContext());
    auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
    if (width == 64)
    {
        auto attr = mlir::IntegerAttr::get(ui64Type, std::numeric_limits<uint64_t>::max());
        return builder.create<sir::ConstOp>(loc, u256, attr);
    }
    if (width < 64)
    {
        uint64_t mask = (width == 0) ? 0ULL : ((1ULL << width) - 1ULL);
        auto attr = mlir::IntegerAttr::get(ui64Type, mask);
        return builder.create<sir::ConstOp>(loc, u256, attr);
    }
    return Value();
}

namespace mlir
{
    namespace ora
    {
        mlir::UnrealizedConversionCastOp createMaterializationCastOp(
            OpBuilder &builder,
            Location loc,
            TypeRange resultTypes,
            ValueRange inputs,
            StringRef kind)
        {
            auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc, resultTypes, inputs);
            cast->setAttr(kOraMaterializationKindAttr, builder.getStringAttr(kind));
            return cast;
        }

        Value createMaterializationCast(
            OpBuilder &builder,
            Location loc,
            Type type,
            Value input,
            StringRef kind)
        {
            return createTaggedCast(builder, loc, type, input, kind);
        }

        Value createMaterializationCast(
            OpBuilder &builder,
            Location loc,
            Type type,
            ValueRange inputs,
            StringRef kind)
        {
            return createTaggedCast(builder, loc, type, inputs, kind);
        }

        bool hasMaterializationKind(
            mlir::UnrealizedConversionCastOp castOp,
            StringRef kind)
        {
            if (!castOp)
                return false;
            auto attr = castOp->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
            return attr && attr.getValue() == kind;
        }

        std::optional<Value> materializePtrCarrierFromOraValue(
            OpBuilder &builder,
            Location loc,
            Type ptrType,
            Value input)
        {
            if (!llvm::isa<sir::PtrType>(ptrType))
                return std::nullopt;

            if (llvm::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(input.getType()))
            {
                return builder.create<sir::BitcastOp>(loc, ptrType, input).getResult();
            }

            if (llvm::isa<ora::TupleType, ora::StructType, ora::AnonymousStructType, ora::StringType, ora::BytesType,
                          mlir::MemRefType, mlir::UnrankedMemRefType>(input.getType()))
            {
                if (auto bitcast = input.getDefiningOp<sir::BitcastOp>())
                {
                    if (llvm::isa<sir::PtrType>(bitcast.getOperand().getType()))
                        return bitcast.getOperand();
                }
            }

            if (auto tupleType = llvm::dyn_cast<ora::TupleType>(input.getType()))
            {
                if (auto castOp = input.getDefiningOp<mlir::UnrealizedConversionCastOp>())
                {
                    if (castOp.getNumOperands() == 1 && llvm::isa<sir::PtrType>(castOp.getOperand(0).getType()))
                        return castOp.getOperand(0);
                }

                if (auto getError = input.getDefiningOp<ora::ErrorGetErrorOp>())
                {
                    if (auto errCast = getError.getValue().getDefiningOp<mlir::UnrealizedConversionCastOp>())
                    {
                        if (errCast.getNumOperands() == 2)
                        {
                            if (llvm::isa<sir::PtrType>(errCast.getOperand(1).getType()))
                                return errCast.getOperand(1);
                            if (llvm::isa<sir::U256Type>(errCast.getOperand(1).getType()) &&
                                tupleType.getElementTypes().size() == 1)
                            {
                                auto u256Type = sir::U256Type::get(builder.getContext());
                                auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                Value size = builder.create<sir::ConstOp>(
                                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
                                Value ptr = builder.create<sir::MallocOp>(loc, ptrType, size);
                                builder.create<sir::StoreOp>(loc, ptr, errCast.getOperand(1));
                                return ptr;
                            }
                        }
                    }
                }

                if (auto tupleCreate = input.getDefiningOp<ora::TupleCreateOp>())
                {
                    auto u256Type = sir::U256Type::get(builder.getContext());
                    auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                    Value size = builder.create<sir::ConstOp>(
                        loc,
                        u256Type,
                        mlir::IntegerAttr::get(
                            ui64Type,
                            static_cast<uint64_t>(tupleCreate.getNumOperands()) * 32ULL));
                    Value ptr = builder.create<sir::MallocOp>(loc, ptrType, size);
                    for (auto [index, element] : llvm::enumerate(tupleCreate.getOperands()))
                    {
                        Value slotPtr = ptr;
                        if (index != 0)
                        {
                            Value offset = builder.create<sir::ConstOp>(
                                loc,
                                u256Type,
                                mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(index) * 32ULL));
                            slotPtr = builder.create<sir::AddPtrOp>(loc, ptrType, ptr, offset);
                        }
                        Value stored = element;
                        if (!llvm::isa<sir::U256Type>(stored.getType()))
                            stored = builder.create<sir::BitcastOp>(loc, u256Type, stored);
                        builder.create<sir::StoreOp>(loc, slotPtr, stored);
                    }
                    return ptr;
                }

                if (tupleType.getElementTypes().size() == 1)
                {
                    Type elemType = tupleType.getElementTypes().front();
                    if (llvm::isa<mlir::IntegerType, ora::IntegerType, ora::BoolType,
                                  ora::AddressType, ora::NonZeroAddressType>(elemType))
                    {
                        if (auto tupleCreate = input.getDefiningOp<ora::TupleCreateOp>())
                        {
                            if (tupleCreate.getNumOperands() == 1)
                            {
                                auto u256Type = sir::U256Type::get(builder.getContext());
                                auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                Value size = builder.create<sir::ConstOp>(
                                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
                                Value ptr = builder.create<sir::MallocOp>(loc, ptrType, size);
                                Value element = tupleCreate.getOperand(0);
                                if (!llvm::isa<sir::U256Type>(element.getType()))
                                    element = builder.create<sir::BitcastOp>(loc, u256Type, element);
                                builder.create<sir::StoreOp>(loc, ptr, element);
                                return ptr;
                            }
                        }
                    }
                }
            }

            if (auto getErrorOp = input.getDefiningOp<ora::ErrorGetErrorOp>())
            {
                if (auto errCast = getErrorOp.getValue().getDefiningOp<mlir::UnrealizedConversionCastOp>())
                {
                    if (errCast.getNumOperands() == 2)
                    {
                        Value payload = errCast.getOperand(1);
                        if (llvm::isa<sir::PtrType>(payload.getType()))
                            return payload;
                        if (llvm::isa<sir::U256Type>(payload.getType()) &&
                            llvm::isa<ora::AnonymousStructType>(input.getType()))
                        {
                            auto anonType = llvm::cast<ora::AnonymousStructType>(input.getType());
                            if (anonType.getFieldTypes().size() == 1)
                            {
                                auto u256Type = sir::U256Type::get(builder.getContext());
                                auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                Value size = builder.create<sir::ConstOp>(
                                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
                                Value ptr = builder.create<sir::MallocOp>(loc, ptrType, size);
                                builder.create<sir::StoreOp>(loc, ptr, payload);
                                return ptr;
                            }
                        }
                        if (llvm::isa<sir::U256Type>(payload.getType()) &&
                            llvm::isa<ora::TupleType>(input.getType()))
                        {
                            auto tupleType = llvm::cast<ora::TupleType>(input.getType());
                            if (tupleType.getElementTypes().size() == 1)
                            {
                                auto u256Type = sir::U256Type::get(builder.getContext());
                                auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                Value size = builder.create<sir::ConstOp>(
                                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, 32));
                                Value ptr = builder.create<sir::MallocOp>(loc, ptrType, size);
                                builder.create<sir::StoreOp>(loc, ptr, payload);
                                return ptr;
                            }
                        }
                    }
                }
            }

            if (auto unwrapOp = input.getDefiningOp<ora::ErrorUnwrapOp>())
            {
                if (auto errCast = unwrapOp.getValue().getDefiningOp<mlir::UnrealizedConversionCastOp>())
                {
                    if (errCast.getNumOperands() == 2)
                    {
                        Value payload = errCast.getOperand(1);
                        if (llvm::isa<sir::PtrType>(payload.getType()))
                            return payload;
                    }
                }
            }

            if (auto structInit = input.getDefiningOp<ora::StructInitOp>())
            {
                auto u256Type = sir::U256Type::get(builder.getContext());
                auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                Value size = builder.create<sir::ConstOp>(
                    loc,
                    u256Type,
                    mlir::IntegerAttr::get(
                        ui64Type,
                        static_cast<uint64_t>(structInit.getFieldValues().size()) * 32ULL));
                Value ptr = builder.create<sir::MallocOp>(loc, ptrType, size);
                for (auto [index, element] : llvm::enumerate(structInit.getFieldValues()))
                {
                    Value slotPtr = ptr;
                    if (index != 0)
                    {
                        Value offset = builder.create<sir::ConstOp>(
                            loc,
                            u256Type,
                            mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(index) * 32ULL));
                        slotPtr = builder.create<sir::AddPtrOp>(loc, ptrType, ptr, offset);
                    }
                    Value stored = element;
                    if (!llvm::isa<sir::U256Type>(stored.getType()))
                        stored = builder.create<sir::BitcastOp>(loc, u256Type, stored);
                    builder.create<sir::StoreOp>(loc, slotPtr, stored);
                }
                return ptr;
            }

            if (auto structInstantiate = input.getDefiningOp<ora::StructInstantiateOp>())
            {
                auto u256Type = sir::U256Type::get(builder.getContext());
                auto ui64Type = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                Value size = builder.create<sir::ConstOp>(
                    loc,
                    u256Type,
                    mlir::IntegerAttr::get(
                        ui64Type,
                        static_cast<uint64_t>(structInstantiate.getFieldValues().size()) * 32ULL));
                Value ptr = builder.create<sir::MallocOp>(loc, ptrType, size);
                for (auto [index, element] : llvm::enumerate(structInstantiate.getFieldValues()))
                {
                    Value slotPtr = ptr;
                    if (index != 0)
                    {
                        Value offset = builder.create<sir::ConstOp>(
                            loc,
                            u256Type,
                            mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(index) * 32ULL));
                        slotPtr = builder.create<sir::AddPtrOp>(loc, ptrType, ptr, offset);
                    }
                    Value stored = element;
                    if (!llvm::isa<sir::U256Type>(stored.getType()))
                        stored = builder.create<sir::BitcastOp>(loc, u256Type, stored);
                    builder.create<sir::StoreOp>(loc, slotPtr, stored);
                }
                return ptr;
            }

            return std::nullopt;
        }

        OraToSIRTypeConverter::OraToSIRTypeConverter()
        {

            // =========================================================================
            // 1. Explicit Ora → SIR type conversions
            // =========================================================================

            // ora.int<N> → sir.u256  (for now: ABI-level erasure)
            addConversion([](ora::IntegerType type) -> Type
                          {
                LLVM_DEBUG(llvm::dbgs() << "[OraToSIRTypeConverter] Converting ora::IntegerType: " << type << "\n");
                auto *ctx = type.getDialect().getContext();
                if (!ctx)
                    return Type();
                return sir::U256Type::get(ctx); });

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

            // ADTs use a two-word compiler-runtime carrier at the conversion
            // boundary: (tag_word: sir.u256, payload_word: sir.u256).
            //
            // The payload word is exact, not best-effort:
            // - scalar payload variants store their payload bits inline
            // - aggregate payload variants store a compiler-managed handle
            //   (ptr-backed tuple/struct/string/bytes/memref value) bitcast to
            //   sir.u256 at the ADT boundary
            //
            // Concrete construct/payload rewrites still fail closed for shapes
            // that do not lower into one of those two cases exactly.
            addConversion([](ora::AdtType type, SmallVectorImpl<Type> &results) -> LogicalResult
                          {
                auto *ctx = type.getDialect().getContext();
                if (!ctx)
                    return failure();
                results.push_back(sir::U256Type::get(ctx));
                results.push_back(getAdtPayloadCarrierType(ctx));
                return success(); });
            addConversion([](ora::AdtType type) -> Type
                          {
                (void)type;
                return Type(); });

            // ora.error_union<T> → sir.u256 or (sir.u256, carrier(T)) depending on payload width
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
                    results.push_back(getWideErrorUnionCarrierType(ctx, type.getSuccessType())); // payload carrier
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
            addConversion([](ora::AnonymousStructType type) -> Type
                          { return sir::PtrType::get(type.getContext(), /*addrSpace*/ 1); });
            addConversion([](ora::TupleType type) -> Type
                          { return sir::PtrType::get(type.getContext(), /*addrSpace*/ 1); });

            // Preserve tensor types until we explicitly enable lowering.
            addConversion([this](UnrankedTensorType type) -> Type
                          {
                if (enableTensorLowering)
                    return sir::U256Type::get(type.getContext());
                return type; });
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
            addConversion([](mlir::IndexType type) -> Type
                          { return sir::U256Type::get(type.getContext()); });

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
                                            if (llvm::isa<mlir::IndexType>(input.getType()))
                                            {
                                                auto i256 = mlir::IntegerType::get(builder.getContext(), 256);
                                                Value asI256 = builder.create<arith::IndexCastOp>(loc, i256, input);
                                                return builder.create<sir::BitcastOp>(loc, type, asI256);
                                            }
                                            if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType()))
                                            {
                                                (void)tensorType;
                                                return createTaggedCast(builder, loc, type, input, mat_kind::kAddressForward);
                                            }
                                            if (llvm::isa<ora::AddressType, ora::NonZeroAddressType>(input.getType()))
                                            {
                                                return createTaggedCast(builder, loc, type, input, mat_kind::kAddressForward);
                                            }
                                        }

                                         if (llvm::isa<sir::PtrType>(type))
                                         {
                                             if (llvm::isa<ora::StringType, ora::BytesType>(input.getType()))
                                             {
                                                 return createTaggedCast(builder, loc, type, input, mat_kind::kPtrView);
                                             }
                                             if (auto materialized = materializePtrCarrierFromOraValue(builder, loc, type, input))
                                                 return *materialized;
                                         }

                                         auto makeMask = [&](unsigned width) -> Value {
                                             return makeMaskValue(builder, loc, width);
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

                                         // Convert sir.u256 to index types
                                         if (llvm::isa<mlir::IndexType>(type))
                                         {
                                             if (llvm::isa<sir::U256Type>(input.getType()))
                                             {
                                                 auto i256 = mlir::IntegerType::get(builder.getContext(), 256);
                                                 Value asI256 = builder.create<sir::BitcastOp>(loc, i256, input);
                                                 return builder.create<arith::IndexCastOp>(loc, type, asI256);
                                             }
                                             if (auto intTy = dyn_cast<mlir::IntegerType>(input.getType()))
                                             {
                                                 return builder.create<arith::IndexCastOp>(loc, type, input);
                                             }
                                         }

                                         return Value(); // Don't handle other cases
                                     });

            // Error union N:1 packing: (sir.u256, carrier(T)) → ora.error_union via
            // unrealized_conversion_cast. This is needed both for wide error unions
            // and for narrow success types that temporarily carry a payload-bearing
            // error path through deferred-return slots.
            addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value
                                     {
                                         if (inputs.size() != 2)
                                             return Value();
                                         auto errType = dyn_cast<ora::ErrorUnionType>(type);
                                         if (!errType)
                                             return Value();
                                         return createTaggedCast(builder, loc, type, inputs, mat_kind::kNormalizedErrorUnion);
                                     });
            addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value
                                     {
                                         if (inputs.size() != 2)
                                             return Value();
                                         auto adtType = dyn_cast<ora::AdtType>(type);
                                         if (!adtType)
                                             return Value();
                                         return createTaggedCast(builder, loc, type, inputs, mat_kind::kNormalizedAdt);
                                     });
            addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value
                                     {
                                         if (inputs.size() != 2)
                                             return Value();
                                         auto errType = dyn_cast<ora::ErrorUnionType>(type);
                                         if (!errType)
                                             return Value();
                                         return createTaggedCast(builder, loc, type, inputs, mat_kind::kNormalizedErrorUnion);
                                     });
            addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) -> Value
                                     {
                                         if (inputs.size() != 2)
                                             return Value();
                                         auto adtType = dyn_cast<ora::AdtType>(type);
                                         if (!adtType)
                                             return Value();
                                         return createTaggedCast(builder, loc, type, inputs, mat_kind::kNormalizedAdt);
                                     });

            // Wide error union 1:N splitting: ora.error_union → (sir.u256, carrier(T)).
            // The conversion framework calls this when it needs to adapt a single
            // error_union value into the two converted values for an op adaptor.
            addTargetMaterialization([](OpBuilder &builder, TypeRange resultTypes, ValueRange inputs, Location loc) -> SmallVector<Value>
                                     {
                                         if (resultTypes.size() != 2 || inputs.size() != 1)
                                             return {};
                                         Value input = inputs[0];
                                         auto errType = dyn_cast<ora::ErrorUnionType>(input.getType());
                                         if (!errType || isNarrowErrorUnion(errType))
                                             return {};
                                         // Look through an existing unrealized_conversion_cast.
                                         if (auto cast = input.getDefiningOp<mlir::UnrealizedConversionCastOp>())
                                         {
                                             if (cast.getNumOperands() == 2 &&
                                                 cast.getNumResults() == 1)
                                             {
                                                 return SmallVector<Value>{cast.getOperand(0), cast.getOperand(1)};
                                             }
                                         }
                                         // Otherwise, decompose via loads from a malloc'd pair.
                                         auto u256 = sir::U256Type::get(builder.getContext());
                                         auto ptrType = sir::PtrType::get(builder.getContext(), 1);
                                         auto ui64 = mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Unsigned);
                                         Value bitcast = builder.create<sir::BitcastOp>(loc, ptrType, input);
                                         Value tag = builder.create<sir::LoadOp>(loc, resultTypes[0], bitcast);
                                         Value off = builder.create<sir::ConstOp>(loc, u256, mlir::IntegerAttr::get(ui64, 32));
                                         Value ptr2 = builder.create<sir::AddPtrOp>(loc, ptrType, bitcast, off);
                                         Value payload = builder.create<sir::LoadOp>(loc, resultTypes[1], ptr2);
                                         return SmallVector<Value>{tag, payload};
                                     });
            addTargetMaterialization([](OpBuilder &builder, TypeRange resultTypes, ValueRange inputs, Location loc) -> SmallVector<Value>
                                     {
                                         (void)builder;
                                         if (resultTypes.size() != 2 || inputs.size() != 1)
                                             return {};
                                         Value input = inputs[0];
                                         auto adtType = dyn_cast<ora::AdtType>(input.getType());
                                         if (!adtType)
                                             return {};
                                         if (auto cast = input.getDefiningOp<mlir::UnrealizedConversionCastOp>())
                                         {
                                             auto kind = cast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
                                             if (kind && kind.getValue() == mat_kind::kNormalizedAdt && cast.getNumOperands() == 2)
                                                 return SmallVector<Value>{cast.getOperand(0), cast.getOperand(1)};
                                             if (kind && kind.getValue() == mat_kind::kAdtHandleView && cast.getNumOperands() == 1)
                                             {
                                                 Value handle = cast.getOperand(0);
                                                 if (!llvm::isa<sir::PtrType, sir::U256Type>(handle.getType()))
                                                     return {};
                                                 auto [tag, payload] = adt_helpers::loadAdtPartsFromHandle(builder, loc, handle);
                                                 return SmallVector<Value>{tag, payload};
                                             }
                                         }
                                         return {};
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
                                             return makeMaskValue(builder, loc, width);
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

                                        if (llvm::isa<mlir::IndexType>(type))
                                        {
                                            if (llvm::isa<sir::U256Type>(input.getType()))
                                            {
                                                auto i256 = mlir::IntegerType::get(builder.getContext(), 256);
                                                Value asI256 = builder.create<sir::BitcastOp>(loc, i256, input);
                                                return builder.create<arith::IndexCastOp>(loc, type, asI256);
                                            }
                                            if (auto intTy = dyn_cast<mlir::IntegerType>(input.getType()))
                                            {
                                                return builder.create<arith::IndexCastOp>(loc, type, input);
                                            }
                                        }

                                        if (llvm::isa<mlir::NoneType>(type) &&
                                            llvm::isa<sir::U256Type>(input.getType()))
                                        {
                                            return createTaggedCast(builder, loc, type, input, "none_forward");
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
                                                return createTaggedCast(builder, loc, type, packed, mat_kind::kNormalizedErrorUnion);
                                            }
                                        }

                                        if (llvm::isa<ora::StructType, ora::AnonymousStructType>(type))
                                        {
                                            if (llvm::isa<sir::U256Type>(input.getType()))
                                            {
                                                return input;
                                            }
                                            if (!enableStructLowering)
                                            {
                                                return Value();
                                            }
                                            if (llvm::isa<sir::PtrType>(input.getType()))
                                            {
                                                return createTaggedCast(builder, loc, type, input, mat_kind::kPtrView);
                                            }
                                        }

                                        if (llvm::isa<ora::TupleType>(type) && llvm::isa<sir::PtrType>(input.getType()))
                                        {
                                            return createTaggedCast(builder, loc, type, input, mat_kind::kPtrView);
                                        }

                                        // ptr -> memref: memrefs are ptrs in SIR, forward via bitcast.
                                        if (enableMemRefLowering &&
                                            llvm::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(type) &&
                                            llvm::isa<sir::PtrType>(input.getType()))
                                        {
                                            return builder.create<sir::BitcastOp>(loc, type, input);
                                        }

                                        if (enableTensorLowering && llvm::isa<ora::AddressType, ora::NonZeroAddressType>(type))
                                        {
                                            if (llvm::isa<sir::U256Type>(input.getType()))
                                            {
                                                return createTaggedCast(builder, loc, type, input, mat_kind::kAddressForward);
                                            }
                                        }

                                        if (llvm::isa<ora::StringType, ora::BytesType>(type))
                                        {
                                            return createTaggedCast(builder, loc, type, input, mat_kind::kPtrView);
                                        }

                                        // Refinement types erase to their base representation;
                                        // materialize via sir.bitcast instead of creating ora ops.
                                        if (llvm::isa<ora::MinValueType, ora::MaxValueType, ora::InRangeType,
                                                     ora::ScaledType, ora::ExactType, ora::NonZeroAddressType>(type))
                                        {
                                            return builder.create<sir::BitcastOp>(loc, type, input).getResult();
                                        }

                                        // If trying to materialize to an Ora type, this is an error
                                        if (type.getDialect().getNamespace() == "ora")
                                        {
                                            LLVM_DEBUG(llvm::dbgs() << "[OraToSIRTypeConverter] ERROR: Attempted to materialize from SIR to Ora type: " << type
                                                                    << " at " << loc << "\n");
                                            return Value();
                                        }
                                         return Value(); // Return null - don't handle other cases
                                     });

        }

    } // namespace ora
} // namespace mlir
