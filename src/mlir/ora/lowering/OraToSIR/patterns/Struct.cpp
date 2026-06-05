#include "Struct.h"

#include "patterns/MissingOps.h"
#include "patterns/Storage.h"
#include "patterns/StorageLayout.h"
#include "OraMaterializationKinds.h"
#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"
#include "patterns/AdtCarrierHelpers.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::ora;

using mlir::ora::lowering::addStorageWordOffset;
using mlir::ora::lowering::getStorageWordCount;
using mlir::ora::lowering::kStorageMemRefViewKind;
using mlir::ora::lowering::kStorageStructCarrierKind;
using mlir::ora::lowering::kStorageStructViewFieldsAttr;

namespace
{
    static LogicalResult getStructFieldsFromDecl(ora::StructDeclOp structDecl, SmallVectorImpl<StringRef> &names, SmallVectorImpl<Type> &types)
    {
        auto fieldNamesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_names");
        auto fieldTypesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_types");
        if (!fieldNamesAttr || !fieldTypesAttr || fieldNamesAttr.size() != fieldTypesAttr.size())
        {
            return failure();
        }

        for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
        {
            auto nameAttr = cast<StringAttr>(fieldNamesAttr[i]);
            auto typeAttr = cast<TypeAttr>(fieldTypesAttr[i]);
            names.push_back(nameAttr.getValue());
            types.push_back(typeAttr.getValue());
        }

        return success();
    }

    static ora::StructDeclOp findStructDecl(Operation *op, StringRef structName)
    {
        ModuleOp module = op->getParentOfType<ModuleOp>();
        if (!module)
        {
            return nullptr;
        }

        ora::StructDeclOp structDecl = nullptr;
        module.walk([&](ora::StructDeclOp declOp)
                    {
            auto nameAttr = declOp->getAttrOfType<StringAttr>("sym_name");
            if (nameAttr && nameAttr.getValue() == structName)
            {
                structDecl = declOp;
                return WalkResult::interrupt();
            }
            return WalkResult::advance(); });

        return structDecl;
    }

    static LogicalResult getStructFields(Operation *op, StringRef structName, SmallVectorImpl<StringRef> &names, SmallVectorImpl<Type> &types)
    {
        auto structDecl = findStructDecl(op, structName);
        if (!structDecl)
        {
            return failure();
        }

        return getStructFieldsFromDecl(structDecl, names, types);
    }

    static bool parseNumericFieldIndex(StringRef fieldName, size_t &fieldIndex)
    {
        uint64_t parsed = 0;
        if (fieldName.empty() || fieldName.getAsInteger(10, parsed))
        {
            return false;
        }
        fieldIndex = static_cast<size_t>(parsed);
        return true;
    }

    static LogicalResult resolveFieldByDeclScan(Operation *op, StringRef fieldName, Type resultTypeHint, size_t &fieldIndex, Type &fieldType)
    {
        ModuleOp module = op->getParentOfType<ModuleOp>();
        if (!module)
        {
            return failure();
        }

        bool found = false;
        bool ambiguous = false;
        module.walk([&](ora::StructDeclOp declOp)
                    {
            SmallVector<StringRef, 8> names;
            SmallVector<Type, 8> types;
            if (failed(getStructFieldsFromDecl(declOp, names, types)))
            {
                return WalkResult::advance();
            }

            for (size_t i = 0; i < names.size(); ++i)
            {
                if (names[i] != fieldName)
                {
                    continue;
                }
                const bool convertedMemRefHint =
                    resultTypeHint && llvm::isa<sir::PtrType>(resultTypeHint) &&
                    llvm::isa<mlir::MemRefType>(types[i]);
                if (resultTypeHint && types[i] != resultTypeHint && !convertedMemRefHint)
                {
                    continue;
                }

                if (found && (fieldIndex != i || fieldType != types[i]))
                {
                    ambiguous = true;
                    return WalkResult::interrupt();
                }

                found = true;
                fieldIndex = i;
                fieldType = types[i];
            }

            return WalkResult::advance(); });

        if (!found || ambiguous)
        {
            return failure();
        }
        return success();
    }

    static std::optional<uint64_t> getStructFieldStorageOffset(Operation *anchor,
                                                               ora::StructType structType,
                                                               StringRef fieldName)
    {
        SmallVector<StringRef, 8> fieldNames;
        SmallVector<Type, 8> fieldTypes;
        if (failed(getStructFields(anchor, structType.getName(), fieldNames, fieldTypes)))
            return std::nullopt;

        uint64_t offset = 0;
        for (size_t i = 0; i < fieldNames.size(); ++i)
        {
            if (fieldNames[i] == fieldName)
                return offset;
            offset += getStorageWordCount(anchor, fieldTypes[i]);
        }

        return std::nullopt;
    }

    static Value getStorageStructValueBaseSlot(Value structValue,
                                               ConversionPatternRewriter &rewriter,
                                               Location loc)
    {
        while (auto cast = structValue.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (cast.getNumOperands() != 1)
                break;
            structValue = cast.getOperand(0);
        }

        if (auto sload = structValue.getDefiningOp<ora::SLoadOp>())
        {
            auto slotIndexOpt = computeGlobalSlot(sload.getGlobalName(), sload.getOperation());
            if (!slotIndexOpt)
                return Value();

            auto *ctx = rewriter.getContext();
            auto u256Type = sir::U256Type::get(ctx);
            auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
            return rewriter.create<sir::ConstOp>(
                loc, u256Type, mlir::IntegerAttr::get(ui64Type, *slotIndexOpt));
        }

        if (auto extract = structValue.getDefiningOp<ora::StructFieldExtractOp>())
        {
            auto parentStructType = llvm::dyn_cast<ora::StructType>(extract.getStructValue().getType());
            if (!parentStructType)
                return Value();

            auto fieldOffset = getStructFieldStorageOffset(
                extract.getOperation(), parentStructType, extract.getFieldName());
            if (!fieldOffset)
                return Value();

            Value parentSlot = getStorageStructValueBaseSlot(extract.getStructValue(), rewriter, loc);
            if (!parentSlot)
                return Value();

            return addStorageWordOffset(loc, parentSlot, *fieldOffset, rewriter);
        }

        return Value();
    }

    static std::optional<Value> materializeAggregateFieldWord(Location loc, ConversionPatternRewriter &rewriter, Value value)
    {
        auto *ctx = rewriter.getContext();
        auto u256Type = sir::U256Type::get(ctx);

        if (auto castOp = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (ora::hasMaterializationKind(castOp, mat_kind::kNormalizedAdt) && castOp.getNumOperands() == 2)
            {
                Value basePtr = ora::adt_helpers::materializeAdtHandle(
                    rewriter, loc, castOp.getOperand(0), castOp.getOperand(1));
                return rewriter.create<sir::BitcastOp>(loc, u256Type, basePtr).getResult();
            }

            if (castOp.getNumOperands() == 1)
            {
                Value src = castOp.getOperand(0);
                if (llvm::isa<sir::PtrType, sir::U256Type>(src.getType()))
                {
                    if (!llvm::isa<sir::U256Type>(src.getType()))
                        src = rewriter.create<sir::BitcastOp>(loc, u256Type, src);
                    return src;
                }
            }
        }

        if (llvm::isa<sir::PtrType>(value.getType()))
            return rewriter.create<sir::BitcastOp>(loc, u256Type, value).getResult();
        if (llvm::isa<sir::U256Type>(value.getType()))
            return value;
        return std::nullopt;
    }

    static bool valueHasMaterializationKind(Value value, StringRef kind)
    {
        while (value)
        {
            Operation *def = value.getDefiningOp();
            if (!def)
                return false;

            auto valueKind = def->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
            if (valueKind && valueKind.getValue() == kind)
                return true;

            if (auto bitcast = dyn_cast<sir::BitcastOp>(def))
            {
                value = bitcast.getOperand();
                continue;
            }
            if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(def);
                cast && cast.getNumOperands() == 1)
            {
                value = cast.getOperand(0);
                continue;
            }

            return false;
        }

        return false;
    }

    static Value coerceStructBasePtr(Location loc, PatternRewriter &rewriter, Value basePtr, bool fallbackBitcast)
    {
        auto *ctx = rewriter.getContext();
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);

        if (llvm::isa<sir::U256Type>(basePtr.getType()))
        {
            basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, basePtr);
        }
        else if (!llvm::isa<sir::PtrType>(basePtr.getType()))
        {
            if (auto castOp = basePtr.getDefiningOp<mlir::UnrealizedConversionCastOp>())
            {
                if (castOp.getNumOperands() == 1)
                {
                    Value src = castOp.getOperand(0);
                    if (llvm::isa<sir::PtrType>(src.getType()))
                    {
                        basePtr = src;
                    }
                    else if (llvm::isa<sir::U256Type>(src.getType()))
                    {
                        basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, src);
                    }
                }
            }
        }

        if (fallbackBitcast && !llvm::isa<sir::PtrType>(basePtr.getType()))
        {
            basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, basePtr);
        }

        return basePtr;
    }

    static Value fieldPtrAtIndex(Location loc, PatternRewriter &rewriter, Value basePtr, uint64_t fieldIndex)
    {
        if (fieldIndex == 0)
            return basePtr;

        auto *ctx = rewriter.getContext();
        auto u256Type = sir::U256Type::get(ctx);
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
        auto offsetAttr = mlir::IntegerAttr::get(ui64Type, fieldIndex * 32ULL);
        Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, offsetAttr);
        return rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, offset);
    }

    static Value buildStructBuffer(Location loc, ConversionPatternRewriter &rewriter, ValueRange fieldValues)
    {
        auto *ctx = rewriter.getContext();
        auto u256Type = sir::U256Type::get(ctx);
        auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

        const uint64_t fieldCount = static_cast<uint64_t>(fieldValues.size());
        const uint64_t byteSize = fieldCount * 32ULL;

        auto sizeAttr = mlir::IntegerAttr::get(ui64Type, byteSize);
        Value sizeVal = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
        Value basePtr = rewriter.create<sir::MallocOp>(loc, ptrType, sizeVal);
        bool hasStorageMemRefField = false;

        for (size_t i = 0; i < fieldValues.size(); ++i)
        {
            Value val = fieldValues[i];
            if (valueHasMaterializationKind(val, kStorageMemRefViewKind) ||
                valueHasMaterializationKind(val, kStorageStructCarrierKind))
                hasStorageMemRefField = true;
            if (auto aggregateWord = materializeAggregateFieldWord(loc, rewriter, val))
                val = *aggregateWord;
            else if (!llvm::isa<sir::U256Type>(val.getType()))
                val = rewriter.create<sir::BitcastOp>(loc, u256Type, val);

            Value fieldPtr = fieldPtrAtIndex(loc, rewriter, basePtr, static_cast<uint64_t>(i));
            rewriter.create<sir::StoreOp>(loc, fieldPtr, val);
        }

        if (hasStorageMemRefField)
        {
            basePtr.getDefiningOp()->setAttr(
                kOraMaterializationKindAttr,
                StringAttr::get(ctx, kStorageStructCarrierKind));
        }

        return basePtr;
    }
} // namespace

LogicalResult ConvertStructInstantiateOp::matchAndRewrite(
    ora::StructInstantiateOp op,
    typename ora::StructInstantiateOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value basePtr = buildStructBuffer(loc, rewriter, adaptor.getFieldValues());
    rewriter.replaceOp(op, basePtr);
    return success();
}

LogicalResult ConvertStructInitOp::matchAndRewrite(
    ora::StructInitOp op,
    typename ora::StructInitOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value basePtr = buildStructBuffer(loc, rewriter, adaptor.getFieldValues());
    rewriter.replaceOp(op, basePtr);
    return success();
}

LogicalResult ConvertTupleCreateOp::matchAndRewrite(
    ora::TupleCreateOp op,
    typename ora::TupleCreateOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    Value basePtr = buildStructBuffer(loc, rewriter, adaptor.getElements());
    rewriter.replaceOp(op, basePtr);
    return success();
}

LogicalResult ConvertTupleExtractOp::matchAndRewrite(
    ora::TupleExtractOp op,
    typename ora::TupleExtractOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    Value basePtr = coerceStructBasePtr(loc, rewriter, adaptor.getTupleValue(), /*fallbackBitcast=*/false);
    const uint64_t fieldIndex = static_cast<uint64_t>(op.getIndex());
    Value slotPtr = fieldPtrAtIndex(loc, rewriter, basePtr, fieldIndex);

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    Value loaded = rewriter.create<sir::LoadOp>(loc, u256Type, slotPtr);
    if (!resultType)
    {
        if (llvm::isa<ora::AdtType>(op.getResult().getType()))
        {
            Value view = ora::createMaterializationCast(rewriter, loc, op.getResult().getType(), loaded, mat_kind::kAdtHandleView);
            rewriter.replaceOp(op, view);
            return success();
        }
        return rewriter.notifyMatchFailure(op, "could not convert tuple extract result type");
    }
    if (resultType == u256Type)
    {
        rewriter.replaceOp(op, loaded);
        return success();
    }
    if (auto intType = llvm::dyn_cast<mlir::IntegerType>(resultType))
    {
        rewriter.replaceOpWithNewOp<sir::BitcastOp>(op, resultType, loaded);
        return success();
    }

    if (llvm::isa<ora::AdtType>(op.getResult().getType()))
    {
        Value view = ora::createMaterializationCast(rewriter, loc, op.getResult().getType(), loaded, mat_kind::kAdtHandleView);
        rewriter.replaceOp(op, view);
        return success();
    }

    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, resultType, loaded);
    return success();
}

LogicalResult LateLowerTupleExtractOp::matchAndRewrite(
    ora::TupleExtractOp op,
    PatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    Value basePtr = coerceStructBasePtr(loc, rewriter, op.getTupleValue(), /*fallbackBitcast=*/false);

    if (!llvm::isa<sir::PtrType>(basePtr.getType()))
        return failure();

    const uint64_t fieldIndex = static_cast<uint64_t>(op.getIndex());
    Value slotPtr = fieldPtrAtIndex(loc, rewriter, basePtr, fieldIndex);

    Type resultType = op.getResult().getType();
    if (typeConverter)
    {
        if (Type converted = typeConverter->convertType(resultType))
            resultType = converted;
    }

    Value loaded = rewriter.create<sir::LoadOp>(loc, u256Type, slotPtr);
    if (resultType == u256Type)
    {
        rewriter.replaceOp(op, loaded);
        return success();
    }
    if (llvm::isa<mlir::IntegerType>(resultType))
    {
        rewriter.replaceOpWithNewOp<sir::BitcastOp>(op, resultType, loaded);
        return success();
    }
    if (llvm::isa<ora::AdtType>(op.getResult().getType()))
    {
        Value view = ora::createMaterializationCast(rewriter, loc, op.getResult().getType(), loaded, mat_kind::kAdtHandleView);
        rewriter.replaceOp(op, view);
        return success();
    }
    if (llvm::isa<sir::PtrType>(resultType))
    {
        rewriter.replaceOpWithNewOp<sir::BitcastOp>(op, resultType, loaded);
        return success();
    }

    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, resultType, loaded);
    return success();
}

LogicalResult ConvertStructFieldExtractOp::matchAndRewrite(
    ora::StructFieldExtractOp op,
    typename ora::StructFieldExtractOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    StringRef fieldName = op.getFieldName();
    size_t fieldIndex = 0;
    bool found = false;
    Type expected = op.getResult().getType();

    if (auto structType = dyn_cast<ora::StructType>(op.getStructValue().getType()))
    {
        SmallVector<StringRef, 8> fieldNames;
        SmallVector<Type, 8> fieldTypes;
        if (succeeded(getStructFields(op.getOperation(), structType.getName(), fieldNames, fieldTypes)))
        {
            for (size_t i = 0; i < fieldNames.size(); ++i)
            {
                if (fieldNames[i] != fieldName)
                {
                    continue;
                }
                fieldIndex = i;
                expected = fieldTypes[i];
                found = true;
                break;
            }
        }
    }
    else if (auto anonType = dyn_cast<ora::AnonymousStructType>(op.getStructValue().getType()))
    {
        auto fieldNames = anonType.getFieldNames();
        auto fieldTypes = anonType.getFieldTypes();
        for (size_t i = 0; i < fieldNames.size(); ++i)
        {
            if (fieldNames[i] != fieldName)
            {
                continue;
            }
            fieldIndex = i;
            expected = fieldTypes[i];
            found = true;
            break;
        }
    }

    if (!found && parseNumericFieldIndex(fieldName, fieldIndex))
    {
        found = true;
    }

    if (!found && succeeded(resolveFieldByDeclScan(op.getOperation(), fieldName, op.getResult().getType(), fieldIndex, expected)))
    {
        found = true;
    }

    if (!found)
    {
        return rewriter.notifyMatchFailure(op, "unknown field in struct_field_extract");
    }

    auto forwardConstructedField = [&](Value fieldValue) -> LogicalResult {
        Type converted = getTypeConverter()->convertType(expected);
        if (!converted)
            converted = expected;
        Value result = fieldValue;
        if (result.getType() != converted)
        {
            if (llvm::isa<sir::U256Type>(result.getType()) &&
                llvm::isa<mlir::IntegerType>(converted) &&
                llvm::cast<mlir::IntegerType>(converted).getWidth() == 256)
            {
                result = rewriter.create<sir::BitcastOp>(loc, converted, result);
            }
            else
            {
                result = getTypeConverter()->materializeTargetConversion(rewriter, loc, converted, result);
                if (!result)
                    return failure();
            }
        }
        rewriter.replaceOp(op, result);
        return success();
    };

    if (auto init = op.getStructValue().getDefiningOp<ora::StructInitOp>())
    {
        if (fieldIndex < init.getFieldValues().size() &&
            succeeded(forwardConstructedField(init.getFieldValues()[fieldIndex])))
            return success();
    }
    if (auto instantiate = op.getStructValue().getDefiningOp<ora::StructInstantiateOp>())
    {
        if (fieldIndex < instantiate.getFieldValues().size() &&
            succeeded(forwardConstructedField(instantiate.getFieldValues()[fieldIndex])))
            return success();
    }

    if (auto memrefType = dyn_cast<mlir::MemRefType>(expected);
        memrefType && !memrefType.hasStaticShape())
    {
        auto forwardDynamicFieldValue = [&](Value fieldValue) -> LogicalResult {
            if (llvm::isa<mlir::MemRefType>(fieldValue.getType()))
            {
                if (auto bitcast = fieldValue.getDefiningOp<sir::BitcastOp>())
                {
                    auto viewKind = bitcast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
                    if (viewKind && viewKind.getValue() == kStorageMemRefViewKind)
                    {
                        Value storageView = rewriter.create<sir::BitcastOp>(loc, op.getResult().getType(), bitcast.getOperand());
                        storageView.getDefiningOp()->setAttr(
                            kOraMaterializationKindAttr,
                            StringAttr::get(ctx, kStorageMemRefViewKind));
                        rewriter.replaceOp(op, storageView);
                        return success();
                    }
                }
            }
            if (llvm::isa<sir::PtrType>(fieldValue.getType()))
            {
                rewriter.replaceOp(op, fieldValue);
                return success();
            }
            if (llvm::isa<sir::U256Type>(fieldValue.getType()))
            {
                Value storageView = rewriter.create<sir::BitcastOp>(loc, op.getResult().getType(), fieldValue);
                if (valueHasMaterializationKind(op.getStructValue(), kStorageStructCarrierKind))
                {
                    storageView.getDefiningOp()->setAttr(
                        kOraMaterializationKindAttr,
                        StringAttr::get(ctx, kStorageMemRefViewKind));
                }
                rewriter.replaceOp(op, storageView);
                return success();
            }
            return failure();
        };

        if (auto init = op.getStructValue().getDefiningOp<ora::StructInitOp>())
        {
            if (fieldIndex < init.getFieldValues().size() &&
                succeeded(forwardDynamicFieldValue(init.getFieldValues()[fieldIndex])))
                return success();
        }
        if (auto instantiate = op.getStructValue().getDefiningOp<ora::StructInstantiateOp>())
        {
            if (fieldIndex < instantiate.getFieldValues().size() &&
                succeeded(forwardDynamicFieldValue(instantiate.getFieldValues()[fieldIndex])))
                return success();
        }

        if (auto structType = dyn_cast<ora::StructType>(op.getStructValue().getType()))
        {
            auto fieldOffset = getStructFieldStorageOffset(
                op.getOperation(), structType, fieldName);
            Value structSlot = fieldOffset
                ? getStorageStructValueBaseSlot(op.getStructValue(), rewriter, loc)
                : Value();
            if (structSlot)
            {
                Value fieldSlot = addStorageWordOffset(loc, structSlot, *fieldOffset, rewriter);
                Value storageView = rewriter.create<sir::BitcastOp>(loc, op.getResult().getType(), fieldSlot);
                storageView.getDefiningOp()->setAttr(
                    kOraMaterializationKindAttr,
                    StringAttr::get(ctx, kStorageMemRefViewKind));
                rewriter.replaceOp(op, storageView);
                return success();
            }
        }

        Value structValue = op.getStructValue();
        while (auto cast = structValue.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        {
            if (cast.getNumOperands() != 1)
                break;
            structValue = cast.getOperand(0);
        }
        if (auto sload = structValue.getDefiningOp<ora::SLoadOp>())
        {
            auto slotIndexOpt = computeGlobalSlot(sload.getGlobalName(), op.getOperation());
            if (!slotIndexOpt)
                return rewriter.notifyMatchFailure(op, "missing slot for dynamic struct storage field");

            Value baseSlot = rewriter.create<sir::ConstOp>(
                loc, u256Type, mlir::IntegerAttr::get(ui64Type, *slotIndexOpt));
            Value fieldSlot = baseSlot;
            if (fieldIndex > 0)
            {
                Value offset = rewriter.create<sir::ConstOp>(
                    loc, u256Type, mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(fieldIndex)));
                fieldSlot = rewriter.create<sir::AddOp>(loc, u256Type, baseSlot, offset);
            }

            Value storageView = rewriter.create<sir::BitcastOp>(loc, op.getResult().getType(), fieldSlot);
            storageView.getDefiningOp()->setAttr(
                kOraMaterializationKindAttr,
                StringAttr::get(ctx, kStorageMemRefViewKind));
            rewriter.replaceOp(op, storageView);
            return success();
        }
    }

    Value basePtr = coerceStructBasePtr(loc, rewriter, adaptor.getStructValue(), /*fallbackBitcast=*/true);

    Value fieldPtr = fieldPtrAtIndex(loc, rewriter, basePtr, static_cast<uint64_t>(fieldIndex));

    Value raw = rewriter.create<sir::LoadOp>(loc, u256Type, fieldPtr);
    if (auto memrefType = dyn_cast<mlir::MemRefType>(expected);
        memrefType && !memrefType.hasStaticShape())
    {
        if (valueHasMaterializationKind(basePtr, kStorageStructCarrierKind))
        {
            Value storageView = rewriter.create<sir::BitcastOp>(loc, op.getResult().getType(), raw);
            storageView.getDefiningOp()->setAttr(
                kOraMaterializationKindAttr,
                StringAttr::get(ctx, kStorageMemRefViewKind));
            rewriter.replaceOp(op, storageView);
            return success();
        }
    }

    if (llvm::isa<ora::StructType>(expected) &&
        valueHasMaterializationKind(basePtr, kStorageStructCarrierKind))
    {
        Value storageCarrier = rewriter.create<sir::BitcastOp>(loc, ptrType, raw);
        storageCarrier.getDefiningOp()->setAttr(
            kOraMaterializationKindAttr,
            StringAttr::get(ctx, kStorageStructCarrierKind));
        rewriter.replaceOp(op, storageCarrier);
        return success();
    }

    Type converted = getTypeConverter()->convertType(expected);
    if (!converted)
    {
        if (llvm::isa<ora::AdtType>(expected))
        {
            Value view = ora::createMaterializationCast(rewriter, loc, expected, raw, mat_kind::kAdtHandleView);
            rewriter.replaceOp(op, view);
            return success();
        }
        converted = u256Type;
    }

    if (llvm::isa<sir::PtrType>(converted) &&
        valueHasMaterializationKind(basePtr, kStorageStructCarrierKind))
    {
        Value storageCarrier = rewriter.create<sir::BitcastOp>(loc, converted, raw);
        storageCarrier.getDefiningOp()->setAttr(
            kOraMaterializationKindAttr,
            StringAttr::get(ctx, kStorageStructCarrierKind));
        rewriter.replaceOp(op, storageCarrier);
        return success();
    }

    Value result = raw;
    if (converted != u256Type)
    {
        result = rewriter.create<sir::BitcastOp>(loc, converted, raw);
    }

    rewriter.replaceOp(op, result);
    return success();
}

LogicalResult ConvertStructFieldStoreOp::matchAndRewrite(
    ora::StructFieldStoreOp op,
    typename ora::StructFieldStoreOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);

    StringRef fieldName = op.getFieldName();
    size_t fieldIndex = 0;
    bool found = false;
    Type expected = adaptor.getValue().getType();

    if (auto structType = dyn_cast<ora::StructType>(op.getStructValue().getType()))
    {
        SmallVector<StringRef, 8> fieldNames;
        SmallVector<Type, 8> fieldTypes;
        if (succeeded(getStructFields(op.getOperation(), structType.getName(), fieldNames, fieldTypes)))
        {
            for (size_t i = 0; i < fieldNames.size(); ++i)
            {
                if (fieldNames[i] != fieldName)
                    continue;
                fieldIndex = i;
                expected = fieldTypes[i];
                found = true;
                break;
            }
        }
    }
    else if (auto anonType = dyn_cast<ora::AnonymousStructType>(op.getStructValue().getType()))
    {
        auto fieldNames = anonType.getFieldNames();
        auto fieldTypes = anonType.getFieldTypes();
        for (size_t i = 0; i < fieldNames.size(); ++i)
        {
            if (fieldNames[i] != fieldName)
                continue;
            fieldIndex = i;
            expected = fieldTypes[i];
            found = true;
            break;
        }
    }

    if (!found && parseNumericFieldIndex(fieldName, fieldIndex))
        found = true;

    if (!found && succeeded(resolveFieldByDeclScan(op.getOperation(), fieldName, adaptor.getValue().getType(), fieldIndex, expected)))
        found = true;

    if (!found)
        return rewriter.notifyMatchFailure(op, "unknown field in struct_field_store");

    Value basePtr = coerceStructBasePtr(loc, rewriter, adaptor.getStructValue(), /*fallbackBitcast=*/true);

    Value fieldPtr = fieldPtrAtIndex(loc, rewriter, basePtr, static_cast<uint64_t>(fieldIndex));

    Value val = adaptor.getValue();
    if (auto aggregateWord = materializeAggregateFieldWord(loc, rewriter, val))
    {
        val = *aggregateWord;
    }
    else
    {
        Type converted = getTypeConverter()->convertType(expected);
        if (converted && converted != val.getType())
            val = rewriter.create<sir::BitcastOp>(loc, converted, val);
        if (!llvm::isa<sir::U256Type>(val.getType()))
            val = rewriter.create<sir::BitcastOp>(loc, u256Type, val);
    }

    rewriter.create<sir::StoreOp>(loc, fieldPtr, val);
    rewriter.eraseOp(op);
    return success();
}

LogicalResult ConvertStructFieldUpdateOp::matchAndRewrite(
    ora::StructFieldUpdateOp op,
    typename ora::StructFieldUpdateOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto u256Type = sir::U256Type::get(ctx);
    auto ptrType = sir::PtrType::get(ctx, /*addrSpace*/ 1);
    auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);

    SmallVector<StringRef, 8> fieldNames;
    SmallVector<Type, 8> fieldTypes;
    if (auto structType = dyn_cast<ora::StructType>(op.getStructValue().getType()))
    {
        if (failed(getStructFields(op.getOperation(), structType.getName(), fieldNames, fieldTypes)))
        {
            return rewriter.notifyMatchFailure(op, "failed to resolve struct fields");
        }
    }
    else if (auto anonType = dyn_cast<ora::AnonymousStructType>(op.getStructValue().getType()))
    {
        fieldNames.append(anonType.getFieldNames().begin(), anonType.getFieldNames().end());
        fieldTypes.append(anonType.getFieldTypes().begin(), anonType.getFieldTypes().end());
    }
    else
    {
        return rewriter.notifyMatchFailure(op, "struct_field_update expects ora.struct or ora.struct_anon type");
    }

    StringRef fieldName = op.getFieldName();
    size_t fieldIndex = 0;
    bool found = false;
    for (size_t i = 0; i < fieldNames.size(); ++i)
    {
        if (fieldNames[i] == fieldName)
        {
            fieldIndex = i;
            found = true;
            break;
        }
    }
    if (!found)
    {
        return rewriter.notifyMatchFailure(op, "unknown field in struct_field_update");
    }

    Value basePtr = coerceStructBasePtr(loc, rewriter, adaptor.getStructValue(), /*fallbackBitcast=*/true);

    // Allocate new struct buffer
    const uint64_t fieldCount = static_cast<uint64_t>(fieldNames.size());
    const uint64_t byteSize = fieldCount * 32ULL;
    auto sizeAttr = mlir::IntegerAttr::get(ui64Type, byteSize);
    Value sizeVal = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
    Value newPtr = rewriter.create<sir::MallocOp>(loc, ptrType, sizeVal);
    Value carrierBase = basePtr;
    while (auto bitcast = carrierBase.getDefiningOp<sir::BitcastOp>())
        carrierBase = bitcast.getOperand();
    auto carrierKind = carrierBase.getDefiningOp()
                           ? carrierBase.getDefiningOp()->getAttrOfType<StringAttr>(kOraMaterializationKindAttr)
                           : StringAttr();
    const bool baseIsStorageStructCarrier =
        carrierKind && carrierKind.getValue() == kStorageStructCarrierKind;
    bool hasStorageMemRefField = false;
    SmallVector<Attribute, 2> storageViewFieldIndices;

    for (size_t i = 0; i < fieldNames.size(); ++i)
    {
        Value fieldPtr = fieldPtrAtIndex(loc, rewriter, basePtr, static_cast<uint64_t>(i));

        Value fieldVal = rewriter.create<sir::LoadOp>(loc, u256Type, fieldPtr);
        if (i == fieldIndex)
        {
            fieldVal = adaptor.getValue();
            if (auto bitcast = fieldVal.getDefiningOp<sir::BitcastOp>())
            {
                auto viewKind = bitcast->getAttrOfType<StringAttr>(kOraMaterializationKindAttr);
                if (viewKind && viewKind.getValue() == kStorageMemRefViewKind)
                    hasStorageMemRefField = true;
            }
            if (auto aggregateWord = materializeAggregateFieldWord(loc, rewriter, fieldVal))
                fieldVal = *aggregateWord;
            else if (!llvm::isa<sir::U256Type>(fieldVal.getType()))
                fieldVal = rewriter.create<sir::BitcastOp>(loc, u256Type, fieldVal);
        }
        else if (i < fieldTypes.size())
        {
            if (auto memrefType = dyn_cast<mlir::MemRefType>(fieldTypes[i]);
                memrefType && !memrefType.hasStaticShape() && baseIsStorageStructCarrier)
            {
                hasStorageMemRefField = true;
                storageViewFieldIndices.push_back(mlir::IntegerAttr::get(
                    ui64Type, static_cast<uint64_t>(i)));
            }
        }

        Value outPtr = fieldPtrAtIndex(loc, rewriter, newPtr, static_cast<uint64_t>(i));
        rewriter.create<sir::StoreOp>(loc, outPtr, fieldVal);
    }

    if (hasStorageMemRefField)
    {
        newPtr.getDefiningOp()->setAttr(
            kOraMaterializationKindAttr,
            StringAttr::get(ctx, kStorageStructCarrierKind));
        if (!storageViewFieldIndices.empty())
        {
            newPtr.getDefiningOp()->setAttr(
                kStorageStructViewFieldsAttr,
                ArrayAttr::get(ctx, storageViewFieldIndices));
        }
    }

    rewriter.replaceOp(op, newPtr);
    return success();
}

LogicalResult ConvertStructDeclOp::matchAndRewrite(
    ora::StructDeclOp op,
    typename ora::StructDeclOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    (void)adaptor;
    rewriter.eraseOp(op);
    return success();
}

LogicalResult StripStructMaterializeOp::matchAndRewrite(
    mlir::UnrealizedConversionCastOp op,
    mlir::UnrealizedConversionCastOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    {
        return rewriter.notifyMatchFailure(op, "expected single operand/result");
    }

    if (!llvm::isa<ora::StructType, ora::AnonymousStructType>(op.getResult(0).getType()))
    {
        return rewriter.notifyMatchFailure(op, "not a struct materialization");
    }

    Value input = adaptor.getOperands()[0];
    if (!llvm::isa<sir::PtrType>(input.getType()))
    {
        return rewriter.notifyMatchFailure(op, "expected sir.ptr operand");
    }

    rewriter.replaceOp(op, input);
    return success();
}

LogicalResult StripAddressMaterializeOp::matchAndRewrite(
    mlir::UnrealizedConversionCastOp op,
    mlir::UnrealizedConversionCastOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    {
        return rewriter.notifyMatchFailure(op, "expected single operand/result");
    }

    Type resultType = op.getResult(0).getType();
    if (!llvm::isa<ora::AddressType, ora::NonZeroAddressType>(resultType))
    {
        return rewriter.notifyMatchFailure(op, "not an address materialization");
    }

    Value input = adaptor.getOperands()[0];
    if (!llvm::isa<sir::U256Type>(input.getType()))
    {
        return rewriter.notifyMatchFailure(op, "expected sir.u256 operand");
    }

    rewriter.replaceOp(op, input);
    return success();
}

LogicalResult StripBytesMaterializeOp::matchAndRewrite(
    mlir::UnrealizedConversionCastOp op,
    mlir::UnrealizedConversionCastOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const
{
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    {
        return rewriter.notifyMatchFailure(op, "expected single operand/result");
    }

    Type resultType = op.getResult(0).getType();
    if (!llvm::isa<ora::StringType, ora::BytesType>(resultType))
    {
        return rewriter.notifyMatchFailure(op, "not a bytes/string materialization");
    }

    Value input = adaptor.getOperands()[0];
    if (!llvm::isa<sir::PtrType>(input.getType()))
    {
        return rewriter.notifyMatchFailure(op, "expected sir.ptr operand");
    }

    rewriter.replaceOp(op, input);
    return success();
}
