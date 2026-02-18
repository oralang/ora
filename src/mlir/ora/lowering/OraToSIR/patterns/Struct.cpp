#include "Struct.h"

#include "OraToSIRTypeConverter.h"
#include "OraDebug.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::ora;

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
            auto nameAttr = declOp->getAttrOfType<StringAttr>("name");
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
                if (resultTypeHint && types[i] != resultTypeHint)
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

        for (size_t i = 0; i < fieldValues.size(); ++i)
        {
            Value val = fieldValues[i];
            if (!llvm::isa<sir::U256Type>(val.getType()))
            {
                val = rewriter.create<sir::BitcastOp>(loc, u256Type, val);
            }

            if (i == 0)
            {
                rewriter.create<sir::StoreOp>(loc, basePtr, val);
                continue;
            }

            auto offsetAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(i) * 32ULL);
            Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, offsetAttr);
            Value ptrOff = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, offset);
            rewriter.create<sir::StoreOp>(loc, ptrOff, val);
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

    Value basePtr = adaptor.getStructValue();
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
    if (!llvm::isa<sir::PtrType>(basePtr.getType()))
    {
        // Last-resort materialization path: tuples/anonymous structs may still
        // appear as non-SIR values at this point; represent them as pointers.
        basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, basePtr);
    }

    Value fieldPtr = basePtr;
    if (fieldIndex > 0)
    {
        auto offsetAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(fieldIndex) * 32ULL);
        Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, offsetAttr);
        fieldPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, offset);
    }

    Value raw = rewriter.create<sir::LoadOp>(loc, u256Type, fieldPtr);
    Type converted = getTypeConverter()->convertType(expected);
    if (!converted)
    {
        converted = u256Type;
    }

    Value result = raw;
    if (converted != u256Type)
    {
        result = rewriter.create<sir::BitcastOp>(loc, converted, raw);
    }

    rewriter.replaceOp(op, result);
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

    auto structType = dyn_cast<ora::StructType>(op.getStructValue().getType());
    if (!structType)
    {
        return rewriter.notifyMatchFailure(op, "struct_field_update expects ora.struct type");
    }

    SmallVector<StringRef, 8> fieldNames;
    SmallVector<Type, 8> fieldTypes;
    if (failed(getStructFields(op.getOperation(), structType.getName(), fieldNames, fieldTypes)))
    {
        return rewriter.notifyMatchFailure(op, "failed to resolve struct fields");
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

    Value basePtr = adaptor.getStructValue();
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
    if (!llvm::isa<sir::PtrType>(basePtr.getType()))
    {
        basePtr = rewriter.create<sir::BitcastOp>(loc, ptrType, basePtr);
    }

    // Allocate new struct buffer
    const uint64_t fieldCount = static_cast<uint64_t>(fieldNames.size());
    const uint64_t byteSize = fieldCount * 32ULL;
    auto sizeAttr = mlir::IntegerAttr::get(ui64Type, byteSize);
    Value sizeVal = rewriter.create<sir::ConstOp>(loc, u256Type, sizeAttr);
    Value newPtr = rewriter.create<sir::MallocOp>(loc, ptrType, sizeVal);

    for (size_t i = 0; i < fieldNames.size(); ++i)
    {
        Value fieldPtr = basePtr;
        if (i > 0)
        {
            auto offsetAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(i) * 32ULL);
            Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, offsetAttr);
            fieldPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, basePtr, offset);
        }

        Value fieldVal = rewriter.create<sir::LoadOp>(loc, u256Type, fieldPtr);
        if (i == fieldIndex)
        {
            fieldVal = adaptor.getValue();
            if (!llvm::isa<sir::U256Type>(fieldVal.getType()))
            {
                fieldVal = rewriter.create<sir::BitcastOp>(loc, u256Type, fieldVal);
            }
        }

        Value outPtr = newPtr;
        if (i > 0)
        {
            auto offsetAttr = mlir::IntegerAttr::get(ui64Type, static_cast<uint64_t>(i) * 32ULL);
            Value offset = rewriter.create<sir::ConstOp>(loc, u256Type, offsetAttr);
            outPtr = rewriter.create<sir::AddPtrOp>(loc, ptrType, newPtr, offset);
        }
        rewriter.create<sir::StoreOp>(loc, outPtr, fieldVal);
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

    if (!llvm::isa<ora::StructType>(op.getResult(0).getType()))
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
