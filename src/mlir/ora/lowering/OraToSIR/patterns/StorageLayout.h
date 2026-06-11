#pragma once

#include "patterns/AdtCarrierLayout.h"
#include "LoweringHelpers.h"

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir::ora::lowering
{
    inline constexpr llvm::StringLiteral kStorageMemRefViewKind{"storage_memref_view"};
    inline constexpr llvm::StringLiteral kStorageStructCarrierKind{"storage_struct_carrier"};
    inline constexpr llvm::StringLiteral kStorageStructViewFieldsAttr{"ora.storage_struct_view_fields"};

    inline ora::StructDeclOp findStructDeclForName(Operation *op, StringRef structName)
    {
        ModuleOp module = op->getParentOfType<ModuleOp>();
        if (!module)
            return nullptr;

        ora::StructDeclOp structDecl = nullptr;
        module.walk([&](ora::StructDeclOp declOp)
                    {
            auto declNameAttr = declOp->getAttrOfType<StringAttr>("sym_name");
            if (declNameAttr && declNameAttr.getValue() == structName)
            {
                structDecl = declOp;
                return WalkResult::interrupt();
            }
            return WalkResult::advance(); });
        return structDecl;
    }

    inline bool getStructFieldAttrs(Operation *anchor,
                                    ora::StructType structType,
                                    ArrayAttr &fieldNamesAttr,
                                    ArrayAttr &fieldTypesAttr)
    {
        auto structDecl = findStructDeclForName(anchor, structType.getName());
        if (!structDecl)
            return false;

        fieldNamesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_names");
        fieldTypesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_types");
        return fieldNamesAttr && fieldTypesAttr && fieldNamesAttr.size() == fieldTypesAttr.size();
    }

    inline LogicalResult getStructFieldsFromDecl(ora::StructDeclOp structDecl,
                                                 SmallVectorImpl<StringRef> &names,
                                                 SmallVectorImpl<Type> &types)
    {
        auto fieldNamesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_names");
        auto fieldTypesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_types");
        if (!fieldNamesAttr || !fieldTypesAttr || fieldNamesAttr.size() != fieldTypesAttr.size())
            return failure();

        for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
        {
            auto nameAttr = cast<StringAttr>(fieldNamesAttr[i]);
            auto typeAttr = cast<TypeAttr>(fieldTypesAttr[i]);
            names.push_back(nameAttr.getValue());
            types.push_back(typeAttr.getValue());
        }

        return success();
    }

    inline LogicalResult getStructFields(Operation *op,
                                         StringRef structName,
                                         SmallVectorImpl<StringRef> &names,
                                         SmallVectorImpl<Type> &types)
    {
        auto structDecl = findStructDeclForName(op, structName);
        if (!structDecl)
            return failure();

        return getStructFieldsFromDecl(structDecl, names, types);
    }

    inline std::optional<uint64_t> getStaticMemRefElementCount(mlir::MemRefType memrefType)
    {
        if (!memrefType.hasStaticShape())
            return std::nullopt;

        uint64_t elements = 1;
        for (int64_t dim : memrefType.getShape())
        {
            if (dim < 0)
                return std::nullopt;
            elements *= static_cast<uint64_t>(dim);
        }
        return elements;
    }

    inline uint64_t getStorageWordCount(Operation *anchor, Type type)
    {
        if (llvm::isa<ora::ErrorUnionType>(type))
            return ::mlir::ora::adt_helpers::kAdtCarrierWordCount;

        if (auto structType = llvm::dyn_cast<ora::StructType>(type))
        {
            ArrayAttr fieldNamesAttr;
            ArrayAttr fieldTypesAttr;
            if (getStructFieldAttrs(anchor, structType, fieldNamesAttr, fieldTypesAttr))
            {
                uint64_t words = 0;
                for (Attribute fieldTypeAttr : fieldTypesAttr)
                {
                    Type fieldType = cast<TypeAttr>(fieldTypeAttr).getValue();
                    words += getStorageWordCount(anchor, fieldType);
                }
                return words == 0 ? 1 : words;
            }
        }

        if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type))
        {
            if (!memrefType.hasStaticShape())
                return 1;
            if (auto elements = getStaticMemRefElementCount(memrefType))
                return *elements * getStorageWordCount(anchor, memrefType.getElementType());
            return 1;
        }

        return 1;
    }

    inline uint64_t getElementWordCount(Operation *anchor, Type elementType)
    {
        return getStorageWordCount(anchor, elementType);
    }

    inline uint64_t getMemRefElementWordCount(Operation *anchor, Type elementType)
    {
        return getStorageWordCount(anchor, elementType);
    }

    inline std::optional<uint64_t> getStaticMemRefWordCount(Operation *anchor, mlir::MemRefType memrefType)
    {
        auto elements = getStaticMemRefElementCount(memrefType);
        if (!elements)
            return std::nullopt;
        return *elements * getStorageWordCount(anchor, memrefType.getElementType());
    }

    inline std::optional<uint64_t> getStructFieldStorageOffset(Operation *anchor,
                                                               ora::StructType structType,
                                                               StringRef fieldName,
                                                               size_t *fieldIndex = nullptr)
    {
        ArrayAttr fieldNamesAttr;
        ArrayAttr fieldTypesAttr;
        if (!getStructFieldAttrs(anchor, structType, fieldNamesAttr, fieldTypesAttr))
            return std::nullopt;

        uint64_t offset = 0;
        for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
        {
            if (cast<StringAttr>(fieldNamesAttr[i]).getValue() == fieldName)
            {
                if (fieldIndex)
                    *fieldIndex = i;
                return offset;
            }
            Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
            offset += getStorageWordCount(anchor, fieldType);
        }

        return std::nullopt;
    }

    inline std::optional<uint64_t> getStructFieldStorageOffsetByScan(Operation *anchor,
                                                                     StringRef fieldName,
                                                                     Type resultType)
    {
        ModuleOp module = anchor ? anchor->getParentOfType<ModuleOp>() : ModuleOp();
        if (!module)
            return std::nullopt;

        std::optional<uint64_t> foundOffset;
        bool ambiguous = false;
        module.walk([&](ora::StructDeclOp declOp)
                    {
            ArrayAttr fieldNamesAttr = declOp->getAttrOfType<ArrayAttr>("ora.field_names");
            ArrayAttr fieldTypesAttr = declOp->getAttrOfType<ArrayAttr>("ora.field_types");
            if (!fieldNamesAttr || !fieldTypesAttr || fieldNamesAttr.size() != fieldTypesAttr.size())
                return WalkResult::advance();

            uint64_t offset = 0;
            for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
            {
                Type fieldType = cast<TypeAttr>(fieldTypesAttr[i]).getValue();
                if (cast<StringAttr>(fieldNamesAttr[i]).getValue() == fieldName &&
                    (!resultType || fieldType == resultType))
                {
                    if (foundOffset && *foundOffset != offset)
                    {
                        ambiguous = true;
                        return WalkResult::interrupt();
                    }
                    foundOffset = offset;
                }
                offset += getStorageWordCount(anchor, fieldType);
            }

            return WalkResult::advance(); });

        if (ambiguous)
            return std::nullopt;
        return foundOffset;
    }

    inline Value addStorageWordOffset(Location loc,
                                      Value slot,
                                      uint64_t offset,
                                      OpBuilder &rewriter)
    {
        if (offset == 0)
            return slot;

        auto *ctx = rewriter.getContext();
        auto u256Type = sir::U256Type::get(ctx);
        auto ui64Type = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
        Value offsetValue = rewriter.create<sir::ConstOp>(
            loc, u256Type, mlir::IntegerAttr::get(ui64Type, offset));
        return rewriter.create<sir::AddOp>(loc, u256Type, slot, offsetValue);
    }
}
