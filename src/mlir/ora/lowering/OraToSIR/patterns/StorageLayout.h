#pragma once

#include "OraDialect.h"
#include "SIR/SIRDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::ora::lowering
{
    inline constexpr llvm::StringLiteral kStorageMemRefViewKind{"storage_memref_view"};
    inline constexpr llvm::StringLiteral kStorageStructCarrierKind{"storage_struct_carrier"};
    inline constexpr llvm::StringLiteral kStorageStructViewFieldsAttr{"ora.storage_struct_view_fields"};

    inline Value ensureU256Value(PatternRewriter &rewriter, Location loc, Value value)
    {
        if (llvm::isa<sir::U256Type>(value.getType()))
            return value;
        auto u256Type = sir::U256Type::get(rewriter.getContext());
        return rewriter.create<sir::BitcastOp>(loc, u256Type, value);
    }

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

    inline uint64_t getStorageWordCount(Operation *anchor, Type type)
    {
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

        if (auto memrefType = llvm::dyn_cast<mlir::MemRefType>(type);
            memrefType && !memrefType.hasStaticShape())
        {
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

    inline Value addStorageWordOffset(Location loc,
                                      Value slot,
                                      uint64_t offset,
                                      ConversionPatternRewriter &rewriter)
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
