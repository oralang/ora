//===- OraDialect.cpp - Ora dialect implementation ----------------------===//
//
// This file implements the Ora MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "OraDialect.h"

// Type definitions
#define GET_TYPEDEF_CLASSES
#include "OraTypes.cpp.inc"

namespace mlir
{
    namespace ora
    {
        /// Initialize the Ora dialect and add operations and types
        void OraDialect::initialize()
        {
            addOperations<
#define GET_OP_LIST
#include "OraOps.cpp.inc"
                >();
            addTypes<
#define GET_TYPEDEF_LIST
#include "OraTypes.cpp.inc"
                >();
        }

        ::mlir::Attribute OraDialect::parseAttribute(::mlir::DialectAsmParser &parser, ::mlir::Type type) const
        {
            return ::mlir::Dialect::parseAttribute(parser, type);
        }

        void OraDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter &printer) const
        {
            ::mlir::Dialect::printAttribute(attr, printer);
        }

    } // namespace ora
} // namespace mlir

namespace mlir
{
    namespace ora
    {
        namespace
        {
            static bool nextTokenIsEllipsis(::mlir::OpAsmParser &parser)
            {
                const char *ptr = parser.getCurrentLocation().getPointer();
                return ptr[0] == '.' && ptr[1] == '.' && ptr[2] == '.';
            }

            static ::mlir::Type getRefinementBaseType(::mlir::Type type)
            {
                if (auto minType = llvm::dyn_cast<MinValueType>(type))
                    return minType.getBaseType();
                if (auto maxType = llvm::dyn_cast<MaxValueType>(type))
                    return maxType.getBaseType();
                if (auto rangeType = llvm::dyn_cast<InRangeType>(type))
                    return rangeType.getBaseType();
                if (auto scaledType = llvm::dyn_cast<ScaledType>(type))
                    return scaledType.getBaseType();
                if (auto exactType = llvm::dyn_cast<ExactType>(type))
                    return exactType.getBaseType();
                if (llvm::isa<NonZeroAddressType>(type))
                    return AddressType::get(type.getContext());
                return {};
            }
        }

        // TupleType: !ora.tuple<type1, type2, ...>
        ::mlir::Type TupleType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::llvm::SmallVector<::mlir::Type> elementTypes;
            if (parser.parseTypeList(elementTypes))
                return {};

            if (parser.parseGreater())
                return {};

            return TupleType::get(parser.getContext(), elementTypes);
        }

        void TupleType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            auto elementTypes = getElementTypes();
            for (size_t i = 0; i < elementTypes.size(); ++i)
            {
                if (i > 0)
                    printer << ", ";
                printer.printType(elementTypes[i]);
            }
            printer << ">";
        }

        // StructType: !ora.struct<"struct_name">
        ::mlir::Type StructType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            std::string name;
            if (parser.parseString(&name))
                return {};

            if (parser.parseGreater())
                return {};

            return StructType::get(parser.getContext(), name);
        }

        void StructType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<\"";
            printer << getName();
            printer << "\">";
        }

        // EnumType: !ora.enum<"enum_name", repr_type>
        ::mlir::Type EnumType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            std::string name;
            if (parser.parseString(&name))
                return {};

            if (parser.parseComma())
                return {};

            ::mlir::Type reprType;
            if (parser.parseType(reprType))
                return {};

            if (parser.parseGreater())
                return {};

            return EnumType::get(parser.getContext(), name, reprType);
        }

        void EnumType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<\"";
            printer << getName();
            printer << "\", ";
            printer.printType(getReprType());
            printer << ">";
        }

        // ContractType: !ora.contract<"contract_name">
        ::mlir::Type ContractType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            std::string name;
            if (parser.parseString(&name))
                return {};

            if (parser.parseGreater())
                return {};

            return ContractType::get(parser.getContext(), name);
        }

        void ContractType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<\"";
            printer << getName();
            printer << "\">";
        }

        // FunctionType: !ora.function<param_types, return_type>
        ::mlir::Type FunctionType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::llvm::SmallVector<::mlir::Type> paramTypes;
            if (parser.parseTypeList(paramTypes))
                return {};

            if (parser.parseComma())
                return {};

            ::mlir::Type returnType;
            if (parser.parseType(returnType))
                return {};

            if (parser.parseGreater())
                return {};

            return FunctionType::get(parser.getContext(), paramTypes, returnType);
        }

        void FunctionType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            auto paramTypes = getParamTypes();
            for (size_t i = 0; i < paramTypes.size(); ++i)
            {
                if (i > 0)
                    printer << ", ";
                printer.printType(paramTypes[i]);
            }
            printer << ", ";
            printer.printType(getReturnType());
            printer << ">";
        }

        // UnionType: !ora.union<type1, type2, ...>
        ::mlir::Type UnionType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::llvm::SmallVector<::mlir::Type> elementTypes;
            if (parser.parseTypeList(elementTypes))
                return {};

            if (parser.parseGreater())
                return {};

            return UnionType::get(parser.getContext(), elementTypes);
        }

        void UnionType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            auto elementTypes = getElementTypes();
            for (size_t i = 0; i < elementTypes.size(); ++i)
            {
                if (i > 0)
                    printer << ", ";
                printer.printType(elementTypes[i]);
            }
            printer << ">";
        }

        // MinValueType: !ora.min_value<base_type, min_high_high, min_high_low, min_low_high, min_low_low>
        ::mlir::Type MinValueType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::mlir::Type baseType;
            if (parser.parseType(baseType))
                return {};

            if (parser.parseComma())
                return {};

            uint64_t minHighHigh, minHighLow, minLowHigh, minLowLow;
            if (parser.parseInteger(minHighHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minHighLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minLowHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minLowLow))
                return {};

            if (parser.parseGreater())
                return {};

            return MinValueType::get(parser.getContext(), baseType, minHighHigh, minHighLow, minLowHigh, minLowLow);
        }

        void MinValueType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            printer.printType(getBaseType());
            printer << ", ";
            printer << getMinHighHigh();
            printer << ", ";
            printer << getMinHighLow();
            printer << ", ";
            printer << getMinLowHigh();
            printer << ", ";
            printer << getMinLowLow();
            printer << ">";
        }

        // MaxValueType: !ora.max_value<base_type, max_high_high, max_high_low, max_low_high, max_low_low>
        ::mlir::Type MaxValueType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::mlir::Type baseType;
            if (parser.parseType(baseType))
                return {};

            if (parser.parseComma())
                return {};

            uint64_t maxHighHigh, maxHighLow, maxLowHigh, maxLowLow;
            if (parser.parseInteger(maxHighHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxHighLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxLowHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxLowLow))
                return {};

            if (parser.parseGreater())
                return {};

            return MaxValueType::get(parser.getContext(), baseType, maxHighHigh, maxHighLow, maxLowHigh, maxLowLow);
        }

        void MaxValueType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            printer.printType(getBaseType());
            printer << ", ";
            printer << getMaxHighHigh();
            printer << ", ";
            printer << getMaxHighLow();
            printer << ", ";
            printer << getMaxLowHigh();
            printer << ", ";
            printer << getMaxLowLow();
            printer << ">";
        }

        // InRangeType: !ora.in_range<base_type, min_high_high, min_high_low, min_low_high, min_low_low, max_high_high, max_high_low, max_low_high, max_low_low>
        ::mlir::Type InRangeType::parse(::mlir::AsmParser &parser)
        {
            if (parser.parseLess())
                return {};

            ::mlir::Type baseType;
            if (parser.parseType(baseType))
                return {};

            if (parser.parseComma())
                return {};

            uint64_t minHighHigh, minHighLow, minLowHigh, minLowLow;
            uint64_t maxHighHigh, maxHighLow, maxLowHigh, maxLowLow;
            if (parser.parseInteger(minHighHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minHighLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minLowHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(minLowLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxHighHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxHighLow))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxLowHigh))
                return {};
            if (parser.parseComma())
                return {};
            if (parser.parseInteger(maxLowLow))
                return {};

            if (parser.parseGreater())
                return {};

            return InRangeType::get(parser.getContext(), baseType, minHighHigh, minHighLow, minLowHigh, minLowLow, maxHighHigh, maxHighLow, maxLowHigh, maxLowLow);
        }

        void InRangeType::print(::mlir::AsmPrinter &printer) const
        {
            printer << "<";
            printer.printType(getBaseType());
            printer << ", ";
            printer << getMinHighHigh();
            printer << ", ";
            printer << getMinHighLow();
            printer << ", ";
            printer << getMinLowHigh();
            printer << ", ";
            printer << getMinLowLow();
            printer << ", ";
            printer << getMaxHighHigh();
            printer << ", ";
            printer << getMaxHighLow();
            printer << ", ";
            printer << getMaxLowHigh();
            printer << ", ";
            printer << getMaxLowLow();
            printer << ">";
        }

        // RefinementToBaseOp: ora.refinement_to_base
        void RefinementToBaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value)
        {
            Type valueType = value.getType();
            Type baseType = getRefinementBaseType(valueType);
            if (!baseType)
                baseType = valueType;

            odsState.addOperands(value);
            odsState.addTypes(baseType);
        }

    } // namespace ora
} // namespace mlir

// Include the generated operation method implementations
// This must come BEFORE custom print/parse methods so the operation classes are fully defined
#define GET_OP_CLASSES
#include "OraOps.cpp.inc"

namespace mlir
{
    namespace ora
    {
        namespace
        {
            static bool isStaticallyZeroIntegerValue(::mlir::Value value)
            {
                if (auto constOp = value.getDefiningOp<arith::ConstantOp>())
                {
                    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                    {
                        return intAttr.getValue().isZero();
                    }
                }
                return false;
            }

            static std::optional<mlir::IntegerAttr> getConstantIntegerAttr(::mlir::Value value)
            {
                if (auto constOp = value.getDefiningOp<arith::ConstantOp>())
                {
                    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                        return intAttr;
                }
                return std::nullopt;
            }

            static GlobalOp lookupGlobalFor(Operation *op, StringRef globalName)
            {
                auto globalAttr = mlir::StringAttr::get(op->getContext(), globalName);
                if (Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(op, globalAttr))
                    return llvm::dyn_cast<GlobalOp>(symbol);
                return nullptr;
            }

            static StructDeclOp findStructDecl(Operation *op, StringRef structName)
            {
                auto structAttr = mlir::StringAttr::get(op->getContext(), structName);
                if (Operation *symbol = mlir::SymbolTable::lookupNearestSymbolFrom(op, structAttr))
                    return llvm::dyn_cast<StructDeclOp>(symbol);
                return nullptr;
            }

            static mlir::LogicalResult replaceOpWithConstant(
                PatternRewriter &rewriter,
                Operation *op,
                ::mlir::Type resultType,
                const llvm::APInt &value)
            {
                auto valueAttr = mlir::IntegerAttr::get(resultType, value);
                auto newConst = rewriter.create<arith::ConstantOp>(op->getLoc(), resultType, valueAttr);
                rewriter.replaceOp(op, newConst.getResult());
                return success();
            }

            static ::mlir::LogicalResult getStructFieldInfo(
                Operation *op,
                StructType structType,
                StringRef fieldName,
                size_t &fieldIndex,
                ::mlir::Type &fieldType)
            {
                auto structDecl = findStructDecl(op, structType.getName());
                if (!structDecl)
                    return op->emitError() << "missing ora.struct.decl for struct '" << structType.getName() << "'";

                auto fieldNamesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_names");
                auto fieldTypesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_types");
                if (!fieldNamesAttr || !fieldTypesAttr || fieldNamesAttr.size() != fieldTypesAttr.size())
                    return op->emitError() << "struct declaration for '" << structType.getName() << "' has malformed field metadata";

                for (size_t i = 0; i < fieldNamesAttr.size(); ++i)
                {
                    auto nameAttr = llvm::dyn_cast<StringAttr>(fieldNamesAttr[i]);
                    auto typeAttr = llvm::dyn_cast<TypeAttr>(fieldTypesAttr[i]);
                    if (!nameAttr || !typeAttr)
                        return op->emitError() << "struct declaration for '" << structType.getName() << "' has invalid field metadata";
                    if (nameAttr.getValue() == fieldName)
                    {
                        fieldIndex = i;
                        fieldType = typeAttr.getValue();
                        return success();
                    }
                }

                return op->emitError() << "unknown field '" << fieldName << "' on struct '" << structType.getName() << "'";
            }
        } // namespace

        ::mlir::LogicalResult DivOp::verify()
        {
            if (isStaticallyZeroIntegerValue(getRhs()))
                return emitOpError("divisor must not be a statically known zero constant");
            return success();
        }

        ::mlir::LogicalResult RemOp::verify()
        {
            if (isStaticallyZeroIntegerValue(getRhs()))
                return emitOpError("divisor must not be a statically known zero constant");
            return success();
        }

        ::mlir::LogicalResult SLoadOp::verify()
        {
            auto global = lookupGlobalFor(*this, getGlobalName());
            if (!global)
                return emitOpError() << "unknown storage global '" << getGlobalName() << "'";

            if (getResult().getType() != global.getGlobalType())
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match storage global type " << global.getGlobalType()
                                     << " for '" << getGlobalName() << "'";
            }

            return success();
        }

        ::mlir::LogicalResult SStoreOp::verify()
        {
            auto global = lookupGlobalFor(*this, getGlobalName());
            if (!global)
                return emitOpError() << "unknown storage global '" << getGlobalName() << "'";

            if (getValue().getType() != global.getGlobalType())
            {
                return emitOpError() << "stored value type " << getValue().getType()
                                     << " does not match storage global type " << global.getGlobalType()
                                     << " for '" << getGlobalName() << "'";
            }

            return success();
        }

        ::mlir::LogicalResult StructInitOp::verify()
        {
            auto structType = llvm::dyn_cast<StructType>(getResult().getType());
            if (!structType)
                return emitOpError("result type must be !ora.struct<...>");

            auto structDecl = findStructDecl(*this, structType.getName());
            if (!structDecl)
                return emitOpError() << "missing ora.struct.decl for struct '" << structType.getName() << "'";

            auto fieldTypesAttr = structDecl->getAttrOfType<ArrayAttr>("ora.field_types");
            if (!fieldTypesAttr)
                return emitOpError() << "struct declaration for '" << structType.getName() << "' is missing ora.field_types";

            if (getFieldValues().size() != fieldTypesAttr.size())
            {
                return emitOpError() << "expected " << fieldTypesAttr.size()
                                     << " field values for struct '" << structType.getName()
                                     << "', got " << getFieldValues().size();
            }

            for (size_t i = 0; i < fieldTypesAttr.size(); ++i)
            {
                auto typeAttr = llvm::dyn_cast<TypeAttr>(fieldTypesAttr[i]);
                if (!typeAttr)
                    return emitOpError() << "struct declaration for '" << structType.getName() << "' has invalid field type metadata";
                if (getFieldValues()[i].getType() != typeAttr.getValue())
                {
                    return emitOpError() << "field operand #" << i << " has type " << getFieldValues()[i].getType()
                                         << " but struct '" << structType.getName() << "' expects " << typeAttr.getValue();
                }
            }

            return success();
        }

        ::mlir::LogicalResult StructFieldExtractOp::verify()
        {
            auto structType = llvm::dyn_cast<StructType>(getStructValue().getType());
            if (!structType)
                return emitOpError("struct operand must have !ora.struct<...> type");

            size_t fieldIndex = 0;
            ::mlir::Type fieldType;
            if (failed(getStructFieldInfo(*this, structType, getFieldName(), fieldIndex, fieldType)))
                return failure();

            if (getResult().getType() != fieldType)
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match field '" << getFieldName()
                                     << "' type " << fieldType;
            }

            return success();
        }

        ::mlir::LogicalResult StructFieldUpdateOp::verify()
        {
            auto structType = llvm::dyn_cast<StructType>(getStructValue().getType());
            if (!structType)
                return emitOpError("struct operand must have !ora.struct<...> type");

            if (getResult().getType() != getStructValue().getType())
                return emitOpError("result type must match the input struct type");

            size_t fieldIndex = 0;
            ::mlir::Type fieldType;
            if (failed(getStructFieldInfo(*this, structType, getFieldName(), fieldIndex, fieldType)))
                return failure();

            if (getValue().getType() != fieldType)
            {
                return emitOpError() << "updated value type " << getValue().getType()
                                     << " does not match field '" << getFieldName()
                                     << "' type " << fieldType;
            }

            return success();
        }

        ::mlir::LogicalResult MapGetOp::verify()
        {
            auto mapType = llvm::dyn_cast<MapType>(getMap().getType());
            if (!mapType)
                return emitOpError("map operand must have !ora.map<key, value> type");

            if (getKey().getType() != mapType.getKeyType())
                return emitOpError() << "key type " << getKey().getType()
                                     << " does not match map key type " << mapType.getKeyType();

            if (getResult().getType() != mapType.getValueType())
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match map value type " << mapType.getValueType();
            }

            return success();
        }

        ::mlir::LogicalResult MapStoreOp::verify()
        {
            auto mapType = llvm::dyn_cast<MapType>(getMap().getType());
            if (!mapType)
                return emitOpError("map operand must have !ora.map<key, value> type");

            if (getKey().getType() != mapType.getKeyType())
                return emitOpError() << "key type " << getKey().getType()
                                     << " does not match map key type " << mapType.getKeyType();

            if (getValue().getType() != mapType.getValueType())
            {
                return emitOpError() << "value type " << getValue().getType()
                                     << " does not match map value type " << mapType.getValueType();
            }

            return success();
        }

        ::mlir::LogicalResult ErrorUnwrapOp::verify()
        {
            auto unionType = llvm::dyn_cast<ErrorUnionType>(getValue().getType());
            if (!unionType)
                return emitOpError("operand must have !ora.error_union<...> type");

            if (getResult().getType() != unionType.getSuccessType())
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match error union success type "
                                     << unionType.getSuccessType();
            }

            return success();
        }

        ::mlir::LogicalResult RefinementToBaseOp::verify()
        {
            auto expectedBaseType = getRefinementBaseType(getValue().getType());
            if (!expectedBaseType)
                return emitOpError("operand must have an Ora refinement type");

            if (getResult().getType() != expectedBaseType)
            {
                return emitOpError() << "result type " << getResult().getType()
                                     << " does not match refinement base type " << expectedBaseType;
            }

            return success();
        }

        ::mlir::LogicalResult BaseToRefinementOp::verify()
        {
            auto expectedBaseType = getRefinementBaseType(getResult().getType());
            if (!expectedBaseType)
                return emitOpError("result type must be an Ora refinement type");

            if (getValue().getType() != expectedBaseType)
            {
                return emitOpError() << "operand type " << getValue().getType()
                                     << " does not match refinement base type " << expectedBaseType;
            }

            return success();
        }

        ::mlir::LogicalResult ReturnOp::verify()
        {
            auto func = (*this)->getParentOfType<mlir::func::FuncOp>();
            if (!func)
                return emitOpError("must be nested inside func.func");

            auto resultTypes = func.getFunctionType().getResults();
            if (getOperands().size() != resultTypes.size())
            {
                return emitOpError() << "expected " << resultTypes.size()
                                     << " return operands to match function signature, got "
                                     << getOperands().size();
            }

            for (size_t i = 0; i < resultTypes.size(); ++i)
            {
                if (getOperands()[i].getType() != resultTypes[i])
                {
                    return emitOpError() << "return operand #" << i << " has type "
                                         << getOperands()[i].getType()
                                         << " but enclosing func.func expects " << resultTypes[i];
                }
            }

            return success();
        }

        ::mlir::LogicalResult IfOp::verify()
        {
            auto verifySingleBlockRegion = [&](::mlir::Region &region, llvm::StringRef regionName) -> ::mlir::LogicalResult {
                if (!region.hasOneBlock())
                    return emitOpError() << regionName << " region must contain exactly one block";
                return success();
            };

            if (failed(verifySingleBlockRegion(getThenRegion(), "then")))
                return failure();
            if (failed(verifySingleBlockRegion(getElseRegion(), "else")))
                return failure();

            auto &thenBlock = getThenRegion().front();
            if (thenBlock.empty())
                return emitOpError("then region must not be empty");
            auto *thenTerminator = thenBlock.getTerminator();
            if (!thenTerminator)
                return emitOpError("then region must terminate with func.return or ora.return");
            if (!llvm::isa<mlir::func::ReturnOp, ora::ReturnOp>(thenTerminator))
            {
                return emitOpError() << "then region must terminate with func.return or ora.return, found '"
                                     << thenTerminator->getName().getStringRef() << "'";
            }

            auto &elseBlock = getElseRegion().front();
            if (elseBlock.empty())
                return emitOpError("else region must not be empty");
            if (elseBlock.getOperations().size() != 1)
                return emitOpError("else region must contain only an empty ora.yield terminator");
            auto *elseTerminator = elseBlock.getTerminator();
            auto elseYield = llvm::dyn_cast_or_null<ora::YieldOp>(elseTerminator);
            if (!elseYield)
            {
                return emitOpError() << "else region must terminate with ora.yield, found '"
                                     << (elseTerminator ? elseTerminator->getName().getStringRef() : "<none>") << "'";
            }
            if (elseYield.getOperands().size() != 0)
                return emitOpError("else region ora.yield must not return values");

            return success();
        }

        ::mlir::ParseResult IfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            ::mlir::OpAsmParser::UnresolvedOperand condition;
            if (parser.parseOperand(condition))
                return ::mlir::failure();

            // Parse "then" keyword
            if (parser.parseKeyword("then"))
                return ::mlir::failure();

            // Parse then region
            std::unique_ptr<::mlir::Region> thenRegion = std::make_unique<::mlir::Region>();
            if (parser.parseRegion(*thenRegion))
                return ::mlir::failure();

            // Parse "else" keyword
            if (parser.parseKeyword("else"))
                return ::mlir::failure();

            // Parse else region
            std::unique_ptr<::mlir::Region> elseRegion = std::make_unique<::mlir::Region>();
            if (parser.parseRegion(*elseRegion))
                return ::mlir::failure();

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            // The printer emits the condition as an MLIR i1 value.
            if (parser.resolveOperand(condition, parser.getBuilder().getI1Type(), result.operands))
                return ::mlir::failure();

            // Add regions
            result.addRegion(std::move(thenRegion));
            result.addRegion(std::move(elseRegion));

            return ::mlir::success();
        }

        void IfOp::print(::mlir::OpAsmPrinter &p)
        {
            p << " ";
            p << getCondition();
            p << " ";

            p << "then ";
            p.printRegion(getThenRegion());

            p << " else ";
            p.printRegion(getElseRegion());

            SmallVector<StringRef> elidedAttrs;
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // ContractOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult ContractOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse @ symbol and symbol name (parseSymbolName handles @ automatically)
            ::mlir::StringAttr symName;
            if (parser.parseSymbolName(symName, "sym_name", result.attributes))
                return ::mlir::failure();

            // Parse body region
            std::unique_ptr<::mlir::Region> body = std::make_unique<::mlir::Region>();
            if (parser.parseRegion(*body))
                return ::mlir::failure();

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            // Add region
            result.addRegion(std::move(body));

            return ::mlir::success();
        }

        void ContractOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print symbol name with space before @ (printSymbolName already includes @)
            p << " ";
            p.printSymbolName(getSymName());

            // Print body region
            p << " ";
            p.printRegion(getBody());

            // Print attributes
            SmallVector<StringRef> elidedAttrs = {"sym_name"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // GlobalOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult GlobalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse format: "name" : type (for maps) or "name" = init : type (for scalars)
            // Parse symbol name
            std::string symName;
            if (parser.parseKeywordOrString(&symName))
                return ::mlir::failure();
            result.addAttribute("sym_name", parser.getBuilder().getStringAttr(symName));

            // Parse optional = init (maps don't have initializers)
            ::mlir::Attribute init;
            if (succeeded(parser.parseOptionalEqual()))
            {
                // Found =, so parse the initializer
                if (parser.parseAttribute(init, "init", result.attributes))
                    return ::mlir::failure();
            }
            else
            {
                // No initializer - use UnitAttr to represent "no initializer"
                init = ::mlir::UnitAttr::get(parser.getContext());
                result.addAttribute("init", init);
            }

            // Parse : type
            ::mlir::Type type;
            if (parser.parseColonType(type))
                return ::mlir::failure();
            result.addAttribute("type", ::mlir::TypeAttr::get(type));

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            return ::mlir::success();
        }

        void GlobalOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print format: "name" : type (initializers are handled by storage, not shown in MLIR)
            p << " ";
            p.printAttributeWithoutType(getSymNameAttr());

            // Never print initializers - they are handled by storage
            // Format: "name" : type {attrs}
            p << " : ";
            p << getType();

            // Print attributes (excluding sym_name, type, init which are already printed)
            SmallVector<StringRef> elidedAttrs = {"sym_name", "type", "init"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // SwitchOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult SwitchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse switch value operand
            ::mlir::OpAsmParser::UnresolvedOperand value;
            if (parser.parseOperand(value))
                return ::mlir::failure();

            // Parse : type(value)
            ::mlir::Type valueType;
            if (parser.parseColonType(valueType))
                return ::mlir::failure();

            // Parse optional -> result_types
            ::llvm::SmallVector<::mlir::Type> resultTypes;
            if (succeeded(parser.parseOptionalArrowTypeList(resultTypes)))
            {
                result.addTypes(resultTypes);
            }

            // Parse { cases }
            if (parser.parseLBrace())
                return ::mlir::failure();

            // Parse cases
            ::llvm::SmallVector<int64_t> caseValues;
            ::llvm::SmallVector<int64_t> rangeStarts;
            ::llvm::SmallVector<int64_t> rangeEnds;
            ::llvm::SmallVector<int64_t> caseKinds; // 0=literal, 1=range, 2=else
            int64_t defaultCaseIndex = -1;
            int caseIndex = 0;

            while (succeeded(parser.parseOptionalKeyword("case")))
            {
                // Try to parse literal value
                int64_t caseStart;
                if (succeeded(parser.parseInteger(caseStart)))
                {
                    int64_t rangeEnd;
                    if (nextTokenIsEllipsis(parser))
                    {
                        if (parser.parseEllipsis())
                            return ::mlir::failure();
                        if (parser.parseInteger(rangeEnd))
                            return ::mlir::failure();

                        rangeStarts.push_back(caseStart);
                        rangeEnds.push_back(rangeEnd);
                        caseValues.push_back(0);
                        caseKinds.push_back(1); // range
                    }
                    else
                    {
                        caseValues.push_back(caseStart);
                        rangeStarts.push_back(0);
                        rangeEnds.push_back(0);
                        caseKinds.push_back(0); // literal
                    }
                }
                else
                {
                    return ::mlir::failure();
                }

                // Parse =>
                if (parser.parseEqual() || parser.parseGreater())
                    return ::mlir::failure();

                // Parse case region
                std::unique_ptr<::mlir::Region> caseRegion = std::make_unique<::mlir::Region>();
                if (parser.parseRegion(*caseRegion))
                    return ::mlir::failure();

                result.addRegion(std::move(caseRegion));
                caseIndex++;
            }

            // Parse optional else/default case
            if (succeeded(parser.parseOptionalKeyword("else")))
            {
                if (parser.parseEqual() || parser.parseGreater())
                    return ::mlir::failure();

                std::unique_ptr<::mlir::Region> elseRegion = std::make_unique<::mlir::Region>();
                if (parser.parseRegion(*elseRegion))
                    return ::mlir::failure();

                result.addRegion(std::move(elseRegion));
                defaultCaseIndex = caseIndex;
                caseKinds.push_back(2); // else
                caseValues.push_back(0);
                rangeStarts.push_back(0);
                rangeEnds.push_back(0);
            }

            // Parse }
            if (parser.parseRBrace())
                return ::mlir::failure();

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            // Resolve value operand
            if (parser.resolveOperand(value, valueType, result.operands))
                return ::mlir::failure();

            // Add case pattern attributes
            if (!caseValues.empty())
            {
                auto caseValuesAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), caseValues);
                result.addAttribute("case_values", caseValuesAttr);
            }

            if (!rangeStarts.empty())
            {
                auto rangeStartsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), rangeStarts);
                result.addAttribute("range_starts", rangeStartsAttr);
            }

            if (!rangeEnds.empty())
            {
                auto rangeEndsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), rangeEnds);
                result.addAttribute("range_ends", rangeEndsAttr);
            }

            if (!caseKinds.empty())
            {
                auto caseKindsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), caseKinds);
                result.addAttribute("case_kinds", caseKindsAttr);
            }

            if (defaultCaseIndex >= 0)
            {
                auto defaultIndexAttr = ::mlir::IntegerAttr::get(
                    ::mlir::IntegerType::get(parser.getContext(), 64), defaultCaseIndex);
                result.addAttribute("default_case_index", defaultIndexAttr);
            }

            return ::mlir::success();
        }

        void SwitchOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print value
            p << " ";
            p << getValue();
            p << " : ";
            p << getValue().getType();

            // Print result types if any
            if (!getResults().empty())
            {
                p << " -> ";
                llvm::interleaveComma(getResults().getTypes(), p, [&](::mlir::Type type)
                                      { p << type; });
            }

            // Print cases
            p << " {";

            auto caseValuesAttr = getCaseValuesAttr();
            auto rangeStartsAttr = getRangeStartsAttr();
            auto rangeEndsAttr = getRangeEndsAttr();
            auto caseKindsAttr = getCaseKindsAttr();
            auto defaultIndexAttr = getDefaultCaseIndexAttr();

            const size_t case_kinds_size = caseKindsAttr ? static_cast<size_t>(caseKindsAttr.size()) : 0;
            const size_t case_values_size = caseValuesAttr ? static_cast<size_t>(caseValuesAttr.size()) : 0;
            const size_t range_starts_size = rangeStartsAttr ? static_cast<size_t>(rangeStartsAttr.size()) : 0;
            const size_t range_ends_size = rangeEndsAttr ? static_cast<size_t>(rangeEndsAttr.size()) : 0;

            for (size_t i = 0; i < getCases().size(); ++i)
            {
                p.printNewline();
                p << "  ";

                int64_t defaultIndex = -1;
                if (defaultIndexAttr)
                {
                    defaultIndex = defaultIndexAttr.getInt();
                }

                if (defaultIndex >= 0 && static_cast<size_t>(defaultIndex) == i)
                {
                    // Default/else case
                    p << "else";
                }
                else
                {
                    // Regular case
                    p << "case ";

                    if (caseKindsAttr && i < case_kinds_size)
                    {
                        int64_t kind = caseKindsAttr[i];
                        if (kind == 0 && caseValuesAttr && i < case_values_size)
                        {
                            // Literal case
                            p << caseValuesAttr[i];
                        }
                        else if (kind == 1 && rangeStartsAttr && rangeEndsAttr &&
                                 i < range_starts_size && i < range_ends_size)
                        {
                            // Range case
                            p << rangeStartsAttr[i] << " ... " << rangeEndsAttr[i];
                        }
                    }
                }

                p << " => ";
                p.printRegion(getCases()[i]);
            }

            p.printNewline();
            p << "}";

            // Print attributes (excluding case pattern attributes)
            SmallVector<StringRef> elidedAttrs = {
                "case_values", "range_starts", "range_ends", "case_kinds", "default_case_index"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // SwitchExprOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult SwitchExprOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse switch value operand
            ::mlir::OpAsmParser::UnresolvedOperand value;
            if (parser.parseOperand(value))
                return ::mlir::failure();

            // Parse : type(value)
            ::mlir::Type valueType;
            if (parser.parseColonType(valueType))
                return ::mlir::failure();

            // Parse -> result_types
            ::llvm::SmallVector<::mlir::Type> resultTypes;
            if (parser.parseArrowTypeList(resultTypes))
                return ::mlir::failure();
            result.addTypes(resultTypes);

            // Parse { cases }
            if (parser.parseLBrace())
                return ::mlir::failure();

            // Parse cases (same as SwitchOp)
            ::llvm::SmallVector<int64_t> caseValues;
            ::llvm::SmallVector<int64_t> rangeStarts;
            ::llvm::SmallVector<int64_t> rangeEnds;
            ::llvm::SmallVector<int64_t> caseKinds;
            int64_t defaultCaseIndex = -1;
            int caseIndex = 0;

            while (succeeded(parser.parseOptionalKeyword("case")))
            {
                int64_t caseStart;
                if (succeeded(parser.parseInteger(caseStart)))
                {
                    int64_t rangeEnd;
                    if (nextTokenIsEllipsis(parser))
                    {
                        if (parser.parseEllipsis())
                            return ::mlir::failure();
                        if (parser.parseInteger(rangeEnd))
                            return ::mlir::failure();

                        rangeStarts.push_back(caseStart);
                        rangeEnds.push_back(rangeEnd);
                        caseValues.push_back(0);
                        caseKinds.push_back(1);
                    }
                    else
                    {
                        caseValues.push_back(caseStart);
                        rangeStarts.push_back(0);
                        rangeEnds.push_back(0);
                        caseKinds.push_back(0);
                    }
                }
                else
                {
                    return ::mlir::failure();
                }

                if (parser.parseEqual() || parser.parseGreater())
                    return ::mlir::failure();

                std::unique_ptr<::mlir::Region> caseRegion = std::make_unique<::mlir::Region>();
                if (parser.parseRegion(*caseRegion))
                    return ::mlir::failure();

                result.addRegion(std::move(caseRegion));
                caseIndex++;
            }

            if (succeeded(parser.parseOptionalKeyword("else")))
            {
                if (parser.parseEqual() || parser.parseGreater())
                    return ::mlir::failure();

                std::unique_ptr<::mlir::Region> elseRegion = std::make_unique<::mlir::Region>();
                if (parser.parseRegion(*elseRegion))
                    return ::mlir::failure();

                result.addRegion(std::move(elseRegion));
                defaultCaseIndex = caseIndex;
                caseKinds.push_back(2);
                caseValues.push_back(0);
                rangeStarts.push_back(0);
                rangeEnds.push_back(0);
            }

            if (parser.parseRBrace())
                return ::mlir::failure();

            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            if (parser.resolveOperand(value, valueType, result.operands))
                return ::mlir::failure();

            if (!caseValues.empty())
            {
                auto caseValuesAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), caseValues);
                result.addAttribute("case_values", caseValuesAttr);
            }

            if (!rangeStarts.empty())
            {
                auto rangeStartsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), rangeStarts);
                result.addAttribute("range_starts", rangeStartsAttr);
            }

            if (!rangeEnds.empty())
            {
                auto rangeEndsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), rangeEnds);
                result.addAttribute("range_ends", rangeEndsAttr);
            }

            if (!caseKinds.empty())
            {
                auto caseKindsAttr = ::mlir::DenseI64ArrayAttr::get(parser.getContext(), caseKinds);
                result.addAttribute("case_kinds", caseKindsAttr);
            }

            if (defaultCaseIndex >= 0)
            {
                auto defaultIndexAttr = ::mlir::IntegerAttr::get(
                    ::mlir::IntegerType::get(parser.getContext(), 64), defaultCaseIndex);
                result.addAttribute("default_case_index", defaultIndexAttr);
            }

            return ::mlir::success();
        }

        void SwitchExprOp::print(::mlir::OpAsmPrinter &p)
        {
            p << " ";
            p << getValue();
            p << " : ";
            p << getValue().getType();

            if (!getResults().empty())
            {
                p << " -> ";
                llvm::interleaveComma(getResults().getTypes(), p, [&](::mlir::Type type)
                                      { p << type; });
            }

            p << " {";

            auto caseValuesAttr = getCaseValuesAttr();
            auto rangeStartsAttr = getRangeStartsAttr();
            auto rangeEndsAttr = getRangeEndsAttr();
            auto caseKindsAttr = getCaseKindsAttr();
            auto defaultIndexAttr = getDefaultCaseIndexAttr();

            const size_t case_kinds_size = caseKindsAttr ? static_cast<size_t>(caseKindsAttr.size()) : 0;
            const size_t case_values_size = caseValuesAttr ? static_cast<size_t>(caseValuesAttr.size()) : 0;
            const size_t range_starts_size = rangeStartsAttr ? static_cast<size_t>(rangeStartsAttr.size()) : 0;
            const size_t range_ends_size = rangeEndsAttr ? static_cast<size_t>(rangeEndsAttr.size()) : 0;

            for (size_t i = 0; i < getCases().size(); ++i)
            {
                p.printNewline();
                p << "  ";

                int64_t defaultIndex = -1;
                if (defaultIndexAttr)
                {
                    defaultIndex = defaultIndexAttr.getInt();
                }

                if (defaultIndex >= 0 && static_cast<size_t>(defaultIndex) == i)
                {
                    p << "else";
                }
                else
                {
                    p << "case ";

                    if (caseKindsAttr && i < case_kinds_size)
                    {
                        int64_t kind = caseKindsAttr[i];
                        if (kind == 0 && caseValuesAttr && i < case_values_size)
                        {
                            p << caseValuesAttr[i];
                        }
                        else if (kind == 1 && rangeStartsAttr && rangeEndsAttr &&
                                 i < range_starts_size && i < range_ends_size)
                        {
                            p << rangeStartsAttr[i] << " ... " << rangeEndsAttr[i];
                        }
                    }
                }

                p << " => ";
                p.printRegion(getCases()[i]);
            }

            p.printNewline();
            p << "}";

            SmallVector<StringRef> elidedAttrs = {
                "case_values", "range_starts", "range_ends", "case_kinds", "default_case_index"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // SLoadOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult SLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse format: "global" : type attr-dict
            std::string globalName;
            if (parser.parseKeywordOrString(&globalName))
                return ::mlir::failure();
            result.addAttribute("global", parser.getBuilder().getStringAttr(globalName));

            ::mlir::Type resultType;
            if (parser.parseColonType(resultType))
                return ::mlir::failure();
            result.addTypes(resultType);

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            return ::mlir::success();
        }

        void SLoadOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print format: "global" : type attr-dict
            p << " ";
            p.printAttributeWithoutType(getGlobalAttr());
            p << " : ";
            p << getResult().getType();

            // Print attributes, but elide internal result name attributes
            SmallVector<StringRef> elidedAttrs = {"global", "ora.result_name_0"};
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // SLoadOp Result Naming (via OpAsmOpInterface)
        //===----------------------------------------------------------------------===//

        void SLoadOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            // Check if we have a stored result name attribute (set via oraOperationSetResultName)
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                // Use the provided name hint
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default: use the global name as the result name hint
                // MLIR will automatically uniquify if needed (e.g., %balances, %balances2)
                setNameFn(getResult(), getGlobal());
            }
        }

        namespace
        {
            struct FoldAddConstants : public OpRewritePattern<AddOp>
            {
                using OpRewritePattern<AddOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override
                {
                    auto lhsVal = getConstantIntegerAttr(op.getLhs());
                    auto rhsVal = getConstantIntegerAttr(op.getRhs());

                    if (!lhsVal || !rhsVal)
                        return failure();

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        lhsVal->getValue() + rhsVal->getValue());
                }
            };

            struct FoldMulConstants : public OpRewritePattern<MulOp>
            {
                using OpRewritePattern<MulOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override
                {
                    auto lhsVal = getConstantIntegerAttr(op.getLhs());
                    auto rhsVal = getConstantIntegerAttr(op.getRhs());

                    if (!lhsVal || !rhsVal)
                        return failure();

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        lhsVal->getValue() * rhsVal->getValue());
                }
            };

            struct FoldSubConstants : public OpRewritePattern<SubOp>
            {
                using OpRewritePattern<SubOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(SubOp op, PatternRewriter &rewriter) const override
                {
                    auto lhsVal = getConstantIntegerAttr(op.getLhs());
                    auto rhsVal = getConstantIntegerAttr(op.getRhs());

                    if (!lhsVal || !rhsVal)
                        return failure();

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        lhsVal->getValue() - rhsVal->getValue());
                }
            };

            struct FoldDivConstants : public OpRewritePattern<DivOp>
            {
                using OpRewritePattern<DivOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(DivOp op, PatternRewriter &rewriter) const override
                {
                    auto lhsVal = getConstantIntegerAttr(op.getLhs());
                    auto rhsVal = getConstantIntegerAttr(op.getRhs());

                    if (!lhsVal || !rhsVal || rhsVal->getValue().isZero())
                        return failure();

                    auto resultOraType = llvm::dyn_cast<ora::IntegerType>(op.getResult().getType());
                    if (!resultOraType)
                        return failure();

                    llvm::APInt result = resultOraType.getIsSigned()
                                             ? lhsVal->getValue().sdiv(rhsVal->getValue())
                                             : lhsVal->getValue().udiv(rhsVal->getValue());

                    return replaceOpWithConstant(
                        rewriter,
                        op,
                        op.getResult().getType(),
                        result);
                }
            };
        }

        void AddOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
        {
            results.add<FoldAddConstants>(context);
        }

        void MulOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
        {
            results.add<FoldMulConstants>(context);
        }

        void SubOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
        {
            results.add<FoldSubConstants>(context);
        }

        void DivOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
        {
            results.add<FoldDivConstants>(context);
        }

        //===----------------------------------------------------------------------===//
        // Memory Effects
        //===----------------------------------------------------------------------===//

        // Storage load: reads from global storage
        void SLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Storage store: writes to global storage
        void SStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        // Memory load: reads from memory
        void MLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Memory store: writes to memory
        void MStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        // Byte memory load: reads from memory
        void MLoad8Op::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Byte memory store: writes to memory
        void MStore8Op::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        // Transient storage load: reads from transient storage
        void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Transient storage store: writes to transient storage
        void TStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        // Map get: reads from storage (maps are storage-based)
        void MapGetOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            // Map operations read from storage (the map is storage-based)
            effects.emplace_back(MemoryEffects::Read::get(), SideEffects::DefaultResource::get());
        }

        // Map store: writes to storage (maps are storage-based)
        void MapStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
        {
            // Map operations write to storage (the map is storage-based)
            effects.emplace_back(MemoryEffects::Write::get(), SideEffects::DefaultResource::get());
        }

        //===----------------------------------------------------------------------===//
        // Result Naming (via OpAsmOpInterface)
        //===----------------------------------------------------------------------===//

        // Arithmetic operations: use semantic names
        void AddOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default semantic name for addition
                setNameFn(getResult(), "sum");
            }
        }

        void SubOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default semantic name for subtraction
                setNameFn(getResult(), "difference");
            }
        }

        void MulOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default semantic name for multiplication
                setNameFn(getResult(), "product");
            }
        }

        void DivOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default semantic name for division
                setNameFn(getResult(), "quotient");
            }
        }

        // Memory load: use variable name
        void MLoadOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default: use the variable name as the result name hint
                setNameFn(getResult(), getVariable());
            }
        }

        // Map get: use map name
        void MapGetOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn)
        {
            auto nameAttr = (*this)->getAttrOfType<StringAttr>("ora.result_name_0");
            if (nameAttr)
            {
                setNameFn(getResult(), nameAttr.getValue());
            }
            else
            {
                // Default: use "value" as semantic name for map get
                setNameFn(getResult(), "value");
            }
        }

    } // namespace ora
} // namespace mlir

// Include the generated dialect definitions implementation
#include "OraDialect.cpp.inc"
