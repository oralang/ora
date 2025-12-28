//===- OraDialect.cpp - Ora dialect implementation ----------------------===//
//
// This file implements the Ora MLIR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/PatternMatch.h"
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
            Type baseType = nullptr;

            // Extract base type from refinement type
            if (auto minValueType = dyn_cast<MinValueType>(valueType))
            {
                baseType = minValueType.getBaseType();
            }
            else if (auto maxValueType = dyn_cast<MaxValueType>(valueType))
            {
                baseType = maxValueType.getBaseType();
            }
            else if (auto inRangeType = dyn_cast<InRangeType>(valueType))
            {
                baseType = inRangeType.getBaseType();
            }
            else if (auto scaledType = dyn_cast<ScaledType>(valueType))
            {
                baseType = scaledType.getBaseType();
            }
            else if (auto exactType = dyn_cast<ExactType>(valueType))
            {
                baseType = exactType.getBaseType();
            }
            else if (isa<NonZeroAddressType>(valueType))
            {
                // NonZeroAddress base type is AddressType
                baseType = AddressType::get(odsBuilder.getContext());
            }
            else
            {
                // Not a refinement type, use the type as-is
                baseType = valueType;
            }

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

            // Parse optional result types
            ::llvm::SmallVector<::mlir::Type, 1> resultTypes;
            if (parser.parseOptionalArrowTypeList(resultTypes))
                return ::mlir::failure();
            result.addTypes(resultTypes);

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            // Resolve condition operand - use Ora BoolType
            if (parser.resolveOperand(condition, BoolType::get(parser.getContext()), result.operands))
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

            if (!getResults().empty())
            {
                p << " -> ";
                llvm::interleaveComma(getResults().getTypes(), p, [&](::mlir::Type type)
                                      { p << type; });
            }

            SmallVector<StringRef> elidedAttrs;
            p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
        }

        //===----------------------------------------------------------------------===//
        // WhileOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult WhileOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse condition operand
            ::mlir::OpAsmParser::UnresolvedOperand condition;
            if (parser.parseOperand(condition))
                return ::mlir::failure();

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            // Parse type of condition
            ::mlir::Type conditionType;
            if (parser.parseColonType(conditionType))
                return ::mlir::failure();

            // Parse body region
            std::unique_ptr<::mlir::Region> body = std::make_unique<::mlir::Region>();
            if (parser.parseRegion(*body))
                return ::mlir::failure();

            // Resolve condition operand
            if (parser.resolveOperand(condition, conditionType, result.operands))
                return ::mlir::failure();

            // Add region
            result.addRegion(std::move(body));

            return ::mlir::success();
        }

        void WhileOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print condition
            p << " ";
            p << getCondition();

            // Print attributes
            p.printOptionalAttrDict((*this)->getAttrs());

            // Print type
            p << " : ";
            p << getCondition().getType();

            // Print body region
            p << " ";
            p.printRegion(getBody());
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
            p.printOptionalAttrDict((*this)->getAttrs());
        }

        //===----------------------------------------------------------------------===//
        // GlobalOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult GlobalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse format: "name" : type (for maps) or "name" = init : type (for scalars)
            // Parse symbol name
            ::mlir::StringAttr symName;
            if (parser.parseAttribute(symName, "sym_name", result.attributes))
                return ::mlir::failure();

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
                int64_t literalValue;
                if (succeeded(parser.parseInteger(literalValue)))
                {
                    // Literal case
                    caseValues.push_back(literalValue);
                    rangeStarts.push_back(0);
                    rangeEnds.push_back(0);
                    caseKinds.push_back(0); // literal
                }
                else
                {
                    // Try to parse range: start...end
                    int64_t rangeStart, rangeEnd;
                    if (succeeded(parser.parseInteger(rangeStart)) &&
                        succeeded(parser.parseEllipsis()) &&
                        succeeded(parser.parseInteger(rangeEnd)))
                    {
                        // Range case
                        rangeStarts.push_back(rangeStart);
                        rangeEnds.push_back(rangeEnd);
                        caseValues.push_back(0);
                        caseKinds.push_back(1); // range
                    }
                    else
                    {
                        return ::mlir::failure();
                    }
                }

                // Parse =>
                if (parser.parseArrow())
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
                if (parser.parseArrow())
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
                            p << rangeStartsAttr[i] << "..." << rangeEndsAttr[i];
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
                int64_t literalValue;
                if (succeeded(parser.parseInteger(literalValue)))
                {
                    caseValues.push_back(literalValue);
                    rangeStarts.push_back(0);
                    rangeEnds.push_back(0);
                    caseKinds.push_back(0);
                }
                else
                {
                    int64_t rangeStart, rangeEnd;
                    if (succeeded(parser.parseInteger(rangeStart)) &&
                        succeeded(parser.parseEllipsis()) &&
                        succeeded(parser.parseInteger(rangeEnd)))
                    {
                        rangeStarts.push_back(rangeStart);
                        rangeEnds.push_back(rangeEnd);
                        caseValues.push_back(0);
                        caseKinds.push_back(1);
                    }
                    else
                    {
                        return ::mlir::failure();
                    }
                }

                if (parser.parseArrow())
                    return ::mlir::failure();

                std::unique_ptr<::mlir::Region> caseRegion = std::make_unique<::mlir::Region>();
                if (parser.parseRegion(*caseRegion))
                    return ::mlir::failure();

                result.addRegion(std::move(caseRegion));
                caseIndex++;
            }

            if (succeeded(parser.parseOptionalKeyword("else")))
            {
                if (parser.parseArrow())
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
                            p << rangeStartsAttr[i] << "..." << rangeEndsAttr[i];
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
            ::mlir::StringAttr globalAttr;
            if (parser.parseAttribute(globalAttr, "global", result.attributes))
                return ::mlir::failure();

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
                    auto getConstantValue = [](Value val) -> std::optional<uint64_t>
                    {
                        if (auto constOp = val.getDefiningOp<arith::ConstantOp>())
                        {
                            auto attr = constOp.getValue();
                            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
                            {
                                return intAttr.getValue().getZExtValue();
                            }
                        }
                        return std::nullopt;
                    };

                    auto lhsVal = getConstantValue(op.getLhs());
                    auto rhsVal = getConstantValue(op.getRhs());

                    if (!lhsVal || !rhsVal)
                        return failure();

                    uint64_t result = *lhsVal + *rhsVal;
                    auto resultType = op.getResult().getType();
                    auto valueAttr = mlir::IntegerAttr::get(resultType, result);
                    auto newConst = rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, valueAttr);
                    rewriter.replaceOp(op, newConst.getResult());
                    return success();
                }
            };

            struct FoldMulConstants : public OpRewritePattern<MulOp>
            {
                using OpRewritePattern<MulOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override
                {
                    auto getConstantValue = [](Value val) -> std::optional<uint64_t>
                    {
                        if (auto constOp = val.getDefiningOp<arith::ConstantOp>())
                        {
                            auto attr = constOp.getValue();
                            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
                            {
                                return intAttr.getValue().getZExtValue();
                            }
                        }
                        return std::nullopt;
                    };

                    auto lhsVal = getConstantValue(op.getLhs());
                    auto rhsVal = getConstantValue(op.getRhs());

                    if (!lhsVal || !rhsVal)
                        return failure();

                    uint64_t result = *lhsVal * *rhsVal;
                    auto resultType = op.getResult().getType();
                    auto valueAttr = mlir::IntegerAttr::get(resultType, result);
                    auto newConst = rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, valueAttr);
                    rewriter.replaceOp(op, newConst.getResult());
                    return success();
                }
            };

            struct FoldSubConstants : public OpRewritePattern<SubOp>
            {
                using OpRewritePattern<SubOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(SubOp op, PatternRewriter &rewriter) const override
                {
                    auto getConstantValue = [](Value val) -> std::optional<uint64_t>
                    {
                        if (auto constOp = val.getDefiningOp<arith::ConstantOp>())
                        {
                            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                            {
                                return intAttr.getValue().getZExtValue();
                            }
                        }
                        return std::nullopt;
                    };

                    auto lhsVal = getConstantValue(op.getLhs());
                    auto rhsVal = getConstantValue(op.getRhs());

                    if (!lhsVal || !rhsVal)
                        return failure();

                    uint64_t result = *lhsVal - *rhsVal;
                    auto resultType = op.getResult().getType();
                    auto valueAttr = mlir::IntegerAttr::get(resultType, result);
                    auto newConst = rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, valueAttr);
                    rewriter.replaceOp(op, newConst.getResult());
                    return success();
                }
            };

            struct FoldDivConstants : public OpRewritePattern<DivOp>
            {
                using OpRewritePattern<DivOp>::OpRewritePattern;

                LogicalResult matchAndRewrite(DivOp op, PatternRewriter &rewriter) const override
                {
                    auto getConstantValue = [](Value val) -> std::optional<uint64_t>
                    {
                        if (auto constOp = val.getDefiningOp<arith::ConstantOp>())
                        {
                            if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
                            {
                                return intAttr.getValue().getZExtValue();
                            }
                        }
                        return std::nullopt;
                    };

                    auto lhsVal = getConstantValue(op.getLhs());
                    auto rhsVal = getConstantValue(op.getRhs());

                    if (!lhsVal || !rhsVal || *rhsVal == 0)
                        return failure();

                    uint64_t result = *lhsVal / *rhsVal;
                    auto resultType = op.getResult().getType();
                    auto valueAttr = mlir::IntegerAttr::get(resultType, result);
                    auto newConst = rewriter.create<arith::ConstantOp>(op.getLoc(), resultType, valueAttr);
                    rewriter.replaceOp(op, newConst.getResult());
                    return success();
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
