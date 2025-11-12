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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

// Include the dialect header which includes operation declarations and dialect definitions
#include "OraDialect.h"

// The header includes OraTypes.h.inc which has the full class definitions
// These should be visible when we include OraTypes.cpp.inc

//===----------------------------------------------------------------------===//
// Include Type Definitions First
//===----------------------------------------------------------------------===//
// According to MLIR docs, we need to include the full type definitions
// before registering them, so the types are fully defined when addTypes<> is called
// The header (OraDialect.h) already includes OraTypes.h.inc with full class definitions,
// so the generated parser should be able to see them
#define GET_TYPEDEF_CLASSES
#include "OraTypes.cpp.inc"

namespace mlir
{
    namespace ora
    {

        //===----------------------------------------------------------------------===//
        // Ora Dialect
        //===----------------------------------------------------------------------===//

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

        // Type parsing/printing is handled by generated code in OraTypes.cpp.inc
        // (included above with GET_TYPEDEF_CLASSES)
        // The generated printType/parseType methods are in OraTypes.cpp.inc
        // Attribute parsing/printing use default implementations
        ::mlir::Attribute OraDialect::parseAttribute(::mlir::DialectAsmParser &parser, ::mlir::Type type) const
        {
            return ::mlir::Dialect::parseAttribute(parser, type);
        }

        void OraDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter &printer) const
        {
            ::mlir::Dialect::printAttribute(attr, printer);
        }

        // Note: printType and parseType are generated in OraTypes.cpp.inc when GET_TYPEDEF_CLASSES is defined
        // They call generatedTypePrinter/generatedTypeParser which dispatch to our custom type printers

    } // namespace ora
} // namespace mlir

//===----------------------------------------------------------------------===//
// Ora IntegerType Custom Methods
//===----------------------------------------------------------------------===//

// Define custom methods for IntegerType
// The types are already defined above (via GET_TYPEDEF_CLASSES), so we can add custom methods
namespace mlir
{
    namespace ora
    {

        // Note: parseType and printType are handled by generated code
        // via the declarative assemblyFormat in OraTypes.td
        // Our Ora types are MLIR types (inherit from Type::TypeBase) and work
        // directly with MLIR infrastructure - no conversion needed

        //===----------------------------------------------------------------------===//
        // Custom parse/print implementations for types with hasCustomAssemblyFormat
        //===----------------------------------------------------------------------===//

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
        //===----------------------------------------------------------------------===//
        // IfOp Custom Parser/Printer
        //===----------------------------------------------------------------------===//

        ::mlir::ParseResult IfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result)
        {
            // Parse condition operand
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
            llvm::errs() << "[DEBUG] ========== IfOp::print() called ==========\n";
            // Print condition
            p << " ";
            p << getCondition();
            p << " ";

            // Print "then" keyword and then region
            p << "then ";
            p.printRegion(getThenRegion());

            // Print "else" keyword and else region
            p << " else ";
            p.printRegion(getElseRegion());

            // Print result types if any
            if (!getResults().empty())
            {
                p << " -> ";
                llvm::interleaveComma(getResults().getTypes(), p, [&](::mlir::Type type)
                                      { p << type; });
            }

            // Print attributes (excluding condition which is already printed)
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
            llvm::errs() << "[DEBUG] ========== WhileOp::print() called ==========\n";
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
            llvm::errs() << "[DEBUG] ========== ContractOp::print() called ==========\n";
            llvm::errs() << "[DEBUG] ContractOp::print() - this pointer: " << (void *)this << "\n";
            llvm::errs() << "[DEBUG] ContractOp symbol name: " << getSymName() << "\n";
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
            llvm::errs() << "[DEBUG] ========== GlobalOp::print() called ==========\n";
            llvm::errs() << "[DEBUG] GlobalOp::print() - this pointer: " << (void *)this << "\n";
            llvm::errs() << "[DEBUG] GlobalOp symbol name: " << getSymName() << "\n";
            llvm::errs() << "[DEBUG] GlobalOp type: " << getType() << "\n";
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

    } // namespace ora
} // namespace mlir

// Include the generated dialect definitions implementation
#include "OraDialect.cpp.inc"
