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

// Include the dialect header which includes operation declarations and dialect definitions
#include "OraDialect.h"

namespace mlir
{
    namespace ora
    {

        //===----------------------------------------------------------------------===//
        // Ora Dialect
        //===----------------------------------------------------------------------===//

        /// Initialize the Ora dialect and add operations
        void OraDialect::initialize()
        {
            addOperations<
#define GET_OP_LIST
#include "OraOps.cpp.inc"
                >();
        }

        // Type and attribute parsing/printing use default implementations
        // These are declared in the header but need implementations
        ::mlir::Attribute OraDialect::parseAttribute(::mlir::DialectAsmParser &parser, ::mlir::Type type) const
        {
            return ::mlir::Dialect::parseAttribute(parser, type);
        }

        void OraDialect::printAttribute(::mlir::Attribute attr, ::mlir::DialectAsmPrinter &printer) const
        {
            ::mlir::Dialect::printAttribute(attr, printer);
        }

        ::mlir::Type OraDialect::parseType(::mlir::DialectAsmParser &parser) const
        {
            return ::mlir::Dialect::parseType(parser);
        }

        void OraDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const
        {
            ::mlir::Dialect::printType(type, printer);
        }

    } // namespace ora
} // namespace mlir

// Include the generated operation method implementations
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

            // Resolve condition operand
            if (parser.resolveOperand(condition, ::mlir::IntegerType::get(parser.getContext(), 1), result.operands))
                return ::mlir::failure();

            // Add regions
            result.addRegion(std::move(thenRegion));
            result.addRegion(std::move(elseRegion));

            return ::mlir::success();
        }

        void IfOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print condition
            p << " ";
            p << getCondition();

            // Print "then" keyword and then region
            p << " then ";
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

            // Print attributes
            p.printOptionalAttrDict((*this)->getAttrs());
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
            // Print @ symbol and symbol name
            p << " @";
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
            // Parse symbol name
            ::mlir::StringAttr symName;
            if (parser.parseAttribute(symName, "sym_name", result.attributes))
                return ::mlir::failure();

            // Parse : type
            ::mlir::Type type;
            if (parser.parseColonType(type))
                return ::mlir::failure();
            result.addAttribute("type", ::mlir::TypeAttr::get(type));

            // Parse = init
            ::mlir::Attribute init;
            if (parser.parseEqual() || parser.parseAttribute(init, "init", result.attributes))
                return ::mlir::failure();

            // Parse optional attributes
            if (parser.parseOptionalAttrDict(result.attributes))
                return ::mlir::failure();

            return ::mlir::success();
        }

        void GlobalOp::print(::mlir::OpAsmPrinter &p)
        {
            // Print symbol name
            p << " ";
            p.printAttributeWithoutType(getSymNameAttr());

            // Print : type
            p << " : ";
            p << getType();

            // Print = init
            p << " = ";
            p.printAttributeWithoutType(getInitAttr());

            // Print attributes
            p.printOptionalAttrDict((*this)->getAttrs());
        }

    } // namespace ora
} // namespace mlir

// Include the generated dialect definitions implementation
#include "OraDialect.cpp.inc"
