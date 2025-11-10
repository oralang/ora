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

// Include the dialect header which includes operation declarations and dialect definitions
#include "OraDialect.h"

// Include the generated operation definitions
#define GET_OP_CLASSES
#include "OraOps.cpp.inc"

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

// Include the generated dialect definitions implementation
#include "OraDialect.cpp.inc"
