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

// Include the generated dialect definitions
#include "OraDialect.h.inc"

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

    } // namespace ora
} // namespace mlir

// Include the generated dialect definitions implementation
#include "OraDialect.cpp.inc"
