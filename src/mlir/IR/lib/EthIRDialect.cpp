//===- EthIRDialect.cpp - EthIR Dialect implementation --------------------===//
//
// This file implements the EthIR dialect and registers its types/ops.
//
//===----------------------------------------------------------------------===//

#include "EthIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace ethir;

// Include the generated type definitions
#define GET_TYPEDEF_CLASSES
#include "EthIRTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

// Include the generated operation definitions
#define GET_OP_CLASSES
#include "EthIROps.cpp.inc"
#undef GET_OP_CLASSES

void EthIRDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "EthIRTypes.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "EthIROps.cpp.inc"
        >();
}
