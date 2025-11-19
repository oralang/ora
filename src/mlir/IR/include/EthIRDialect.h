//===- EthIRDialect.h - EthIR dialect definition -------------------------===//
//
// This file defines the EthIR MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ETHIR_DIALECT_H
#define ETHIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

// Include the generated dialect declarations
#include "EthIRDialect.h.inc"

// Include the generated type declarations
#define GET_TYPEDEF_CLASSES
#include "EthIRTypes.h.inc"
#undef GET_TYPEDEF_CLASSES

// Include the generated operation declarations
#define GET_OP_CLASSES
#include "EthIROps.h.inc"
#undef GET_OP_CLASSES

#endif // ETHIR_DIALECT_H
