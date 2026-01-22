//===- SIRDialect.h - SIR dialect definition -------------------------===//
//
// This file defines the SIR MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SIR_DIALECT_H
#define SIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

// Include the generated dialect declarations
#include "SIRDialect.h.inc"

// Include the generated type declarations
#define GET_TYPEDEF_CLASSES
#include "SIRTypes.h.inc"
#undef GET_TYPEDEF_CLASSES

// Include the generated operation declarations
#define GET_OP_CLASSES
#include "SIROps.h.inc"
#undef GET_OP_CLASSES

#endif // SIR_DIALECT_H
