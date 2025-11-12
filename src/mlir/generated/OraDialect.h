//===- OraDialect.h - Ora dialect definition -------------------------===//
//
// This file defines the Ora MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_DIALECT_H
#define ORA_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

// Include the generated dialect declarations
#include "OraDialect.h.inc"

// Include the generated type declarations
// Note: We need GET_TYPEDEF_CLASSES to get full class definitions, not just forward declarations
#define GET_TYPEDEF_CLASSES
#include "OraTypes.h.inc"
#undef GET_TYPEDEF_CLASSES

// Define GET_OP_CLASSES to include operation class declarations
#define GET_OP_CLASSES
#include "OraOps.h.inc"

#endif // ORA_DIALECT_H
