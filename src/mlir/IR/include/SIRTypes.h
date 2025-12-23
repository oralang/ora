//===- SIRTypes.h - SIR dialect types -------------------------===//
//
// This file defines the SIR MLIR dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef SIR_TYPES_H
#define SIR_TYPES_H

#include "SIRDialect.h"

// Include the generated type declarations
#define GET_TYPEDEF_CLASSES
#include "SIRTypes.h.inc"
#undef GET_TYPEDEF_CLASSES

#endif // SIR_TYPES_H

