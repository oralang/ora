//===- SIROps.h - SIR dialect operations -------------------------===//
//
// This file defines the SIR MLIR dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef SIR_OPS_H
#define SIR_OPS_H

#include "SIRDialect.h"

// Include the generated operation declarations
#define GET_OP_CLASSES
#include "SIROps.h.inc"
#undef GET_OP_CLASSES

#endif // SIR_OPS_H

