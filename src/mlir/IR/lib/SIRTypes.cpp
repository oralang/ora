//===- SIRTypes.cpp - SIR Dialect Types -----------------------------===//
//
// This file implements parsing/printing for SIR types.
//
//===----------------------------------------------------------------------===//

#include "SIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace sir;

Type PtrType::parse(AsmParser &parser)
{
  if (parser.parseLess())
    return Type();
  unsigned space;
  if (parser.parseInteger(space) || parser.parseGreater())
    return Type();
  return PtrType::get(parser.getContext(), space);
}

void PtrType::print(AsmPrinter &printer) const
{
  printer << "<" << getAddrSpace() << ">";
}

// U256Type has no parameters, so it uses default parsing/printing
// No custom parse/print methods needed
