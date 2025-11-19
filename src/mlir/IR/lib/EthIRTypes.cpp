//===- EthIRTypes.cpp - EthIR Dialect Types -----------------------------===//
//
// This file implements parsing/printing for EthIR types.
//
//===----------------------------------------------------------------------===//

#include "EthIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace ethir;

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
