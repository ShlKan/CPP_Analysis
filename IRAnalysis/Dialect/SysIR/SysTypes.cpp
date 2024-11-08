//===----------------------------------------------------------------------===//
//
// This file defines the types in the SysIR dialect.
//
//===----------------------------------------------------------------------===//
#include "SysIR/Dialect/IR/SysTypes.h"
#include "mlir/IR/OpImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "SysIR/Dialect/IR/SysOpsTypes.cpp.inc"

using namespace ::mlir::sys;

void SIntType::print(::mlir::AsmPrinter &printer) const {}
