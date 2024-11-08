//===------------------ SMDialect.h ------------------===//
//===-------------------------------------------------===//
// This file declares the SM (System Modeling) Dialect.
//===-------------------------------------------------===//
#ifndef MLIR_SYSTEM_MODELING_DIALECT_
#define MLIR_SYSTEM_MODELING_DIALECT_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"

#include "CIR/Dialect/IR/SMOpsDialect.h.inc"
#include "CIR/Dialect/IR/SMTypes.h"

#define GET_OP_CLASSES
#include "CIR/Dialect/IR/SMOps.h.inc"

#endif