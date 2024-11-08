//===------------------ SMDialect.h ------------------===//
//===-------------------------------------------------===//
// This file declares the SM (System Modeling) Dialect.
//===-------------------------------------------------===//
#ifndef MLIR_SYSTEM_MODELING_DIALECT_
#define MLIR_SYSTEM_MODELING_DIALECT_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"

#include "SysIR/Dialect/IR/SMOpsDialect.h.inc"
#include "SysIR/Dialect/IR/SMTypes.h"

#define GET_OP_CLASSES
#include "SysIR/Dialect/IR/SMOps.h.inc"

#endif