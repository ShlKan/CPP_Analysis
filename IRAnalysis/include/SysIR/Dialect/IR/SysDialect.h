//===------------------ SMDialect.h ------------------===//
//===-------------------------------------------------===//
// This file declares the SM (System Modeling) Dialect.
//===-------------------------------------------------===//
#ifndef MLIR_SYSTEM_MODELING_DIALECT_
#define MLIR_SYSTEM_MODELING_DIALECT_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"

#include "SysIR/Dialect/IR/SysOpsDialect.h.inc"
#include "SysIR/Dialect/IR/SysTypes.h"

#define GET_OP_CLASSES
#include "SysIR/Dialect/IR/SysOps.h.inc"

#endif