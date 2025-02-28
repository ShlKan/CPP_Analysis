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
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "SysIR/Dialect/IR/SysOpsDialect.h.inc"
#include "SysIR/Dialect/IR/SysOpsEnums.h.inc"
#include "SysIR/Dialect/IR/SysTypes.h"

#include "CIR/Dialect/IR/CIRTypes.h"

#define GET_OP_CLASSES
#include "SysIR/Dialect/IR/SysOps.h.inc"

#endif