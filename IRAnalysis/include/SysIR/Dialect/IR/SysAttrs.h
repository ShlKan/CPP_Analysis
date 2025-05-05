

#ifndef MLIR_DIALECT_SYS_IR_ATTRS_H_
#define MLIR_DIALECT_SYS_IR_ATTRS_H_

#include "SysIR/Dialect/IR/SysTypes.h"

#include "SysIR/Dialect/IR/SysOpsEnums.h.inc"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
// CIR Dialect Attrs
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "SysIR/Dialect/IR/SysOpsAttributes.h.inc"

#endif // MLIR_DIALECT_CIR_IR_CIRATTRS_H_
