//===- SysDialect.cpp - MLIR Sys ops implementation -----------------------===//

#include "SysIR/Dialect/IR/SysDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

using namespace ::mlir;
using namespace ::mlir::sys;

#include "SysIR/Dialect/IR/SysOpsDialect.cpp.inc"

void SysDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "SysIR/Dialect/IR/SysOps.cpp.inc"
      >();
}

//===---     ProcDefOp      ---===//

void ProcDefOP::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState, StringRef name,
                      SProcessType type) {
  mlir::ArrayAttr attr;
  build(odsBuilder, odsState, type, name, type, attr);
}

//===---     ProcRegisterOp     ---====//

Attribute SysDialect::parseAttribute(mlir::DialectAsmParser &,
                                     mlir::Type) const {
  // TODO
}

void SysDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
  // TODO
}

Type SysDialect::parseType(DialectAsmParser &parser) const {
  // TODO
}

void SysDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // TODO
}

#define GET_OP_CLASSES
#include "SysIR/Dialect/IR/SysOps.cpp.inc"
