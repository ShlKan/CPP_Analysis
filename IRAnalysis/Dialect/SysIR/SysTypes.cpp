//===----------------------------------------------------------------------===//
//
// This file defines the types in the SysIR dialect.
//
//===----------------------------------------------------------------------===//
#include "SysIR/Dialect/IR/SysTypes.h"
#include "SysIR/Dialect/IR/SysDialect.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

//===----------------------------------------------------------------------====//
//  Custom parser and printer.
//===----------------------------------------------------------------------====//

static mlir::ParseResult parseProcArgs(mlir::AsmParser &p,
                                       llvm::SmallVector<mlir::Type> &args);

static void printProcArgs(mlir::AsmPrinter &p, llvm::ArrayRef<mlir::Type> args);

//===----------------------------------------------------------------------====//
//  ODS Sys dialect type implementation.
//===----------------------------------------------------------------------====//

#define GET_TYPEDEF_CLASSES
#include "SysIR/Dialect/IR/SysOpsTypes.cpp.inc"

using namespace ::mlir::sys;

void SIntType::print(::mlir::AsmPrinter &odsPrinter) const {
  if (getIsSigned())
    odsPrinter << "SInt<";
  else
    odsPrinter << "UInt<";
  odsPrinter << this->getWidth();
  odsPrinter << ">";
}

mlir::ParseResult parseProcArgs(mlir::AsmParser &p,
                                llvm::SmallVector<mlir::Type> &args) {
  mlir::Type type;
  if (p.parseType(type))
    return mlir::failure();
  args.push_back(type);
  while (succeeded(p.parseOptionalComma())) {
    if (p.parseType(type))
      return mlir::failure();
    args.push_back(type);
  }
  return mlir::success();
}

void printProcArgs(mlir::AsmPrinter &p, llvm::ArrayRef<mlir::Type> args) {
  llvm::interleaveComma(args, p, [&p](mlir::Type type) { p.printType(type); });
}
