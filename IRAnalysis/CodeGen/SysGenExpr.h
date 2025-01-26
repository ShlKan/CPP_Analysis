//===---- SysGenExp.h - Sys IR Module Generation ----===//
//===-----------------------------------------------------------------===//

#ifndef MLIR_SYS_EXP_GEN_H
#define MLIR_SYS_EXP_GEN_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <map>
#include <memory>

#include "CIRGenBuilder.h"
#include "CIRGenModule.h"

#include "CPPFrontend/CIROptions.h"

#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "SysMatcher.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

namespace sys {

/**  This class inherits from CIRGenModule, in order to reuse
     its build methds for non-systemc-specific constructs. */
class SysGenExpr {
private:
  cir::CIRGenBuilderTy &builder;
  cir::CIRGenModule *theModule;
  std::unique_ptr<sys::SysMatcher> sysMatcher;
  clang::ASTContext &astCtx;

public:
  SysGenExpr(cir::CIRGenBuilderTy &builder, cir::CIRGenModule *theModule,
             clang::ASTContext &astCtx)
      : builder(builder), theModule(theModule), astCtx(astCtx) {
    sysMatcher = std::make_unique<sys::SysMatcher>(astCtx);
  }
  mlir::Value buildExpr(clang::Expr *expr, mlir::Operation *context);
  mlir::Value buildBinOp(clang::BinaryOperator *binExpr,
                         mlir::Operation *context);
};
} // namespace sys

#endif