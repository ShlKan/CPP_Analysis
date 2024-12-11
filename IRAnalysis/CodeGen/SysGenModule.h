//===---- SysGenModule.h - Sys IR Module Generation ----===//
//===-----------------------------------------------------------------===//

#ifndef MLIR_SYS_MODULE_GEN_H
#define MLIR_SYS_MODULE_GEN_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/Diagnostic.h"

#include "CPPFrontend/CIROptions.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace sys {

class SysGenModule {
  SysGenModule(SysGenModule &) = delete;
  SysGenModule &operator=(SysGenModule &) = delete;

public:
  SysGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx,
               const cir::CIROptions &CIROption,
               clang::DiagnosticsEngine &Diags);

  ~SysGenModule();

private:
  const cir::CIROptions &cirOptions;
  clang::DiagnosticsEngine &diags;
  clang::ASTContext &astCtx;

  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;

public:
  mlir::ModuleOp getModule() { return theModule; }
  clang::DiagnosticsEngine &getDiag() { return diags; }
  const cir::CIROptions &getCIROptions() { return cirOptions; }
  void buildSysModule(clang::CXXRecordDecl *moduleDecl);
};
} // namespace sys

#endif