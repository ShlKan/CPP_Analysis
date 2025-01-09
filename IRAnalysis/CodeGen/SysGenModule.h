//===---- SysGenModule.h - Sys IR Module Generation ----===//
//===-----------------------------------------------------------------===//

#ifndef MLIR_SYS_MODULE_GEN_H
#define MLIR_SYS_MODULE_GEN_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallVector.h"

#include "CIRGenModule.h"

#include "CPPFrontend/CIROptions.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

namespace sys {

/**  This class inherits from CIRGenModule, in order to reuse
     its build methds for non-systemc-specific constructs. */
class SysGenModule : public cir::CIRGenModule {
  SysGenModule(SysGenModule &) = delete;
  SysGenModule &operator=(SysGenModule &) = delete;

public:
  SysGenModule(mlir::MLIRContext &context, clang::ASTContext &astctx,
               const clang::CodeGenOptions &codeGenOpts,
               const cir::CIROptions &CIROption,
               clang::DiagnosticsEngine &Diags);

  ~SysGenModule();

private:
  llvm::SmallVector<clang::StringLiteral *, 4> processNames;

private:
  void collectProcess(clang::CXXRecordDecl *moduleDecl);

public:
  void buildSysModule(clang::CXXRecordDecl *moduleDecl);
};
} // namespace sys

#endif