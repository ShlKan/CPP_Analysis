//===---- SysGenModule.h - Sys IR Module Generation ----===//
//===-----------------------------------------------------------------===//

#ifndef MLIR_SYS_MODULE_GEN_H
#define MLIR_SYS_MODULE_GEN_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallVector.h"

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
  llvm::SmallVector<clang::StringLiteral *, 4> processNames;

  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;

private:
  void collectProcess(clang::CXXRecordDecl *moduleDecl);

public:
  mlir::ModuleOp getModule() { return theModule; }
  clang::DiagnosticsEngine &getDiag() { return diags; }
  const cir::CIROptions &getCIROptions() { return cirOptions; }
  void buildSysModule(clang::CXXRecordDecl *moduleDecl);

  mlir::Location getLoc(clang::SourceLocation SLoc) {
    assert(SLoc.isValid() && "expected valid source location");
    const clang::SourceManager &SM = astCtx.getSourceManager();
    clang::PresumedLoc PLoc = SM.getPresumedLoc(SLoc);
    llvm::StringRef Filename = PLoc.getFilename();
    return mlir::FileLineColLoc::get(builder.getStringAttr(Filename),
                                     PLoc.getLine(), PLoc.getColumn());
  }

  mlir::Location
  getLoc(clang::SourceRange SLoc) { // Some AST nodes might contain invalid
                                    // source locations (e.g.
    // CXXDefaultArgExpr), workaround that to still get something out.
    if (SLoc.isValid()) {
      mlir::Location B = getLoc(SLoc.getBegin());
      mlir::Location E = getLoc(SLoc.getEnd());
      llvm::SmallVector<mlir::Location, 2> locs = {B, E};
      mlir::Attribute metadata;
      return mlir::FusedLoc::get(locs, metadata, builder.getContext());
    }
    return builder.getUnknownLoc();
  };
};
} // namespace sys

#endif