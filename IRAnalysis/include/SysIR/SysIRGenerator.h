//===- SysIRGenerator.h - Sys IR Generation from Clang AST --------------===//

#ifndef CLANG_SYSIRGENERATOR_H_
#define CLANG_SYSIRGENERATOR_H_

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/CodeGenOptions.h"

#include "CPPFrontend/CIROptions.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <clang/Frontend/CompilerInstance.h>

#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
class OwningModuleRef;
} // namespace mlir

namespace clang {
class ASTContext;
class DeclGroupRef;
class FunctionDecl;
} // namespace clang

namespace sys {

class SysGenModule;

class SysIRGenerator : public clang::ASTConsumer {
  clang::DiagnosticsEngine &Diags;
  clang::ASTContext *astCtx;
  const cir::CIROptions cirOpts;
  std::unique_ptr<SysGenModule> sysMG;

protected:
  std::unique_ptr<mlir::MLIRContext> mlirCtx;

public:
  SysIRGenerator(cir::CIROptions cirOpts, clang::DiagnosticsEngine &diags);
  ~SysIRGenerator();
  void Initialize(clang::ASTContext &Context) override;

  bool HandleTopLevelDecl(clang::DeclGroupRef D) override;
  void HandleTranslationUnit(clang::ASTContext &Ctx) override;

  mlir::ModuleOp getModule();
  std::unique_ptr<mlir::MLIRContext> takeContext() {
    return std::move(mlirCtx);
  };

  bool verifyModule();
};

} // namespace sys

#endif // CLANG_CIRGENERATOR_H_
