//===---- SysGenAction.h - Sys IR Code Generation Frontend Action ----===//
//===-----------------------------------------------------------------===//

#ifndef MLIR_SYS_GEN_ACTION_H
#define MLIR_SYS_GEN_ACTION_H

#include <SysIR/SysIRGenerator.h>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/StringRef.h"
#include <clang/Frontend/FrontendAction.h>

#include <memory>

namespace sys {
class SysGenConsumer;

class SysGenAction : public clang::ASTFrontendAction {

public:
  enum class OutputType { EmitSySIR, None };

protected:
  SysGenAction(OutputType emitType);

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci,
                    llvm::StringRef InFile) override;

  void ExecuteAction() override;
  void EndSourceFileAction() override;

public:
  ~SysGenAction();

  SysGenConsumer *sysConsumer;
  OutputType action;
};

class EmitSysGenAction : public SysGenAction {
public:
  EmitSysGenAction(mlir::MLIRContext *mlirCtx = nullptr);
};

} // namespace sys

#endif
