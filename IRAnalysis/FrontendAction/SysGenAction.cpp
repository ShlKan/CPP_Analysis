//===--- SysIR generation Frontend Action ---------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"

#include <SysIRFrontendAction/SysGenAction.h>
#include <iostream>

#include <memory>

using namespace sys;

namespace sys {
class SysGenConsumer : public clang ::ASTConsumer {

private:
  SysGenAction::OutputType action;
  std::unique_ptr<sys::SysIRGenerator> generator;
  clang::ASTContext *astContext{nullptr};

public:
  SysGenConsumer(SysGenAction::OutputType action,
                 const cir::CIROptions &cirOptions,
                 clang::DiagnosticsEngine &diags)
      : action(action),
        generator(std::make_unique<sys::SysIRGenerator>(cirOptions, diags)) {}

  void Initialize(clang::ASTContext &ctx) override {
    // TODO
    astContext = &ctx;
    generator->Initialize(ctx);
  }

  bool HandleTopLevelDecl(clang::DeclGroupRef D) override {
    generator->HandleTopLevelDecl(D);
    return true;
  }

  void HandleTranslationUnit(clang::ASTContext &ctx) override {}
};

SysGenAction::SysGenAction(OutputType emitType) : action(emitType) {}
SysGenAction::~SysGenAction() {
  // TODO
}

std::unique_ptr<clang::ASTConsumer>
SysGenAction::CreateASTConsumer(clang::CompilerInstance &ci,
                                llvm::StringRef InFile) {
  auto result =
      std::make_unique<SysGenConsumer>(action, *cirOption, ci.getDiagnostics());
  sysConsumer = result.get();
  result->Initialize(ci.getASTContext());
  return std::move(result);
}

void SysGenAction::ExecuteAction() { this->ASTFrontendAction::ExecuteAction(); }

void SysGenAction::EndSourceFileAction() {
  // TODO
}

EmitSysGenAction::EmitSysGenAction(mlir::MLIRContext *mlirCtx)
    : SysGenAction(SysGenAction::OutputType::EmitSySIR) {}
} // namespace sys