//===--- SysIR generation Frontend Action ---------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"

#include <SysIR/SysIRGenerator.h>
#include <SysIRFrontendAction/SysGenAction.h>

#include <memory>

using namespace sys;

class SysGenConsumer : public clang ::ASTConsumer {

private:
  SysGenAction::OutputType action;
  std::unique_ptr<sys::SysIRGenerator> generator;
  clang::ASTContext *astContext{nullptr};

public:
  SysGenConsumer(SysGenAction::OutputType action,
                 clang::DiagnosticsEngine &diags)
      : action(action),
        generator(std::make_unique<sys::SysIRGenerator>(diags)) {}

  void Initialize(clang::ASTContext &ctx) override {
    // TODO
    astContext = &ctx;
    generator->Initialize(ctx);
  }

  void HandleTranslationUnit(clang::ASTContext &ctx) override {
    // TODO
  }
};

SysGenAction::SysGenAction(OutputType emitType) : action(emitType) {}
SysGenAction::~SysGenAction() {
  // TODO
}

std::unique_ptr<clang::ASTConsumer>
SysGenAction::CreateASTConsumer(clang::CompilerInstance &ci,
                                llvm::StringRef InFile) {
  return std::make_unique<SysGenConsumer>(action, ci.getDiagnostics());
}

void SysGenAction::ExecuteAction() {
  // TODO
}

void SysGenAction::EndSourceFileAction() {
  // TODO
}

EmitSysGenAction::EmitSysGenAction(mlir::MLIRContext *_MLIRContext)
    : SysGenAction(SysGenAction::OutputType::EmitSySIR) {}
