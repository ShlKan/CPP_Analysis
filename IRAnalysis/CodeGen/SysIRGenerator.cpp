//===--- SysIRGenerator.cpp - Emit Sys IR from ASTs ----------------------===//

#include "SysIR/SysIRGenerator.h"
#include "CIR/Dialect/IR/CIRDialect.h"
#include "SysGenModule.h"
#include "SysIR/Dialect/IR/SysDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/CodeGenOptions.h"
#include "llvm/Support/Casting.h"
#include <memory>

using namespace sys;
using namespace clang;

SysIRGenerator::SysIRGenerator(const cir::CIROptions cirOpts,
                               DiagnosticsEngine &diags)
    : cirOpts(cirOpts), Diags(diags) {}

SysIRGenerator::~SysIRGenerator() {
  // TODO
}

void SysIRGenerator::Initialize(clang::ASTContext &Context) {
  this->astCtx = &Context;
  mlirCtx = std::make_unique<mlir::MLIRContext>();
  mlirCtx->getOrLoadDialect<mlir::sys::SysDialect>();
  mlirCtx->getOrLoadDialect<mlir::cir::CIRDialect>();
  mlirCtx->getLoadedDialect<mlir::cf::ControlFlowDialect>();
  clang::CodeGenOptions codeOpts;
  sysMG = std::make_unique<SysGenModule>(*mlirCtx, *astCtx, codeOpts, cirOpts,
                                         Diags);
}

bool SysIRGenerator::HandleTopLevelDecl(clang::DeclGroupRef D) {
  for (const auto &topDecl : D) {
    auto decl = llvm::dyn_cast<CXXRecordDecl>(topDecl);
    if (decl == nullptr)
      continue;
    if (!decl->hasDefinition())
      continue;
    // If base type contains ::sc_core::sc_module, then this module is a systemc
    // module, and it should be processed by module generation.
    for (const auto &base : decl->bases()) {
      if (base.getType().getAsString() == "::sc_core::sc_module") {
        sysMG->buildSysModule(decl);
      }
    }
  }
  return true;
}

void SysIRGenerator::HandleTranslationUnit(clang::ASTContext &Ctx) {
  // TODO
}

mlir::ModuleOp SysIRGenerator::getModule() {
  // TODO
  return mlir::ModuleOp();
}

bool SysIRGenerator::verifyModule() {
  // TODO
  return true;
}
