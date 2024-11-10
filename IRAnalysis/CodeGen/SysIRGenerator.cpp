//===--- SysIRGenerator.cpp - Emit Sys IR from ASTs ----------------------===//

#include "SysIR/SysIRGenerator.h"
#include "mlir/IR/BuiltinOps.h"

using namespace sys;
using namespace clang;

SysIRGenerator::SysIRGenerator(DiagnosticsEngine &diags) : Diags(diags) {}

SysIRGenerator::~SysIRGenerator() {
  // TODO
}

void SysIRGenerator::Initialize(clang::ASTContext &Context) {
  // TODO
}

bool SysIRGenerator::HandleTopLevelDecl(clang::DeclGroupRef D) {
  // TODO
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
