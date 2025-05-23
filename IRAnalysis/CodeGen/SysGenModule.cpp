/*
 * Created Date: Th Jan 2025
 * Author: Shuanglong Kan
 * -----
 * Last Modified: Fri Jan 17 2025
 * Modified By: Shuanglong Kan
 * -----
 * Copyright (c) 2025 Shuanglong Kan
 * ---------------------------------------------------------
 */

#include "SysGenModule.h"
#include "CIR/Dialect/IR/CIRDialect.h"
#include "CIR/Dialect/IR/CIROps.h.inc"
#include "CIR/Dialect/IR/CIRTypes.h"
#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "SysGenProcess.h"
#include "SysIR/Dialect/IR/SysAttrs.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "SysMatcher.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/AST/APValue.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"

#include "CIRGenFunction.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace sys {

SysGenModule::SysGenModule(mlir::MLIRContext &context,
                           clang::ASTContext &astCtx,
                           const clang::CodeGenOptions &codeGenOpts,
                           const cir::CIROptions &CIROptions,
                           clang::DiagnosticsEngine &diags)
    : cir::CIRGenModule(context, astCtx, codeGenOpts, CIROptions, diags),
      sysMatcher(std::make_unique<sys::SysMatcher>(astCtx)) {}

void SysGenModule::collectProcess(const clang::CXXRecordDecl *moduleDecl) {
  for (const auto &method : moduleDecl->methods()) {
    if (!llvm::isa<clang::CXXConstructorDecl>(method))
      continue;
    auto stmt = method->getBody();
    if (!stmt)
      continue;
    for (const auto &childStmt : stmt->children()) {
      auto memCall = llvm::cast<clang::CXXMemberCallExpr>(childStmt);
      if (!memCall)
        continue;
      // When this method in the constructor is a declare process statement,
      // We register this process in the module.
      if (memCall->getMethodDecl()->getNameAsString() !=
          "declare_thread_process")
        continue;
      // declare process statement's first non-object (second) argument is the
      // process name.
      auto procVar = llvm::cast<clang::ImplicitCastExpr>(memCall->getArg(1));
      auto procName = llvm::cast<clang::StringLiteral>(procVar->getSubExpr());
      processNames.push_back(procName);
    }
  }
}

mlir::sys::ConstantOp SysGenModule::getConstSysInt(mlir::Location loc,
                                                   mlir::Type ty,
                                                   llvm::APInt &val) {
  auto ConstantOp = builder.create<mlir::sys::ConstantOp>(
      loc, ty, mlir::sys::IntAttr::get(ty, val));
  return ConstantOp;
}

mlir::sys::ConstantOp SysGenModule::getConstSysBV(mlir::Location loc,
                                                  mlir::Type ty,
                                                  llvm::StringRef &val) {
  auto ConstantOp = builder.create<mlir::sys::ConstantOp>(
      loc, ty, mlir::sys::BitVecAttr::get(ty, val));
  return ConstantOp;
}

mlir::sys::ConstantOp SysGenModule::getConstSysBVL(mlir::Location loc,
                                                   mlir::Type ty,
                                                   llvm::StringRef &val) {
  auto ConstantOp = builder.create<mlir::sys::ConstantOp>(
      loc, ty, mlir::sys::BitVecLAttr::get(ty, val));
  return ConstantOp;
}

void SysGenModule::buildSysModule(const clang::CXXRecordDecl *moduleDecl) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  theModule.setName(moduleDecl->getDeclName().getAsString());

  SysGenProcess genProcess(*this, builder);
  builder.setInsertionPointToEnd(theModule.getBody());

  for (const auto &field : moduleDecl->fields()) {
    buildFieldDecl(field);
  }

  collectProcess(moduleDecl);
  llvm::SmallVector<mlir::sys::ProcDefOP, 4> processOPs;
  for (const auto &method : moduleDecl->methods()) {
    if (std::find_if(processNames.begin(), processNames.end(),
                     [&method](const clang::StringLiteral *procName) {
                       return procName->getString() ==
                              method->getDeclName().getAsString();
                     }) != processNames.end()) {
      cir::CIRGenFunction cFunGen(*this, builder);
      CurCGF = &cFunGen;
      auto proc = genProcess.buildProcess(method);
      processOPs.push_back(proc);
      CurCGF = nullptr;
    }
  }

  for (const auto &procOP : processOPs)
    genProcess.buildProcessRegister(procOP);
  theModule.print(llvm::outs());
}

mlir::sys::SIntType SysGenModule::getSSignedIntType(uint32_t size) {
  if (sSignedIntTyMap.count(size))
    return sSignedIntTyMap[size];
  auto ty = mlir::sys::SIntType::get(builder.getContext(), size, true);
  sSignedIntTyMap[size] = ty;
  return ty;
}

mlir::sys::SBitVecType SysGenModule::getBitVecType(uint32_t size) {
  if (sBitVecTyMap.count(size))
    return sBitVecTyMap[size];
  auto ty = mlir::sys::SBitVecType::get(builder.getContext(), size);
  sBitVecTyMap[size] = ty;
  return ty;
}

mlir::sys::SBitVecLType SysGenModule::getBitVecLType(uint32_t size) {
  if (sBitVecLTyMap.count(size))
    return sBitVecLTyMap[size];
  auto ty = mlir::sys::SBitVecLType::get(builder.getContext(), size);
  sBitVecLTyMap[size] = ty;
  return ty;
}

mlir::sys::SIntType SysGenModule::getSUSignedIntType(uint32_t size) {
  if (sUnsigendIntTyMap.count(size))
    return sUnsigendIntTyMap[size];
  auto ty = mlir::sys::SIntType::get(builder.getContext(), size, false);
  sUnsigendIntTyMap[size] = ty;
  return ty;
}

mlir::Type SysGenModule::convertType(const clang::QualType type) {
  auto plainType = type.getDesugaredType(astCtx);
  if (llvm::isa<BuiltinType>(plainType)) {
    return genTypes.ConvertType(plainType);
  }
  if (auto sizeOpt = sysMatcher->matchSysInt(plainType)) {
    return getSSignedIntType(sizeOpt.value());
  }
  llvm_unreachable("Unsupported Type.");
}

SysGenModule::~SysGenModule() {}

} // namespace sys