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
#include "SysGenExpr.h"
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

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
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
      sysMatcher(std::make_unique<sys::SysMatcher>(astCtx)),
      sysGenExpr(std::make_unique<sys::SysGenExpr>(builder, this, astCtx)) {}

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
                                                   llvm::StringRef varName,
                                                   mlir::Type ty,
                                                   llvm::APInt &val) {
  auto ConstantOp = builder.create<mlir::sys::ConstantOp>(
      loc, ty, mlir::sys::IntAttr::get(ty, val));
  mlir::SymbolTable::setSymbolName(ConstantOp, varName);
}

void SysGenModule::buildFieldDeclBuiltin(mlir::Location loc,
                                         llvm::StringRef varName,
                                         clang::BuiltinType::Kind &kind,
                                         llvm::APInt &val) {
  switch (kind) {
  case clang::BuiltinType::Int:
  case clang::BuiltinType::Int128:
  case clang::BuiltinType::Long:
  case clang::BuiltinType::LongLong: {
    auto constantOp = builder.getConstInt(loc, llvm::APSInt(val, false));
    mlir::SymbolTable::setSymbolName(constantOp, varName);
    break;
  }
  case clang::BuiltinType::UInt:
  case clang::BuiltinType::UInt128:
  case clang::BuiltinType::ULong:
  case clang::BuiltinType::ULongLong: {
    auto constantOp = builder.getConstInt(loc, val);
    mlir::SymbolTable::setSymbolName(constantOp, varName);
    break;
  }
  case clang::BuiltinType::Bool: {
    break;
  }
  default:
    break;
  }
}

void SysGenModule::buildSysModule(const clang::CXXRecordDecl *moduleDecl) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  theModule.setName(moduleDecl->getDeclName().getAsString());

  SysGenProcess genProcess(*this, builder);
  builder.setInsertionPointToEnd(theModule.getBody());

  for (const auto &field : moduleDecl->fields()) {
    // TODO : currently only support constant OP. When expr in sysIR is defined,
    if (auto kindOpt = sysMatcher->matchBuiltinInt(field->getType());
        kindOpt.has_value()) {
      if (auto initVal = sysMatcher->matchFieldInitAPInt(*field))
        buildFieldDeclBuiltin(getLoc(field->getLocation()),
                              field->getDeclName().getAsString(),
                              kindOpt.value(), initVal.value());
    } else if (auto sizeOpt = sysMatcher->matchSysInt(field->getType());
               sizeOpt != std::nullopt) {
      if (auto initVal = sysMatcher->matchFieldInitAPInt(*field))
        getConstSysInt(getLoc(field->getLocation()),
                       field->getDeclName().getAsString(),
                       getSSignedIntType(sizeOpt.value()), initVal.value());
      else {
        sysGenExpr->buildExpr(field->getInClassInitializer(), theModule);
      }
    } else
      llvm_unreachable("Unsupport type.");
  }

  collectProcess(moduleDecl);
  llvm::SmallVector<mlir::sys::ProcDefOP, 4> processOPs;
  for (const auto &method : moduleDecl->methods()) {
    if (std::find_if(processNames.begin(), processNames.end(),
                     [&method](const clang::StringLiteral *procName) {
                       return procName->getString() ==
                              method->getDeclName().getAsString();
                     }) != processNames.end()) {
      auto proc = genProcess.buildProcess(method);
      processOPs.push_back(proc);
    }
  }
  for (const auto &procOP : processOPs)
    genProcess.buildProcessRegister(procOP);
  theModule->dump();
}

mlir::sys::SIntType SysGenModule::getSSignedIntType(uint32_t size) {
  if (sSignedIntTyMap.count(size))
    return sSignedIntTyMap[size];
  auto ty = mlir::sys::SIntType::get(builder.getContext(), size, true);
  sSignedIntTyMap[size] = ty;
  return ty;
}

mlir::sys::SIntType SysGenModule::getSUSignedIntType(uint32_t size) {
  if (sUnsigendIntTyMap.count(size))
    return sUnsigendIntTyMap[size];
  auto ty = mlir::sys::SIntType::get(builder.getContext(), size, false);
  sUnsigendIntTyMap[size] = ty;
  return ty;
}

SysGenModule::~SysGenModule() {}

} // namespace sys