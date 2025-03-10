//===---- SysGenModule.h - Sys IR Module Generation ----===//
//===-----------------------------------------------------------------===//

#ifndef MLIR_SYS_MODULE_GEN_H
#define MLIR_SYS_MODULE_GEN_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <map>
#include <memory>

#include "CIRGenModule.h"

#include "CPPFrontend/CIROptions.h"

#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "SysMatcher.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
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

  mlir::sys::SIntType getSSignedIntType(uint32_t size);
  mlir::sys::SIntType getSUSignedIntType(uint32_t size);
  SysMatcher *getSysMatcher() { return sysMatcher.get(); }

  ~SysGenModule();

private:
  llvm::SmallVector<clang::StringLiteral *, 4> processNames;
  std::unordered_map<uint32_t, mlir::sys::SIntType> sSignedIntTyMap;
  std::unordered_map<uint32_t, mlir::sys::SIntType> sUnsigendIntTyMap;
  void collectProcess(const clang::CXXRecordDecl *moduleDecl);
  std::unique_ptr<sys::SysMatcher> sysMatcher;

public:
  mlir::sys::ConstantOp getConstSysInt(mlir::Location loc, mlir::Type ty,
                                       llvm::APInt &val);
  void buildSysModule(const clang::CXXRecordDecl *moduleDecl);
  void buildFieldDecl(const clang::FieldDecl *decl);

  mlir::Type convertType(const clang::QualType type);
  mlir::Value
  buildExpr(clang::Expr *expr, mlir::Operation *context,
            llvm::ScopedHashTable<const clang::Decl *, mlir::Value> &symTable);
};
} // namespace sys

#endif