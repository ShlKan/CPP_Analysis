//===---- SysGenModule.h - Sys IR Module Generation ----===//
//===-----------------------------------------------------------------===//

#ifndef MLIR_SYS_MODULE_GEN_H
#define MLIR_SYS_MODULE_GEN_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <map>

#include "CIRGenModule.h"

#include "CPPFrontend/CIROptions.h"

#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
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

  ~SysGenModule();

private:
  llvm::SmallVector<clang::StringLiteral *, 4> processNames;
  std::map<uint32_t, mlir::sys::SIntType> sSignedIntTyMap;
  std::map<uint32_t, mlir::sys::SIntType> sUnsigendIntTyMap;
  void collectProcess(clang::CXXRecordDecl *moduleDecl);

public:
  mlir::sys::ConstantOp getConstInt(mlir::Location loc, mlir::Type ty,
                                    uint64_t val);
  void buildSysModule(clang::CXXRecordDecl *moduleDecl);
};
} // namespace sys

#endif