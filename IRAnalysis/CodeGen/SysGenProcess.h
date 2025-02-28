// Copyright 2024 kanshl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===---- SysGenProcess.h - Sys IR Process Generation ----===//
//===-----------------------------------------------------===//

//===---- SysGenModule.h - Sys IR Module Generation ----===//
//===-----------------------------------------------------------------===//

#ifndef MLIR_SYS_PROCESS_GEN_H
#define MLIR_SYS_PROCESS_GEN_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Stmt.h"

#include "SysGenModule.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

namespace sys {
class SysGenProcess {
private:
  SysGenModule &SGM;
  cir::CIRGenBuilderTy &builder;
  using SymTableTy = llvm::ScopedHashTable<const clang::Decl *, mlir::Value>;
  SymTableTy symbolTable;

protected:
  SysGenProcess(SysGenProcess &) = delete;
  SysGenProcess &operator=(SysGenProcess &) = delete;

public:
  SysGenProcess(SysGenModule &SGM, cir::CIRGenBuilderTy &builder)
      : SGM(SGM), builder(builder) {}
  mlir::sys::ProcDefOP buildProcess(clang::CXXMethodDecl *);
  mlir::sys::ProcRegisterOP buildProcessRegister(mlir::sys::ProcDefOP);
  mlir::Block *buildCompoundStmt(mlir::Region &parent,
                                 clang::CompoundStmt *compoundStmt);
  void buildDeclStmt(clang::DeclStmt *);
  void buildStmt(mlir::Region &parent, clang::Stmt *stmt);
  void buildVarDecl(clang::VarDecl *);
  void buildIfStmt(clang::IfStmt *ifStmt);
  void buildLoopStmt(clang::ForStmt *loopStmt);
};
} // namespace sys

#endif