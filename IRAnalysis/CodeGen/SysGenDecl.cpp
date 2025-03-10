// Copyright 2025 kanshl
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

//===---- SysGenDecl.h - Sys IR Declaration Generation ----===//
//          Translation of all possible declarations in C++.
//===-----------------------------------------------------------------===//

#ifndef MLIR_SYS_DECLARATION_GEN_H
#define MLIR_SYS_DECLARATION_GEN_H

#include "SysGenModule.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Casting.h"

namespace sys {

void SysGenModule::buildFieldDecl(const clang::FieldDecl *field) {
  llvm::ScopedHashTable<const clang::Decl *, mlir::Value> hashTable;
  auto expr = buildExpr(field->getInClassInitializer(), theModule, hashTable);
  mlir::SymbolTable::setSymbolName(expr.getDefiningOp(),
                                   field->getDeclName().getAsString());
}

} // namespace sys

#endif