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
#include "SysIR/Dialect/IR/SysAttrs.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace sys {

void SysGenModule::buildFieldDecl(const clang::FieldDecl *field) {
  llvm::ScopedHashTable<const clang::Decl *, mlir::Value> hashTable;
  if (field->getInClassInitializer() == nullptr) {
    if (this->getSysMatcher()->matchSCEventTy(field->getType())) {
      // Create a new event
      mlir::sys::SEventType eventType =
          mlir::sys::SEventType::get(builder.getContext());
      auto eventAttr = mlir::sys::EventAttr::get(
          builder.getContext(), eventType, field->getDeclName().getAsString());
      builder.create<mlir::sys::ConstantOp>(getLoc(field->getLocation()),
                                            eventType, eventAttr);
    } else {
      llvm_unreachable(
          "The field type is not supported in the current version of SysGen.");
    }
    return;
  }
  auto expr = buildExpr(field->getInClassInitializer(), theModule, hashTable);
  mlir::SymbolTable::setSymbolName(expr.getDefiningOp(),
                                   field->getDeclName().getAsString());
}

} // namespace sys

#endif