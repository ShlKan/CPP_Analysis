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

//===--- SysIR Process generation ---------===//

#include "SysGenProcess.h"
#include "SysGenModule.h"
#include "SysIR/Dialect/IR/SysDialect.h"
#include "SysIR/Dialect/IR/SysTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/APInt.h"

namespace sys {

// TODO: CIRGenFunction.cpp has similar definition. Think about to remove one of
// them and keep unique.

mlir::sys::ProcDefOP SysGenProcess::buildProcess(clang::CXXMethodDecl *method) {
  // TODO incorrect type
  llvm::ArrayRef<mlir::Type> argTys{};
  auto procType = mlir::sys::SProcessType::get(builder.getContext(), argTys);
  auto process = builder.create<mlir::sys::ProcDefOP>(
      SGM.getLoc(method->getLocation()), method->getDeclName().getAsString(),
      procType);
  // Body Generation.
  buildStmt(process.getBody(), method->getBody());

  builder.setInsertionPointToEnd(SGM.getModule().getBody());
  return process;
}

mlir::sys::ProcRegisterOP
SysGenProcess::buildProcessRegister(mlir::sys::ProcDefOP procOP) {
  mlir::ValueRange args{};
  return builder.create<mlir::sys::ProcRegisterOP>(procOP->getLoc(), procOP,
                                                   args);
}

} // namespace sys