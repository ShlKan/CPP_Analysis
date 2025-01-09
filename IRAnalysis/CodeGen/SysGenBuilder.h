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

#ifndef MLIR_SYS_GEN_BUILDER_H
#define MLIR_SYS_GEN_BUILDER_H

#include "CIRGenBuilder.h"

namespace sys {

/* SysGenBuilder inherits from CIRGenBuilderTy because it also used to generate
 * CIR operations. */
class SysGenBuilder : public cir::CIRGenBuilderTy {

public:
  using cir::CIRGenBuilderTy::CIRGenBuilderTy;
};

} // namespace sys

#endif