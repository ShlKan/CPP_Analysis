//===----------------------------------------------------------------------===//
//
// Created by Shuanglong Kan, 25.10.2024
// This file declares the SM dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef _SYSMTEM_SMTYPE_H
#define _SYSMTEM_SMTYPE_H

include "CIR/Dialect/IR/SMDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

//===----------------------------------------------------------------------===//
// SM Type
//===----------------------------------------------------------------------===//

class SM_Type<string name, string typeMnemonic, list<Trait> traits =[], 
        string baseCppClass = "::mlir::Type" >
    : TypeDef<SM_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// SM Integer Type
//===----------------------------------------------------------------------===//
def SM_IntType : SM_Type<"SInt", "s_int", 
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "System integer type with arbitrary precision up to a fixed limit";

  let description = "SInt is different from C++ builtin integer types. "
                    "Especially for the treatment of overflow."
                    "For example, if x + y (x and y are unsigned integers) is overflow then "
                    "C++ builtin integers will wrap round the result. But SInt will extend one bit"
                    "to represent the accurate value.";
  let parameters = (ins "unsigned":$width, "bool":$isSigned);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{

    /// Return whether the type is signed.
    bool isSigned() const { return getIsSigned(); }

    /// Return whether the type is unsigned.
    bool isUnsigned() const { return !getIsSigned(); }

  }];
  let genVerifyDecl = 1;
}

def SysUIntType
  : Type <And<[
      CPred<"::mlir::isa<::mlir::sm::SIntType>($_self)">,
      CPred<"::mlir::cast<::mlir::sm::SIntType>($_self).isUnsigned()">,
      CPred<"::mlir::cast<::mlir::sm::SIntType>($_self).getWidth() > 0">
    ]>, "Unsigend system integer",  "::mlir::sm::SIntType">;

//System Signed integer.
def SysSIntType
  : Type <And<[
      CPred<"::mlir::isa<::mlir::sm::SIntType>($_self)">,
      CPred<"::mlir::cast<::mlir::sm::SIntType>($_self).isSigned()">,
      CPred<"::mlir::cast<::mlir::sm::SIntType>($_self).getWidth() > 0 " >
    ]>, "Sigend system integer", "::mlir::sm::SIntType">;


def SysProcType : SM_Type<"SInt", "s_int"> {
  let summary = "The type of system process";
  let description = [{
    This is type of system processes.
    System processes contain only input parameters and always return void.
  }];

  let parameters = (ins ArrayRefParameter<"Type">:$inputs);

  let assemblyFormat = [{
    `P` ` ` `(` custom<ProcArgs>($inputs) `)`
  }];

  let builders = [
    TypeBuilder<(ins "ArrayRef<Type>":$inputs), [{
      return $_get($_ctxt, inputs);
    }]>
  ];
}

def SM_AnyType : AnyTypeOf<[
  SM_IntType
]>;

#endif

