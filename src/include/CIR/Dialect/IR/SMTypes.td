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

  let description = "SInt is different from C++ builtin integers. "
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

//System Unsigned integer.
class SysUInt<int width>
  : Type <And<[
      CPred<"::mlir::isa<::mlir::sm::SIntType>($_self)">,
      CPred<"::mlir::cast<::mlir::sm::SIntType>($_self).isUnsigned()">,
      CPred<"::mlir::cast<::mlir::sm::SIntType>($_self).getWidth() == " # width>
    ]>, width # "-bit unsigend system integer", "::mlir::sm::SInt">,
    BuildableType<"::mlir::sm::SIntType::get($_builder.getContext(), " # width # " /*isSigned=*/false)"> {
  int bitwidth = width;
}

//System Signed integer.
class SysSInt<int width>
  : Type <And<[
      CPred<"::mlir::isa<::mlir::sm::SIntType>($_self)">,
      CPred<"::mlir::cast<::mlir::sm::SIntType>($_self).isSigned()">,
      CPred<"::mlir::cast<::mlir::sm::SIntType>($_self).getWidth() == " # width>
    ]>, width # "-bit unsigend system integer", "::mlir::sm::SInt">,
    BuildableType<"::mlir::sm::SIntType::get($_builder.getContext(), " # width # " /*isSigned=*/false)"> {
  int bitwidth = width;
}


#endif

