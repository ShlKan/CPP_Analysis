//===----------------------------------------------------------------------===//
//
// Created by Shuanglong Kan, 25.10.2024
// This file declares the SYS dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef _SYSMTEM_SYSTYPE_H
#define _SYSMTEM_SYSTYPE_H

include "SysIR/Dialect/IR/SysDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "CIR/Dialect/IR/CIRTypes.td"
include "SysIR/Dialect/IR/SysAttrs.td"



//===----------------------------------------------------------------------===//
// Sys Type
//===----------------------------------------------------------------------===//

class Sys_Type<string name, string typeMnemonic, list<Trait> traits =[],
        string baseCppClass = "::mlir::Type" >
    : TypeDef<Sys_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Sys Integer Type
//===----------------------------------------------------------------------===//
def Sys_IntType : Sys_Type<"SInt", "s_int",
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

class SysUIntType<int width>
  : Type <And<[
      CPred<"::mlir::isa<::mlir::sys::SIntType>($_self)">,
      CPred<"::mlir::cast<::mlir::sys::SIntType>($_self).isUnsigned()">,
      CPred<"::mlir::cast<::mlir::sys::SIntType>($_self).getWidth() > 0">
    ]>, "Unsigend system integer",  "::mlir::sys::SIntType">,
    BuildableType<
      "mlir::sys::SIntType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/false)"> {
  int bitwidth = width;
}

//System Signed integer.
class SysSIntType<int width>
  : Type <And<[
      CPred<"::mlir::isa<::mlir::sys::SIntType>($_self)">,
      CPred<"::mlir::cast<::mlir::sys::SIntType>($_self).isSigned()">,
      CPred<"::mlir::cast<::mlir::sys::SIntType>($_self).getWidth() > 0 " >
    ]>, "Sigend system integer", "::mlir::sys::SIntType">,
    BuildableType<
      "mlir::sys::SIntType::get($_builder.getContext(), "
      # width # ", /*isSigned=*/true)"> {
  int bitwidth = width;
}


//===----------------------------------------------------------------------===//
// Sys Bit Vector Type
//===----------------------------------------------------------------------===//
def Sys_BitVecType : Sys_Type<"SBitVec", "s_bv",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "System bit vector type with only 0 and 1 as values";

  let description = "SBitVec is a bit vector type that supports a set of bitvector operations."
                    "It is configured by the bitwidth of the vector. ";
  let parameters = (ins "unsigned":$width);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Sys Bit Vector 4 Value Type
//===----------------------------------------------------------------------===//
def Sys_BitVecLType : Sys_Type<"SBitVecL", "s_bvl",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "System bit vector type with four kinds of bit values: 0, 1, Z (high-impedance), and X.";

  let description = "SBitVecL is a bit vector type that supports a set of bitvector operations."
                    "It is configured by the bitwidth of the vector. ";
  let parameters = (ins "unsigned":$width);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===-----------------------------------------------------------------------===//
// Sys Event Type
//===-----------------------------------------------------------------------===//
def Sys_EventType : Sys_Type<"SEvent", "s_event"> {
  let summary = "System event type.";
  let description = "SEvent is the event type (sc_event) in SystemC.";
  let assemblyFormat = [{`event`}];
}



//===-----------------------------------------------------------------------===//
// Sys Time Type
//===-----------------------------------------------------------------------===//
def Sys_TimeType : Sys_Type<"STime", "s_time"> {
  let summary = "System time type.";
  let description = "STime is the time.";
  let assemblyFormat = [{`time`}];
}


def SysProcType : Sys_Type<"SProcess", "s_proc"> {
  let summary = "The type of system process";
  let description = [{
    This is type of system processes.
    System processes contain only input parameters and always return void.
  }];

  let parameters = (ins ArrayRefParameter<"Type">:$inputs);

  let assemblyFormat = [{
    `Proc` ` ` `(` custom<ProcArgs>($inputs) `)`
  }];

}



def Sys_AnyType : AnyTypeOf<[
  Sys_IntType, Sys_BitVecType, Sys_BitVecLType, CIR_BoolType,
  CIR_IntType, Sys_EventType, Sys_TimeType
]>;

def Sys_BVType : AnyTypeOf<[
  Sys_BitVecType, Sys_BitVecLType
]>;


def Sys_EventAndTime : AnyTypeOf <[
  Sys_EventType, Sys_TimeType
]>;

#endif

