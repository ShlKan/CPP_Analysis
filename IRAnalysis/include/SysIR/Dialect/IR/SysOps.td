//===----------------------------------------------------------------------===//
//
// Created by Shuanglong Kan, 25.10.2024
// This file declares the Sys dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef _SYSMTEM_SYSOPS_H
#define _SYSMTEM_SYSOPS_H

include "SysIR/Dialect/IR/SysTypes.td"
include "SySIR/Dialect/IR/SysDialect.td"
include "SysIR/Dialect/IR/SysAttrs.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

class SysOP<string mnemoic, list<Trait> traits=[]> :
    Op<Sys_Dialect, mnemoic, traits>;

def ProcDefOP : SysOP<"ProcDef"> {
    let summary = "Define a process.";
    
    let description = [{
        This OP corresponds to a definition of a process, 
        which may run in parallel with other processes in the system.
    }];

    let arguments = (ins 
        SymbolNameAttr:$proc_name,
        TypeAttrOf<SysProcType>:$proc_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs
    );

    let results = (outs SysProcType:$result);

    let regions = (region AnyRegion:$body);
    
    let builders = [OpBuilder<(ins
        "StringRef":$name,
        "SProcessType":$type
        //CArg wraps a type and its default value
        //CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs
        )>];

    let assemblyFormat = "$proc_name $body attr-dict `:` type($result)";
}

def ProcRegisterOP : SysOP<"ProcRegister"> {
    let summary = "Register a process into the system";
    let description = [{
        With the definition of processes, we also need to register these processes 
        into the system. When the system starts, all registered processes will start to 
        run in parallel.
    }];

    let arguments = (ins SysProcType:$proc, Variadic<Sys_AnyType>:$args);
    let assemblyFormat = "$proc $args attr-dict `[` type($proc) ` ` type($args) `]` `:` `void`";
}

def ConstantOp : SysOP<"const",
    [ConstantLike, Pure, AllTypesMatch<["value", "res"]>]> {
  // type repetition in the assembly form.

  let summary = "Defines a SysIR constant";
  let description = [{
    The `sys.const` operation turns a literal into an SSA value. The data is
    attached to the operation as an attribute.

    ```mlir
      %0 = sys.const 42 : i32
      %1 = sys.const 4.2 : f32
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins TypedAttrInterface:$value);

  // The constant operation returns a single value of Sys_AnyType.
  let results = (outs Sys_AnyType:$res);

  let assemblyFormat = "attr-dict $value";

  //let hasVerifier = 1;

  //let hasFolder = 1;
}

#endif
