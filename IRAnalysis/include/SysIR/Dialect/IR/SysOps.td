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

//===--------------- Binary Operation for Sys Integers -----------------====/
def SBinOpKind_Mul: I32EnumAttrCase<"SMul", 1, "smul">;
def SBinOpKind_Div : I32EnumAttrCase<"SDiv", 2, "sdiv">;
def SBinOpKind_Rem : I32EnumAttrCase<"SRem", 3, "srem">;
def SBinOpKind_Add : I32EnumAttrCase<"SAdd", 4, "sadd">;
def SBinOpKind_Sub : I32EnumAttrCase<"SSub", 5, "ssub">;
def SBinOpKind_And : I32EnumAttrCase<"SAnd", 6, "sand">;
def SBinOpKind_Xor : I32EnumAttrCase<"SXor", 7, "sxor">;
def SBinOpKind_Or  : I32EnumAttrCase<"SOr", 8, "sor">;

def SBinOpKind : I32EnumAttr<
    "SBinOpKind",
    "binary operation (arith and logic) kind for Sys Integers",
    [SBinOpKind_Mul, SBinOpKind_Div, SBinOpKind_Rem,
     SBinOpKind_Add, SBinOpKind_Sub,
     SBinOpKind_And, SBinOpKind_Xor,
     SBinOpKind_Or]> {
  let cppNamespace = "::mlir::sys";
}

def BinOp : SysOP<"sbinop", [Pure,
  SameTypeOperands, SameOperandsAndResultType]> {

  let summary = "Binary operations (arith and logic) for system integers.";
  let description = [{
    sys.sbinop performs the binary operation according to
    the specified opcode kind: [mul, div, rem, add, sub,
    and, xor, or].

    It requires two input operands and has one result, all types
    should be the same.

    ```mlir
    %7 = sys.sbinop(sadd, %1, %2) : !sys_s32i
    %7 = sys.sbinop(smul, %1, %2) : !sys_u8i
    ```
  }];

  let results = (outs Sys_AnyType:$result);
  let arguments = (ins Arg<SBinOpKind, "sbinop kind">:$kind,
                       Sys_AnyType:$lhs, Sys_AnyType:$rhs,
                       UnitAttr:$no_unsigned_wrap,
                       UnitAttr:$no_signed_wrap);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)`
    (`nsw` $no_signed_wrap^)?
    (`nuw` $no_unsigned_wrap^)?
    `:` type($lhs) attr-dict
  }];

  let hasVerifier = 1;
}

#endif
