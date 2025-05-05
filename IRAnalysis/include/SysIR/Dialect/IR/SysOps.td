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
include "mlir/Interfaces/ControlFlowInterfaces.td"

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

//===--------------- Unary operator for BitVector ----------------------====/
def SUOpKind_AndReduce: I32EnumAttrCase<"AndRed", 31, "and_reduce">;
def SUOpKind_OrReduce: I32EnumAttrCase<"OrRed", 32, "or_reduce">;
def SUOpKind_NandReduce: I32EnumAttrCase<"NandRed", 33, "nand_reduce">;
def SUOpKind_NorReduce: I32EnumAttrCase<"NorRed", 34, "nor_reduce">;
def SUOpKind_XorReduce: I32EnumAttrCase<"XorRed", 35, "xor_reduce">;
def SUOpKind_XnorReduce: I32EnumAttrCase<"XnorRed", 36, "xnor_reduce">;

def SUnaryOpKind : I32EnumAttr<
  "SUnaryOpKind",
  "unary operation kind for Sys bit vectors",
  [SUOpKind_AndReduce, SUOpKind_OrReduce, SUOpKind_NandReduce,
   SUOpKind_NorReduce, SUOpKind_XorReduce, SUOpKind_XnorReduce]> {
 let cppNamespace = "::mlir::sys";
}

def UnaryOp : SysOP<"suaryop", [Pure]> {
  let summary = "Unary operations for system bit vectors.";
  let description =  [{
    sys.suaryop performs the unary operation according to
    the specified opcode kind: [add_reduce, nand_reduce, nor_reduce, xor_reduce, xnor_reduce].

    It requires one input operans and has one result.

    ```mlir
    %7 = sys.suaryop(add_reduce, %1)
    %8 = sys.suaryop(nand_reduce, %1)
    ```
  }];
  let results = (outs Sys_AnyType:$result);
  let arguments = (ins Arg<SUnaryOpKind, "sunaryop kind">:$kind,
                       Arg<Sys_AnyType>:$input);

  let assemblyFormat =
  [{
      `(` $kind `,` $input `)` `:` type($input) `,` type($result) attr-dict
  }];

  let hasVerifier = 1;
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
def SBinOpKind_ShiftL  : I32EnumAttrCase<"SShiftL", 9, "sshiftl">;
def SBinOpKind_ShiftR  : I32EnumAttrCase<"SShiftR", 10, "sshiftr">;
def SBinOpKind_Greater  : I32EnumAttrCase<"SGt", 11, "sgt">;

def SBinOpKind : I32EnumAttr<
    "SBinOpKind",
    "binary operation (arith and logic) kind for Sys Integers",
    [SBinOpKind_Mul, SBinOpKind_Div, SBinOpKind_Rem,
     SBinOpKind_Add, SBinOpKind_Sub,
     SBinOpKind_And, SBinOpKind_Xor,
     SBinOpKind_ShiftL, SBinOpKind_ShiftR,
     SBinOpKind_Or, SBinOpKind_Greater]> {
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


def CmpOpKind_LT : I32EnumAttrCase<"lt", 1>;
def CmpOpKind_LE : I32EnumAttrCase<"le", 2>;
def CmpOpKind_GT : I32EnumAttrCase<"gt", 3>;
def CmpOpKind_GE : I32EnumAttrCase<"ge", 4>;
def CmpOpKind_EQ : I32EnumAttrCase<"eq", 5>;
def CmpOpKind_NE : I32EnumAttrCase<"ne", 6>;

def CmpOpKind : I32EnumAttr<
    "CmpOpKind",
    "compare operation kind",
    [CmpOpKind_LT, CmpOpKind_LE, CmpOpKind_GT,
     CmpOpKind_GE, CmpOpKind_EQ, CmpOpKind_NE]> {
  let cppNamespace = "::mlir::sys";
}

def CmpOp : SysOP<"cmp", [Pure, SameTypeOperands]> {

  let summary = "Compare values two sys values and produce a boolean result";
  let description = [{
    `sys.cmp` compares two input operands of same sys type and produces a
    `cir.bool` result. The kinds of comparison available are:
    [lt,gt,ge,eq,ne]

    Note that, cir.bool is reused in Sys_Dialect.

    ```mlir
    %7 = sys.cmp(gt, %1, %2) : i32, !cir.bool
    ```
  }];

  // TODO: get more accurate than CIR_AnyType
  let results = (outs CIR_AnyType:$result);
  let arguments = (ins Arg<CmpOpKind, "cmp kind">:$kind,
                       Sys_AnyType:$lhs, Sys_AnyType:$rhs);

  let assemblyFormat = [{
    `(` $kind `,` $lhs `,` $rhs  `)` `:` type($lhs) `,` type($result) attr-dict
  }];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// Shift Operations
//===----------------------------------------------------------------------===//

 def ShiftOp : SysOP<"shift", [Pure]> {
  let summary = "Shift";
  let description = [{
    Shift `left` or `right`, according to the first operand. Second operand is
    the shift target and the third the amount.

    ```mlir
    %7 = sys.shift(left, %1, %4)
    ```
  }];

  let results = (outs Sys_AnyType:$result);
  let arguments = (ins Sys_AnyType:$value, CIR_IntType:$amount,
                       UnitAttr:$isShiftleft);

  let assemblyFormat = [{
    `(`
      (`left` $isShiftleft^) : (`right`)?
      `,` $value `:` type($value)
      `,` $amount `:` type($amount)
    `)` `->` type($result) attr-dict
  }];

  // Already covered by the traits
  let hasVerifier = 0;
}


//===----------------------------------------------------------------------===//
// Shift Operations
//===----------------------------------------------------------------------===//

 def RangeOp : SysOP<"range", [Pure]> {
  let summary = "Range";
  let description = [{
    extract a range from a bit vector.

    ```mlir
    %7 = sys.range(%1, 3, 2);
    ```
  }];

  let results = (outs Sys_BVType:$result);
  let arguments = (ins Sys_BVType:$value, IntAttr:$high, IntAttr:$low);

  let assemblyFormat = [{
    `(`
      $value `:` type($value)
      attr-dict
    `)` `->` type($result)
  }];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// ConditionOp
// It follows the idea of SCF, but we use the type.
// in Dialect Sys. It can be used to represent for-loop, while-loop, and do-while-loop.
//===----------------------------------------------------------------------===//

def LoopOp : SysOP<"loop",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands"]>,
     DeclareOpInterfaceMethods<LoopLikeOpInterface,
        ["getRegionIterArgs", "getYieldedValuesMutable"]>,
     RecursiveMemoryEffects, SingleBlock]> {
  let summary = "a generic loop";
  let description = [{
    This operation represents a generic loop that keeps
    iterating as long as a condition is satisfied. There is no restriction on
    the complexity of the condition. It consists of two regions (with single
    block each): "before" region and "after" region.

    The "before" region terminates with a special operation, `sys.condition`,
    that accepts as its first operand an `CIR_BoolType` value indicating whether to
    proceed to the "after" region (value is `true`) or not. The two regions
    communicate by means of region arguments. Initially, the "before" region
    accepts as arguments the operands of the `sys.loop` operation and uses them
    to evaluate the condition. It forwards the trailing, non-condition operands
    of the `sys.condition` terminator either to the "after" region if the
    control flow is transferred there or to results of the `sys.loop` operation
    otherwise. The "after" region takes as arguments the values produced by the
    "before" region and uses `sys.yield` to supply new arguments for the
    "before" region, into which it transfers the control flow unconditionally.

    A simple "for" loop:
    ```C++
    for (int i = 0 ; i < 10 ; i ++ ) {
      y = x * i;
    }
    ```
    can be represented as follows.

    ```mlir
    %i = sys.const (0)
    %res = scf.loop (%arg1 = %i) : (i32) -> i32 {
      // "Before" region.
      %condition = cir.binop(<, %arg1, 10)
      sys.condition(%condition) %arg1 : i32
    } do {
      ^bb0(%arg2: i32) :
      %y = cir.binop(*, %x, %arg2)
      %i = cir.binop(+, %arg2, 1)
      sys.yield %i : i32
    }
    ```

    A simple "while" loop can be represented as follows.
    ```mlir
      %res = sys.loop (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "while" loop, this region computes the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Forward the argument (as result or "after" region argument).
      sys.condition(%condition) %arg1 : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // In a "while" loop, this region is the loop body.
      %next = call @payload(%arg2) : (f32) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `sys.while` operands.
      sys.yield %next : f32
    }
    ```

    A simple "do-while" loop can be represented by reducing the "after" block
    to a simple forwarder.

    ```mlir
    %res = sys.loop (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "do-while" loop, this region contains the loop body.
      %next = call @payload(%arg1) : (f32) -> f32

      // And also evaluates the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Loop through the "after" region.
      sys.condition(%condition) %next : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // Forwards the values back to "before" region unmodified.
      sys.yield %arg2 : f32
    }
  ```
  Note that the types of region arguments need not to match with each other.
    The op expects the operand types to match with argument types of the
    "before" region; the result types to match with the trailing operand types
    of the terminator of the "before" region, and with the argument types of the
    "after" region. The following scheme can be used to share the results of
    some operations executed in the "before" region with the "after" region,
    avoiding the need to recompute them.

    ```mlir
    %res = sys.loop (%arg1 = %init1) : (f32) -> i64 {
      // One can perform some computations, e.g., necessary to evaluate the
      // condition, in the "before" region and forward their results to the
      // "after" region.
      %shared = call @shared_compute(%arg1) : (f32) -> i64

      // Evaluate the condition.
      %condition = call @evaluate_condition(%arg1, %shared) : (f32, i64) -> i1

      // Forward the result of the shared computation to the "after" region.
      // The types must match the arguments of the "after" region as well as
      // those of the `sys.while` results.
      sys.condition(%condition) %shared : i64

    } do {
    ^bb0(%arg2: i64) {
      // Use the partial result to compute the rest of the payload in the
      // "after" region.
      %res = call @payload(%arg2) : (i64) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `sys.loop` operands.
      sys.yield %res : f32
    }
    }
    ```

  }];
  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$operands,
      "function_ref<void(OpBuilder &, Location, ValueRange)>":$beforeBuilder,
      "function_ref<void(OpBuilder &, Location, ValueRange)>":$afterBuilder)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, ValueRange)>;

    ConditionOp getConditionOp();
    YieldOp getYieldOp();

    Block::BlockArgListType getBeforeArguments();
    Block::BlockArgListType getAfterArguments();
    Block *getBeforeBody() { return &getBefore().front(); }
    Block *getAfterBody() { return &getAfter().front(); }
  }];

  let hasCanonicalizer = 1;
//  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

def ConditionOp : SysOP<"condition", [
  HasParent<"LoopOp">,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
    ["getSuccessorRegions"]>,
  Pure,
  Terminator
]> {
  let summary = "loop continuation condition";
  let description = [{
    This operation accepts the continuation (i.e., inverse of exit) condition
    of the `sys.loop` construct. If its first argument is true, the "after"
    region of `sys.loop` is executed, with the remaining arguments forwarded
    to the entry block of the region. Otherwise, the loop terminates.
  }];

  let arguments = (ins CIR_BoolType:$condition, Variadic<AnyType>:$args);

  let assemblyFormat =
      [{ `(` $condition `)` attr-dict ($args^ `:` type($args))? }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : SysOP<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["LoopOp"]>]> {
  let summary = "loop yield and termination operation";
  let description = [{
    The `sys.yield` operation yields an SSA value from the SCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If `sys.yield` has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the `scf.yield` may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  //let assemblyFormat =
  //    [{  attr-dict ($results^ `:` type($results))? }];
}


//===--------------------------------------------------===//
// Wait operation
//===--------------------------------------------------===//
def EventAnd: I32EnumAttrCase<"And", 1, "and">;
def EventOr: I32EnumAttrCase<"Or", 2, "or">;

def EventCombKind : I32EnumAttr<
  "SEventCombKind",
  "Combination kind for events",
  [EventAnd, EventOr]> {
 let cppNamespace = "::mlir::sys";
}
def WaitOp : SysOP<"wait", [Pure, ParentOneOf<["ProcDefOP"]>]> {
  let summary = "Wait operation for events";
  let description = [{
    The `sys.wait` is an wait operation for some events to happen.
  }];

  let arguments = (ins Variadic<Sys_EventAndTime>:$args,
                       Arg<EventCombKind, "event comb kind">:$kind);

  let assemblyFormat =
      [{ `(` $args `:` type($args) `:` $kind `)`  attr-dict}];
}

def NotifyOp : SysOP<"notify", [Pure, ParentOneOf<["ProcDefOP"]>]> {
  let summary = "Notify operation for events";
  let description = [{
    The `sys.notify` is a notify operation for some events to happen.
  }];

  let arguments = (ins Sys_EventAndTime:$arg);

  let assemblyFormat =
      [{ `(` $arg `:` type($arg) `)`  attr-dict}];
}

#endif
