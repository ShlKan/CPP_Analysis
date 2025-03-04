// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(emptyProcess);
}
// CHECK: !u32i = !cir.int<u, 32>
// CHECK:  "builtin.module"() <{sym_name = "Process"}> ({
void emptyProcess() {
  // CHECK:    %0 = "sys.ProcDef"() <{proc_name = "emptyProcess", proc_type =
  int x = 0;
  // CHECK:    %1 = "cir.const"() <{value = #cir.int<0> : !u32i}> : () -> !u32i
  int y = 0;
  // CHECK:    %2 = "cir.const"() <{value = #cir.int<0> : !u32i}> : () -> !u32i
  y = x + x;
}
// CHECK: %3 = "cir.binop"(%1, %1) <{kind = 4 : i32}> : (!u32i, !u32i) -> !u32i
// CHECK: }) : () -> !sys<s_proc Proc ()>
// CHECK: "sys.ProcRegister"(%0) : (!sys<s_proc Proc ()>) -> ()
}
;
// CHECK: }) : () -> ()