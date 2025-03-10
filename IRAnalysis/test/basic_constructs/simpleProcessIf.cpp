// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

// CHECK1: !s32i = !cir.int<s, 32>
// CHECK1: !u32i = !cir.int<u, 32>
SC_MODULE(Process){// CHECK: "builtin.module"() <{sym_name = "Process"}> ({
                   SC_CTOR(Process){SC_THREAD(emptyProcess);
// CHECK: %0 = "sys.ProcDef"() <{proc_name = "emptyProcess", proc_type =
// !sys<s_proc Proc ()>}> ({
}
void emptyProcess() {
  sc_dt::sc_int<8> i = 2;
  // CHECK:  %1 = "sys.const"() <{value = #sys.int<2> : !sys.s_int<s, 8>}> : ()
  // -> !sys.s_int<s, 8>
  sc_dt::sc_int<8> j = 3;
  // CHECK: %2 = "sys.const"() <{value = #sys.int<3> : !sys.s_int<s, 8>}> : ()
  // -> !sys.s_int<s, 8>
  if (i > j) {
    sc_dt::sc_int<8> x = 0;
  }
  // CHECK: %3 = "sys.cmp"(%1, %2) <{kind = 3 : i32}> : (!sys.s_int<s, 8>,
  // !sys.s_int<s, 8>) -> !cir.bool
  // CHECK: "cir.brcond"(%3)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1,
  // 0, 0>}> : (!cir.bool) -> ()
  // CHECK: ^bb1: // pred: ^bb0
  // CHECK: %4 = "sys.const"() <{value = #sys.int<0> : !sys.s_int<s, 8>}> : ()
  // -> !sys.s_int<s, 8>
  // CHECK: }) : () -> !sys<s_proc Proc ()>
  // CHECK: "sys.ProcRegister"(%0) : (!sys<s_proc Proc ()>) -> ()
  // CHECK: }) : () -> ()
}
}
;