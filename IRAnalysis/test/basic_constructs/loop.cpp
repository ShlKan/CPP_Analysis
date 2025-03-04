// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(emptyProcess);
}
// CHECK: !s32i = !cir.int<s, 32>
// CHECK: !u32i = !cir.int<u, 32>
// CHECK:  "builtin.module"() <{sym_name = "Process"}> ({
void emptyProcess() {
  for (int i = 0; i < 10; i = i + 1) {
    sc_dt::sc_int<8> k = 7;
  }
  // CHECK: %2 = "sys.loop"(%1) ({
  // CHECK:  ^bb0(%arg1: !u32i):
  // CHECK:   %6 = "cir.const"() <{value = #cir.int<10> : !u32i}> : () -> !u32i
  // CHECK    %7 = "cir.cmp"(%arg1, %6) <{kind = 1 : i32}> : (!u32i, !u32i) ->
  // !cir.bool
  // CHECK:   "sys.condition"(%7, %arg1) : (!cir.bool, !u32i) -> ()
  // CHECK: ^bb0(%arg0: !s32i):
  // CHECK: %3 = "sys.const"() <{value = #sys.int<7> : !sys.s_int<s, 8>}> : ()
  // -> !sys.s_int<s, 8>
  // CHECK: %4 = "cir.const"() <{value = #cir.int<1> : !u32i}> : () -> !u32i
  // CHECK: %5 = "cir.binop"(%1, %4) <{kind = 4 : i32}> : (!u32i, !u32i) ->
  // !u32i
}
}
;
// CHECK: }) : () -> ()