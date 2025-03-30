// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(emptyProcess);
}
void emptyProcess() {
  sc_dt::sc_int<8> i = 2;
  sc_dt::sc_int<8> j = 3;
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
  // CHECK: ^bb2:  // pred: ^bb0
}
}
;