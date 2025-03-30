// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(emptyProcess);
}
void emptyProcess() {
  int x = 0;
  // CHECK:    %1 = "cir.const"() <{value = #cir.int<0> : !u32i}> : () -> !u32i
  int y = 0;
  // CHECK:    %2 = "cir.const"() <{value = #cir.int<0> : !u32i}> : () -> !u32i
  y = x + x;
}
// CHECK: %3 = "cir.binop"(%1, %1) <{kind = 4 : i32}> : (!u32i, !u32i) -> !u32i
}
;
// CHECK: }) : () -> ()