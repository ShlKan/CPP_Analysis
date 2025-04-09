// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

using namespace sc_dt;

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(emptyProcess);
}
void emptyProcess() {
  sc_bv<8> b1 = "01111011";
  sc_bv<8> b2 = b1 >> 2;
  // CHECK: %3 = "sys.sbinop"(%1, %2) <{kind = 10 : i32}> : (!sys.s_bv<8>,
  // CHECK: !u32i) -> !sys.s_bv<8>
}
}
;