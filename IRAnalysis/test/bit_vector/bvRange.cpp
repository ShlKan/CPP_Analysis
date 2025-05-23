// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

using namespace sc_dt;

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(emptyProcess);
}
void emptyProcess() {
  sc_bv<8> b1 = "01111011";
  sc_bv<3> b2 = b1.range(2, 0);
  // CHECK: %2 = "sys.range"(%1) <{high = #sys.int<2> : !sys.s_int<u, 32>, low =
  // CHECK: #sys.int<0> : !sys.s_int<u, 32>}> : (!sys.s_bv<8>) -> !sys.s_bv<3>
}
}
;