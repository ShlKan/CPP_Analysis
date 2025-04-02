// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

using namespace sc_dt;

SC_MODULE(Process) {
  SC_CTOR(Process) { SC_THREAD(emptyProcess); }
  sc_bv<8> b1 = "01111011";
  // CHECK: %0 = "sys.const"() <{value = #sys<bv[0;1;1;1;1;0;1;1]> :
  // CHECK: !sys.s_bv<8>}> {sym_name = "b1"} : () -> !sys.s_bv<8>
  void emptyProcess() {}
};