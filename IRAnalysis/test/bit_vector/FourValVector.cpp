// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

using namespace sc_dt;

SC_MODULE(Process) {
  SC_CTOR(Process) { SC_THREAD(emptyProcess); }

  sc_lv<8> b1 = "0111Z01x";
  // CHECK: %0 = "sys.const"() <{value = #sys<bvl[0;1;1;1;z;0;1;x]> :
  // CHECK: !sys.s_bvl<8>}> {sym_name = "b1"} : () -> !sys.s_bvl<8>
  sc_logic a = SC_LOGIC_0;
  // CHECK: %1 = "sys.const"() <{value = #sys<bvl[0]> : !sys.s_bvl<1>}>
  // CHECK: {sym_name = "a"} : () -> !sys.s_bvl<1>
  sc_logic b = SC_LOGIC_1;
  // CHECK: %2 = "sys.const"() <{value = #sys<bvl[1]> : !sys.s_bvl<1>}>
  // CHECK: {sym_name = "b"} : () -> !sys.s_bvl<1>
  sc_logic c = SC_LOGIC_X;
  // CHECK: %3 = "sys.const"() <{value = #sys<bvl[x]> : !sys.s_bvl<1>}>
  // CHECK: {sym_name = "c"} : () -> !sys.s_bvl<1>
  sc_logic d = SC_LOGIC_Z;
  // CHECK: %4 = "sys.const"() <{value = #sys<bvl[z]> : !sys.s_bvl<1>}>
  // CHECK: {sym_name = "d"} : () -> !sys.s_bvl<1>
  void emptyProcess() {}
};