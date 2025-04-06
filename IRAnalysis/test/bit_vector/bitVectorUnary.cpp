// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

using namespace sc_dt;

SC_MODULE(Process){SC_CTOR(Process){SC_THREAD(emptyProcess);
}
void emptyProcess() {
  sc_bv<8> b1 = "01111011";
  sc_bv<1> b2 = b1.and_reduce();
  // CHECK: %2 = "sys.suaryop"(%1) <{kind = 31 : i32}> : (!sys.s_bv<8>) ->
  // CHECK: !sys.s_bv<1>
  sc_bv<1> b3 = b1.or_reduce();
  // CHECK: %3 = "sys.suaryop"(%1) <{kind = 32 : i32}> : (!sys.s_bv<8>) ->
  // CHECK: !sys.s_bv<1>
  sc_bv<1> b4 = b1.nand_reduce();
  // CHECK: %4 = "sys.suaryop"(%1) <{kind = 33 : i32}> : (!sys.s_bv<8>) ->
  // CHECK: !sys.s_bv<1>
  sc_bv<1> b5 = b1.nor_reduce();
  // CHECK: %5 = "sys.suaryop"(%1) <{kind = 34 : i32}> : (!sys.s_bv<8>) ->
  // CHECK: !sys.s_bv<1>
  sc_bv<1> b6 = b1.xor_reduce();
  // CHECK: %6 = "sys.suaryop"(%1) <{kind = 35 : i32}> : (!sys.s_bv<8>) ->
  // CHECK: !sys.s_bv<1>
  sc_bv<1> b7 = b1.xnor_reduce();
  // CHECK: %7 = "sys.suaryop"(%1) <{kind = 36 : i32}> : (!sys.s_bv<8>) ->
  // CHECK: !sys.s_bv<1>

  sc_lv<8> c1 = "0111Z01x";
  sc_lv<1> c2 = c1.and_reduce();
  // CHECK: %9 = "sys.suaryop"(%8) <{kind = 31 : i32}> : (!sys.s_bvl<8>) ->
  // CHECK: !sys.s_bvl<1>
  sc_lv<1> c3 = c1.xnor_reduce();
  // CHECK: %10 = "sys.suaryop"(%8) <{kind = 36 : i32}> : (!sys.s_bvl<8>) ->
  // CHECK: !sys.s_bvl<1>
}
}
;