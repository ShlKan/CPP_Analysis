// RUN: %cpp_analysis %args %s | FileCheck %s

#include <sys/wait.h>
#include <systemc>

using namespace sc_dt;
using namespace sc_core;

SC_MODULE(Process) {
  SC_CTOR(Process) {
    SC_THREAD(P1);
    SC_THREAD(P2);
  }
  sc_event e1;
  void P1() { wait(e1); }
  void P2() { e1.notify(); }
};

// CHECK: %0 = "sys.const"() <{value = #sys<event[e1]> : !sys<s_event event>}>
// CHECK: {sym_name = "e1"} : () -> !sys<s_event event>
// CHECK: "sys.wait"(%0) <{kind = 1 : i32}> : (!sys<s_event event>) -> ()
// CHECK: "sys.notify"(%0) : (!sys<s_event event>) -> ()