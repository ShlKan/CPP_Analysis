#include <systemc>

SC_MODULE(Process) {

  SC_CTOR(Process) {
    SC_THREAD(emptyProcess);
    SC_THREAD(emptyProcess1);
  }

  int x = 9;
  sc_dt::sc_int<33> y = 32;
  // bool z;
  void emptyProcess() {
    // Do nothing.
  }

  void emptyProcess1(void) {
    // Do nothing.
  }
};
