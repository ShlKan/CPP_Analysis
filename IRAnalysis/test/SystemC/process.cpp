#include <systemc>

SC_MODULE(Process) {

  SC_CTOR(Process) {
    SC_THREAD(emptyProcess);
    SC_THREAD(emptyProcess1);
  }

  sc_dt::sc_int<33> y = 32;
  sc_dt::sc_int<33> z = 32;
  sc_dt::sc_int<33> t = y + z + z + y;
  // bool z;
  void emptyProcess() {
    int x = 0;
    sc_dt::sc_int<8> i = 2;
    sc_dt::sc_int<8> j = 3;
    if (i > j) {
      sc_dt::sc_int<8> x = 0;
      sc_dt::sc_int<8> y = 0;
      sc_dt::sc_int<8> z = x + y;
    } else {
      sc_dt::sc_int<8> x = 0;
    }
  }

  void emptyProcess1(void) {
    sc_dt::sc_int<8> i = 2;
    sc_dt::sc_int<8> j = 3;
    if (i > j) {
      sc_dt::sc_int<8> x = 0;
      sc_dt::sc_int<8> y = 0;
      sc_dt::sc_int<8> z = x + y;
    }
  }
};
