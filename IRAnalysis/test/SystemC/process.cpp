#include <systemc>

SC_MODULE(Process) {

  SC_CTOR(Process) {
    SC_THREAD(emptyProcess);
    SC_THREAD(emptyProcess1);
  }

  int x = 9;
  void emptyProcess() {
    // Do nothing.
  }

  void emptyProcess1(void) {
    // Do nothing.
  }
};
