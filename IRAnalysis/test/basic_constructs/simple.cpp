// RUN: %cpp_analysis %args %s | FileCheck %s

#include <systemc>

SC_MODULE(Process){
    // CHECK: module @Process {
};
// CHECK: }