#include <iostream>
#include <stdio.h>
#include <chrono>
#include "common.h"

extern int forward_pass(REAL *input);
extern int forward_pass_avx(REAL *input);

int main() {
  REAL input[NR_FEAT];
  int num_reps = 1;
  for(int i = 0; i < NR_FEAT; i++) {
    input[i] = i * 2;
  }
  double time = 0.0;
  for(int i = 0; i < num_reps; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    // operation to be timed ...
    forward_pass(input);
    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
    std::cout << "Reference time: " << duration << "ns\n";
  }
  time = 0.0;

  for(int i = 0; i < num_reps; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    // operation to be timed ...
    forward_pass_avx(input);

    auto finish = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
    std::cout << "AVX time: " << duration << "ns\n";
  }
  return 0;
}
