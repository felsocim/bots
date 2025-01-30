#include <atomic>
#include "atomic.hpp"

int atomic_compare(int * v1, int * v2) {
  return atomic_compare_exchange_strong((std::atomic_int *)v1, v2, 1);
}

void atomic_add(int * dest, int val) {
  atomic_fetch_add((std::atomic_int *)dest, val);
}
