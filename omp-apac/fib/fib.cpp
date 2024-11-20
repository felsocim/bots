#include "fib.hpp"

#include "bots.h"

long long int fib_results[41] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155};

long long int fib_seq(int n) {
  long long int x, y;
  if (n < 2) return n;
  x = fib_seq(n - 1);
  y = fib_seq(n - 2);
  return x + y;
}

long long int fib(int n) {
  long long int __apac_result;
#pragma omp taskgroup
  {
    long long int x, y;
    if (n < 2) {
      __apac_result = n;
      goto __apac_exit;
    }
#pragma omp task default(shared) depend(in : n) depend(inout : x)
    x = fib(n - 1);
#pragma omp task default(shared) depend(in : n) depend(inout : y)
    y = fib(n - 2);
#pragma omp taskwait
    __apac_result = x + y;
    goto __apac_exit;
  __apac_exit:;
  }
  return __apac_result;
}

long long int par_res;

long long int seq_res;

void fib0(int n) {
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
  {
#pragma omp task default(shared) depend(inout : n, par_res)
    {
#pragma omp critical
      {
        par_res = fib(n);
        bots_message("Fibonacci result for %d is %lld\n", n, par_res);
      }
    }
  __apac_exit:;
  }
}

void fib0_seq(int n) {
  seq_res = fib_seq(n);
  bots_message("Fibonacci result for %d is %lld\n", n, seq_res);
}

long long int fib_verify_value(int n) {
  if (n < 41) return fib_results[n];
  return fib_verify_value(n - 1) + fib_verify_value(n - 2);
}

int fib_verify(int n) {
  int result;
  if (bots_sequential_flag) {
#pragma omp critical
    if (par_res == seq_res)
      result = 1;
    else
      result = 2;
  } else {
    seq_res = fib_verify_value(n);
#pragma omp critical
    if (par_res == seq_res)
      result = 1;
    else
      result = 2;
  }
  return result;
}
