#include <omp.h>
#include <stdlib.h>

#include "bots.h"
#include "fib.hpp"
const static int __apac_count_infinite = getenv("APAC_TASK_COUNT_INFINITE") ? 1 : 0;

const static int __apac_count_max = getenv("APAC_TASK_COUNT_MAX") ? atoi(getenv("APAC_TASK_COUNT_MAX")) : omp_get_max_threads() * 10;

int __apac_count = 0;

const static int __apac_depth_infinite = getenv("APAC_TASK_DEPTH_INFINITE") ? 1 : 0;

const static int __apac_depth_max = getenv("APAC_TASK_DEPTH_MAX") ? atoi(getenv("APAC_TASK_DEPTH_MAX")) : 5;

int __apac_depth = 0;

#pragma omp threadprivate(__apac_depth)

long long int fib_results[41] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155};

long long int fib_seq(int n) {
  long long int x;
  long long int y;
  if (n < 2) return n;
  x = fib_seq(n - 1);
  y = fib_seq(n - 2);
  return x + y;
}

long long int fib(int n) {
  int __apac_count_ok = __apac_count_infinite || __apac_count < __apac_count_max;
  int __apac_depth_local = __apac_depth;
  int __apac_depth_ok = __apac_depth_infinite || __apac_depth_local < __apac_depth_max;
  if (__apac_depth_ok) {
    long long int __apac_result;
#pragma omp taskgroup
    {
      long long int x;
      long long int y;
      if (n < 2) {
        __apac_result = n;
        goto __apac_exit;
      }
      if (__apac_count_ok) {
#pragma omp atomic
        __apac_count++;
      }
#pragma omp task default(shared) depend(in : n) depend(inout : x) firstprivate(__apac_depth_local) if (__apac_count_ok || __apac_depth_ok)
      {
        if (__apac_count_ok || __apac_depth_ok) {
          __apac_depth = __apac_depth_local + 1;
        }
        x = fib(n - 1);
        if (__apac_count_ok) {
#pragma omp atomic
          __apac_count--;
        }
      }
      if (__apac_count_ok) {
#pragma omp atomic
        __apac_count++;
      }
#pragma omp task default(shared) depend(in : n) depend(inout : y) firstprivate(__apac_depth_local) if (__apac_count_ok || __apac_depth_ok)
      {
        if (__apac_count_ok || __apac_depth_ok) {
          __apac_depth = __apac_depth_local + 1;
        }
        y = fib(n - 2);
        if (__apac_count_ok) {
#pragma omp atomic
          __apac_count--;
        }
      }
#pragma omp taskwait
      __apac_result = x + y;
      goto __apac_exit;
    __apac_exit:;
    }
    return __apac_result;
  } else {
    return fib_seq(n);
  }
}

long long int par_res;

long long int seq_res;

void fib0(int n) {
  int __apac_count_ok = __apac_count_infinite || __apac_count < __apac_count_max;
  int __apac_depth_local = __apac_depth;
  int __apac_depth_ok = __apac_depth_infinite || __apac_depth_local < __apac_depth_max;
  if (__apac_depth_ok) {
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
    {
      if (__apac_count_ok) {
#pragma omp atomic
        __apac_count++;
      }
#pragma omp task default(shared) depend(in : n) depend(inout : par_res) firstprivate(__apac_depth_local) if (__apac_count_ok || __apac_depth_ok)
      {
        if (__apac_count_ok || __apac_depth_ok) {
          __apac_depth = __apac_depth_local + 1;
        }
#pragma omp critical
        {
          par_res = fib(n);
          bots_message("Fibonacci result for %d is %lld\n", n, par_res);
        }
        if (__apac_count_ok) {
#pragma omp atomic
          __apac_count--;
        }
      }
    __apac_exit:;
    }
  } else {
    fib0_seq(n);
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
