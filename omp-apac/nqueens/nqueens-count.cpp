#include <alloca.h>
#include <memory.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "app-desc.hpp"
#include "bots.h"

const static int __apac_count_infinite = getenv("APAC_TASK_COUNT_INFINITE") ? 1 : 0;

const static int __apac_count_max = getenv("APAC_TASK_COUNT_MAX") ? atoi(getenv("APAC_TASK_COUNT_MAX")) : omp_get_max_threads() * 10;

int __apac_count = 0;

int solutions[] = {1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596};

int total_count;

int ok(int n, char* a) {
  int i;
  int j;
  char p;
  char q;
  for (i = 0; i < n; i++) {
    p = a[i];
    for (j = i + 1; j < n; j++) {
      q = a[j];
      if (q == p || q == p - (j - i) || q == p + (j - i)) return 0;
    }
  }
  return 1;
}

void nqueens_ser(int n, int j, char* a, int* solutions) {
  int res;
  int i;
  if (n == j) {
    *solutions = 1;
    return;
  }
  *solutions = 0;
  for (i = 0; i < n; i++) {
    a[j] = (char)i;
    if (ok(j + 1, a)) {
      nqueens_ser(n, j + 1, a, &res);
      *solutions += res;
    }
  }
}

void nqueens(int n, int j, char* a, int* solutions) {
  int __apac_count_ok = __apac_count_infinite || __apac_count < __apac_count_max;
#pragma omp taskgroup
  {
    int* csols;
    int i;
    if (n == j) {
      *solutions = 1;
      goto __apac_exit;
    }
    *solutions = 0;
    csols = (int*)__builtin_alloca(n * sizeof(int));
    memset(csols, 0, n * sizeof(int));
    for (i = 0; i < n; i++) {
      char* b;
      b = (char*)malloc(n * sizeof(char));
      memcpy(b, a, j * sizeof(char));
      b[j] = (char)i;
#pragma omp taskwait depend(in : b, j) depend(inout : b[0])
      if (ok(j + 1, b)) {
        if (__apac_count_ok) {
#pragma omp atomic
          __apac_count++;
        }
#pragma omp task default(shared) depend(in : b, csols, j, n) depend(inout : b[0], csols[i]) firstprivate(i) if (__apac_count_ok)
        {
          nqueens(n, j + 1, b, &csols[i]);
          if (__apac_count_ok) {
#pragma omp atomic
            __apac_count--;
          }
        }
      }
#pragma omp taskwait depend(in : b) depend(inout : b[0])
      free(b);
    }
#pragma omp taskwait
    for (i = 0; i < n; i++) {
      *solutions += csols[i];
    }
  __apac_exit:;
  }
}

void find_queens(int size) {
  int __apac_count_ok = __apac_count_infinite || __apac_count < __apac_count_max;
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
  {
#pragma omp critical
    total_count = 0;
    bots_message("Computing N-Queens algorithm (n=%d) ", size);
    char* a;
    a = (char*)__builtin_alloca(size * sizeof(char));
    if (__apac_count_ok) {
#pragma omp atomic
      __apac_count++;
    }
#pragma omp task default(shared) depend(in : a, size) depend(inout : a[0], total_count) if (__apac_count_ok)
    {
#pragma omp critical
      nqueens(size, 0, a, &total_count);
      if (__apac_count_ok) {
#pragma omp atomic
        __apac_count--;
      }
    }
    bots_message(" completed!\n");
  __apac_exit:;
  }
}

int verify_queens(int size) {
  if (size > sizeof(solutions) / sizeof(int)) return 0;
#pragma omp critical
  if (total_count == solutions[size - 1]) return 1;
  return 2;
}
