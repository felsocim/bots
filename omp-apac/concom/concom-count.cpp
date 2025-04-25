#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "app-desc.hpp"
#include "atomic.hpp"
#include "bots.h"
const static int __apac_count_infinite = getenv("APAC_TASK_COUNT_INFINITE") ? 1 : 0;

const static int __apac_count_max = getenv("APAC_TASK_COUNT_MAX") ? atoi(getenv("APAC_TASK_COUNT_MAX")) : omp_get_max_threads() * 10;

int __apac_count = 0;

node* nodes;

int* visited;

int* components;

int linkable(int N1, int N2) {
  int i;
  if (N1 == N2) return 0;
  if (nodes[N1].n >= bots_arg_size_1) return 0;
  if (nodes[N2].n >= bots_arg_size_1) return 0;
  for (i = 0; i < nodes[N1].n; i++)
    if (nodes[N1].neighbor[i] == N2) return 0;
  return 1;
}

void initialize() {
  int i;
  int l1;
  int l2;
  int N1;
  int N2;
  double RN;
  nodes = (node*)malloc(bots_arg_size * sizeof(node));
  visited = (int*)malloc(bots_arg_size * sizeof(int));
  components = (int*)malloc(bots_arg_size * sizeof(int));
  for (i = 0; i < bots_arg_size; i++) {
    nodes[i].n = 0;
    nodes[i].neighbor = (int*)malloc(bots_arg_size_1 * sizeof(int));
  }
  for (i = 0; i < bots_arg_size_2; i++) {
    RN = rand() / (double)2147483647;
    N1 = (int)((bots_arg_size - 1) * RN);
    RN = rand() / (double)2147483647;
    N2 = (int)((bots_arg_size - 1) * RN);
    if (linkable(N1, N2)) {
      l1 = nodes[N1].n;
      l2 = nodes[N2].n;
      nodes[N1].neighbor[l1] = N2;
      nodes[N2].neighbor[l2] = N1;
      nodes[N1].n += 1;
      nodes[N2].n += 1;
    }
  }
}

void write_outputs(int n, int cc) {
  int i;
  printf("Graph %d, Number of components %d\n", n, cc);
  if (bots_verbose_mode)
    for (i = 0; i < cc; i++) printf("Component %d       Size: %d\n", i, components[i]);
}

void cc_core(int i, int cc) {
  int __apac_count_ok = __apac_count_infinite || __apac_count < __apac_count_max;
#pragma omp taskgroup
  {
    int j;
    int n;
    int expected = 0;
    if (atomic_compare(&visited[i], &expected)) {
      if (bots_verbose_mode) {
        printf("Adding node %d to component %d\n", i, cc);
      }
      atomic_add(&components[cc], 1);
      for (j = 0; j < nodes[i].n; j++) {
        if (__apac_count_ok) {
#pragma omp atomic
          __apac_count++;
        }
#pragma omp task default(shared) depend(in : cc, i, nodes) depend(inout : n) firstprivate(j) if (__apac_count_ok)
        {
          n = nodes[i].neighbor[j];
          cc_core(n, cc);
          if (__apac_count_ok) {
#pragma omp atomic
            __apac_count--;
          }
        }
      }
    }
  __apac_exit:;
  }
}

void cc_core_seq(int i, int cc) {
  int j;
  int n;
  if (visited[i] == 0) {
    if (bots_verbose_mode) printf("Adding node %d to component %d\n", i, cc);
    visited[i] = 1;
    components[cc]++;
    for (j = 0; j < nodes[i].n; j++) {
      n = nodes[i].neighbor[j];
      cc_core_seq(n, cc);
    }
  }
}

void cc_init() {
  int i;
  for (i = 0; i < bots_arg_size; i++) {
    visited[i] = 0;
    components[i] = 0;
  }
}

void cc(int* cc) {
  int __apac_count_ok = __apac_count_infinite || __apac_count < __apac_count_max;
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
  {
    int i;
    int expected = 0;
    *cc = 0;
    for (i = 0; i < bots_arg_size; i++) {
      if (atomic_compare(&visited[i], &expected)) {
        if (__apac_count_ok) {
#pragma omp atomic
          __apac_count++;
        }
#pragma omp task default(shared) depend(in : cc) depend(inout : cc[0]) firstprivate(i) if (__apac_count_ok)
        {
          cc_core(i, *cc);
          (*cc)++;
          if (__apac_count_ok) {
#pragma omp atomic
            __apac_count--;
          }
        }
      }
    }
  __apac_exit:;
  }
}

void cc_seq(int* cc) {
  int i;
  *cc = 0;
  for (i = 0; i < bots_arg_size; i++) {
    if (visited[i] == 0) {
      cc_core_seq(i, *cc);
      (*cc)++;
    }
  }
}

int cc_check(int ccs, int ccp) {
  if (bots_verbose_mode) fprintf(stdout, "Sequential = %d CC, Parallel =%d CC\n", ccs, ccp);
  if (ccs == ccp)
    return 1;
  else
    return 2;
}
