#include <limits.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app-desc.hpp"
#include "bots.h"
#include "compare.hpp"
const static int __apac_count_infinite = getenv("APAC_TASK_COUNT_INFINITE") ? 1 : 0;

const static int __apac_depth_infinite = getenv("APAC_TASK_DEPTH_INFINITE") ? 1 : 0;

const static int __apac_count_max = getenv("APAC_TASK_COUNT_MAX") ? atoi(getenv("APAC_TASK_COUNT_MAX")) : omp_get_max_threads() * 10;

const static int __apac_depth_max = getenv("APAC_TASK_DEPTH_MAX") ? atoi(getenv("APAC_TASK_DEPTH_MAX")) : 5;

int __apac_count = 0;

int __apac_depth = 0;

#pragma omp threadprivate(__apac_depth)

int best_so_far;

int number_of_tasks;

int read_input(const char* filename, item_t* items, int* capacity, int* n) {
  int i;
  FILE* f;
  if (filename == NULL) filename = "";
  f = fopen(filename, "r");
  if (f == NULL) {
    fprintf(stderr, "open_input('%s') failed\n", filename);
    return -1;
  }
  fscanf(f, "%d", n);
  fscanf(f, "%d", capacity);
  for (i = 0; i < *n; ++i) fscanf(f, "%d %d", &items[i].value, &items[i].weight);
  fclose(f);
  qsort(items, *n, sizeof(item_t), &compare);
  return 0;
}

void knapsack_par(item_t* e, int c, int n, int v, int* sol, int l) {
#pragma omp taskgroup
  {
    int __apac_count_ok = __apac_count_infinite || __apac_count < __apac_count_max;
    int __apac_depth_local = __apac_depth;
    int __apac_depth_ok = __apac_depth_infinite || __apac_depth_local < __apac_depth_max;
    int with;
    int without;
    int best;
    double ub;
#pragma omp critical
    number_of_tasks++;
    if (c < 0) {
      *sol = -2147483647 - 1;
      goto __apac_exit;
    }
    if (n == 0 || c == 0) {
      *sol = v;
      goto __apac_exit;
    }
    ub = (double)v + c * e->value / e->weight;
#pragma omp critical
    if (ub < best_so_far) {
      *sol = -2147483647 - 1;
      goto __apac_exit;
    }
    if (__apac_count_ok) {
#pragma omp atomic
      __apac_count++;
    }
#pragma omp task default(shared) depend(in : c, e, e[0], l, n, v) depend(inout : without) firstprivate(__apac_depth_local) if (__apac_count_ok || __apac_depth_ok)
    {
      if (__apac_count_ok || __apac_depth_ok) {
        __apac_depth = __apac_depth_local + 1;
      }
      knapsack_par(e + 1, c, n - 1, v, &without, l + 1);
      if (__apac_count_ok) {
#pragma omp atomic
        __apac_count--;
      }
    }
    if (__apac_count_ok) {
#pragma omp atomic
      __apac_count++;
    }
#pragma omp task default(shared) depend(in : c, e, e[0], l, n, v) depend(inout : with) firstprivate(__apac_depth_local) if (__apac_count_ok || __apac_depth_ok)
    {
      if (__apac_count_ok || __apac_depth_ok) {
        __apac_depth = __apac_depth_local + 1;
      }
      knapsack_par(e + 1, c - e->weight, n - 1, v + e->value, &with, l + 1);
      if (__apac_count_ok) {
#pragma omp atomic
        __apac_count--;
      }
    }
#pragma omp taskwait
    best = (with > without ? with : without);
#pragma omp critical
    if (best > best_so_far) {
      best_so_far = best;
    }
    *sol = best;
  __apac_exit:;
  }
}

void knapsack_seq(item_t* e, int c, int n, int v, int* sol) {
  int with;
  int without;
  int best;
  double ub;
#pragma omp critical
  number_of_tasks++;
  if (c < 0) {
    *sol = -2147483647 - 1;
    return;
  }
  if (n == 0 || c == 0) {
    *sol = v;
    return;
  }
  ub = (double)v + c * e->value / e->weight;
#pragma omp critical
  if (ub < best_so_far) {
    *sol = -2147483647 - 1;
    return;
  }
  knapsack_seq(e + 1, c, n - 1, v, &without);
  knapsack_seq(e + 1, c - e->weight, n - 1, v + e->value, &with);
  best = (with > without ? with : without);
#pragma omp critical
  if (best > best_so_far) best_so_far = best;
  *sol = best;
}

void knapsack_main_par(item_t* e, int c, int n, int* sol) {
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
  {
    int __apac_count_ok = __apac_count_infinite || __apac_count < __apac_count_max;
    int __apac_depth_local = __apac_depth;
    int __apac_depth_ok = __apac_depth_infinite || __apac_depth_local < __apac_depth_max;
#pragma omp critical
    {
      best_so_far = -2147483647 - 1;
      number_of_tasks = 0;
      bots_number_of_tasks += number_of_tasks;
    }
    if (__apac_count_ok) {
#pragma omp atomic
      __apac_count++;
    }
#pragma omp task default(shared) depend(in : c, e, e[0], n, sol) depend(inout : sol[0]) firstprivate(__apac_depth_local) if (__apac_count_ok || __apac_depth_ok)
    {
      if (__apac_count_ok || __apac_depth_ok) {
        __apac_depth = __apac_depth_local + 1;
      }
      knapsack_par(e, c, n, 0, sol, 0);
      if (__apac_count_ok) {
#pragma omp atomic
        __apac_count--;
      }
    }
#pragma omp taskwait
    if (bots_verbose_mode) {
      printf("Best value for parallel execution is %d\n\n", *sol);
    }
  __apac_exit:;
  }
}

void knapsack_main_seq(item_t* e, int c, int n, int* sol) {
#pragma omp critical
  {
    best_so_far = -2147483647 - 1;
    number_of_tasks = 0;
  }
  knapsack_seq(e, c, n, 0, sol);
  if (bots_verbose_mode) printf("Best value for sequential execution is %d\n\n", *sol);
}

int knapsack_check(int sol_seq, int sol_par) {
  if (sol_seq == sol_par)
    return 1;
  else
    return 2;
}
