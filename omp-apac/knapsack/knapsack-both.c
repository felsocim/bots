#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app-desc.h"
#include "bots.h"

int best_so_far;

int number_of_tasks;

int compare(void* p1, void* p2) {
  item_t* a = (item_t*)p1;
  item_t* b = (item_t*)p2;
  double c = (double)a->value / a->weight - (double)b->value / b->weight;
  if (c > 0) return -1;
  if (c < 0) return 1;
  return 0;
}

int read_input(const char* filename, item_t* items, int* capacity, int* n) {
  int i;
  FILE* f;
  if (filename == (void*)0) filename = "";
  f = fopen(filename, "r");
  if (f == (void*)0) {
    fprintf(stderr, "open_input('%s') failed\n", filename);
    return -1;
  }
  fscanf(f, "%d", n);
  fscanf(f, "%d", capacity);
  for (i = 0; i < *n; ++i)
    fscanf(f, "%d %d", &items[i].value, &items[i].weight);
  fclose(f);
  qsort(items, *n, sizeof(item_t), compare);
  return 0;
}

void knapsack_par(item_t* e, int c, int n, int v, int* sol, int l) {
#pragma omp taskgroup
  {
    int with, without, best;
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
#pragma omp task default(shared) depend(in                     \
                                        : c, e, e[0], l, n, v) \
    depend(inout                                               \
           : without)
    knapsack_par(e + 1, c, n - 1, v, &without, l + 1);
#pragma omp task default(shared) depend(in                                  \
                                        : c, e, e[0], l, n, v) depend(inout \
                                                                      : with)
    knapsack_par(e + 1, c - e->weight, n - 1, v + e->value, &with, l + 1);
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
  int with, without, best;
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
#pragma omp critical
    {
      best_so_far = -2147483647 - 1;
      number_of_tasks = 0;
      bots_number_of_tasks += number_of_tasks;
    }
#pragma omp task default(shared) depend(in                                 \
                                        : c, e, e[0], n, sol) depend(inout \
                                                                     : sol[0])
    knapsack_par(e, c, n, 0, sol, 0);
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
  if (bots_verbose_mode)
    printf("Best value for sequential execution is %d\n\n", *sol);
}

int knapsack_check(int sol_seq, int sol_par) {
  if (sol_seq == sol_par)
    return 1;
  else
    return 2;
}
