#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app-desc.hpp"
#include "bots.h"
#include "inlines.cpp"
const double __apac_cutoff = getenv("APAC_EXECUTION_TIME_CUTOFF") ? atof(getenv("APAC_EXECUTION_TIME_CUTOFF")) : 2.22100e-6;

const static int __apac_count_infinite = getenv("APAC_TASK_COUNT_INFINITE") ? 1 : 0;

const static int __apac_count_max = getenv("APAC_TASK_COUNT_MAX") ? atoi(getenv("APAC_TASK_COUNT_MAX")) : omp_get_max_threads() * 10;

int __apac_count = 0;

ELM* array;

ELM* tmp;

ELM* seqpart(ELM* low, ELM* high) {
  ELM pivot;
  ELM h;
  ELM l;
  ELM* curr_low = low;
  ELM* curr_high = high;
  pivot = choose_pivot(low, high);
  while (1) {
    while ((h = *curr_high) > pivot) curr_high--;
    while ((l = *curr_low) < pivot) curr_low++;
    if (curr_low >= curr_high) break;
    *curr_high = l;
    curr_high--;
    *curr_low = h;
    curr_low++;
  }
  if (curr_high < high)
    return curr_high;
  else
    return curr_high - 1;
}

void insertion_sort(ELM* low, ELM* high) {
  ELM* p;
  ELM* q;
  ELM a;
  ELM b;
  for (q = low + 1; q <= high; ++q) {
    a = q[0];
    for (p = q - 1; p >= low && (b = p[0]) > a; p--) p[1] = b;
    p[1] = a;
  }
}

void seqquick(ELM* low, ELM* high) {
  ELM* p;
  while (high - low >= bots_app_cutoff_value_2) {
    p = seqpart(low, high);
    seqquick(low, p);
    low = p + 1;
  }
  insertion_sort(low, high);
}

void seqmerge(ELM* low1, ELM* high1, ELM* low2, ELM* high2, ELM* lowdest) {
  ELM a1;
  ELM a2;
  if (low1 < high1 && low2 < high2) {
    a1 = *low1;
    a2 = *low2;
    while (1) {
      if (a1 < a2) {
        *lowdest = a1;
        lowdest++;
        ++low1;
        a1 = *low1;
        if (low1 >= high1) break;
      } else {
        *lowdest = a2;
        lowdest++;
        ++low2;
        a2 = *low2;
        if (low2 >= high2) break;
      }
    }
  }
  if (low1 <= high1 && low2 <= high2) {
    a1 = *low1;
    a2 = *low2;
    while (1) {
      if (a1 < a2) {
        *lowdest = a1;
        lowdest++;
        ++low1;
        if (low1 > high1) break;
        a1 = *low1;
      } else {
        *lowdest = a2;
        lowdest++;
        ++low2;
        if (low2 > high2) break;
        a2 = *low2;
      }
    }
  }
  if (low1 > high1) {
    memcpy(lowdest, low2, sizeof(ELM) * (high2 - low2 + 1));
  } else {
    memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1 + 1));
  }
}

ELM* binsplit(ELM val, ELM* low, ELM* high) {
  ELM* mid;
  while (low != high) {
    mid = low + (high - low + 1 >> 1);
    if (val <= *mid)
      high = mid - 1;
    else
      low = mid;
  }
  if (*low > val)
    return low - 1;
  else
    return low;
}

void cilkmerge(ELM* low1, ELM* high1, ELM* low2, ELM* high2, ELM* lowdest) {
  ELM* split1;
  ELM* split2;
  ELM* tmp;
  long int lowsize;
  if (high2 - low2 > high1 - low1) {
    tmp = low1;
    low1 = low2;
    low2 = tmp;
    tmp = high1;
    high1 = high2;
    high2 = tmp;
  }
  if (high2 < low2) {
    memcpy(lowdest, low1, sizeof(ELM) * (high1 - low1));
    return;
  }
  if (high2 - low2 < bots_app_cutoff_value) {
    seqmerge(low1, high1, low2, high2, lowdest);
    return;
  }
  split1 = (high1 - low1 + 1) / 2 + low1;
  split2 = binsplit(split1[0], low2, high2);
  lowsize = split1 - low1 + split2 - low2;
  lowdest[lowsize + 1] = split1[0];
  cilkmerge(low1, split1 - 1, low2, split2, lowdest);
  cilkmerge(split1 + 1, high1, split2 + 1, high2, lowdest + lowsize + 2);
  return;
}

void cilksort(ELM* low, ELM* tmp, long int size) {
  int __apac_count_ok = __apac_count_infinite || __apac_count < __apac_count_max;
#pragma omp taskgroup
  {
    long int quarter = size / 4;
    ELM* A;
    ELM* B;
    ELM* C;
    ELM* D;
    ELM* tmpA;
    ELM* tmpB;
    ELM* tmpC;
    ELM* tmpD;
    if (size < bots_app_cutoff_value_1) {
      seqquick(low, low + size - 1);
      goto __apac_exit;
    }
    A = low;
    tmpA = tmp;
    B = A + quarter;
    C = B + quarter;
    D = C + quarter;
    tmpB = tmpA + quarter;
    tmpC = tmpB + quarter;
    tmpD = tmpC + quarter;
    if (__apac_count_ok) {
#pragma omp atomic
      __apac_count++;
    }
#pragma omp task default(shared) depend(in : low[0], quarter, size) depend(inout : low, tmp) if (__apac_count_ok && 0.00296263979006 + (size - 3 * quarter) * 3.59540049706e-06 > __apac_cutoff)
    {
      cilksort(A, tmpA, quarter);
      cilksort(B, tmpB, quarter);
      cilksort(C, tmpC, quarter);
      cilksort(D, tmpD, size - 3 * quarter);
      cilkmerge(A, A + quarter - 1, B, B + quarter - 1, tmpA);
      cilkmerge(C, C + quarter - 1, D, low + size - 1, tmpC);
      cilkmerge(tmpA, tmpC - 1, tmpC, tmpA + size - 1, A);
      if (__apac_count_ok) {
#pragma omp atomic
        __apac_count--;
      }
    }
  __apac_exit:;
  }
}

void scramble_array(ELM* array) {
  long unsigned int i;
  long unsigned int j;
  ELM tmp;
  for (i = 0; i < bots_arg_size; ++i) {
    j = my_rand();
    j = j % bots_arg_size;
    tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
  }
}

void fill_array(ELM* array) {
  long unsigned int i;
  my_srand(1);
  for (i = 0; i < bots_arg_size; ++i) {
    array[i] = i;
  }
}

void sort_init() {
  if (bots_arg_size < 4) {
    bots_message("%s can not be less than 4, using 4 as a parameter.\n", "Array size");
    bots_arg_size = 4;
  }
  if (bots_app_cutoff_value < 2) {
    bots_message("%s can not be less than 2, using 2 as a parameter.\n", "Sequential Merge cutoff value");
    bots_app_cutoff_value = 2;
  } else if (bots_app_cutoff_value > bots_arg_size) {
    bots_message("%s can not be greather than vector size, using %d as a parameter.\n", "Sequential Merge cutoff value", bots_arg_size);
    bots_app_cutoff_value = bots_arg_size;
  }
  if (bots_app_cutoff_value_1 > bots_arg_size) {
    bots_message("%s can not be greather than vector size, using %d as a parameter.\n", "Sequential Quicksort cutoff value", bots_arg_size);
    bots_app_cutoff_value_1 = bots_arg_size;
  }
  if (bots_app_cutoff_value_2 > bots_arg_size) {
    bots_message("%s can not be greather than vector size, using %d as a parameter.\n", "Sequential Insertion cutoff value", bots_arg_size);
    bots_app_cutoff_value_2 = bots_arg_size;
  }
  if (bots_app_cutoff_value_2 > bots_app_cutoff_value_1) {
    bots_message("%s can not be greather than %s, using %d as a parameter.\n", "Sequential Insertion cutoff value", "Sequential Quicksort cutoff value", bots_app_cutoff_value_1);
    bots_app_cutoff_value_2 = bots_app_cutoff_value_1;
  }
  array = (ELM*)malloc(bots_arg_size * sizeof(ELM));
  tmp = (ELM*)malloc(bots_arg_size * sizeof(ELM));
  fill_array(array);
  scramble_array(array);
}

void sort() {
  bots_message("Computing multisort algorithm (n=%d) ", bots_arg_size);
  cilksort(array, tmp, bots_arg_size);
  bots_message(" completed!\n");
}

int sort_verify() {
  int i;
  int success = 1;
  for (i = 0; i < bots_arg_size; ++i)
    if (array[i] != i) success = 0;
  return (success ? 1 : 2);
}
