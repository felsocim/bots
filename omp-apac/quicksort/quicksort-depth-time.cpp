#include <errno.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "bots.h"
#include "quicksort.hpp"
const double __apac_cutoff = getenv("APAC_EXECUTION_TIME_CUTOFF") ? atof(getenv("APAC_EXECUTION_TIME_CUTOFF")) : 2.22100e-6;

const static int __apac_depth_infinite = getenv("APAC_TASK_DEPTH_INFINITE") ? 1 : 0;

const static int __apac_depth_max = getenv("APAC_TASK_DEPTH_MAX") ? atoi(getenv("APAC_TASK_DEPTH_MAX")) : 5;

int __apac_depth = 0;

#pragma omp threadprivate(__apac_depth)

void partition(int* out_pivot, int* arr, int right_limit) {
  int pivot = arr[right_limit - 1];
  int idx_left = -1;
  int idx_iter;
  int tmp;
  for (idx_iter = 0; idx_iter < right_limit - 1; idx_iter++) {
    if (arr[idx_iter] < pivot) {
      idx_left++;
      tmp = arr[idx_left];
      arr[idx_left] = arr[idx_iter];
      arr[idx_iter] = tmp;
    }
  }
  tmp = arr[idx_left + 1];
  arr[idx_left + 1] = arr[right_limit - 1];
  arr[right_limit - 1] = tmp;
  *out_pivot = idx_left + 1;
}

void insertion_sort(int* arr, int right_limit) {
  for (int idx = 0; idx < right_limit - 1; ++idx) {
    int idx_min = idx;
    int idx_iter;
    for (idx_iter = idx_min + 1; idx_iter < right_limit; ++idx_iter) {
      if (arr[idx_min] > arr[idx_iter]) {
        idx_min = idx_iter;
      }
    }
    int tmp = arr[idx];
    arr[idx] = arr[idx_min];
    arr[idx_min] = tmp;
  }
}

void __apac_sequential_sort_core(int* in_out_data, int right_limit) {
  if (0 >= right_limit) {
    return;
  }
  if (right_limit <= 256) {
    insertion_sort(in_out_data, right_limit);
  } else {
    int pivot;
    partition(&pivot, in_out_data, right_limit);
    __apac_sequential_sort_core(&in_out_data[0], pivot);
    __apac_sequential_sort_core(&in_out_data[pivot + 1], right_limit - (pivot + 1));
  }
}

void sort_core(int* in_out_data, int right_limit) {
  int __apac_depth_local = __apac_depth;
  int __apac_depth_ok = __apac_depth_infinite || __apac_depth_local < __apac_depth_max;
  if (__apac_depth_ok) {
#pragma omp taskgroup
    {
      if (0 >= right_limit) {
        goto __apac_exit;
      }
      if (right_limit <= 256) {
        insertion_sort(in_out_data, right_limit);
      } else {
        int* pivot = new int();
        partition(pivot, in_out_data, right_limit);
#pragma omp task default(shared) depend(in : in_out_data, pivot[0], pivot) depend(inout : in_out_data[0]) firstprivate(__apac_depth_local) if (__apac_depth_ok && -0.000433323553631 + *pivot * 1.69026131977e-07 > __apac_cutoff)
        {
          if (__apac_depth_ok) {
            __apac_depth = __apac_depth_local + 1;
          }
          sort_core(&in_out_data[0], *pivot);
        }
#pragma omp task default(shared) depend(in : in_out_data, pivot[0], right_limit, pivot) depend(inout : in_out_data[*pivot + 1]) firstprivate(__apac_depth_local) if (__apac_depth_ok && -0.000707797413477 + (right_limit - (*pivot + 1)) * 1.73940866457e-07 > __apac_cutoff)
        {
          if (__apac_depth_ok) {
            __apac_depth = __apac_depth_local + 1;
          }
          sort_core(&in_out_data[*pivot + 1], right_limit - (*pivot + 1));
        }
#pragma omp task default(shared) depend(inout : pivot) firstprivate(__apac_depth_local) if (__apac_depth_ok)
        {
          if (__apac_depth_ok) {
            __apac_depth = __apac_depth_local + 1;
          }
          delete pivot;
        }
      }
    __apac_exit:;
    }
  } else {
    __apac_sequential_sort_core(in_out_data, right_limit);
  }
}

void __apac_sequential_sort(int* in_out_data, int in_size) { __apac_sequential_sort_core(in_out_data, in_size); }

void sort(int* in_out_data, int in_size) {
  int __apac_depth_local = __apac_depth;
  int __apac_depth_ok = __apac_depth_infinite || __apac_depth_local < __apac_depth_max;
  if (__apac_depth_ok) {
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
    {
#pragma omp task default(shared) depend(in : in_out_data, in_size) depend(inout : in_out_data[0]) firstprivate(__apac_depth_local) if (__apac_depth_ok)
      {
        if (__apac_depth_ok) {
          __apac_depth = __apac_depth_local + 1;
        }
        sort_core(in_out_data, in_size);
      }
    __apac_exit:;
    }
  } else {
    __apac_sequential_sort(in_out_data, in_size);
  }
}

int* init(int in_size) {
  int* data = (int*)malloc((size_t)in_size * sizeof(int));
  if (!data) {
    bots_error(*__errno_location(), "unable to allocate the array of items to sort");
  }
  srand(time((time_t*)0));
  for (int idx = 0; idx < in_size; idx++) {
    data[idx] = rand();
  }
  return data;
}

int check(int* in_out_data, int in_size) {
  for (int idx = 1; idx < in_size; idx++) {
    if (in_out_data[idx - 1] > in_out_data[idx]) {
      return 2;
    }
  }
  return 1;
}
