#include <libgen.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bots.h"
#include "sparselu.hpp"
const static int __apac_depth_infinite = getenv("APAC_TASK_DEPTH_INFINITE") ? 1 : 0;

const static int __apac_depth_max = getenv("APAC_TASK_DEPTH_MAX") ? atoi(getenv("APAC_TASK_DEPTH_MAX")) : 5;

int __apac_depth = 0;

#pragma omp threadprivate(__apac_depth)

int checkmat(float* M, float* N) {
  int i;
  int j;
  float r_err;
  for (i = 0; i < bots_arg_size_1; i++) {
    for (j = 0; j < bots_arg_size_1; j++) {
      r_err = M[i * bots_arg_size_1 + j] - N[i * bots_arg_size_1 + j];
      if (r_err == 0.) continue;
      if (r_err < 0.) r_err = -r_err;
      if (M[i * bots_arg_size_1 + j] == 0) {
        bots_message("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; \n", i, j, M[i * bots_arg_size_1 + j], i, j, N[i * bots_arg_size_1 + j]);
        return 0;
      }
      r_err = r_err / M[i * bots_arg_size_1 + j];
      if (r_err > 1e-06) {
        bots_message("Checking failure: A[%d][%d]=%f  B[%d][%d]=%f; Relative Error=%f\n", i, j, M[i * bots_arg_size_1 + j], i, j, N[i * bots_arg_size_1 + j], r_err);
        return 0;
      }
    }
  }
  return 1;
}

void genmat(float* M[]) {
  int null_entry;
  int init_val;
  int i;
  int j;
  int ii;
  int jj;
  float* p;
  init_val = 1325;
  for (ii = 0; ii < bots_arg_size; ii++) {
    for (jj = 0; jj < bots_arg_size; jj++) {
      null_entry = 0;
      if (ii < jj && ii % 3 != 0) null_entry = 1;
      if (ii > jj && jj % 3 != 0) null_entry = 1;
      if (ii % 2 == 1) null_entry = 1;
      if (jj % 2 == 1) null_entry = 1;
      if (ii == jj) null_entry = 0;
      if (ii == jj - 1) null_entry = 0;
      if (ii - 1 == jj) null_entry = 0;
      if (null_entry == 0) {
        M[ii * bots_arg_size + jj] = (float*)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
        if (M[ii * bots_arg_size + jj] == (float*)0) {
          bots_message("Error: Out of memory\n");
          exit(101);
        }
        p = M[ii * bots_arg_size + jj];
        for (i = 0; i < bots_arg_size_1; i++) {
          for (j = 0; j < bots_arg_size_1; j++) {
            init_val = 3125 * init_val % 65536;
            *p = (float)((init_val - 32768.) / 16384.);
            p++;
          }
        }
      } else {
        M[ii * bots_arg_size + jj] = (float*)0;
      }
    }
  }
}

void print_structure(const char* name, float* M[]) {
  int ii;
  int jj;
  bots_message("Structure for matrix %s @ 0x%p\n", name, M);
  for (ii = 0; ii < bots_arg_size; ii++) {
    for (jj = 0; jj < bots_arg_size; jj++) {
      if (M[ii * bots_arg_size + jj]) {
        bots_message("x");
      } else
        bots_message(" ");
    }
    bots_message("\n");
  }
  bots_message("\n");
}

float* allocate_clean_block() {
  int i;
  int j;
  float* p;
  float* q;
  p = (float*)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
  q = p;
  if (p) {
    for (i = 0; i < bots_arg_size_1; i++)
      for (j = 0; j < bots_arg_size_1; j++) {
        *p = 0.f;
        p++;
      }
  } else {
    bots_message("Error: Out of memory\n");
    exit(101);
  }
  return q;
}

void lu0(float* diag) {
  int i;
  int j;
  int k;
  for (k = 0; k < bots_arg_size_1; k++)
    for (i = k + 1; i < bots_arg_size_1; i++) {
      diag[i * bots_arg_size_1 + k] = diag[i * bots_arg_size_1 + k] / diag[k * bots_arg_size_1 + k];
      for (j = k + 1; j < bots_arg_size_1; j++) diag[i * bots_arg_size_1 + j] = diag[i * bots_arg_size_1 + j] - diag[i * bots_arg_size_1 + k] * diag[k * bots_arg_size_1 + j];
    }
}

void bdiv(float* diag, float* row) {
  int i;
  int j;
  int k;
  for (i = 0; i < bots_arg_size_1; i++)
    for (k = 0; k < bots_arg_size_1; k++) {
      row[i * bots_arg_size_1 + k] = row[i * bots_arg_size_1 + k] / diag[k * bots_arg_size_1 + k];
      for (j = k + 1; j < bots_arg_size_1; j++) row[i * bots_arg_size_1 + j] = row[i * bots_arg_size_1 + j] - row[i * bots_arg_size_1 + k] * diag[k * bots_arg_size_1 + j];
    }
}

void bmod(float* row, float* col, float* inner) {
  int i;
  int j;
  int k;
  for (i = 0; i < bots_arg_size_1; i++)
    for (j = 0; j < bots_arg_size_1; j++)
      for (k = 0; k < bots_arg_size_1; k++) inner[i * bots_arg_size_1 + j] = inner[i * bots_arg_size_1 + j] - row[i * bots_arg_size_1 + k] * col[k * bots_arg_size_1 + j];
}

void fwd(float* diag, float* col) {
  int i;
  int j;
  int k;
  for (j = 0; j < bots_arg_size_1; j++)
    for (k = 0; k < bots_arg_size_1; k++)
      for (i = k + 1; i < bots_arg_size_1; i++) col[i * bots_arg_size_1 + j] = col[i * bots_arg_size_1 + j] - diag[i * bots_arg_size_1 + k] * col[k * bots_arg_size_1 + j];
}

void prealloc_sparselu(float** BENCH, int* timestamp) {
  bots_message("Pre-allocating factorized matrix");
  for (int ii = 0; ii < bots_arg_size; ii++)
    for (int jj = 0; jj < bots_arg_size; jj++)
      if (BENCH[ii * bots_arg_size + jj]) timestamp[ii * bots_arg_size + jj] = -1;
  for (int kk = 0; kk < bots_arg_size; kk++) {
    for (int ii = kk + 1; ii < bots_arg_size; ii++)
      if (BENCH[ii * bots_arg_size + kk])
        for (int jj = kk + 1; jj < bots_arg_size; jj++)
          if (BENCH[kk * bots_arg_size + jj]) {
            if (BENCH[ii * bots_arg_size + jj] == (float*)0) {
              timestamp[ii * bots_arg_size + jj] = kk;
              BENCH[ii * bots_arg_size + jj] = allocate_clean_block();
            }
          }
  }
  bots_message(" completed!\n");
}

void sparselu_init(float*** pBENCH, const char* pass, int** timestamp) {
  *pBENCH = (float**)malloc(bots_arg_size * bots_arg_size * sizeof(float*));
  genmat(*pBENCH);
  print_structure(pass, *pBENCH);
  *timestamp = (int*)calloc(bots_arg_size * bots_arg_size, sizeof(int));
  prealloc_sparselu(*pBENCH, *timestamp);
}

void sparselu(float** BENCH, int* timestamp) {
  int __apac_depth_local = __apac_depth;
  int __apac_depth_ok = __apac_depth_infinite || __apac_depth_local < __apac_depth_max;
  if (__apac_depth_ok) {
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
    {
      bots_message("Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ", bots_arg_size, bots_arg_size, bots_arg_size_1, bots_arg_size_1);
      for (int kk = 0; kk < bots_arg_size; kk++) {
#pragma omp taskwait depend(in : BENCH, BENCH[kk * bots_arg_size + kk], kk) depend(inout : BENCH[kk * bots_arg_size + kk][0])
        lu0(BENCH[kk * bots_arg_size + kk]);
        for (int jj = kk + 1; jj < bots_arg_size; jj++) {
          if (BENCH[kk * bots_arg_size + jj] && timestamp[kk * bots_arg_size + jj] < kk) {
#pragma omp taskwait depend(in : BENCH, BENCH[kk * bots_arg_size + jj], BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + kk][0], jj, kk) depend(inout : BENCH[kk * bots_arg_size + jj][0])
            fwd(BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj]);
          }
        }
        for (int ii = kk + 1; ii < bots_arg_size; ii++) {
          if (BENCH[ii * bots_arg_size + kk] && timestamp[ii * bots_arg_size + kk] < kk) {
#pragma omp task default(shared) depend(in : BENCH, BENCH[ii * bots_arg_size + kk], BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + kk][0]) depend(inout : BENCH[ii * bots_arg_size + kk][0]) firstprivate(__apac_depth_local, kk, ii) if (__apac_depth_ok)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              bdiv(BENCH[kk * bots_arg_size + kk], BENCH[ii * bots_arg_size + kk]);
            }
          }
        }
        for (int ii = kk + 1; ii < bots_arg_size; ii++) {
          if (BENCH[ii * bots_arg_size + kk] && timestamp[ii * bots_arg_size + kk] < kk) {
            for (int jj = kk + 1; jj < bots_arg_size; jj++) {
              if (BENCH[kk * bots_arg_size + jj] && timestamp[kk * bots_arg_size + jj] < kk) {
#pragma omp task default(shared) depend(in : BENCH, BENCH[ii * bots_arg_size + jj], BENCH[ii * bots_arg_size + kk], BENCH[ii * bots_arg_size + kk][0], BENCH[kk * bots_arg_size + jj], BENCH[kk * bots_arg_size + jj][0]) depend(inout : BENCH[ii * bots_arg_size + jj][0]) firstprivate(__apac_depth_local, kk, jj, ii) if (__apac_depth_ok)
                {
                  if (__apac_depth_ok) {
                    __apac_depth = __apac_depth_local + 1;
                  }
                  bmod(BENCH[ii * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj], BENCH[ii * bots_arg_size + jj]);
                }
              }
            }
          }
        }
      }
      bots_message(" completed!\n");
    __apac_exit:;
    }
  } else {
    sparselu_seq(BENCH, timestamp);
  }
}

void sparselu_seq(float** BENCH, int* timestamp) {
  for (int kk = 0; kk < bots_arg_size; kk++) {
    lu0(BENCH[kk * bots_arg_size + kk]);
    for (int jj = kk + 1; jj < bots_arg_size; jj++)
      if (BENCH[kk * bots_arg_size + jj] && timestamp[kk * bots_arg_size + jj] < kk) {
        fwd(BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj]);
      }
    for (int ii = kk + 1; ii < bots_arg_size; ii++)
      if (BENCH[ii * bots_arg_size + kk] && timestamp[ii * bots_arg_size + kk] < kk) {
        bdiv(BENCH[kk * bots_arg_size + kk], BENCH[ii * bots_arg_size + kk]);
      }
    for (int ii = kk + 1; ii < bots_arg_size; ii++)
      if (BENCH[ii * bots_arg_size + kk] && timestamp[ii * bots_arg_size + kk] < kk)
        for (int jj = kk + 1; jj < bots_arg_size; jj++)
          if (BENCH[kk * bots_arg_size + jj] && timestamp[kk * bots_arg_size + jj] < kk) {
            bmod(BENCH[ii * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj], BENCH[ii * bots_arg_size + jj]);
          }
  }
}

void sparselu_fini(float** BENCH, const char* pass, int** timestamp) {
  print_structure(pass, BENCH);
  free(*timestamp);
  *timestamp = (int*)0;
}

int sparselu_check(float** SEQ, float** BENCH) {
  int ii;
  int jj;
  int ok = 1;
  for (ii = 0; ii < bots_arg_size && ok; ii++) {
    for (jj = 0; jj < bots_arg_size && ok; jj++) {
      if (SEQ[ii * bots_arg_size + jj] == (float*)0 && BENCH[ii * bots_arg_size + jj]) ok = 0;
      if (SEQ[ii * bots_arg_size + jj] && BENCH[ii * bots_arg_size + jj] == (float*)0) ok = 0;
      if (SEQ[ii * bots_arg_size + jj] && BENCH[ii * bots_arg_size + jj]) ok = checkmat(SEQ[ii * bots_arg_size + jj], BENCH[ii * bots_arg_size + jj]);
    }
  }
  if (ok)
    return 1;
  else
    return 2;
}
