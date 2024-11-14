#include <libgen.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bots.h"
#include "sparselu.h"

const double __apac_cutoff = getenv("APAC_EXECUTION_TIME_CUTOFF") ? atof(getenv("APAC_EXECUTION_TIME_CUTOFF")) : 2.22100e-6;

template <class T>
T apac_fpow(int exp, const T& base) {
  T result = T(1);
  T pow = base;
  int i = exp;
  while (i) {
    if (i & 1) {
      result *= pow;
    }
    pow *= pow;
    i /= 2;
  }
  return result;
}

int checkmat(float* M, float* N) {
  int i, j;
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
  int null_entry, init_val, i, j, ii, jj;
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

void print_structure(char* name, float* M[]) {
  int ii, jj;
  bots_message("Structure for matrix %s @ 0x%p\n", name, M);
  for (ii = 0; ii < bots_arg_size; ii++) {
    for (jj = 0; jj < bots_arg_size; jj++) {
      if (M[ii * bots_arg_size + jj] != (float*)0) {
        bots_message("x");
      } else
        bots_message(" ");
    }
    bots_message("\n");
  }
  bots_message("\n");
}

float* allocate_clean_block() {
  int i, j;
  float *p, *q;
  p = (float*)malloc(bots_arg_size_1 * bots_arg_size_1 * sizeof(float));
  q = p;
  if (p != (float*)0) {
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
  int i, j, k;
  for (k = 0; k < bots_arg_size_1; k++)
    for (i = k + 1; i < bots_arg_size_1; i++) {
      diag[i * bots_arg_size_1 + k] = diag[i * bots_arg_size_1 + k] / diag[k * bots_arg_size_1 + k];
      for (j = k + 1; j < bots_arg_size_1; j++) diag[i * bots_arg_size_1 + j] = diag[i * bots_arg_size_1 + j] - diag[i * bots_arg_size_1 + k] * diag[k * bots_arg_size_1 + j];
    }
}

void bdiv(float* diag, float* row) {
  int i, j, k;
  for (i = 0; i < bots_arg_size_1; i++)
    for (k = 0; k < bots_arg_size_1; k++) {
      row[i * bots_arg_size_1 + k] = row[i * bots_arg_size_1 + k] / diag[k * bots_arg_size_1 + k];
      for (j = k + 1; j < bots_arg_size_1; j++) row[i * bots_arg_size_1 + j] = row[i * bots_arg_size_1 + j] - row[i * bots_arg_size_1 + k] * diag[k * bots_arg_size_1 + j];
    }
}

void bmod(float* row, float* col, float* inner) {
  int i, j, k;
  for (i = 0; i < bots_arg_size_1; i++)
    for (j = 0; j < bots_arg_size_1; j++)
      for (k = 0; k < bots_arg_size_1; k++) inner[i * bots_arg_size_1 + j] = inner[i * bots_arg_size_1 + j] - row[i * bots_arg_size_1 + k] * col[k * bots_arg_size_1 + j];
}

void fwd(float* diag, float* col) {
  int i, j, k;
  for (j = 0; j < bots_arg_size_1; j++)
    for (k = 0; k < bots_arg_size_1; k++)
      for (i = k + 1; i < bots_arg_size_1; i++) col[i * bots_arg_size_1 + j] = col[i * bots_arg_size_1 + j] - diag[i * bots_arg_size_1 + k] * col[k * bots_arg_size_1 + j];
}

void sparselu_init(float*** pBENCH, char* pass) {
  *pBENCH = (float**)malloc(bots_arg_size * bots_arg_size * sizeof(float*));
  genmat(*pBENCH);
  print_structure(pass, *pBENCH);
}

void sparselu_par_call(float** BENCH) {
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
  {
    int ii, jj, kk;
    bots_message("Computing SparseLU Factorization (%dx%d matrix with %dx%d blocks) ", bots_arg_size, bots_arg_size, bots_arg_size_1, bots_arg_size_1);
    for (kk = 0; kk < bots_arg_size; kk++) {
#pragma omp task default(shared) depend(in : BENCH, BENCH[kk * bots_arg_size + kk]) depend(inout : BENCH[kk * bots_arg_size + kk][0]) firstprivate(kk)
      lu0(BENCH[kk * bots_arg_size + kk]);
#pragma omp taskwait depend(inout : jj)
      for (jj = kk + 1; jj < bots_arg_size; jj++) {
#pragma omp taskwait depend(in : BENCH, BENCH[kk * bots_arg_size + jj])
        if (BENCH[kk * bots_arg_size + jj] != (float*)0) {
#pragma omp task default(shared) depend(in : BENCH, BENCH[kk * bots_arg_size + jj], BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + kk][0]) depend(inout : BENCH[kk * bots_arg_size + jj][0]) firstprivate(kk, jj)
          fwd(BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj]);
        }
      }
#pragma omp taskwait depend(inout : ii)
      for (ii = kk + 1; ii < bots_arg_size; ii++) {
#pragma omp taskwait depend(in : BENCH, BENCH[ii * bots_arg_size + kk])
        if (BENCH[ii * bots_arg_size + kk] != (float*)0) {
#pragma omp task default(shared) depend(in : BENCH, BENCH[ii * bots_arg_size + kk], BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + kk][0]) depend(inout : BENCH[ii * bots_arg_size + kk][0]) firstprivate(kk, ii)
          bdiv(BENCH[kk * bots_arg_size + kk], BENCH[ii * bots_arg_size + kk]);
        }
      }
#pragma omp taskwait depend(inout : ii)
      for (ii = kk + 1; ii < bots_arg_size; ii++) {
#pragma omp taskwait depend(in : BENCH, BENCH[ii * bots_arg_size + kk])
        if (BENCH[ii * bots_arg_size + kk] != (float*)0) {
#pragma omp taskwait depend(inout : jj)
          for (jj = kk + 1; jj < bots_arg_size; jj++) {
#pragma omp taskwait depend(in : BENCH, BENCH[kk * bots_arg_size + jj])
            if (BENCH[kk * bots_arg_size + jj] != (float*)0) {
#pragma omp taskwait depend(in : BENCH) depend(inout : BENCH[ii * bots_arg_size + jj])
              if (BENCH[ii * bots_arg_size + jj] == (float*)0) {
#pragma omp task default(shared) depend(in : BENCH) depend(inout : BENCH[ii * bots_arg_size + jj]) firstprivate(jj, ii)
                BENCH[ii * bots_arg_size + jj] = allocate_clean_block();
              }
#pragma omp task default(shared) depend(in : BENCH, BENCH[ii * bots_arg_size + jj], BENCH[ii * bots_arg_size + kk], BENCH[ii * bots_arg_size + kk][0], BENCH[kk * bots_arg_size + jj], BENCH[kk * bots_arg_size + jj][0]) depend(inout : BENCH[ii * bots_arg_size + jj][0]) firstprivate(kk, jj, ii)
              bmod(BENCH[ii * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj], BENCH[ii * bots_arg_size + jj]);
            }
          }
        }
      }
    }
    bots_message(" completed!\n");
  __apac_exit:;
  }
}

void sparselu_seq(float** BENCH) {
  int ii, jj, kk;
  for (kk = 0; kk < bots_arg_size; kk++) {
    lu0(BENCH[kk * bots_arg_size + kk]);
    for (jj = kk + 1; jj < bots_arg_size; jj++)
      if (BENCH[kk * bots_arg_size + jj] != (float*)0) {
        fwd(BENCH[kk * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj]);
      }
    for (ii = kk + 1; ii < bots_arg_size; ii++)
      if (BENCH[ii * bots_arg_size + kk] != (float*)0) {
        bdiv(BENCH[kk * bots_arg_size + kk], BENCH[ii * bots_arg_size + kk]);
      }
    for (ii = kk + 1; ii < bots_arg_size; ii++)
      if (BENCH[ii * bots_arg_size + kk] != (float*)0)
        for (jj = kk + 1; jj < bots_arg_size; jj++)
          if (BENCH[kk * bots_arg_size + jj] != (float*)0) {
            if (BENCH[ii * bots_arg_size + jj] == (float*)0) BENCH[ii * bots_arg_size + jj] = allocate_clean_block();
            bmod(BENCH[ii * bots_arg_size + kk], BENCH[kk * bots_arg_size + jj], BENCH[ii * bots_arg_size + jj]);
          }
  }
}

void sparselu_fini(float** BENCH, char* pass) { print_structure(pass, BENCH); }

int sparselu_check(float** SEQ, float** BENCH) {
  int ii, jj, ok = 1;
  for (ii = 0; ii < bots_arg_size && ok; ii++) {
    for (jj = 0; jj < bots_arg_size && ok; jj++) {
      if (SEQ[ii * bots_arg_size + jj] == (float*)0 && BENCH[ii * bots_arg_size + jj] != (float*)0) ok = 0;
      if (SEQ[ii * bots_arg_size + jj] != (float*)0 && BENCH[ii * bots_arg_size + jj] == (float*)0) ok = 0;
      if (SEQ[ii * bots_arg_size + jj] != (float*)0 && BENCH[ii * bots_arg_size + jj] != (float*)0) ok = checkmat(SEQ[ii * bots_arg_size + jj], BENCH[ii * bots_arg_size + jj]);
    }
  }
  if (ok)
    return 1;
  else
    return 2;
}
