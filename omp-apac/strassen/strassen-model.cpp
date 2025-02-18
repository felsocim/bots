#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "app-desc.hpp"
#include "bots.h"
#include "strassen.hpp"

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

void matrixmul(int n, REAL* A, int an, REAL* B, int bn, REAL* C, int cn) {
  int i;
  int j;
  int k;
  REAL s;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      s = 0.;
      for (k = 0; k < n; ++k) s += A[i * an + k] * B[k * bn + j];
      C[i * cn + j] = s;
    }
  }
}

void FastNaiveMatrixMultiply(REAL* C, REAL* A, REAL* B, unsigned int MatrixSize, unsigned int RowWidthC, unsigned int RowWidthA, unsigned int RowWidthB) {
  PTR RowWidthBInBytes = RowWidthB << 3;
  PTR RowWidthAInBytes = RowWidthA << 3;
  PTR MatrixWidthInBytes = MatrixSize << 3;
  PTR RowIncrementC = RowWidthC - MatrixSize << 3;
  unsigned int Horizontal;
  unsigned int Vertical;
  REAL* ARowStart = A;
  for (Vertical = 0; Vertical < MatrixSize; Vertical++) {
    for (Horizontal = 0; Horizontal < MatrixSize; Horizontal += 8) {
      REAL* BColumnStart = B + Horizontal;
      REAL FirstARowValue = *ARowStart++;
      REAL Sum0 = FirstARowValue * *BColumnStart;
      REAL Sum1 = FirstARowValue * *(BColumnStart + 1);
      REAL Sum2 = FirstARowValue * *(BColumnStart + 2);
      REAL Sum3 = FirstARowValue * *(BColumnStart + 3);
      REAL Sum4 = FirstARowValue * *(BColumnStart + 4);
      REAL Sum5 = FirstARowValue * *(BColumnStart + 5);
      REAL Sum6 = FirstARowValue * *(BColumnStart + 6);
      REAL Sum7 = FirstARowValue * *(BColumnStart + 7);
      unsigned int Products;
      for (Products = 1; Products < MatrixSize; Products++) {
        REAL ARowValue = *ARowStart++;
        BColumnStart = (REAL*)((PTR)BColumnStart + RowWidthBInBytes);
        Sum0 += ARowValue * *BColumnStart;
        Sum1 += ARowValue * *(BColumnStart + 1);
        Sum2 += ARowValue * *(BColumnStart + 2);
        Sum3 += ARowValue * *(BColumnStart + 3);
        Sum4 += ARowValue * *(BColumnStart + 4);
        Sum5 += ARowValue * *(BColumnStart + 5);
        Sum6 += ARowValue * *(BColumnStart + 6);
        Sum7 += ARowValue * *(BColumnStart + 7);
      }
      ARowStart = (REAL*)((PTR)ARowStart - MatrixWidthInBytes);
      *C = Sum0;
      *(C + 1) = Sum1;
      *(C + 2) = Sum2;
      *(C + 3) = Sum3;
      *(C + 4) = Sum4;
      *(C + 5) = Sum5;
      *(C + 6) = Sum6;
      *(C + 7) = Sum7;
      C += 8;
    }
    ARowStart = (REAL*)((PTR)ARowStart + RowWidthAInBytes);
    C = (REAL*)((PTR)C + RowIncrementC);
  }
}

void FastAdditiveNaiveMatrixMultiply(REAL* C, REAL* A, REAL* B, unsigned int MatrixSize, unsigned int RowWidthC, unsigned int RowWidthA, unsigned int RowWidthB) {
  PTR RowWidthBInBytes = RowWidthB << 3;
  PTR RowWidthAInBytes = RowWidthA << 3;
  PTR MatrixWidthInBytes = MatrixSize << 3;
  PTR RowIncrementC = RowWidthC - MatrixSize << 3;
  unsigned int Horizontal;
  unsigned int Vertical;
  REAL* ARowStart = A;
  for (Vertical = 0; Vertical < MatrixSize; Vertical++) {
    for (Horizontal = 0; Horizontal < MatrixSize; Horizontal += 8) {
      REAL* BColumnStart = B + Horizontal;
      REAL Sum0 = *C;
      REAL Sum1 = *(C + 1);
      REAL Sum2 = *(C + 2);
      REAL Sum3 = *(C + 3);
      REAL Sum4 = *(C + 4);
      REAL Sum5 = *(C + 5);
      REAL Sum6 = *(C + 6);
      REAL Sum7 = *(C + 7);
      unsigned int Products;
      for (Products = 0; Products < MatrixSize; Products++) {
        REAL ARowValue = *ARowStart++;
        Sum0 += ARowValue * *BColumnStart;
        Sum1 += ARowValue * *(BColumnStart + 1);
        Sum2 += ARowValue * *(BColumnStart + 2);
        Sum3 += ARowValue * *(BColumnStart + 3);
        Sum4 += ARowValue * *(BColumnStart + 4);
        Sum5 += ARowValue * *(BColumnStart + 5);
        Sum6 += ARowValue * *(BColumnStart + 6);
        Sum7 += ARowValue * *(BColumnStart + 7);
        BColumnStart = (REAL*)((PTR)BColumnStart + RowWidthBInBytes);
      }
      ARowStart = (REAL*)((PTR)ARowStart - MatrixWidthInBytes);
      *C = Sum0;
      *(C + 1) = Sum1;
      *(C + 2) = Sum2;
      *(C + 3) = Sum3;
      *(C + 4) = Sum4;
      *(C + 5) = Sum5;
      *(C + 6) = Sum6;
      *(C + 7) = Sum7;
      C += 8;
    }
    ARowStart = (REAL*)((PTR)ARowStart + RowWidthAInBytes);
    C = (REAL*)((PTR)C + RowIncrementC);
  }
}

void MultiplyByDivideAndConquer(REAL* C, REAL* A, REAL* B, unsigned int MatrixSize, unsigned int RowWidthC, unsigned int RowWidthA, unsigned int RowWidthB, int AdditiveMode) {
#pragma omp taskgroup
  {
    REAL* A01;
    REAL* A10;
    REAL* A11;
    REAL* B01;
    REAL* B10;
    REAL* B11;
    REAL* C01;
    REAL* C10;
    REAL* C11;
    unsigned int QuadrantSize = MatrixSize >> 1;
    A01 = A + QuadrantSize;
    A10 = A + RowWidthA * QuadrantSize;
    A11 = A10 + QuadrantSize;
    B01 = B + QuadrantSize;
    B10 = B + RowWidthB * QuadrantSize;
    B11 = B10 + QuadrantSize;
    C01 = C + QuadrantSize;
    C10 = C + RowWidthC * QuadrantSize;
    C11 = C10 + QuadrantSize;
    if (QuadrantSize > 16) {
      MultiplyByDivideAndConquer(C, A, B, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, AdditiveMode);
      MultiplyByDivideAndConquer(C01, A, B01, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, AdditiveMode);
      MultiplyByDivideAndConquer(C11, A10, B01, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, AdditiveMode);
      MultiplyByDivideAndConquer(C10, A10, B, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, AdditiveMode);
      MultiplyByDivideAndConquer(C, A01, B10, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, 1);
      MultiplyByDivideAndConquer(C01, A01, B11, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, 1);
      MultiplyByDivideAndConquer(C11, A11, B11, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, 1);
      MultiplyByDivideAndConquer(C10, A11, B10, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, 1);
    } else {
      if (AdditiveMode) {
        FastAdditiveNaiveMatrixMultiply(C, A, B, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
        FastAdditiveNaiveMatrixMultiply(C01, A, B01, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
        FastAdditiveNaiveMatrixMultiply(C11, A10, B01, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
        FastAdditiveNaiveMatrixMultiply(C10, A10, B, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
      } else {
        FastNaiveMatrixMultiply(C, A, B, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
        FastNaiveMatrixMultiply(C01, A, B01, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
        FastNaiveMatrixMultiply(C11, A10, B01, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
        FastNaiveMatrixMultiply(C10, A10, B, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
      }
      FastAdditiveNaiveMatrixMultiply(C, A01, B10, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
      FastAdditiveNaiveMatrixMultiply(C01, A01, B11, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
      FastAdditiveNaiveMatrixMultiply(C11, A11, B11, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
      FastAdditiveNaiveMatrixMultiply(C10, A11, B10, QuadrantSize, RowWidthC, RowWidthA, RowWidthB);
    }
    goto __apac_exit;
  __apac_exit:;
  }
}

void OptimizedStrassenMultiply_seq(REAL* C, REAL* A, REAL* B, unsigned int MatrixSize, unsigned int RowWidthC, unsigned int RowWidthA, unsigned int RowWidthB, int Depth) {
  unsigned int QuadrantSize = MatrixSize >> 1;
  unsigned int QuadrantSizeInBytes = sizeof(REAL) * QuadrantSize * QuadrantSize + 32;
  unsigned int Column;
  unsigned int Row;
  REAL* A12;
  REAL* B12;
  REAL* C12;
  REAL* A21;
  REAL* B21;
  REAL* C21;
  REAL* A22;
  REAL* B22;
  REAL* C22;
  REAL* S1;
  REAL* S2;
  REAL* S3;
  REAL* S4;
  REAL* S5;
  REAL* S6;
  REAL* S7;
  REAL* S8;
  REAL* M2;
  REAL* M5;
  REAL* T1sMULT;
  PTR TempMatrixOffset = 0;
  PTR MatrixOffsetA = 0;
  PTR MatrixOffsetB = 0;
  char* Heap;
  void* StartHeap;
  PTR RowIncrementA = RowWidthA - QuadrantSize << 3;
  PTR RowIncrementB = RowWidthB - QuadrantSize << 3;
  PTR RowIncrementC = RowWidthC - QuadrantSize << 3;
  if (MatrixSize <= bots_app_cutoff_value) {
    MultiplyByDivideAndConquer(C, A, B, MatrixSize, RowWidthC, RowWidthA, RowWidthB, 0);
    return;
  }
  A12 = A + QuadrantSize;
  B12 = B + QuadrantSize;
  C12 = C + QuadrantSize;
  A21 = A + RowWidthA * QuadrantSize;
  B21 = B + RowWidthB * QuadrantSize;
  C21 = C + RowWidthC * QuadrantSize;
  A22 = A21 + QuadrantSize;
  B22 = B21 + QuadrantSize;
  C22 = C21 + QuadrantSize;
  StartHeap = Heap = (char*)malloc(QuadrantSizeInBytes * 11);
  if ((PTR)Heap & 31) Heap = (char*)((PTR)Heap + 32 - ((PTR)Heap & 31));
  S1 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  S2 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  S3 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  S4 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  S5 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  S6 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  S7 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  S8 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  M2 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  M5 = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  T1sMULT = (REAL*)Heap;
  Heap += QuadrantSizeInBytes;
  for (Row = 0; Row < QuadrantSize; Row++) {
    for (Column = 0; Column < QuadrantSize; Column++) {
      *((REAL*)((PTR)S4 + TempMatrixOffset)) = *((REAL*)((PTR)A12 + MatrixOffsetA)) - (*((REAL*)((PTR)S2 + TempMatrixOffset)) = (*((REAL*)((PTR)S1 + TempMatrixOffset)) = *((REAL*)((PTR)A21 + MatrixOffsetA)) + *((REAL*)((PTR)A22 + MatrixOffsetA))) - *((REAL*)((PTR)A + MatrixOffsetA)));
      *((REAL*)((PTR)S8 + TempMatrixOffset)) = (*((REAL*)((PTR)S6 + TempMatrixOffset)) = *((REAL*)((PTR)B22 + MatrixOffsetB)) - (*((REAL*)((PTR)S5 + TempMatrixOffset)) = *((REAL*)((PTR)B12 + MatrixOffsetB)) - *((REAL*)((PTR)B + MatrixOffsetB)))) - *((REAL*)((PTR)B21 + MatrixOffsetB));
      *((REAL*)((PTR)S3 + TempMatrixOffset)) = *((REAL*)((PTR)A + MatrixOffsetA)) - *((REAL*)((PTR)A21 + MatrixOffsetA));
      *((REAL*)((PTR)S7 + TempMatrixOffset)) = *((REAL*)((PTR)B22 + MatrixOffsetB)) - *((REAL*)((PTR)B12 + MatrixOffsetB));
      TempMatrixOffset += sizeof(REAL);
      MatrixOffsetA += sizeof(REAL);
      MatrixOffsetB += sizeof(REAL);
    }
    MatrixOffsetA += RowIncrementA;
    MatrixOffsetB += RowIncrementB;
  }
  OptimizedStrassenMultiply_seq(M2, A, B, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth + 1);
  OptimizedStrassenMultiply_seq(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth + 1);
  OptimizedStrassenMultiply_seq(T1sMULT, S2, S6, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth + 1);
  OptimizedStrassenMultiply_seq(C22, S3, S7, QuadrantSize, RowWidthC, QuadrantSize, QuadrantSize, Depth + 1);
  OptimizedStrassenMultiply_seq(C, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth + 1);
  OptimizedStrassenMultiply_seq(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth + 1);
  OptimizedStrassenMultiply_seq(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth + 1);
  for (Row = 0; Row < QuadrantSize; Row++) {
    for (Column = 0; Column < QuadrantSize; Column += 4) {
      REAL LocalM5_0 = *M5;
      REAL LocalM5_1 = *(M5 + 1);
      REAL LocalM5_2 = *(M5 + 2);
      REAL LocalM5_3 = *(M5 + 3);
      REAL LocalM2_0 = *M2;
      REAL LocalM2_1 = *(M2 + 1);
      REAL LocalM2_2 = *(M2 + 2);
      REAL LocalM2_3 = *(M2 + 3);
      REAL T1_0 = *T1sMULT + LocalM2_0;
      REAL T1_1 = *(T1sMULT + 1) + LocalM2_1;
      REAL T1_2 = *(T1sMULT + 2) + LocalM2_2;
      REAL T1_3 = *(T1sMULT + 3) + LocalM2_3;
      REAL T2_0 = *C22 + T1_0;
      REAL T2_1 = *(C22 + 1) + T1_1;
      REAL T2_2 = *(C22 + 2) + T1_2;
      REAL T2_3 = *(C22 + 3) + T1_3;
      *C += LocalM2_0;
      *(C + 1) += LocalM2_1;
      *(C + 2) += LocalM2_2;
      *(C + 3) += LocalM2_3;
      *C12 += LocalM5_0 + T1_0;
      *(C12 + 1) += LocalM5_1 + T1_1;
      *(C12 + 2) += LocalM5_2 + T1_2;
      *(C12 + 3) += LocalM5_3 + T1_3;
      *C22 = LocalM5_0 + T2_0;
      *(C22 + 1) = LocalM5_1 + T2_1;
      *(C22 + 2) = LocalM5_2 + T2_2;
      *(C22 + 3) = LocalM5_3 + T2_3;
      *C21 = -*C21 + T2_0;
      *(C21 + 1) = -*(C21 + 1) + T2_1;
      *(C21 + 2) = -*(C21 + 2) + T2_2;
      *(C21 + 3) = -*(C21 + 3) + T2_3;
      M5 += 4;
      M2 += 4;
      T1sMULT += 4;
      C += 4;
      C12 += 4;
      C21 += 4;
      C22 += 4;
    }
    C = (REAL*)((PTR)C + RowIncrementC);
    C12 = (REAL*)((PTR)C12 + RowIncrementC);
    C21 = (REAL*)((PTR)C21 + RowIncrementC);
    C22 = (REAL*)((PTR)C22 + RowIncrementC);
  }
  free(StartHeap);
}

void OptimizedStrassenMultiply(REAL* C, REAL* A, REAL* B, unsigned int MatrixSize, unsigned int RowWidthC, unsigned int RowWidthA, unsigned int RowWidthB, int Depth) {
#pragma omp taskgroup
  {
    unsigned int QuadrantSize = MatrixSize >> 1;
    unsigned int QuadrantSizeInBytes = sizeof(REAL) * QuadrantSize * QuadrantSize + 32;
    unsigned int Column;
    unsigned int Row;
    REAL* A12;
    REAL* B12;
    REAL* C12;
    REAL* A21;
    REAL* B21;
    REAL* C21;
    REAL* A22;
    REAL* B22;
    REAL* C22;
    REAL* S1;
    REAL* S2;
    REAL* S3;
    REAL* S4;
    REAL* S5;
    REAL* S6;
    REAL* S7;
    REAL* S8;
    REAL* M2;
    REAL* M5;
    REAL* T1sMULT;
    PTR TempMatrixOffset = 0;
    PTR MatrixOffsetA = 0;
    PTR MatrixOffsetB = 0;
    char* Heap;
    void* StartHeap;
    PTR RowIncrementA = RowWidthA - QuadrantSize << 3;
    PTR RowIncrementB = RowWidthB - QuadrantSize << 3;
    PTR RowIncrementC = RowWidthC - QuadrantSize << 3;
    if (MatrixSize <= bots_app_cutoff_value) {
      MultiplyByDivideAndConquer(C, A, B, MatrixSize, RowWidthC, RowWidthA, RowWidthB, 0);
      goto __apac_exit;
    }
    A12 = A + QuadrantSize;
    A21 = A + RowWidthA * QuadrantSize;
    A22 = A21 + QuadrantSize;
    B12 = B + QuadrantSize;
    B21 = B + RowWidthB * QuadrantSize;
    B22 = B21 + QuadrantSize;
    C12 = C + QuadrantSize;
    C21 = C + RowWidthC * QuadrantSize;
    C22 = C21 + QuadrantSize;
    StartHeap = Heap = (char*)malloc(QuadrantSizeInBytes * 11);
    if ((PTR)Heap & 31) {
      Heap = (char*)((PTR)Heap + 32 - ((PTR)Heap & 31));
    }
    S1 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    S2 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    S3 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    S4 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    S5 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    S6 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    S7 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    S8 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    M2 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    M5 = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    T1sMULT = (REAL*)Heap;
    Heap += QuadrantSizeInBytes;
    for (Row = 0; Row < QuadrantSize; Row++) {
      for (Column = 0; Column < QuadrantSize; Column++) {
        *((REAL*)((PTR)S4 + TempMatrixOffset)) = *((REAL*)((PTR)A12 + MatrixOffsetA)) - (*((REAL*)((PTR)S2 + TempMatrixOffset)) = (*((REAL*)((PTR)S1 + TempMatrixOffset)) = *((REAL*)((PTR)A21 + MatrixOffsetA)) + *((REAL*)((PTR)A22 + MatrixOffsetA))) - *((REAL*)((PTR)A + MatrixOffsetA)));
        *((REAL*)((PTR)S8 + TempMatrixOffset)) = (*((REAL*)((PTR)S6 + TempMatrixOffset)) = *((REAL*)((PTR)B22 + MatrixOffsetB)) - (*((REAL*)((PTR)S5 + TempMatrixOffset)) = *((REAL*)((PTR)B12 + MatrixOffsetB)) - *((REAL*)((PTR)B + MatrixOffsetB)))) - *((REAL*)((PTR)B21 + MatrixOffsetB));
        *((REAL*)((PTR)S3 + TempMatrixOffset)) = *((REAL*)((PTR)A + MatrixOffsetA)) - *((REAL*)((PTR)A21 + MatrixOffsetA));
        *((REAL*)((PTR)S7 + TempMatrixOffset)) = *((REAL*)((PTR)B22 + MatrixOffsetB)) - *((REAL*)((PTR)B12 + MatrixOffsetB));
        TempMatrixOffset += sizeof(REAL);
        MatrixOffsetA += sizeof(REAL);
        MatrixOffsetB += sizeof(REAL);
      }
      MatrixOffsetA += RowIncrementA;
      MatrixOffsetB += RowIncrementB;
    }
#pragma omp task default(shared) depend(in : A, B, Depth, QuadrantSize, RowWidthA, RowWidthB) depend(inout : A[0], B[0], Heap) if (-0.00852048915189 + *QuadrantSize * 0.00160425053739 + apac_fpow(2, Depth + 1) * 1.47406346101e-07 > __apac_cutoff)
    OptimizedStrassenMultiply(M2, A, B, QuadrantSize, QuadrantSize, RowWidthA, RowWidthB, Depth + 1);
#pragma omp task default(shared) depend(in : Depth, QuadrantSize) depend(inout : A, B, Heap) if (0.00397850657798 + 0. + apac_fpow(2, *QuadrantSize) * 2.80786503052e-07 + *QuadrantSize * *QuadrantSize * -1.5410733005e-05 > __apac_cutoff)
    {
      OptimizedStrassenMultiply(M5, S1, S5, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth + 1);
      OptimizedStrassenMultiply(T1sMULT, S2, S6, QuadrantSize, QuadrantSize, QuadrantSize, QuadrantSize, Depth + 1);
    }
#pragma omp task default(shared) depend(in : Depth, QuadrantSize, RowWidthA, RowWidthB, RowWidthC) depend(inout : A, B, C, C[0]) if (0.0070585935097 + 0. + (Depth + 1) * *QuadrantSize * 3.15413017601e-08 + RowWidthA * *QuadrantSize * 3.15413017601e-08 + RowWidthA * *QuadrantSize * 3.15413017601e-08 + RowWidthA * *QuadrantSize * 3.15413017601e-08 + RowWidthC * *QuadrantSize * 3.15413017601e-08 + RowWidthC * *QuadrantSize * 3.15413017601e-08 + RowWidthC * *QuadrantSize * 3.15413017601e-08 + RowWidthC * *QuadrantSize * 3.15413017601e-08 + apac_fpow(2, Depth + 1) * 3.15413017601e-08 + (Depth + 1) * *QuadrantSize * 3.15413017601e-08 + *QuadrantSize * 3.15413017601e-08 + *QuadrantSize * *QuadrantSize * 3.15413017601e-08 + *QuadrantSize * *QuadrantSize * 3.15413017601e-08 + *QuadrantSize * 3.15413017601e-08 + *QuadrantSize * *QuadrantSize * -2.7685166584e-05 + *QuadrantSize * 3.15413017601e-08 + *QuadrantSize * *QuadrantSize * 3.15413017601e-08 + *QuadrantSize * 3.15413017601e-08 > __apac_cutoff
    )
    {
      OptimizedStrassenMultiply(C22, S3, S7, QuadrantSize, RowWidthC, QuadrantSize, QuadrantSize, Depth + 1);
      OptimizedStrassenMultiply(C, A12, B21, QuadrantSize, RowWidthC, RowWidthA, RowWidthB, Depth + 1);
      OptimizedStrassenMultiply(C12, S4, B22, QuadrantSize, RowWidthC, QuadrantSize, RowWidthB, Depth + 1);
      OptimizedStrassenMultiply(C21, A22, S8, QuadrantSize, RowWidthC, RowWidthA, QuadrantSize, Depth + 1);
    }
#pragma omp taskwait
    for (Row = 0; Row < QuadrantSize; Row++) {
      for (Column = 0; Column < QuadrantSize; Column += 4) {
        REAL LocalM5_0 = *M5;
        REAL LocalM5_1 = *(M5 + 1);
        REAL LocalM5_2 = *(M5 + 2);
        REAL LocalM5_3 = *(M5 + 3);
        REAL LocalM2_0 = *M2;
        REAL LocalM2_1 = *(M2 + 1);
        REAL LocalM2_2 = *(M2 + 2);
        REAL LocalM2_3 = *(M2 + 3);
        REAL T1_0 = *T1sMULT + LocalM2_0;
        REAL T1_1 = *(T1sMULT + 1) + LocalM2_1;
        REAL T1_2 = *(T1sMULT + 2) + LocalM2_2;
        REAL T1_3 = *(T1sMULT + 3) + LocalM2_3;
        REAL T2_0 = *C22 + T1_0;
        REAL T2_1 = *(C22 + 1) + T1_1;
        REAL T2_2 = *(C22 + 2) + T1_2;
        REAL T2_3 = *(C22 + 3) + T1_3;
        *C += LocalM2_0;
        *(C + 1) += LocalM2_1;
        *(C + 2) += LocalM2_2;
        *(C + 3) += LocalM2_3;
        *C12 += LocalM5_0 + T1_0;
        *(C12 + 1) += LocalM5_1 + T1_1;
        *(C12 + 2) += LocalM5_2 + T1_2;
        *(C12 + 3) += LocalM5_3 + T1_3;
        *C22 = LocalM5_0 + T2_0;
        *(C22 + 1) = LocalM5_1 + T2_1;
        *(C22 + 2) = LocalM5_2 + T2_2;
        *(C22 + 3) = LocalM5_3 + T2_3;
        *C21 = -*C21 + T2_0;
        *(C21 + 1) = -*(C21 + 1) + T2_1;
        *(C21 + 2) = -*(C21 + 2) + T2_2;
        *(C21 + 3) = -*(C21 + 3) + T2_3;
        M5 += 4;
        M2 += 4;
        T1sMULT += 4;
        C += 4;
        C12 += 4;
        C21 += 4;
        C22 += 4;
      }
      C = (REAL*)((PTR)C + RowIncrementC);
      C12 = (REAL*)((PTR)C12 + RowIncrementC);
      C21 = (REAL*)((PTR)C21 + RowIncrementC);
      C22 = (REAL*)((PTR)C22 + RowIncrementC);
    }
    free(StartHeap);
  __apac_exit:;
  }
}

void init_matrix(int n, REAL* A, int an) {
  int i;
  int j;
  for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j) A[i * an + j] = (double)rand() / (double)2147483647;
}

int compare_matrix(int n, REAL* A, int an, REAL* B, int bn) {
  int i;
  int j;
  REAL c;
  for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j) {
      c = A[i * an + j] - B[i * bn + j];
      if (c < 0.) c = -c;
      c = c / A[i * an + j];
      if (c > 1e-06) {
        bots_message("Strassen: Wrong answer!\n");
        return 2;
      }
    }
  return 1;
}

REAL* alloc_matrix(int n) { return (REAL*)malloc(n * n * sizeof(REAL)); }

void strassen_main(REAL* A, REAL* B, REAL* C, int n) {
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
  {
    bots_message("Computing parallel Strassen algorithm (n=%d) ", n);
#pragma omp task default(shared) depend(in : A, B, C, n) depend(inout : A[0], B[0], C[0])
    OptimizedStrassenMultiply(C, A, B, n, n, n, n, 1);
    bots_message(" completed!\n");
  __apac_exit:;
  }
}

void strassen_main_seq(REAL* A, REAL* B, REAL* C, int n) {
  bots_message("Computing sequential Strassen algorithm (n=%d) ", n);
  OptimizedStrassenMultiply_seq(C, A, B, n, n, n, n, 1);
  bots_message(" completed!\n");
}
