#ifndef SPARSELU_H
#define SPARSELU_H

#define EPSILON 1.0E-6

int checkmat (float *M, float *N);
void genmat (float *M[]);
void print_structure(const char *name, float *M[]);
float * allocate_clean_block();
void lu0(float *diag);
void bdiv(float *diag, float *row);
void bmod(float *row, float *col, float *inner);
void fwd(float *diag, float *col);

void sparselu_init (float ***pBENCH, const char *pass, int **timestamp); 
void sparselu_fini (float **BENCH, const char *pass, int **timestamp); 

void sparselu_seq(float **BENCH, int* timestamp);
void sparselu(float **BENCH, int* timestamp);

int sparselu_check(float **SEQ, float **BENCH);

#endif
