/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

#include "omp-tasks-app.h"

#define BOTS_APP_NAME "SparseLU (Single version)"
#define BOTS_APP_PARAMETERS_DESC "S1=%dx%d, S2=%dx%d"
#define BOTS_APP_PARAMETERS_LIST ,bots_arg_size,bots_arg_size,bots_arg_size_1,bots_arg_size_1

#define BOTS_APP_USES_ARG_SIZE
#define BOTS_APP_DEF_ARG_SIZE 50
#define BOTS_APP_DESC_ARG_SIZE "Matrix Size"

#define BOTS_APP_USES_ARG_SIZE_1
#define BOTS_APP_DEF_ARG_SIZE_1 100
#define BOTS_APP_DESC_ARG_SIZE_1 "Submatrix Size"

#define BOTS_APP_INIT float **SEQ,**BENCH; int* timestamp = (int*)0;

void sparselu_init(float ***pM, char *pass, int **timestamp);
void sparselu_fini(float **M, char *pass, int **timestamp);
void sparselu_seq(float **SEQ, int *timestamp);
void sparselu_par_call(float **BENCH, int *timestamp);
int sparselu_check(float **SEQ, float **BENCH);

#define KERNEL_INIT sparselu_init(&BENCH,"benchmark",&timestamp);
#define KERNEL_CALL sparselu_par_call(BENCH,timestamp);
#define KERNEL_FINI sparselu_fini(BENCH,"benchmark",&timestamp);

#define KERNEL_SEQ_INIT sparselu_init(&SEQ,"serial",&timestamp);
#define KERNEL_SEQ_CALL sparselu_seq(SEQ,timestamp);
#define KERNEL_SEQ_FINI sparselu_fini(SEQ,"serial",&timestamp);

#define BOTS_APP_CHECK_USES_SEQ_RESULT
#define KERNEL_CHECK sparselu_check(SEQ,BENCH);

