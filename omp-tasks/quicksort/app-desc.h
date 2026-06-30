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

#define BOTS_APP_NAME "Quicksort"
#define BOTS_APP_PARAMETERS_DESC "N=%d"
#define BOTS_APP_PARAMETERS_LIST ,bots_arg_size

#define BOTS_APP_USES_ARG_SIZE
#define BOTS_APP_DEF_ARG_SIZE 10000000
#define BOTS_APP_DESC_ARG_SIZE "Number of items to sort"

int * init(int);
void sort(int *, int);
void sort_seq(int *, int);
int check(int *, int);

#define BOTS_APP_INIT \
  int * data_par = NULL, * data_seq = NULL; \
  data_par = init(bots_arg_size); \
  data_seq = (int*) malloc((size_t) bots_arg_size * sizeof(int)); \
  for(int i = 0; i < bots_arg_size; i++) data_seq[i] = data_par[i]
#define KERNEL_SEQ_CALL sort_seq(data_seq, bots_arg_size)
#define KERNEL_CALL sort(data_par, bots_arg_size)
#define KERNEL_CHECK \
  ((bots_sequential_flag && \
    check(data_par, bots_arg_size) == BOTS_RESULT_SUCCESSFUL && \
    check(data_seq, bots_arg_size) == BOTS_RESULT_SUCCESSFUL) || \
   (!bots_sequential_flag && \
    check(data_par, bots_arg_size) == BOTS_RESULT_SUCCESSFUL)) ? \
   BOTS_RESULT_SUCCESSFUL : BOTS_RESULT_UNSUCCESSFUL
#define BOTS_APP_FINI free(data_seq); free(data_par)
