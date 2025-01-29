#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "app-desc.hpp"
#include "bots.h"

int solution = -1;

FILE* inputFile;

typedef struct {
  int n;
  int** alt;
  int top;
  int bot;
  int lhs;
  int rhs;
  int left;
  int above;
  int next;
} cell;

cell* gcells;

int MIN_AREA;

char BEST_BOARD[64][64];

int MIN_FOOTPRINT[2];

int N;

int starts(int id, int shape, int** NWS, cell* cells) {
  int i;
  int n;
  int top;
  int bot;
  int lhs;
  int rhs;
  int rows;
  int cols;
  int left;
  int above;
  rows = cells[id].alt[shape][0];
  cols = cells[id].alt[shape][1];
  left = cells[id].left;
  above = cells[id].above;
  if (left >= 0 && above >= 0) {
    top = cells[above].bot + 1;
    lhs = cells[left].rhs + 1;
    bot = top + rows;
    rhs = lhs + cols;
    if (top <= cells[left].bot && bot >= cells[left].top && lhs <= cells[above].rhs && rhs >= cells[above].lhs) {
      n = 1;
      NWS[0][0] = top;
      NWS[0][1] = lhs;
    } else {
      n = 0;
    }
  } else if (left >= 0) {
    top = (cells[left].top - rows + 1 > 0 ? cells[left].top - rows + 1 : 0);
    bot = (cells[left].bot < 64 ? cells[left].bot : 64);
    n = bot - top + 1;
    for (i = 0; i < n; i++) {
      NWS[i][0] = i + top;
      NWS[i][1] = cells[left].rhs + 1;
    }
  } else {
    lhs = (cells[above].lhs - cols + 1 > 0 ? cells[above].lhs - cols + 1 : 0);
    rhs = (cells[above].rhs < 64 ? cells[above].rhs : 64);
    n = rhs - lhs + 1;
    for (i = 0; i < n; i++) {
      NWS[i][0] = cells[above].bot + 1;
      NWS[i][1] = i + lhs;
    }
  }
  return n;
}

int lay_down(int id, char board[64][64], cell* cells) {
  int i;
  int j;
  int top;
  int bot;
  int lhs;
  int rhs;
  top = cells[id].top;
  bot = cells[id].bot;
  lhs = cells[id].lhs;
  rhs = cells[id].rhs;
  for (i = top; i <= bot; i++) {
    for (j = lhs; j <= rhs; j++) {
      if (board[i][j] == 0)
        board[i][j] = (char)id;
      else
        return 0;
    }
  }
  return 1;
}

void read_integer(FILE* file, int* var) {
  if (fscanf(file, "%d", var) == -1) {
    bots_message(" Bogus input file\n");
    exit(-1);
  }
}

void read_inputs() {
  int i;
  int j;
  int n;
  read_integer(inputFile, &n);
  N = n;
#pragma omp critical
  {
    gcells = (cell*)malloc((n + 1) * sizeof(cell));
    gcells[0].n = 0;
    gcells[0].alt = 0;
    gcells[0].top = 0;
    gcells[0].bot = 0;
    gcells[0].lhs = -1;
    gcells[0].rhs = -1;
    gcells[0].left = 0;
    gcells[0].above = 0;
    gcells[0].next = 0;
  }
  for (i = 1; i < n + 1; i++) {
#pragma omp critical
    {
      read_integer(inputFile, &gcells[i].n);
      gcells[i].alt = (int**)malloc(gcells[i].n * sizeof(int*));
      for (j = 0; j < gcells[i].n; j++) {
        gcells[i].alt[j] = (int*)malloc(2 * sizeof(int));
        read_integer(inputFile, &gcells[i].alt[j][0]);
        read_integer(inputFile, &gcells[i].alt[j][1]);
      }
      read_integer(inputFile, &gcells[i].left);
      read_integer(inputFile, &gcells[i].above);
      read_integer(inputFile, &gcells[i].next);
    }
  }
  if (!feof(inputFile)) {
    read_integer(inputFile, &solution);
  }
}

void write_outputs() {
  int i;
  int j;
#pragma omp critical
  {
    bots_message("Minimum area = %d\n\n", MIN_AREA);
    for (i = 0; i < MIN_FOOTPRINT[0]; i++) {
      for (j = 0; j < MIN_FOOTPRINT[1]; j++) {
        if (BEST_BOARD[i][j] == 0) {
          bots_message(" ");
        } else
          bots_message("%c", 65 + BEST_BOARD[i][j] - 1);
      }
      bots_message("\n");
    }
  }
}

int add_cell(int id, int FOOTPRINT[2], char BOARD[64][64], cell* CELLS) {
  int __apac_result;
#pragma omp taskgroup
  {
    int i;
    int j;
    int nn;
    int area;
    int nnc;
    int nnl;
    char board[64][64];
    int footprint[2];
    int** NWS;
    NWS = (int**)malloc(64 * sizeof(int*));
    for (i = 0; i < 64; i++) {
      NWS[i] = (int*)malloc(2 * sizeof(int));
    }
    nnc = nnl = 0;
    for (i = 0; i < CELLS[id].n; i++) {
#pragma omp task default(shared) depend(in : CELLS, NWS, id) depend(inout : CELLS[0], NWS[0], NWS[0][0], nn) firstprivate(i)
      nn = starts(id, i, NWS, CELLS);
#pragma omp taskwait depend(in : nn) depend(inout : nnl)
      nnl += nn;
#pragma omp taskwait depend(in : nn) depend(inout : j)
      for (j = 0; j < nn; j++) {
        cell* cells = new cell[N + 1]();
        memcpy(cells, CELLS, sizeof(cell) * (N + 1));
        cells[id].top = NWS[j][0];
        cells[id].bot = cells[id].top + cells[id].alt[i][0] - 1;
        cells[id].lhs = NWS[j][1];
        cells[id].rhs = cells[id].lhs + cells[id].alt[i][1] - 1;
#pragma omp taskwait depend(in : BOARD, board) depend(inout : BOARD[0], BOARD[0][0], board[0], board[0][0])
        memcpy(board, BOARD, (size_t)64 * 64 * sizeof(char));
#pragma omp taskwait depend(in : board, cells, id) depend(inout : board[0], board[0][0], cells[0])
        if (!lay_down(id, board, cells)) {
          bots_debug("Chip %d, shape %d does not fit\n", id, i);
          goto _end;
        }
#pragma omp taskwait depend(in : FOOTPRINT, FOOTPRINT[0], cells, footprint, id) depend(inout : footprint[0])
        footprint[0] = (FOOTPRINT[0] > cells[id].bot + 1 ? FOOTPRINT[0] : cells[id].bot + 1);
#pragma omp taskwait depend(in : FOOTPRINT, FOOTPRINT[1], cells, footprint, id) depend(inout : footprint[1])
        footprint[1] = (FOOTPRINT[1] > cells[id].rhs + 1 ? FOOTPRINT[1] : cells[id].rhs + 1);
        area = footprint[0] * footprint[1];
        if (cells[id].next == 0) {
#pragma omp critical
          if (area < MIN_AREA) {
            if (area < MIN_AREA) {
              MIN_AREA = area;
              bots_debug("N  %d\n", MIN_AREA);
              MIN_FOOTPRINT[0] = footprint[0];
              MIN_FOOTPRINT[1] = footprint[1];
#pragma omp taskwait depend(in : BEST_BOARD, board) depend(inout : BEST_BOARD[0], BEST_BOARD[0][0], board[0], board[0][0])
              memcpy(BEST_BOARD, board, (size_t)64 * 64 * sizeof(char));
            }
          }
        } else {
#pragma omp critical
          if (area < MIN_AREA) {
#pragma omp task default(shared) depend(in : board, cells, footprint, footprint[0], id) depend(inout : board[0], board[0][0], cells[0], nnc)
            nnc += add_cell(cells[id].next, footprint, board, cells);
          } else {
            bots_debug("T  %d, %d\n", area, MIN_AREA);
          }
        }
      _end:;
#pragma omp task default(shared) depend(inout : cells)
        delete[] cells;
      }
    }
#pragma omp taskwait
    for (i = 0; i < 64; i++) {
      free(NWS[i]);
    }
    free(NWS);
    __apac_result = nnc + nnl;
    goto __apac_exit;
  __apac_exit:;
  }
  return __apac_result;
}

char board[64][64];

void floorplan_init(char* filename) {
  int i;
  int j;
  inputFile = fopen(filename, "r");
  if (NULL == inputFile) {
    bots_message("Couldn't open %s file for reading\n", filename);
    exit(1);
  }
  read_inputs();
#pragma omp critical
  MIN_AREA = 64 * 64;
  for (i = 0; i < 64; i++)
    for (j = 0; j < 64; j++) board[i][j] = 0;
}

void compute_floorplan() {
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
  {
    int footprint[2];
    footprint[0] = 0;
    footprint[1] = 0;
    bots_message("Computing floorplan ");
#pragma omp task default(shared) depend(in : board, footprint, footprint[0], gcells) depend(inout : board[0], board[0][0], gcells[0])
    {
#pragma omp critical
      bots_number_of_tasks = add_cell(1, footprint, board, gcells);
    }
    bots_message(" completed!\n");
  __apac_exit:;
  }
}

void floorplan_end() { write_outputs(); }

int floorplan_verify() {
  if (solution != -1)
    return (MIN_AREA == solution ? 1 : 2);
  else
    return 0;
}
