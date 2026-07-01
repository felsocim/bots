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

#include "serial-app.h"

#define BOTS_APP_NAME "Molecular Dyn"
#define BOTS_APP_PARAMETERS_DESC "size=%d, time_step=%d, steps=%d"
#define BOTS_APP_PARAMETERS_LIST ,bots_arg_size,bots_arg_size_1,bots_arg_size_2

#define BOTS_APP_USES_ARG_SIZE
#define BOTS_APP_DEF_ARG_SIZE 20000
#define BOTS_APP_DESC_ARG_SIZE "Size"

#define BOTS_APP_USES_ARG_SIZE_1
#define BOTS_APP_DEF_ARG_SIZE_1 1
#define BOTS_APP_DESC_ARG_SIZE_1 "Time step (s)"

#define BOTS_APP_USES_ARG_SIZE_2
#define BOTS_APP_DEF_ARG_SIZE_2 50
#define BOTS_APP_DESC_ARG_SIZE_2 "Number of steps"

typedef struct {
  double x, y, z;
  double weight;
  double vx, vy, vz;
} Particle_symb;

typedef struct {
  double fx, fy, fz;
} Particle_forces;

typedef struct {
  Particle_symb * particles_symb;
  Particle_forces * particles_forces;
  int size, capacity;
} Cell;

Cell cell_create(int);
void fill_cell_with_rand_particles(Cell*, double, int);
int grid_create(
  double, double, const Particle_symb*, const Particle_forces*, const int,
  int**, Particle_symb***, Particle_forces***
);
void cell_destroy(Cell*);
void grid_compute(const int, int*, Particle_symb**, Particle_forces**);
void grid_update(
  const int, const double, const double, double, int**,
  Particle_symb***, Particle_forces***
);
void grid_destroy(const int, int**, Particle_symb***, Particle_forces***);
void compute(
  const int, const int, const double, const double, double, int**,
  Particle_symb***, Particle_forces***
);

#define BOTS_APP_INIT\
  const double box_width = 1;\
  const double cell_width = 0.20;\
  const int steps = bots_arg_size_2;\
  const double time_step =  (double) bots_arg_size_1 / 1000.;\
  const int size = bots_arg_size;\
  int nb_cells_per_dim = -1;\
  Cell cell = cell_create(size);\
  fill_cell_with_rand_particles(&cell, box_width, size);\
  int* sizes = NULL;\
  Particle_symb** particles_symb = NULL;\
  Particle_forces** particles_forces = NULL;

#define KERNEL_INIT\
  nb_cells_per_dim =\
    grid_create(\
      box_width, cell_width,\
      cell.particles_symb, cell.particles_forces, cell.size,\
      &sizes, &particles_symb, &particles_forces\
    )
#define KERNEL_CALL compute(steps, nb_cells_per_dim, box_width,\
  cell_width, time_step, &sizes, &particles_symb, &particles_forces)

#define KERNEL_CHECK BOTS_RESULT_NA

#define BOTS_APP_FINI\
  grid_destroy(\
    nb_cells_per_dim, &sizes, &particles_symb, &particles_forces\
  );\
  cell_destroy(&cell)
