#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "moldyn.hpp"
#include "bots.h"
#include "tools.hpp"

Cell cell_create(int capacity) {
  Cell new_cell;
  memset(&new_cell, 0, sizeof(Cell));

  new_cell.particles_symb = (Particle_symb *) malloc(capacity * sizeof(Particle_symb));
  if(!new_cell.particles_symb) { return new_cell; }

  new_cell.particles_forces = (Particle_forces *) malloc(capacity * sizeof(Particle_forces));
  if(!new_cell.particles_forces) { return new_cell; }

  new_cell.capacity = capacity;
  new_cell.size = 0L;

  return new_cell;
}

void cell_destroy(Cell* cell) {
  if(!cell) { return; }

  if(cell->particles_symb) {
    free(cell->particles_symb);
  }
  if(cell->particles_forces) {
    free(cell->particles_forces);
  }
  cell->capacity = 0;
  cell->size = 0;
}

void cell_add_particle(Cell * cell, Particle_symb particle_symb, Particle_forces particle_forces) {
  if(!cell) { return; }

  if(cell->size == cell->capacity) {
    int new_capacity = cell->capacity * 2;

    cell->particles_symb = (Particle_symb *)
      realloc(cell->particles_symb, new_capacity * sizeof(Particle_symb));
    if(!cell->particles_symb) { return; }

    cell->particles_forces = (Particle_forces *)
      realloc(cell->particles_forces, new_capacity * sizeof(Particle_forces));
    if(!cell->particles_forces) { return; }

    cell->capacity = new_capacity;
  }
  cell->particles_symb[cell->size] = particle_symb;
  cell->particles_forces[cell->size] = particle_forces;
  cell->size += 1;
}

void cell_self_compute(const Particle_symb* particles_symb, Particle_forces* particles_forces, const int size) {  
  for(int idxTgt = 0; idxTgt < size; idxTgt++) {
    for(int idxSrc = 0; idxSrc < idxTgt+1; idxSrc++) {
      const double dx = particles_symb[idxSrc].x - particles_symb[idxTgt].x;
      const double dy = particles_symb[idxSrc].y - particles_symb[idxTgt].y;
      const double dz = particles_symb[idxSrc].z - particles_symb[idxTgt].z;

      const double square_distance = (dx * dx + dy * dy + dz * dz + 0.00001);
      const double distance = sqrt(square_distance);
      const double cube_distance = square_distance * distance;

      const double coef = particles_symb[idxTgt].weight * particles_symb[idxSrc].weight / cube_distance;

      const double fx = dx * coef;
      const double fy = dy * coef;
      const double fz = dz * coef;

      particles_forces[idxTgt].fx += fx;
      particles_forces[idxTgt].fy += fy;
      particles_forces[idxTgt].fz += fz;

      particles_forces[idxSrc].fx -= fx;
      particles_forces[idxSrc].fy -= fy;
      particles_forces[idxSrc].fz -= fz;
    }
  }
}

void cell_neighbor_compute(const Particle_symb* particles_symb, Particle_forces* particles_forces, const int size,
                             const Particle_symb* particles_symbNeigh, const int sizeNeighbor) {  
  for(int idxTgt = 0; idxTgt < size; idxTgt++) {
    for(int idxSrc = 0; idxSrc < sizeNeighbor; idxSrc++) {
      const double dx = particles_symbNeigh[idxSrc].x - particles_symb[idxTgt].x;
      const double dy = particles_symbNeigh[idxSrc].y - particles_symb[idxTgt].y;
      const double dz = particles_symbNeigh[idxSrc].z - particles_symb[idxTgt].z;

      const double square_distance = (dx * dx + dy * dy + dz * dz + 0.00001);
      const double distance = sqrt(square_distance);
      const double cube_distance = square_distance * distance;

      const double coef = particles_symb[idxTgt].weight * particles_symbNeigh[idxSrc].weight / cube_distance;

      const double fx = dx * coef;
      const double fy = dy * coef;
      const double fz = dz * coef;

      particles_forces[idxTgt].fx += fx;
      particles_forces[idxTgt].fy += fy;
      particles_forces[idxTgt].fz += fz;
    }
  }
}

void cell_update(Particle_symb* particles_symb, Particle_forces* particles_forces, const int size,
                   double time_step){  
  for(int idxPart = 0; idxPart < size; idxPart++) {
    particles_symb[idxPart].vx +=
      (particles_forces[idxPart].fx / particles_symb[idxPart].weight) * time_step;
    particles_symb[idxPart].vy +=
      (particles_forces[idxPart].fy / particles_symb[idxPart].weight) * time_step;
    particles_symb[idxPart].vz +=
      (particles_forces[idxPart].fz / particles_symb[idxPart].weight) * time_step;
    
    particles_symb[idxPart].x += particles_symb[idxPart].vx * time_step;
    particles_symb[idxPart].y += particles_symb[idxPart].vy * time_step;
    particles_symb[idxPart].z += particles_symb[idxPart].vz * time_step;
  }
}

int grid_cell_idx_from_position(const double cell_width, const int nb_cells_per_dim, Particle_symb particle) {  
  int x = (int) (particle.x / cell_width),
      y = (int) (particle.y / cell_width),
      z = (int) (particle.z / cell_width);
  return (x * nb_cells_per_dim + y) * nb_cells_per_dim + z;
}

int grid_create(double box_width, double cell_width, const Particle_symb* src_particles_symb, const Particle_forces* src_particles_forces,
                const int src_size,
                 int** sizes, Particle_symb*** particles_symb, Particle_forces*** particles_forces) {  
  const int nb_cells_per_dim = (int) (box_width / cell_width);
  const int capacity =
    nb_cells_per_dim *
    nb_cells_per_dim *
    nb_cells_per_dim;

  *sizes = (int*)calloc(capacity, sizeof(int));
  *particles_symb = (Particle_symb**)calloc(capacity, sizeof(Particle_symb*));
  *particles_forces = (Particle_forces**)calloc(capacity, sizeof(Particle_forces*));

  for(int idxPart = 0; idxPart < src_size; idxPart++) {
    int cell_idx =
      grid_cell_idx_from_position(cell_width, nb_cells_per_dim, src_particles_symb[idxPart]);
    (*sizes)[cell_idx]++;
  }
  for(int idx_x = 0; idx_x < nb_cells_per_dim; idx_x++) {
    for(int idx_y = 0; idx_y < nb_cells_per_dim; idx_y++) {
      for(int idx_z = 0; idx_z < nb_cells_per_dim; idx_z++) {
        const int cell_idx = (idx_x * nb_cells_per_dim + idx_y)
              * nb_cells_per_dim + idx_z;
        (*particles_symb)[cell_idx] = (Particle_symb*)calloc((*sizes)[cell_idx], sizeof(Particle_symb));
        (*particles_forces)[cell_idx] = (Particle_forces*)calloc((*sizes)[cell_idx], sizeof(Particle_forces));
      }
    }
  }

  int* cpt = (int*)calloc(capacity, sizeof(int));

  for(int idxPart = 0; idxPart < src_size; idxPart++) {
    int cell_idx =
      grid_cell_idx_from_position(cell_width, nb_cells_per_dim, src_particles_symb[idxPart]);
    (*particles_symb)[cell_idx][cpt[cell_idx]] = src_particles_symb[idxPart];
    (*particles_forces)[cell_idx][cpt[cell_idx]] = src_particles_forces[idxPart];
    cpt[cell_idx]++;
  }
  free(cpt);
    
  return nb_cells_per_dim;
}

void grid_destroy(const int nb_cells_per_dim, int** sizes, Particle_symb*** particles_symb, Particle_forces*** particles_forces) {
  for(int idx_x = 0; idx_x < nb_cells_per_dim; idx_x++) {
    for(int idx_y = 0; idx_y < nb_cells_per_dim; idx_y++) {
      for(int idx_z = 0; idx_z < nb_cells_per_dim; idx_z++) {
        const int me = (idx_x * nb_cells_per_dim + idx_y)
              * nb_cells_per_dim + idx_z;
        free((*particles_symb)[me]);
        free((*particles_forces)[me]);
      }
    }
  }
  free(*sizes);
  free(*particles_symb);
  free(*particles_forces);
  *sizes = NULL;
  *particles_symb = NULL;
  *particles_forces = NULL;
}

void grid_compute(const int nb_cells_per_dim, int* sizes, Particle_symb** particles_symb, Particle_forces** particles_forces) {
  if(!sizes || !particles_symb || !particles_forces ) { return; }

  int me, neighbor;
  for(int idx_x = 0; idx_x < nb_cells_per_dim; idx_x++) {
    for(int idx_y = 0; idx_y < nb_cells_per_dim; idx_y++) {
      for(int idx_z = 0; idx_z < nb_cells_per_dim; idx_z++) {
        me = (idx_x * nb_cells_per_dim + idx_y)
              * nb_cells_per_dim + idx_z;
        
        cell_self_compute(particles_symb[me], particles_forces[me], sizes[me]);

        for(int idx_x_neigh = -1; idx_x_neigh <= 1; idx_x_neigh++) {
          for(int idx_y_neigh = -1; idx_y_neigh <= 1; idx_y_neigh++) {
            for(int idx_z_neigh = -1; idx_z_neigh <= 1; idx_z_neigh++) {
              neighbor = 
                (((idx_x + idx_x_neigh + nb_cells_per_dim) 
                    % nb_cells_per_dim) * nb_cells_per_dim
                + ((idx_y + idx_y_neigh + nb_cells_per_dim) 
                    % nb_cells_per_dim)) * nb_cells_per_dim
                + ((idx_z + idx_z_neigh + nb_cells_per_dim) 
                    % nb_cells_per_dim);   

              cell_neighbor_compute(particles_symb[me], particles_forces[me], sizes[me],
                                    particles_symb[neighbor], sizes[neighbor]);
            }
          }
        }
      }
    }
  }
}

void grid_update(const int nb_particles, const int nb_cells_per_dim, const double box_width, const double cell_width, double time_step,
                 int** sizes, Particle_symb*** particles_symb, Particle_forces*** particles_forces) {
  int* src_sizes = *sizes;

  const int capacity =
    nb_cells_per_dim *
    nb_cells_per_dim *
    nb_cells_per_dim;

  for(int idx_x = 0; idx_x < nb_cells_per_dim; idx_x++) {
    for(int idx_y = 0; idx_y < nb_cells_per_dim; idx_y++) {
      for(int idx_z = 0; idx_z < nb_cells_per_dim; idx_z++) {
        int cell_idx = (idx_x * nb_cells_per_dim + idx_y)
              * nb_cells_per_dim + idx_z;
        for(int idxPart = 0; idxPart < src_sizes[cell_idx]; idxPart++) {
          // Update position
          (*particles_symb)[cell_idx][idxPart].vx +=
            ((*particles_forces)[cell_idx][idxPart].fx / (*particles_symb)[cell_idx][idxPart].weight) * time_step;
          (*particles_symb)[cell_idx][idxPart].vy +=
            ((*particles_forces)[cell_idx][idxPart].fy / (*particles_symb)[cell_idx][idxPart].weight) * time_step;
          (*particles_symb)[cell_idx][idxPart].vz +=
            ((*particles_forces)[cell_idx][idxPart].fz / (*particles_symb)[cell_idx][idxPart].weight) * time_step;
          (*particles_symb)[cell_idx][idxPart].x += (*particles_symb)[cell_idx][idxPart].vx * time_step;
          (*particles_symb)[cell_idx][idxPart].y += (*particles_symb)[cell_idx][idxPart].vy * time_step;
          (*particles_symb)[cell_idx][idxPart].z += (*particles_symb)[cell_idx][idxPart].vz * time_step;

          while((*particles_symb)[cell_idx][idxPart].x < 0){
            (*particles_symb)[cell_idx][idxPart].x += box_width;
          }
          while((*particles_symb)[cell_idx][idxPart].x >= box_width){
            (*particles_symb)[cell_idx][idxPart].x -= box_width;
          }
          while((*particles_symb)[cell_idx][idxPart].y < 0){
            (*particles_symb)[cell_idx][idxPart].y += box_width;
          }
          while((*particles_symb)[cell_idx][idxPart].y >= box_width){
            (*particles_symb)[cell_idx][idxPart].y -= box_width;
          }
          while((*particles_symb)[cell_idx][idxPart].z < 0){
            (*particles_symb)[cell_idx][idxPart].z += box_width;
          }
          while((*particles_symb)[cell_idx][idxPart].z >= box_width){
            (*particles_symb)[cell_idx][idxPart].z -= box_width;
          }

          // Compute new target cell
          int up_cell_idx =
            grid_cell_idx_from_position(cell_width, nb_cells_per_dim, (*particles_symb)[cell_idx][idxPart]);
          (*sizes)[up_cell_idx]++;
        }
      }
    }
  }


  Particle_symb** src_particles_symb = *particles_symb;
  Particle_forces** src_particles_forces = *particles_forces;

  *sizes = (int*)calloc(capacity, sizeof(int));
  *particles_symb = (Particle_symb**)calloc(capacity, sizeof(Particle_symb*));
  *particles_forces = (Particle_forces**)calloc(capacity, sizeof(Particle_forces*));

  for(int idx_x = 0; idx_x < nb_cells_per_dim; idx_x++) {
    for(int idx_y = 0; idx_y < nb_cells_per_dim; idx_y++) {
      for(int idx_z = 0; idx_z < nb_cells_per_dim; idx_z++) {
        const int cell_idx = (idx_x * nb_cells_per_dim + idx_y)
              * nb_cells_per_dim + idx_z;
        (*particles_symb)[cell_idx] = (Particle_symb*)calloc((*sizes)[cell_idx], sizeof(Particle_symb));
        (*particles_forces)[cell_idx] = (Particle_forces*)calloc((*sizes)[cell_idx], sizeof(Particle_forces));
      }
    }
  }

  int* cpt = (int*)calloc(capacity, sizeof(int));

  for(int idx_x = 0; idx_x < nb_cells_per_dim; idx_x++) {
    for(int idx_y = 0; idx_y < nb_cells_per_dim; idx_y++) {
      for(int idx_z = 0; idx_z < nb_cells_per_dim; idx_z++) {
        const int cell_idx = (idx_x * nb_cells_per_dim + idx_y)
              * nb_cells_per_dim + idx_z;
        for(int idxPart = 0; idxPart < src_sizes[cell_idx]; idxPart++) {
          int up_cell_idx =
            grid_cell_idx_from_position(cell_width, nb_cells_per_dim, src_particles_symb[cell_idx][idxPart]);
          (*particles_symb)[up_cell_idx][cpt[up_cell_idx]] = src_particles_symb[cell_idx][idxPart];
          (*particles_forces)[up_cell_idx][cpt[up_cell_idx]] = src_particles_forces[cell_idx][idxPart];
          cpt[up_cell_idx]++;
        }
      }
    }
  }
  free(cpt);

  grid_destroy(nb_cells_per_dim, &src_sizes, &src_particles_symb, &src_particles_forces);
}

void fill_cell_with_rand_particles(Cell* inCell, double box_width, int size){
  initialize_random_number_generator(box_width);

  for(int idx = 0 ; idx < size ; idx++){
    Particle_symb particle;
    particle.x = random_number();
    particle.y = random_number();
    particle.z = random_number();
    particle.weight = 1.0;
    particle.vx = 0.0;
    particle.vy = 0.0;
    particle.vz = 0.0;
    Particle_forces particle_forces;
    particle_forces.fx = random_number();
    particle_forces.fy = random_number();
    particle_forces.fz = random_number();
    cell_add_particle(inCell, particle, particle_forces);
  }
}

void compute(
  const int steps, const int nb_particles, const int nb_cells_per_dim,
  const double box_width, const double cell_width, double time_step,
  int** sizes, Particle_symb*** particles_symb, 
  Particle_forces*** particles_forces
) {
  for(int idx = 0; idx < steps; idx++) {
    grid_compute(nb_cells_per_dim, *sizes, *particles_symb, *particles_forces);
    grid_update(
      nb_particles, nb_cells_per_dim, box_width, cell_width, time_step, sizes,
      particles_symb, particles_forces
    );
  }
}

static int check_symb(Particle_symb p1, Particle_symb p2) {
  return (p1.x == p2.x) && (p1.y == p2.y) && (p1.z == p2.z) && 
    (p1.weight == p2.weight) && (p1.vx == p2.vx) && (p1.vy == p2.vy) && 
    (p1.vz == p2.vz);
}

static int check_force(Particle_forces p1, Particle_forces p2) {
  return (p1.fx == p2.fx) && (p1.fy == p2.fy) && (p1.fz == p2.fz);
}

int check(
  const int nb_cells_per_dim_seq, const int nb_cells_per_dim_par, int* sizes, 
  Particle_symb** particles_symb_seq, Particle_forces** particles_forces_seq, 
  Particle_symb** particles_symb_par, Particle_forces** particles_forces_par
) {
  if(nb_cells_per_dim_par != nb_cells_per_dim_seq)
    return BOTS_RESULT_UNSUCCESSFUL;
  const int nb_cells_per_dim = nb_cells_per_dim_par;
  for(int idx_x = 0; idx_x < nb_cells_per_dim; idx_x++) {
    for(int idx_y = 0; idx_y < nb_cells_per_dim; idx_y++) {
      for(int idx_z = 0; idx_z < nb_cells_per_dim; idx_z++) {
        const int me = (idx_x * nb_cells_per_dim + idx_y)
              * nb_cells_per_dim + idx_z;
        Particle_symb * particle_sym_seq = particles_symb_seq[me];
        Particle_symb * particle_sym_par = particles_symb_par[me];
        Particle_forces * particle_forces_seq = particles_forces_seq[me];
        Particle_forces * particle_forces_par = particles_forces_par[me];
        const int size = sizes[me];
        for(int idx = 0; idx < size; idx++) {
          if(!check_symb(particle_sym_par[idx], particle_sym_seq[idx])) {
            return BOTS_RESULT_UNSUCCESSFUL;
          }
          if(!check_force(particle_forces_par[idx], particle_forces_seq[idx])) {
            return BOTS_RESULT_UNSUCCESSFUL;
          }
        }
      }
    }
  }

  return BOTS_RESULT_SUCCESSFUL;
}