#ifndef MOLECULAR_DYN_H
#define MOLECULAR_DYN_H

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
void cell_destroy(Cell*);
void cell_add_particle(Cell*, Particle_symb, Particle_forces);
void cell_self_compute(const Particle_symb*, Particle_forces*, const int);
void cell_neighbor_compute(
  const Particle_symb*, Particle_forces*, const int, const Particle_symb*, 
  const int
);
void cell_update(Particle_symb*, Particle_forces*, const int, double);
int grid_cell_idx_from_position(const double, const int, Particle_symb);
int grid_create(
  double, double, const Particle_symb*, const Particle_forces*, const int,
  int**, Particle_symb***, Particle_forces***
);
void grid_destroy(const int, int**, Particle_symb***, Particle_forces***);
void grid_compute(const int, int*, Particle_symb**, Particle_forces**);
void grid_update(
  const int, const int, const double, const double, double, int**,
  Particle_symb***, Particle_forces***
);
void fill_cell_with_rand_particles(Cell*, double, int);
void compute(
  const int, const int, const int, const double, const double, double, int**,
  Particle_symb***, Particle_forces***
);
int check(
  const int, int*, Particle_symb**, Particle_forces**, Particle_symb**, 
  Particle_forces**
);

#endif