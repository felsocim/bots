#ifndef ATOMIC_HPP
#define ATOMIC_HPP

int atomic_compare(int * v1, int * v2);
int atomic_load(int * val);
void atomic_add(int * dest, int val);

#endif