#include "uts.hpp"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "app-desc.hpp"
#include "bots.h"

long long unsigned int nLeaves = 0;

int maxTreeDepth = 0;

double b_0 = 4.;

int rootId = 0;

int nonLeafBF = 4;

double nonLeafProb = 15. / 64.;

int computeGranularity = 1;

long long unsigned int exp_tree_size = 0;

int exp_tree_depth = 0;

long long unsigned int exp_num_leaves = 0;

double rng_toProb(int n) {
  if (n < 0) {
    printf("*** toProb: rand n = %d out of range\n", n);
  }
  return (n < 0 ? 0. : (double)n / 2147483648.);
}

void uts_initRoot(Node* root) {
  root->height = 0;
  root->numChildren = -1;
  rng_init(root->state.state, rootId);
  bots_message("Root node at %p\n", root);
}

int uts_numChildren_bin(Node* parent) {
  int v = rng_rand(parent->state.state);
  double d = rng_toProb(v);
  return (d < nonLeafProb ? nonLeafBF : 0);
}

int uts_numChildren(Node* parent) {
  int numChildren = 0;
  if (parent->height == 0)
    numChildren = (int)floor(b_0);
  else
    numChildren = uts_numChildren_bin(parent);
  if (parent->height == 0) {
    int rootBF = (int)ceil(b_0);
    if (numChildren > rootBF) {
      bots_debug("*** Number of children of root truncated from %d to %d\n", numChildren, rootBF);
      numChildren = rootBF;
    }
  } else {
    if (numChildren > 100) {
      bots_debug("*** Number of children truncated from %d to %d\n", numChildren, 100);
      numChildren = 100;
    }
  }
  return numChildren;
}

long long unsigned int parallel_uts(Node* root) {
  long long unsigned int __apac_result;
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
  {
    long long unsigned int num_nodes = 0;
#pragma omp task default(shared) depend(inout : root, root[0])
    root->numChildren = uts_numChildren(root);
    bots_message("Computing Unbalance Tree Search algorithm ");
#pragma omp task default(shared) depend(in : root) depend(inout : num_nodes, root[0])
    num_nodes = parTreeSearch(0, root, root->numChildren);
#pragma omp taskwait
    bots_message(" completed!");
    __apac_result = num_nodes;
    goto __apac_exit;
  __apac_exit:;
  }
  return __apac_result;
}

long long unsigned int parTreeSearch(int depth, Node* parent, int numChildren) {
  long long unsigned int __apac_result;
#pragma omp taskgroup
  {
    Node n[numChildren];
    Node* nodePtr;
    int i, j;
    long long unsigned int subtreesize = 1;
    long long unsigned int partialCount[numChildren];
    for (i = 0; i < numChildren; i++) {
      nodePtr = &n[i];
      nodePtr->height = parent->height + 1;
      for (j = 0; j < computeGranularity; j++) {
#pragma omp taskwait depend(inout : i, n, parent)
        rng_spawn(parent->state.state, nodePtr->state.state, i);
      }
#pragma omp task default(shared) depend(in : partialCount, depth) depend(inout : partialCount[i], n) firstprivate(i)
      {
        nodePtr->numChildren = uts_numChildren(nodePtr);
        partialCount[i] = parTreeSearch(depth + 1, nodePtr, nodePtr->numChildren);
      }
    }
#pragma omp taskwait
    for (i = 0; i < numChildren; i++) {
      subtreesize += partialCount[i];
    }
    __apac_result = subtreesize;
    goto __apac_exit;
  __apac_exit:;
  }
  return __apac_result;
}

void uts_read_file(char* filename) {
  FILE* fin;
  if ((fin = fopen(filename, "r")) == NULL) {
    bots_message("Could not open input file (%s)\n", filename);
    exit(-1);
  }
  fscanf(fin, "%lf %lf %d %d %d %llu %d %llu", &b_0, &nonLeafProb, &nonLeafBF, &rootId, &computeGranularity, &exp_tree_size, &exp_tree_depth, &exp_num_leaves);
  fclose(fin);
  computeGranularity = (1 > computeGranularity ? 1 : computeGranularity);
  bots_message("\n");
  bots_message("Root branching factor                = %f\n", b_0);
  bots_message("Root seed (0 <= 2^31)                = %d\n", rootId);
  bots_message("Probability of non-leaf node         = %f\n", nonLeafProb);
  bots_message("Number of children for non-leaf node = %d\n", nonLeafBF);
  bots_message("E(n)                                 = %f\n", (double)(nonLeafProb * nonLeafBF));
  bots_message("E(s)                                 = %f\n", (double)(1. / (1. - nonLeafProb * nonLeafBF)));
  bots_message("Compute granularity                  = %d\n", computeGranularity);
  bots_message("Random number generator              = ");
  rng_showtype();
}

void uts_show_stats() {
  int nPes = atoi(bots_resources);
  int chunkSize = 0;
  bots_message("\n");
  bots_message("Tree size                            = %llu\n", (long long unsigned int)bots_number_of_tasks);
  bots_message("Maximum tree depth                   = %d\n", maxTreeDepth);
  bots_message("Chunk size                           = %d\n", chunkSize);
  bots_message("Number of leaves                     = %llu (%.2f%%)\n", nLeaves, nLeaves / (float)bots_number_of_tasks * 100.);
  bots_message("Number of PE's                       = %.4d threads\n", nPes);
  bots_message("Wallclock time                       = %.3f sec\n", bots_time_program);
  bots_message("Overall performance                  = %.0f nodes/sec\n", bots_number_of_tasks / bots_time_program);
  bots_message("Performance per PE                   = %.0f nodes/sec\n", bots_number_of_tasks / bots_time_program / nPes);
}

int uts_check_result() {
  int answer = 1;
  if (bots_number_of_tasks != exp_tree_size) {
    answer = 2;
    bots_message("Incorrect tree size result (%llu instead of %llu).\n", bots_number_of_tasks, exp_tree_size);
  }
  return answer;
}