#include <libgen.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "alignment.hpp"
#include "bots.h"
#include "param.hpp"
#include "sequence.hpp"
const double __apac_cutoff = getenv("APAC_EXECUTION_TIME_CUTOFF") ? atof(getenv("APAC_EXECUTION_TIME_CUTOFF")) : 2.22100e-6;

const static int __apac_depth_infinite = getenv("APAC_TASK_DEPTH_INFINITE") ? 1 : 0;

const static int __apac_depth_max = getenv("APAC_TASK_DEPTH_MAX") ? atoi(getenv("APAC_TASK_DEPTH_MAX")) : 5;

int __apac_depth = 0;

#pragma omp threadprivate(__apac_depth)

int ktup;

int window;

int signif;

int prot_ktup;

int prot_window;

int prot_signif;

int gap_pos1;

int gap_pos2;

int nseqs;

int max_aa;

int* seqlen_array;

int def_aa_xref[32 + 1];

int* bench_output;

int* seq_output;

double gap_open;

double gap_extend;

double prot_gap_open;

double prot_gap_extend;

double pw_go_penalty;

double pw_ge_penalty;

double prot_pw_go_penalty;

double prot_pw_ge_penalty;

char** args;

char** names;

char** seq_array;

double gap_open_scale;

double gap_extend_scale;

int dnaFlag = 0;

int clustalw = 0;

void del(int k, int* print_ptr, int* last_print, int* displ) {
  if (*last_print < 0)
    *last_print = displ[*print_ptr - 1] -= k;
  else
    *last_print = displ[(*print_ptr)++] = -k;
}

void add(int v, int* print_ptr, int* last_print, int* displ) {
  if (*last_print < 0) {
    displ[*print_ptr - 1] = v;
    displ[(*print_ptr)++] = *last_print;
  } else {
    *last_print = displ[(*print_ptr)++] = v;
  }
}

int calc_score(int iat, int jat, int v1, int v2, int seq1, int seq2, int** matrix) {
  int i;
  int j;
  int ipos;
  int jpos;
  ipos = v1 + iat;
  jpos = v2 + jat;
  i = seq_array[seq1][ipos];
  j = seq_array[seq2][jpos];
  return matrix[i][j];
}

int get_matrix(int* matptr, int* xref, int scale, int** matrix, int* mat_avscore) {
  int gg_score = 0;
  int gr_score = 0;
  int i;
  int j;
  int k;
  int ti;
  int tj;
  int ix;
  int av1;
  int av2;
  int av3;
  int min;
  int max;
  int maxres;
  for (i = 0; i <= max_aa; i++)
    for (j = 0; j <= max_aa; j++) matrix[i][j] = 0;
  ix = 0;
  maxres = 0;
  for (i = 0; i <= max_aa; i++) {
    ti = xref[i];
    for (j = 0; j <= i; j++) {
      tj = xref[j];
      if (ti != -1 && tj != -1) {
        k = matptr[ix];
        if (ti == tj) {
          matrix[ti][ti] = k * scale;
          maxres++;
        } else {
          matrix[ti][tj] = k * scale;
          matrix[tj][ti] = k * scale;
        }
        ix++;
      }
    }
  }
  maxres--;
  av1 = av2 = av3 = 0;
  for (i = 0; i <= max_aa; i++) {
    for (j = 0; j <= i; j++) {
      av1 += matrix[i][j];
      if (i == j)
        av2 += matrix[i][j];
      else
        av3 += matrix[i][j];
    }
  }
  av1 /= maxres * maxres / 2;
  av2 /= maxres;
  av3 /= (int)((double)(maxres * maxres - maxres) / 2);
  *mat_avscore = -av3;
  min = max = matrix[0][0];
  for (i = 0; i <= max_aa; i++)
    for (j = 1; j <= i; j++) {
      if (matrix[i][j] < min) min = matrix[i][j];
      if (matrix[i][j] > max) max = matrix[i][j];
    }
  for (i = 0; i < gap_pos1; i++) {
    matrix[i][gap_pos1] = gr_score;
    matrix[gap_pos1][i] = gr_score;
    matrix[i][gap_pos2] = gr_score;
    matrix[gap_pos2][i] = gr_score;
  }
  matrix[gap_pos1][gap_pos1] = gg_score;
  matrix[gap_pos2][gap_pos2] = gg_score;
  matrix[gap_pos2][gap_pos1] = gg_score;
  matrix[gap_pos1][gap_pos2] = gg_score;
  maxres += 2;
  return maxres;
}

void forward_pass(char* ia, char* ib, int n, int m, int* se1, int* se2, int* maxscore, int g, int gh, int** matrix) {
  int i;
  int j;
  int f;
  int p;
  int t;
  int hh;
  int HH[5000];
  int DD[5000];
  *maxscore = 0;
  *se1 = *se2 = 0;
  for (i = 0; i <= m; i++) {
    HH[i] = 0;
    DD[i] = -g;
  }
  for (i = 1; i <= n; i++) {
    hh = p = 0;
    f = -g;
    for (j = 1; j <= m; j++) {
      f -= gh;
      t = hh - g - gh;
      if (f < t) f = t;
      DD[j] -= gh;
      t = HH[j] - g - gh;
      if (DD[j] < t) DD[j] = t;
      hh = p + matrix[(int)ia[i]][(int)ib[j]];
      if (hh < f) hh = f;
      if (hh < DD[j]) hh = DD[j];
      if (hh < 0) hh = 0;
      p = HH[j];
      HH[j] = hh;
      if (hh > *maxscore) {
        *maxscore = hh;
        *se1 = i;
        *se2 = j;
      }
    }
  }
}

void reverse_pass(char* ia, char* ib, int se1, int se2, int* sb1, int* sb2, int maxscore, int g, int gh, int** matrix) {
  int i;
  int j;
  int f;
  int p;
  int t;
  int hh;
  int cost;
  int HH[5000];
  int DD[5000];
  cost = 0;
  *sb1 = *sb2 = 1;
  for (i = se2; i > 0; i--) {
    HH[i] = -1;
    DD[i] = -1;
  }
  for (i = se1; i > 0; i--) {
    hh = f = -1;
    if (i == se1)
      p = 0;
    else
      p = -1;
    for (j = se2; j > 0; j--) {
      f -= gh;
      t = hh - g - gh;
      if (f < t) f = t;
      DD[j] -= gh;
      t = HH[j] - g - gh;
      if (DD[j] < t) DD[j] = t;
      hh = p + matrix[(int)ia[i]][(int)ib[j]];
      if (hh < f) hh = f;
      if (hh < DD[j]) hh = DD[j];
      p = HH[j];
      HH[j] = hh;
      if (hh > cost) {
        cost = hh;
        *sb1 = i;
        *sb2 = j;
        if (cost >= maxscore) break;
      }
    }
    if (cost >= maxscore) break;
  }
}

int diff(int A, int B, int M, int N, int tb, int te, int* print_ptr, int* last_print, int* displ, int seq1, int seq2, int g, int gh, int** matrix) {
  int __apac_depth_local = __apac_depth;
  int __apac_depth_ok = __apac_depth_infinite || __apac_depth_local < __apac_depth_max;
  if (__apac_depth_ok) {
    int __apac_result;
#pragma omp taskgroup
    {
      int i;
      int j;
      int f;
      int e;
      int s;
      int t;
      int hh;
      int midi;
      int midj;
      int midh;
      int type;
      int HH[5000];
      int DD[5000];
      int RR[5000];
      int SS[5000];
      if (N <= 0) {
        if (M > 0) {
          del(M, print_ptr, last_print, displ);
        }
        __apac_result = -((int)((M <= 0 ? 0 : tb + gh * M)));
        goto __apac_exit;
      }
      if (M <= 1) {
        if (M <= 0) {
          add(N, print_ptr, last_print, displ);
          __apac_result = -((int)((N <= 0 ? 0 : tb + gh * N)));
          goto __apac_exit;
        }
        midh = -(tb + gh) - ((N <= 0 ? 0 : te + gh * N));
        hh = -(te + gh) - ((N <= 0 ? 0 : tb + gh * N));
        if (hh > midh) {
          midh = hh;
        }
        midj = 0;
        for (j = 1; j <= N; j++) {
          hh = calc_score(1, j, A, B, seq1, seq2, matrix) - ((N - j <= 0 ? 0 : te + gh * (N - j))) - ((j - 1 <= 0 ? 0 : tb + gh * (j - 1)));
          if (hh > midh) {
            midh = hh;
            midj = j;
          }
        }
        if (midj == 0) {
          del(1, print_ptr, last_print, displ);
          add(N, print_ptr, last_print, displ);
        } else {
          if (midj > 1) {
            add(midj - 1, print_ptr, last_print, displ);
          }
          displ[(*print_ptr)++] = *last_print = 0;
          if (midj < N) {
            add(N - midj, print_ptr, last_print, displ);
          }
        }
        __apac_result = midh;
        goto __apac_exit;
      }
      midi = M / 2;
      HH[0] = 0.;
      t = -tb;
      for (j = 1; j <= N; j++) {
        HH[j] = t = t - gh;
        DD[j] = t - g;
      }
      t = -tb;
      for (i = 1; i <= midi; i++) {
        s = HH[0];
        HH[0] = hh = t = t - gh;
        f = t - g;
        for (j = 1; j <= N; j++) {
          if ((hh = hh - g - gh) > (f = f - gh)) {
            f = hh;
          }
          if ((hh = HH[j] - g - gh) > (e = DD[j] - gh)) {
            e = hh;
          }
          hh = s + calc_score(i, j, A, B, seq1, seq2, matrix);
          if (f > hh) {
            hh = f;
          }
          if (e > hh) {
            hh = e;
          }
          s = HH[j];
          HH[j] = hh;
          DD[j] = e;
        }
      }
      DD[0] = HH[0];
      RR[N] = 0;
      t = -te;
      for (j = N - 1; j >= 0; j--) {
        RR[j] = t = t - gh;
        SS[j] = t - g;
      }
      t = -te;
      for (i = M - 1; i >= midi; i--) {
        s = RR[N];
        RR[N] = hh = t = t - gh;
        f = t - g;
        for (j = N - 1; j >= 0; j--) {
          if ((hh = hh - g - gh) > (f = f - gh)) {
            f = hh;
          }
          if ((hh = RR[j] - g - gh) > (e = SS[j] - gh)) {
            e = hh;
          }
          hh = s + calc_score(i + 1, j + 1, A, B, seq1, seq2, matrix);
          if (f > hh) {
            hh = f;
          }
          if (e > hh) {
            hh = e;
          }
          s = RR[j];
          RR[j] = hh;
          SS[j] = e;
        }
      }
      SS[N] = RR[N];
      midh = HH[0] + RR[0];
      midj = 0;
      type = 1;
      for (j = 0; j <= N; j++) {
        hh = HH[j] + RR[j];
        if (hh >= midh) {
          if (hh > midh || HH[j] != DD[j] && RR[j] == SS[j]) {
            midh = hh;
            midj = j;
          }
        }
      }
      for (j = N; j >= 0; j--) {
        hh = DD[j] + SS[j] + g;
        if (hh > midh) {
          midh = hh;
          midj = j;
          type = 2;
        }
      }
      if (type == 1) {
#pragma omp task default(shared) depend(in : A, B, M, N, displ, g, gh, last_print, matrix, matrix[0], matrix[0][0], midi, midj, print_ptr, seq1, seq2, tb, te) depend(inout : displ[0], last_print[0], print_ptr[0]) firstprivate(__apac_depth_local) if (__apac_depth_ok && 0.000240472025916 + seq2 * seq1 * 1.00550419223e-05 > __apac_cutoff)
        {
          if (__apac_depth_ok) {
            __apac_depth = __apac_depth_local + 1;
          }
          diff(A, B, midi, midj, tb, g, print_ptr, last_print, displ, seq1, seq2, g, gh, matrix);
          diff(A + midi, B + midj, M - midi, N - midj, g, te, print_ptr, last_print, displ, seq1, seq2, g, gh, matrix);
        }
      } else {
#pragma omp task default(shared) depend(in : A, B, M, N, displ, g, gh, last_print, matrix, matrix[0], matrix[0][0], midi, midj, print_ptr, seq1, seq2, tb, te) depend(inout : displ[0], last_print[0], print_ptr[0]) firstprivate(__apac_depth_local) if (__apac_depth_ok && 0.00123006849756 + seq2 * (A + midi + 1) * 8.109146256e-06 > __apac_cutoff)
        {
          if (__apac_depth_ok) {
            __apac_depth = __apac_depth_local + 1;
          }
          diff(A, B, midi - 1, midj, tb, 0., print_ptr, last_print, displ, seq1, seq2, g, gh, matrix);
          del(2, print_ptr, last_print, displ);
          diff(A + midi + 1, B + midj, M - midi - 1, N - midj, 0., te, print_ptr, last_print, displ, seq1, seq2, g, gh, matrix);
        }
      }
      __apac_result = midh;
      goto __apac_exit;
    __apac_exit:;
    }
    return __apac_result;
  } else {
    return diff_seq(A, B, M, N, tb, te, print_ptr, last_print, displ, seq1, seq2, g, gh, matrix);
  }
}

int diff_seq(int A, int B, int M, int N, int tb, int te, int* print_ptr, int* last_print, int* displ, int seq1, int seq2, int g, int gh, int** matrix) {
  int i;
  int j;
  int f;
  int e;
  int s;
  int t;
  int hh;
  int midi;
  int midj;
  int midh;
  int type;
  int HH[5000];
  int DD[5000];
  int RR[5000];
  int SS[5000];
  if (N <= 0) {
    if (M > 0) del(M, print_ptr, last_print, displ);
    return -((int)((M <= 0 ? 0 : tb + gh * M)));
  }
  if (M <= 1) {
    if (M <= 0) {
      add(N, print_ptr, last_print, displ);
      return -((int)((N <= 0 ? 0 : tb + gh * N)));
    }
    midh = -(tb + gh) - ((N <= 0 ? 0 : te + gh * N));
    hh = -(te + gh) - ((N <= 0 ? 0 : tb + gh * N));
    if (hh > midh) midh = hh;
    midj = 0;
    for (j = 1; j <= N; j++) {
      hh = calc_score(1, j, A, B, seq1, seq2, matrix) - ((N - j <= 0 ? 0 : te + gh * (N - j))) - ((j - 1 <= 0 ? 0 : tb + gh * (j - 1)));
      if (hh > midh) {
        midh = hh;
        midj = j;
      }
    }
    if (midj == 0) {
      del(1, print_ptr, last_print, displ);
      add(N, print_ptr, last_print, displ);
    } else {
      if (midj > 1) add(midj - 1, print_ptr, last_print, displ);
      displ[(*print_ptr)++] = *last_print = 0;
      if (midj < N) add(N - midj, print_ptr, last_print, displ);
    }
    return midh;
  }
  midi = M / 2;
  HH[0] = 0.;
  t = -tb;
  for (j = 1; j <= N; j++) {
    HH[j] = t = t - gh;
    DD[j] = t - g;
  }
  t = -tb;
  for (i = 1; i <= midi; i++) {
    s = HH[0];
    HH[0] = hh = t = t - gh;
    f = t - g;
    for (j = 1; j <= N; j++) {
      if ((hh = hh - g - gh) > (f = f - gh)) f = hh;
      if ((hh = HH[j] - g - gh) > (e = DD[j] - gh)) e = hh;
      hh = s + calc_score(i, j, A, B, seq1, seq2, matrix);
      if (f > hh) hh = f;
      if (e > hh) hh = e;
      s = HH[j];
      HH[j] = hh;
      DD[j] = e;
    }
  }
  DD[0] = HH[0];
  RR[N] = 0;
  t = -te;
  for (j = N - 1; j >= 0; j--) {
    RR[j] = t = t - gh;
    SS[j] = t - g;
  }
  t = -te;
  for (i = M - 1; i >= midi; i--) {
    s = RR[N];
    RR[N] = hh = t = t - gh;
    f = t - g;
    for (j = N - 1; j >= 0; j--) {
      if ((hh = hh - g - gh) > (f = f - gh)) f = hh;
      if ((hh = RR[j] - g - gh) > (e = SS[j] - gh)) e = hh;
      hh = s + calc_score(i + 1, j + 1, A, B, seq1, seq2, matrix);
      if (f > hh) hh = f;
      if (e > hh) hh = e;
      s = RR[j];
      RR[j] = hh;
      SS[j] = e;
    }
  }
  SS[N] = RR[N];
  midh = HH[0] + RR[0];
  midj = 0;
  type = 1;
  for (j = 0; j <= N; j++) {
    hh = HH[j] + RR[j];
    if (hh >= midh)
      if (hh > midh || HH[j] != DD[j] && RR[j] == SS[j]) {
        midh = hh;
        midj = j;
      }
  }
  for (j = N; j >= 0; j--) {
    hh = DD[j] + SS[j] + g;
    if (hh > midh) {
      midh = hh;
      midj = j;
      type = 2;
    }
  }
  if (type == 1) {
    diff_seq(A, B, midi, midj, tb, g, print_ptr, last_print, displ, seq1, seq2, g, gh, matrix);
    diff_seq(A + midi, B + midj, M - midi, N - midj, g, te, print_ptr, last_print, displ, seq1, seq2, g, gh, matrix);
  } else {
    diff_seq(A, B, midi - 1, midj, tb, 0., print_ptr, last_print, displ, seq1, seq2, g, gh, matrix);
    del(2, print_ptr, last_print, displ);
    diff_seq(A + midi + 1, B + midj, M - midi - 1, N - midj, 0., te, print_ptr, last_print, displ, seq1, seq2, g, gh, matrix);
  }
  return midh;
}

double tracepath(int tsb1, int tsb2, int* print_ptr, int* displ, int seq1, int seq2) {
  int i;
  int k;
  int i1 = tsb1;
  int i2 = tsb2;
  int pos = 0;
  int count = 0;
  for (i = 1; i <= *print_ptr - 1; ++i) {
    if (displ[i] == 0) {
      char c1 = seq_array[seq1][i1];
      char c2 = seq_array[seq2][i2];
      if (c1 != gap_pos1 && c1 != gap_pos2 && c1 == c2) count++;
      ++i1;
      ++i2;
      ++pos;
    } else if ((k = displ[i]) > 0) {
      i2 += k;
      pos += k;
    } else {
      i1 -= k;
      pos -= k;
    }
  }
  return 100. * (double)count;
}

int pairalign() {
  int __apac_depth_local = __apac_depth;
  int __apac_depth_ok = __apac_depth_infinite || __apac_depth_local < __apac_depth_max;
  if (__apac_depth_ok) {
    int __apac_result;
#pragma omp taskgroup
    {
      int i;
      int n;
      int m;
      int si;
      int sj;
      int len1;
      int len2;
      int maxres;
      int mat_avscore;
      double gg;
      double mm_score;
      int** matrix;
      int* mat_xref;
      int* matptr;
      matptr = gon250mt;
      mat_xref = def_aa_xref;
      matrix = (int**)malloc(32 * sizeof(int*));
      for (int a = 0; a < 32; a++) {
        matrix[a] = (int*)malloc(32 * sizeof(int));
      }
      maxres = get_matrix(matptr, mat_xref, 10, matrix, &mat_avscore);
      if (maxres == 0) {
        __apac_result = -1;
        goto __apac_exit;
      }
      bots_message("Start aligning ");
      for (si = 0; si < nseqs; si++) {
        n = seqlen_array[si + 1];
        len1 = 0;
        for (i = 1; i <= n; i++) {
          char c = seq_array[si + 1][i];
          if (c != gap_pos1 && c != gap_pos2) {
            len1++;
          }
        }
        for (sj = si + 1; sj < nseqs; sj++) {
          m = seqlen_array[sj + 1];
          if (n == 0 || m == 0) {
            bench_output[si * nseqs + sj] = (int)1.;
          } else {
            int* se1 = new int();
            int* se2 = new int();
            int* sb1 = new int();
            int* sb2 = new int();
            int* maxscore = new int();
            int* seq1 = new int();
            int* seq2 = new int();
            int* g = new int();
            int* gh = new int();
            int* displ = new int[2 * 5000 + 1]();
            int* print_ptr = new int();
            int* last_print = new int();
            len2 = 0;
            for (i = 1; i <= m; i++) {
              char c = seq_array[sj + 1][i];
              if (c != gap_pos1 && c != gap_pos2) {
                len2++;
              }
            }
            if (dnaFlag == 1) {
#pragma omp taskwait depend(in : gap_open_scale, pw_go_penalty) depend(inout : g[0])
              *g = (int)(2 * 100 * pw_go_penalty * gap_open_scale);
#pragma omp taskwait depend(in : gap_extend_scale, pw_ge_penalty) depend(inout : gh[0])
              *gh = (int)(100 * pw_ge_penalty * gap_extend_scale);
            } else {
#pragma omp taskwait depend(in : gap_open_scale, m, mat_avscore, n, pw_go_penalty) depend(inout : g[0], gg)
              gg = pw_go_penalty + log((double)((n < m ? n : m)));
              *g = (int)((mat_avscore <= 0 ? 2 * 100 * gg : 2 * mat_avscore * gg * gap_open_scale));
#pragma omp taskwait depend(in : pw_ge_penalty) depend(inout : gh[0])
              *gh = (int)(100 * pw_ge_penalty);
            }
#pragma omp taskwait depend(in : si) depend(inout : seq1[0])
            *seq1 = si + 1;
#pragma omp taskwait depend(in : sj) depend(inout : seq2[0])
            *seq2 = sj + 1;
#pragma omp taskwait depend(in : g[0], gh[0], m, matrix, matrix[0], matrix[0][0], n, seq1[0], seq2[0], seq_array, seq_array[*seq1], seq_array[*seq1][0], seq_array[*seq2], seq_array[*seq2][0]) depend(inout : maxscore[0], se1[0], se2[0])
            forward_pass(&seq_array[*seq1][0], &seq_array[*seq2][0], n, m, se1, se2, maxscore, *g, *gh, matrix);
#pragma omp taskwait depend(in : g[0], gh[0], matrix, matrix[0], matrix[0][0], maxscore[0], se1[0], se2[0], seq1[0], seq2[0], seq_array, seq_array[*seq1], seq_array[*seq1][0], seq_array[*seq2], seq_array[*seq2][0]) depend(inout : sb1[0], sb2[0])
            reverse_pass(&seq_array[*seq1][0], &seq_array[*seq2][0], *se1, *se2, sb1, sb2, *maxscore, *g, *gh, matrix);
#pragma omp taskwait depend(inout : print_ptr[0])
            *print_ptr = 1;
#pragma omp taskwait depend(inout : last_print[0])
            *last_print = 0;
#pragma omp task default(shared) depend(in : displ, g[0], gh[0], matrix, matrix[0], matrix[0][0], sb1[0], sb2[0], se1[0], se2[0], seq1[0], seq2[0]) depend(inout : displ[0], last_print[0], mm_score, print_ptr[0]) firstprivate(__apac_depth_local) if (__apac_depth_ok && -0.000364026370248 + *sb2 * *sb1 * 5.25870589025e-06 > __apac_cutoff) firstprivate(se1) firstprivate(se2) firstprivate(sb1) firstprivate(sb2) firstprivate(seq1) firstprivate(seq2) firstprivate(g) firstprivate(gh) firstprivate(print_ptr) firstprivate(last_print)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              diff(*sb1 - 1, *sb2 - 1, *se1 - *sb1 + 1, *se2 - *sb2 + 1, 0, 0, print_ptr, last_print, displ, *seq1, *seq2, *g, *gh, matrix);
              mm_score = tracepath(*sb1, *sb2, print_ptr, displ, *seq1, *seq2);
            }
            if (len1 == 0 || len2 == 0) {
#pragma omp taskwait depend(inout : mm_score)
              mm_score = 0.;
            } else {
#pragma omp taskwait depend(in : len1, len2) depend(inout : mm_score)
              mm_score /= (double)((len1 < len2 ? len1 : len2));
            }
#pragma omp taskwait depend(in : bench_output, mm_score, nseqs, si, sj) depend(inout : bench_output[si * nseqs + sj])
            bench_output[si * nseqs + sj] = (int)mm_score;
#pragma omp task default(shared) depend(inout : se1[0]) if (__apac_depth_ok) firstprivate(se1)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete se1;
            }
#pragma omp task default(shared) depend(inout : se2[0]) if (__apac_depth_ok) firstprivate(se2)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete se2;
            }
#pragma omp task default(shared) depend(inout : sb1[0]) if (__apac_depth_ok) firstprivate(sb1)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete sb1;
            }
#pragma omp task default(shared) depend(inout : sb2[0]) if (__apac_depth_ok) firstprivate(sb2)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete sb2;
            }
#pragma omp task default(shared) depend(inout : maxscore[0]) if (__apac_depth_ok) firstprivate(maxscore)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete maxscore;
            }
#pragma omp task default(shared) depend(inout : seq1[0]) if (__apac_depth_ok) firstprivate(seq1)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete seq1;
            }
#pragma omp task default(shared) depend(inout : seq2[0]) if (__apac_depth_ok) firstprivate(seq2)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete seq2;
            }
#pragma omp task default(shared) depend(inout : g[0]) if (__apac_depth_ok) firstprivate(g)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete g;
            }
#pragma omp task default(shared) depend(inout : gh[0]) if (__apac_depth_ok) firstprivate(gh)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete gh;
            }
#pragma omp task default(shared) depend(inout : displ[0]) if (__apac_depth_ok) firstprivate(displ)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete[] displ;
            }
#pragma omp task default(shared) depend(inout : print_ptr[0]) if (__apac_depth_ok) firstprivate(print_ptr)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete print_ptr;
            }
#pragma omp task default(shared) depend(inout : last_print[0]) if (__apac_depth_ok) firstprivate(last_print)
            {
              if (__apac_depth_ok) {
                __apac_depth = __apac_depth_local + 1;
              }
              delete last_print;
            }
          }
        }
      }
#pragma omp taskwait
      for (int a = 0; a < 32; a++) {
        free(matrix[a]);
      }
      free(matrix);
      bots_message(" completed!\n");
      __apac_result = 0;
      goto __apac_exit;
    __apac_exit:;
    }
    return __apac_result;
  } else {
    return pairalign_seq();
  }
}

int pairalign_seq() {
  int i;
  int n;
  int m;
  int si;
  int sj;
  int len1;
  int len2;
  int maxres;
  int mat_avscore;
  double gg;
  double mm_score;
  int** matrix;
  int* mat_xref;
  int* matptr;
  matptr = gon250mt;
  mat_xref = def_aa_xref;
  matrix = (int**)malloc(32 * sizeof(int*));
  for (int a = 0; a < 32; a++) matrix[a] = (int*)malloc(32 * sizeof(int));
  maxres = get_matrix(matptr, mat_xref, 10, matrix, &mat_avscore);
  if (maxres == 0) return -1;
  bots_message("Start aligning ");
  for (si = 0; si < nseqs; si++) {
    n = seqlen_array[si + 1];
    len1 = 0;
    for (i = 1; i <= n; i++) {
      char c = seq_array[si + 1][i];
      if (c != gap_pos1 && c != gap_pos2) len1++;
    }
    for (sj = si + 1; sj < nseqs; sj++) {
      m = seqlen_array[sj + 1];
      if (n == 0 || m == 0) {
        seq_output[si * nseqs + sj] = (int)1.;
      } else {
        int se1;
        int se2;
        int sb1;
        int sb2;
        int maxscore;
        int seq1;
        int seq2;
        int g;
        int gh;
        int displ[2 * 5000 + 1];
        int print_ptr;
        int last_print;
        len2 = 0;
        for (i = 1; i <= m; i++) {
          char c = seq_array[sj + 1][i];
          if (c != gap_pos1 && c != gap_pos2) len2++;
        }
        if (dnaFlag == 1) {
          g = (int)(2 * 100 * pw_go_penalty * gap_open_scale);
          gh = (int)(100 * pw_ge_penalty * gap_extend_scale);
        } else {
          gg = pw_go_penalty + log((double)((n < m ? n : m)));
          g = (int)((mat_avscore <= 0 ? 2 * 100 * gg : 2 * mat_avscore * gg * gap_open_scale));
          gh = (int)(100 * pw_ge_penalty);
        }
        seq1 = si + 1;
        seq2 = sj + 1;
        forward_pass(&seq_array[seq1][0], &seq_array[seq2][0], n, m, &se1, &se2, &maxscore, g, gh, matrix);
        reverse_pass(&seq_array[seq1][0], &seq_array[seq2][0], se1, se2, &sb1, &sb2, maxscore, g, gh, matrix);
        print_ptr = 1;
        last_print = 0;
        diff_seq(sb1 - 1, sb2 - 1, se1 - sb1 + 1, se2 - sb2 + 1, 0, 0, &print_ptr, &last_print, displ, seq1, seq2, g, gh, matrix);
        mm_score = tracepath(sb1, sb2, &print_ptr, displ, seq1, seq2);
        if (len1 == 0 || len2 == 0)
          mm_score = 0.;
        else
          mm_score /= (double)((len1 < len2 ? len1 : len2));
        seq_output[si * nseqs + sj] = (int)mm_score;
      }
    }
  }
  for (int a = 0; a < 32; a++) free(matrix[a]);
  free(matrix);
  bots_message(" completed!\n");
  return 0;
}

void init_matrix() {
  int i;
  int j;
  char c1;
  char c2;
  gap_pos1 = 32 - 2;
  gap_pos2 = 32 - 1;
  max_aa = strlen(amino_acid_codes) - 2;
  for (i = 0; i < 32; i++) def_aa_xref[i] = -1;
  for (i = 0; c1 = amino_acid_order[i]; i++)
    for (j = 0; c2 = amino_acid_codes[j]; j++)
      if (c1 == c2) {
        def_aa_xref[i] = j;
        break;
      }
}

void pairalign_init(char* filename) {
  int i;
  if (!filename || !filename[0]) {
    bots_error(0, "Please specify an input file with the -f option\n");
  }
  init_matrix();
  nseqs = readseqs(filename);
  bots_message("Multiple Pairwise Alignment (%d sequences)\n", nseqs);
  for (i = 1; i <= nseqs; i++) bots_debug("Sequence %d: %s %6.d aa\n", i, names[i], seqlen_array[i]);
  if (clustalw == 1) {
    gap_open_scale = 0.6667;
    gap_extend_scale = 0.751;
  } else {
    gap_open_scale = 1.;
    gap_extend_scale = 1.;
  }
  if (dnaFlag == 1) {
    ktup = 2;
    window = 4;
    signif = 4;
    gap_open = 15.;
    gap_extend = 6.66;
    pw_go_penalty = 15.;
    pw_ge_penalty = 6.66;
  } else {
    ktup = 1;
    window = 5;
    signif = 5;
    gap_open = 10.;
    gap_extend = 0.2;
    pw_go_penalty = 10.;
    pw_ge_penalty = 0.1;
  }
}

void align_init() {
  int i;
  int j;
  bench_output = (int*)malloc(sizeof(int) * nseqs * nseqs);
  for (i = 0; i < nseqs; i++)
    for (j = 0; j < nseqs; j++) bench_output[i * nseqs + j] = 0;
}

void align() {
#pragma omp parallel
#pragma omp master
#pragma omp taskgroup
  {
    pairalign();
  __apac_exit:;
  }
}

void align_init_seq() {
  int i;
  int j;
  seq_output = (int*)malloc(sizeof(int) * nseqs * nseqs);
  bench_output = (int*)malloc(sizeof(int) * nseqs * nseqs);
  for (i = 0; i < nseqs; i++)
    for (j = 0; j < nseqs; j++) seq_output[i * nseqs + j] = 0;
}

void align_seq() { pairalign_seq(); }

void align_end() {
  int i;
  int j;
  for (i = 0; i < nseqs; i++)
    for (j = 0; j < nseqs; j++)
      if (bench_output[i * nseqs + j] != 0) bots_debug("Benchmark sequences (%d:%d) Aligned. Score: %d\n", i + 1, j + 1, (int)bench_output[i * nseqs + j]);
}

int align_verify() {
  int i;
  int j;
  int result = 1;
  for (i = 0; i < nseqs; i++) {
    for (j = 0; j < nseqs; j++) {
      if (bench_output[i * nseqs + j] != seq_output[i * nseqs + j]) {
        bots_message("Error: Optimized prot. (%3d:%3d)=%5d Sequential prot. (%3d:%3d)=%5d\n", i + 1, j + 1, (int)bench_output[i * nseqs + j], i + 1, j + 1, (int)seq_output[i * nseqs + j]);
        result = 2;
      }
    }
  }
  return result;
}
