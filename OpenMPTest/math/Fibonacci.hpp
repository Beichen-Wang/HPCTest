#include <omp.h>

int OmpFib(int n) {
  int x, y;
  if (n < 2) {
    return n;
  }
#pragma omp task shared(x) untied
  { x = OmpFib(n - 1); }
#pragma omp task shared(y) untied
  { y = OmpFib(n - 2); }
#pragma omp taskwait
  return x + y;
}

int NorFib(int n) {
  int x, y;
  if (n < 2)
    return n;
  x = NorFib(n - 1);
  y = NorFib(n - 2);
  return x + y;
}
