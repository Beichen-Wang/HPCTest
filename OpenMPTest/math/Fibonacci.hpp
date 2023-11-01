#include <omp.h>

int OmpFib(int n) {
  int x, y;
  if (n < 2) {
    return n;
  }
  if(n > 20){
#pragma omp task shared(x) untied
  { x = OmpFib(n - 1); }
#pragma omp task shared(y) untied
  { y = OmpFib(n - 2); }
#pragma omp taskwait
  } else {
    x = OmpFib(n - 1);
    y = OmpFib(n - 2);
  }
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
