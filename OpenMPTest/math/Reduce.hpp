#include <cstddef>

class SubProcessBase {
public:
  virtual size_t operator()(size_t n) {
    size_t sum = 0;
    for (size_t i = 0; i < n - 1; i++) {
      sum += (i * (i + 1));
    }
    return sum;
  }
};

class GrainSizeSubProcess final: public SubProcessBase {
public:
  size_t operator()(size_t n) override {
    size_t sum = 0;
#pragma omp taskloop grainsize(100000) shared(sum)
    for (size_t i = 0; i < n - 1; i++) {
      sum += (i * (i + 1));
    }
    return sum;
  }
};

class NumTasksSubProcess final: public SubProcessBase {
public:
  size_t operator()(size_t n) override {
    size_t sum = 0;
#pragma omp taskloop num_tasks(50) shared(sum)
    for (size_t i = 0; i < n - 1; i++) {
      sum += (i * (i + 1));
    }
    return sum;
  }
};

class NumTasksSIMDSubProcess final: public SubProcessBase {
public:
  size_t operator()(size_t n) override {
    size_t sum = 0;
#pragma omp taskloop simd num_tasks(50) shared(sum)
    for (size_t i = 0; i < n - 1; i++) {
      sum += (i * (i + 1));
    }
    return sum;
  }
};

class SIMDSubProcess final: public SubProcessBase {
public:
  size_t operator()(size_t n) override {
    size_t sum = 0;
#pragma omp simd safelen(4)
    for (size_t i = 0; i < n - 1; i++) {
      sum += (i * (i + 1));
    }
    return sum;
  }
};

class GrainSizeReduceSubProcess final: public SubProcessBase {
public:
  size_t operator()(size_t n) override {
    size_t sum = 0;
#pragma omp taskloop grainsize(100000) reduction(+: sum)
    for (size_t i = 0; i < n - 1; i++) {
      sum += (i * (i + 1));
    }
    return sum;
  }
};

class Reduce {
public:
  template <typename SubProcessFunc = SubProcessBase>
  size_t OMPProcess(size_t n) {
    static_assert(std::is_base_of<SubProcessBase, SubProcessFunc>::value,
                  " the SubProcessFunc must derived from SubProcessBase");
    return SubProcessFunc()(n);
  }
  size_t NorProcess(size_t n) { return SubProcessBase()(n); }

private:
};