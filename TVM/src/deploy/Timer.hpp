#include <chrono>
#include <iostream>
#include <omp.h>

namespace util{
class Timer {
public:
  void start() { timeStart = std::chrono::high_resolution_clock::now(); }
  void stop() { timeStop = std::chrono::high_resolution_clock::now(); }
  double duration() {
    auto duration =
        std::chrono::duration<double, std::milli>(timeStop - timeStart).count();
    return duration;
  }

private:
  decltype(std::chrono::high_resolution_clock::now()) timeStart;
  decltype(std::chrono::high_resolution_clock::now()) timeStop;
};
}