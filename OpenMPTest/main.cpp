#include "util/Timer.hpp"
#include "math/Fibonacci.hpp"
#include "math/unbound_loop.hpp"
#include <gtest/gtest.h>

TEST(MyTestSuite, omp_tast_test_fib){
  std::cout << "************************************" << std::endl;
  std::cout << "profile for the omp tast Fib" << std::endl;
  EXPECT_EQ(OmpFib(5), NorFib(5));
  util::Timer timer[2];
  int Num = 30;

  timer[0].start();
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    { OmpFib(Num); }
  }
  timer[0].stop();
  timer[1].start();
  NorFib(Num);
  timer[1].stop();

  std::cout << "OMP duration is :" << timer[0].duration() << "ms" << std::endl;
  std::cout << "Nor duration is :" << timer[1].duration() << "ms" << std::endl;
  std::cout << "************************************" << std::endl;
}

TEST(MyTestSuite, omp_tast_test_unbound_loop){
  std::cout << "************************************" << std::endl;
  std::cout << "profile for the omp tast unbound_loop" << std::endl;

  Unboundloop test(50000);
  EXPECT_EQ(test.OMPProcess(), test.NorProcess());
  util::Timer timer[2];
  timer[0].start();
  test.OMPProcess();
  timer[0].stop();
  timer[1].start();
  test.NorProcess();
  timer[1].stop();

  std::cout << "OMP duration is :" << timer[0].duration() << "ms" << std::endl;
  std::cout << "Nor duration is :" << timer[1].duration() << "ms" << std::endl;
  std::cout << "************************************" << std::endl;

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}