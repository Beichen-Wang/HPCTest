#include "util/Timer.hpp"
#include "math/Fibonacci.hpp"
#include "math/unbound_loop.hpp"
#include <gtest/gtest.h>
#include "Reduce.hpp"

#define SUFFIX(time1, time2, doc) \
std::cout << doc << std::endl; \
std::cout << "OMP duration is :" << time1.duration() << "ms" << std::endl; \
std::cout << "Nor duration is :" << time2.duration() << "ms" << std::endl; \
std::cout << std::endl;

TEST(MyTestSuite, omp_tast_test_fib){
  std::cout << "************************************" << std::endl;
  std::cout << "profile for the omp tast Fib" << std::endl;
  EXPECT_EQ(OmpFib(30), NorFib(30));
  util::Timer timer[2];
  int Num = 30;

  timer[0].start();
#pragma omp parallel num_threads(4)
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

  Unboundloop test(100);
  EXPECT_EQ(test.OMPProcess<decltype(&SubProcess::SubProcess4)>(), test.NorProcess<decltype(&SubProcess::SubProcess4)>());
  util::Timer timer[2];
  timer[0].start();
  test.OMPProcess<decltype(&SubProcess::SubProcess1)>();
  timer[0].stop();
  timer[1].start();
  test.NorProcess<decltype(&SubProcess::SubProcess1)>();
  timer[1].stop();
  SUFFIX(timer[0], timer[1], "SubProcess1")
  timer[0].start();
  test.OMPProcess<decltype(&SubProcess::SubProcess2)>();
  timer[0].stop();
  timer[1].start();
  test.NorProcess<decltype(&SubProcess::SubProcess2)>();
  timer[1].stop();
  SUFFIX(timer[0], timer[1], "SubProcess2")
  timer[0].start();
  test.OMPProcess<decltype(&SubProcess::SubProcess3)>();
  timer[0].stop();
  timer[1].start();
  test.NorProcess<decltype(&SubProcess::SubProcess3)>();
  timer[1].stop();
  SUFFIX(timer[0], timer[1], "SubProcess3")
  timer[0].start();
  test.OMPProcess<decltype(&SubProcess::SubProcess4)>();
  timer[0].stop();
  timer[1].start();
  test.NorProcess<decltype(&SubProcess::SubProcess4)>();
  timer[1].stop();
  SUFFIX(timer[0], timer[1], "SubProcess4")
  std::cout << "************************************" << std::endl;

}

TEST(MyTestSuite, omp_task_test_taskloop){
    std::cout << "************************************" << std::endl;
    std::cout << "profile for the omp tast taskloop" << std::endl;
    Reduce reduce;
    EXPECT_EQ(reduce.OMPProcess<GrainSizeSubProcess>(50), reduce.NorProcess(50));
    util::Timer timer[2];
    size_t Num = 5000000;
    timer[0].start();
    reduce.OMPProcess<GrainSizeSubProcess>(Num);
    timer[0].stop();
    timer[1].start();
    reduce.NorProcess(Num);
    timer[1].stop();

    SUFFIX(timer[0], timer[1], "GrainSizeSubProcess TaskLoop");

    EXPECT_EQ(reduce.OMPProcess<NumTasksSIMDSubProcess>(50), reduce.NorProcess(50));
    timer[0].start();
    reduce.OMPProcess<NumTasksSubProcess>(Num);
    timer[0].stop();
    timer[1].start();
    reduce.NorProcess(Num);
    timer[1].stop();

    SUFFIX(timer[0], timer[1], "NumTasksSubProcess TaskLoop");

    EXPECT_EQ(reduce.OMPProcess<NumTasksSIMDSubProcess>(50), reduce.NorProcess(50));
    timer[0].start();
    reduce.OMPProcess<NumTasksSIMDSubProcess>(Num);
    timer[0].stop();
    timer[1].start();
    reduce.NorProcess(Num);
    timer[1].stop();

    SUFFIX(timer[0], timer[1], "NumTasksSIMDSubProcess TaskLoop");

    EXPECT_EQ(reduce.OMPProcess<SIMDSubProcess>(50), reduce.NorProcess(50));
    timer[0].start();
    reduce.OMPProcess<SIMDSubProcess>(Num);
    timer[0].stop();
    timer[1].start();
    reduce.NorProcess(Num);
    timer[1].stop();

    SUFFIX(timer[0], timer[1], "SIMDSubProcess TaskLoop");

    EXPECT_EQ(reduce.OMPProcess<GrainSizeReduceSubProcess>(50), reduce.NorProcess(50));
    timer[0].start();
    reduce.OMPProcess<GrainSizeReduceSubProcess>(Num);
    timer[0].stop();
    timer[1].start();
    reduce.NorProcess(Num);
    timer[1].stop();

    SUFFIX(timer[0], timer[1], "GrainSizeReduceSubProcess TaskLoop");
    std::cout << "************************************" << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}