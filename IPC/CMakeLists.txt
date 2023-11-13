cmake_minimum_required(VERSION 3.12)
project(IPCTest)

# Set C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

# Add include directories
include_directories(SharedFiles/include)

add_compile_options(-g -pthread)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Add executable target
add_executable(Producer SharedFiles/src/producer.cpp)
add_executable(Consumer SharedFiles/src/consumer.cpp)

add_executable(Writer SharedMemory/src/Writer.cpp)
target_link_libraries(Writer rt)
add_executable(Reader SharedMemory/src/Reader.cpp)
target_link_libraries(Reader rt)

add_executable(SemaProducer Semaphores/src/Producer.cpp)
target_link_libraries(SemaProducer rt pthread)
add_executable(SemaConsumer Semaphores/src/Consumer.cpp)
target_link_libraries(SemaConsumer rt pthread)

add_executable(Pipe Pipes/Pipe.cpp)
target_link_libraries(Pipe rt pthread)

add_executable(FifoWriter Pipes/FifoWriter.cpp)
target_link_libraries(FifoWriter rt pthread)

add_executable(FifoReader Pipes/FifoReader.cpp)
target_link_libraries(FifoReader rt pthread)

add_executable(MessageQueueSender MessageQueue/MessageQueueSender.cpp)
target_link_libraries(MessageQueueSender rt pthread)

add_executable(MessageQueueReciver MessageQueue/MessageQueueReciver.cpp)
target_link_libraries(MessageQueueReciver rt pthread)

add_executable(SocketServer Socket/Server.cpp)
target_link_libraries(SocketServer rt pthread)

add_executable(SocketClient Socket/Client.cpp)
target_link_libraries(SocketClient rt pthread)

add_executable(SignalTerm Signal/SignalTerm.cpp)
target_link_libraries(SignalTerm rt pthread)

add_executable(SignalUsr1 Signal/SignalUsr1.cpp)
if(THREADS_HAVE_PTHREAD_ARG)
  set_property(TARGET SignalUsr1 PROPERTY COMPILE_OPTIONS "-pthread")
  set_property(TARGET SignalUsr1 PROPERTY INTERFACE_COMPILE_OPTIONS "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(SignalUsr1 "${CMAKE_THREAD_LIBS_INIT}")
endif()