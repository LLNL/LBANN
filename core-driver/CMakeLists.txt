cmake_minimum_required(VERSION 3.21.0)
project(lbann-test-driver CXX)
find_package(LBANN 0.102.0 REQUIRED)
add_executable(lbann-test-driver main.cpp)
target_link_libraries(lbann-test-driver PRIVATE LBANN::lbann)

set_target_properties(lbann-test-driver
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
