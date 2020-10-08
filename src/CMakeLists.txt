cmake_minimum_required(VERSION 3.17)
project(CuMat_Compiler)

set(CMAKE_CXX_COMPILER clang++)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_subdirectory(inc)
add_subdirectory(src)