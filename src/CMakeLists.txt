cmake_minimum_required(VERSION 3.12)
project(CuMat_Compiler)

find_program(CLANG_10 "clang++-10")
IF(CLANG_10)
  set(CMAKE_CXX_COMPILER clang++-10)
ELSE()
  set(CMAKE_CXX_COMPILER clang++)
ENDIF()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

add_subdirectory(inc)
add_subdirectory(src)