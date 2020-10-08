cmake_minimum_required(VERSION 3.17)
project(CS407)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_subdirectory(inc)
add_subdirectory(src)