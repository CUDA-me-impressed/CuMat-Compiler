cmake_minimum_required(VERSION 3.12)
project(CuMat_Compiler)

find_program(CLANG_10 "clang++-10")
IF (CLANG_10)
    set(CMAKE_CXX_COMPILER clang++-10)
    set(CMAKE_CXX_CLANG_TID clang-tidy-10)
ELSE ()
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_CXX_CLANG_TIDY clang-tidy)
ENDIF ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif ()

link_libraries(-lstdc++fs)
add_subdirectory(src)
