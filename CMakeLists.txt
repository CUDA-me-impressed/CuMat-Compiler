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

find_package(LLVM 10 REQUIRED CONFIG
        # try the llvm-10 directory first to avoid other versions
        PATHS /usr/lib/llvm-10/lib/cmake/llvm/)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
if (${LLVM_PACKAGE_VERSION} VERSION_LESS 10 OR ${LLVM_PACKAGE_VERSION} VERSION_GREATER_EQUAL 11)
    message(FATAL_ERROR "LLVM Version is ${LLVM_PACKAGE_VERSION}, not 10")
endif ()

# include all of the components, may need to include nvvm at some point
llvm_map_components_to_libnames(llvm_libs support core irreader)
add_library(llvm_interface INTERFACE)
target_link_libraries(llvm_interface INTERFACE ${llvm_libs})
target_include_directories(llvm_interface INTERFACE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(llvm_interface INTERFACE ${LLVM_DEFINITIONS})

if ($ENV{CLION_IDE})
    message(STATUS "Using extra llvm include files as CLion can't follow WSL symlinks")
    target_include_directories(llvm_interface INTERFACE "/usr/include/llvm-10")
endif ()


link_libraries(-lstdc++fs)
add_subdirectory(src)