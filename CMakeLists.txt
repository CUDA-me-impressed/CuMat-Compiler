cmake_minimum_required(VERSION 3.12)
project(CuMat_Compiler LANGUAGES CXX)
enable_testing()

find_program(CLANG_10 "clang++-10")
if (CLANG_10)
    set(CMAKE_CXX_COMPILER clang++-10)
    set(CMAKE_CXX_CLANG_TID clang-tidy-10)
elseif ()
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_CXX_CLANG_TIDY clang-tidy)
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif ()

find_package(LLVM 10 REQUIRED CONFIG
        # try the llvm-10 directory first to avoid other versions
        PATHS /usr/lib/llvm-10/lib/cmake/llvm/)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Normal LLVM Include headers found in: ${LLVM_INCLUDE_DIRS}")
if (${LLVM_PACKAGE_VERSION} VERSION_LESS 10 OR ${LLVM_PACKAGE_VERSION} VERSION_GREATER_EQUAL 11)
    message(FATAL_ERROR "LLVM Version is ${LLVM_PACKAGE_VERSION}, not 10")
endif ()

# include all of the components, may need to include nvvm at some point
llvm_map_components_to_libnames(llvm_libs support core irreader)
add_library(llvm_interface INTERFACE)
target_link_libraries(llvm_interface INTERFACE ${llvm_libs})
target_include_directories(llvm_interface INTERFACE ${LLVM_INCLUDE_DIRS})
if ($ENV{CLION_IDE})
    message(STATUS "Using extra llvm include files as CLion can't follow WSL symlinks")
    target_include_directories(llvm_interface INTERFACE "/usr/include/llvm-10")
endif ()
target_compile_definitions(llvm_interface INTERFACE ${LLVM_DEFINITIONS})

# Include Catch2
include(FetchContent)
FetchContent_Declare(Catch2_download
        GIT_REPOSITORY      https://github.com/catchorg/Catch2.git
        GIT_TAG             v2.13.3)
FetchContent_Populate(Catch2_download)
FetchContent_GetProperties(Catch2_download
        SOURCE_DIR Catch2_source_dir)
message(STATUS "Downloaded Catch2 to ${Catch2_source_dir}")
add_library(Catch2 INTERFACE)
target_include_directories(Catch2 INTERFACE
        "${Catch2_source_dir}/single_include")

# Include trompeloeil
FetchContent_Declare(Trompeloeil_download
        GIT_REPOSITORY      https://github.com/rollbear/trompeloeil.git
        GIT_TAG             v39)
FetchContent_Populate(Trompeloeil_download)
FetchContent_GetProperties(Trompeloeil_download
        SOURCE_DIR Trompeloeil_source_dir)
message(STATUS "Downloaded Trompeloeil to ${Trompeloeil_source_dir}")
add_library(Trompeloeil INTERFACE)
target_include_directories(Trompeloeil INTERFACE
        "${Trompeloeil_source_dir}/include")

link_libraries(-lstdc++fs)
add_subdirectory(src)