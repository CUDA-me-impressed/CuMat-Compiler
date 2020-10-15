include_directories(${PROJECT_SOURCE_DIR}/src/compiler/inc)
include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(CuMatComp "main.cpp")
target_link_libraries(CuMatComp PreProcessor)