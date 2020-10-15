include_directories(${PROJECT_SOURCE_DIR}/src/compiler/inc)

add_executable(CuMatComp "main.cpp")
target_link_libraries(CuMatComp PreProcessor)