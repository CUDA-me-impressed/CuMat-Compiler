include_directories(${PROJECT_SOURCE_DIR}/compiler/inc)

add_executable(CuMatComp "main.cpp")
target_link_libraries(CuMatComp PreProcessor)