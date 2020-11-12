add_executable(CuMatComp main.cpp)
target_include_directories(CuMatComp PUBLIC ../inc)
target_link_libraries(CuMatComp PUBLIC PreProcessor Parser)