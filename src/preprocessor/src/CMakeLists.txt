
add_library(PreProcessor Preprocessor.cpp ProgramGraph.cpp)
target_include_directories(PreProcessor PUBLIC ${PROJECT_SOURCE_DIR}/preprocessor/inc)
