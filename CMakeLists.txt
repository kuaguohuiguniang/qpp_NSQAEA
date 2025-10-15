cmake_minimum_required(VERSION 3.20.0)
project(NS_QAEA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_EXTENSIONS OFF)
# Set the output path for the executable
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Collect source files
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

# Create the executable target
#add_executable(qpp ${SRC_LIST})
#add_executable(test ${SRC_LIST})
add_executable(NSQAEA ${SRC_LIST})


include_directories(${PROJECT_SOURCE_DIR}/include)
   