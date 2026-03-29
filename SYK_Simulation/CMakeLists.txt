cmake_minimum_required(VERSION 3.16)
project(SYK_Simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/external/eigen-5.0.0)

add_executable(run
    run.cpp
    hamiltonian.cpp
    utils.cpp
)
add_executable(test test.cpp)