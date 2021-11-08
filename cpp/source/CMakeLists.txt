cmake_minimum_required(VERSION 3.4)

project(RelaxationLabeling)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
include_directories(Eigen3::Eigen)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(problems)
add_subdirectory(relax)
