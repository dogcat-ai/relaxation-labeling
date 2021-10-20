cmake_minimum_required(VERSION 3.4)

project(RelaxationLabeling)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
include_directories(Eigen3::Eigen)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(relax relax.cpp relax.hpp compatibility_background_foreground.hpp compatibility.hpp compatibility.cpp compatibility_background_foreground.cpp compatibility_dummy.hpp compatibility_dummy.cpp)
target_link_libraries(relax Eigen3::Eigen ${OpenCV_LIBS} opencv_ximgproc)

add_executable(relax_demo relax_demo.cpp)
target_link_libraries(relax_demo relax)
