cmake_minimum_required(VERSION 3.4)

add_library(relax relax.hpp relax.cpp)
target_link_libraries(relax compatibility_points3d)

add_executable(relax_demo relax_demo.cpp)
target_link_libraries(relax_demo relax)
