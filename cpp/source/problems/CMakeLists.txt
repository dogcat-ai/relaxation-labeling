cmake_minimum_required(VERSION 3.4)

add_library(compatibility compatibility.hpp compatibility.cpp)
target_link_libraries(compatibility Eigen3::Eigen)

add_subdirectory(points3d)
add_subdirectory(chessboard)
