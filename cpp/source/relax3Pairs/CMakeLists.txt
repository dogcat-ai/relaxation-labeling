cmake_minimum_required(VERSION 3.4)

add_library(relax3Pairs relax3Pairs.hpp relax3Pairs.cpp)
target_link_libraries(relax3Pairs compatibility_points3d_3pairs)

add_executable(relax3Pairs_demo relax3Pairs_demo.cpp)
target_link_libraries(relax3Pairs_demo relax3Pairs)
