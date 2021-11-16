find_package(Eigen3 REQUIRED)
include_directories(Eigen3::Eigen)

add_library(compatibility_dummy compatibility_dummy.hpp compatibility_dummy.cpp)
