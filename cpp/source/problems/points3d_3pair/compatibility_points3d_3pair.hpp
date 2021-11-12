#ifndef _COMPATIBILITY_POINTS3D_3PAIR_HPP
#define _COMPATIBILITY_POINTS3D_3PAIR_HPP

#include <vector>

#include <Eigen/Dense>

#include "../compatibility.hpp"

struct CompatibilityPoints3d3pair : Compatibility3Pairs {
    CompatibilityPoints3d(size_t numObjects, size_t numLabels, const std::vector<Eigen::Vector3d>& objects, const std::vector<Eigen::Vector3d>& labels);
    void calculate();
    const std::vector<Eigen::Vector3d>& objects; 
    const std::vector<Eigen::Vector3d>& labels;
};

#endif // _COMPATIBILITY_POINTS3D_3PAIR_HPP
