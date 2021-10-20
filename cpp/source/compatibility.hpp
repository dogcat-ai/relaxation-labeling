#ifndef _COMPATIBILITY_HPP
#define _COMPATIBILITY_HPP

#include <unsupported/Eigen/CXX11/Tensor>

struct Compatibility {
    size_t numObjects;
    size_t numLabels;
    Eigen::Tensor<double, 4> compatibility;
    bool save;
    int verbose;
    
    Compatibility(size_t numObjects, size_t numLabels);
    virtual void calculate() = 0;
};

#endif // _COMPATIBILITY_HPP
