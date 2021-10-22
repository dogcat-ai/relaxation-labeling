#ifndef _COMPATIBILITY_HPP
#define _COMPATIBILITY_HPP

#include <unsupported/Eigen/CXX11/Tensor>

struct Compatibility {
    size_t numObjects;
    size_t numLabels;
    bool save;
    int verbose;
    
    Compatibility(size_t numObjects, size_t numLabels);
    virtual void calculate() = 0;
};

struct Compatibility2Pairs : Compatibility {
    Eigen::Tensor<double, 4> compatibility;

    Compatibility2Pairs(size_t numObjects, size_t numLabels);
};

struct Compatibility3Pairs : Compatibility {
    Eigen::Tensor<double, 6> compatibility;

    Compatibility3Pairs(size_t numObjects, size_t numLabels);
};

#endif // _COMPATIBILITY_HPP
