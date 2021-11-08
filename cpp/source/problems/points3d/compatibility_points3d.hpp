#ifndef _COMPATIBILITY_POINTS3D_HPP
#define _COMPATIBILITY_POINTS3D_HPP

#include "../compatibility.hpp"

struct CompatibilityPoints3d : Compatibility2Pairs {
    CompatibilityPoints3d(size_t numObjects, size_t numLabels);
    void calculate();
};

#endif // _COMPATIBILITY_POINTS3D_HPP
