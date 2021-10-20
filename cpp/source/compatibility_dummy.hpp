#ifndef _COMPATIBILITY_DUMMY_HPP
#define _COMPATIBILITY_DUMMY_HPP

#include "compatibility.hpp"

struct CompatibilityDummy : Compatibility {
    CompatibilityDummy(size_t numObjects, size_t numLabels);
    void calculate();
};

#endif // _COMPATIBILITY_DUMMY_HPP
