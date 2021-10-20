#ifndef _COMPATIBILITY_BACKGROUND_FOREGROUND
#define _COMPATIBILITY_BACKGROUND_FOREGROUND

#include "compatibility.hpp"

struct CompatibilityBackgroundForeground : Compatibility {
    CompatibilityBackgroundForeground(size_t numObjects, size_t numLabels);
    void calculate();
};

#endif// _COMPATIBILITY_BACKGROUND_FOREGROUND

