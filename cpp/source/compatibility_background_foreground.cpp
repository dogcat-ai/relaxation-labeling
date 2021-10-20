#include <iostream>

#include "compatibility_background_foreground.hpp"

CompatibilityBackgroundForeground::CompatibilityBackgroundForeground(size_t numObjects, size_t numLabels) :
    Compatibility(numObjects, numLabels)
{
    calculate();
}

void CompatibilityBackgroundForeground::calculate()
{
    std::cerr << "ERROR: CompatibilityBackgroundForeground::calculate() has not been written yet." << std::endl;
    exit(EXIT_FAILURE);
}
