#include "compatibility.hpp"

Compatibility::Compatibility(size_t numObjects, size_t numLabels) :
    numObjects(numObjects),
    numLabels(numLabels),
    compatibility(numObjects,numLabels,numObjects,numLabels),
    save(true),
    verbose(2)
{
    compatibility.setZero(); // TODO: MGLOVER - OPTIMIZATION - You may want to leave everything default-initialized.
}
