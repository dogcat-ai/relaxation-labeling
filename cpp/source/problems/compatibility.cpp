#include "compatibility.hpp"

Compatibility::Compatibility(size_t numObjects, size_t numLabels) :
    numObjects(numObjects),
    numLabels(numLabels),
    save(true),
    verbose(2)
{
}

Compatibility2Pairs::Compatibility2Pairs(size_t numObjects, size_t numLabels) :
	Compatibility(numObjects, numLabels),
    compatibility(numObjects, numLabels, numObjects, numLabels)
{
	compatibility.setZero(); // TODO: MGLOVER - OPTIMIZATION - You may want to leave everything default-initialized.
}

Compatibility3Pairs::Compatibility3Pairs(size_t numObjects, size_t numLabels) :
	Compatibility(numObjects, numLabels),
    compatibility(numObjects, numLabels, numObjects, numLabels, numObjects, numLabels)
{
	compatibility.setZero(); // TODO: MGLOVER - OPTIMIZATION - You may want to leave everything default-initialized.
}
