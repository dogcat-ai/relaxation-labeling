#include <iostream>
#include <limits>

#include "relax.hpp"

RelaxationLabeling::RelaxationLabeling(const Eigen::Tensor<double, 4>& compatibility) :
    compatibility(compatibility),
    supportFactor(0.5),
    iteration(0),
    iterationLimit(20),
    save(true),
    verbose(0)
{
    numObjects = compatibility.dimension(0);
    numLabels = compatibility.dimension(1);
    
    double initialStrength = 1.0/numLabels;
    strength = Eigen::MatrixXd::Constant(numObjects, numLabels, initialStrength);
    support = Eigen::MatrixXd::Constant(numObjects, numLabels, 0.0);

    while (iteration < iterationLimit)
    {
        iterate();
    }
    for (size_t i = 0; i < numObjects; ++i)
    {
        std::cout << "final strength for object " << i << std::endl << strength.row(i) << std::endl;
    }
}

void RelaxationLabeling::iterate()
{
    updateSupport();
    updateStrength();
    ++iteration;
}

void RelaxationLabeling::updateSupport()
{
    std::array<long, 4> extent = {1,1,numObjects,numLabels};  // Size of array to slice from starting point

    for (size_t i = 0; i < numObjects; ++i)
    {
        for (size_t j = 0; j < numLabels; ++j)
        {
            std::array<long, 4> offset = {long(i),long(j),0,0};  // Starting point
            Eigen::Tensor<double, 4> compatSliceTensor = compatibility.slice(offset, extent);

            Eigen::MatrixXd compatSlice = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> (compatSliceTensor.data(), numObjects, numLabels);

            support(i,j) = (compatSlice.array()*strength.array()).sum();
        }
        if (verbose > 1)
        {
            std::cout << "raw support for object " << i << std::endl << support.row(i) << std::endl;
        }
    }
    normalizeSupport(SupportNormalizationMethod::ALL_IJ);
}

void RelaxationLabeling::normalizeSupport(SupportNormalizationMethod snm)
{
    if (snm == SupportNormalizationMethod::ALL_IJ)
    {
        double minElement = support.minCoeff();
        double maxElement = support.maxCoeff();
        support = (support.array() - minElement)/maxElement;
    }
}

void RelaxationLabeling::updateStrength()
{
    strength = strength.array() + strength.array()*support.array()*supportFactor;
    strength.rowwise().normalize();
}

void RelaxationLabeling::normalizeStrength(size_t i, double sumStrengths)
{
    for (size_t j = 0; j < numLabels; ++j)
    {
        strength(i, j) /= sumStrengths;
    }
}

void RelaxationLabeling::assign()
{
    
}
