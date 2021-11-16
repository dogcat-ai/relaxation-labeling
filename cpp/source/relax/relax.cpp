#include <iostream>
#include <limits>

#include "relax.hpp"

RelaxationLabeling::RelaxationLabeling(const Eigen::Tensor<double, 4>& compatibility) :
    compatibility(compatibility),
    supportFactor(0.5),
    iteration(0),
    iterationLimit(20),
    save(true),
    verbose(2)
{
    numObjects = compatibility.dimension(0);
    numLabels = compatibility.dimension(1);
    
    double initialStrength = 1.0/numLabels;
    strength = Eigen::MatrixXd::Constant(numObjects, numLabels, initialStrength);
    support = Eigen::MatrixXd::Constant(numObjects, numLabels, 0.0);
    
    std::vector<std::unique_ptr<std::ofstream> > strengthFiles(numObjects);
    while (iteration < iterationLimit)
    {
        iterate();
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
    support.array() = 0.0;

    for (size_t i = 0; i < numObjects; ++i)
    {
        for (size_t j = 0; j < numLabels; ++j)
        {
            for (size_t k = 0; k < numObjects; ++k)
            {
                for (size_t l = 0; l < numLabels; ++l)
                {
                    support(i,j) += compatibility(i,j,k,l)*strength(k,l);
                }
            }
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
    for (size_t i = 0; i < numObjects; ++i)
    {
        double sumStrengths = 0;
        for (size_t j = 0; j < numLabels; ++j)
        {
            strength(i,j) *= (1.0+support(i,j)*supportFactor);
            sumStrengths += strength(i,j);
        }
        normalizeStrength(i, sumStrengths);
        if (verbose > 1)
        {
            std::cout << "strength for object " << i << std::endl << strength.row(i) << std::endl;
        }
    }
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
