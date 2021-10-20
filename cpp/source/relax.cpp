#include <iostream>
#include <limits>

#include "relax.hpp"

RelaxationLabeling::RelaxationLabeling(const Eigen::Tensor<double, 4>& compatibility) :
    compatibility(compatibility),
    supportFactor(0.2),
    iteration(0),
    iterationLimit(10),
    save(true),
    verbose(2)
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
}

void RelaxationLabeling::iterate()
{
    updateSupport();
    updateStrength();
    ++iteration;
}

void RelaxationLabeling::updateSupport()
{
    for (size_t i = 0; i < numObjects; ++i)
    {
        double iMinimumSupport = std::numeric_limits<double>::max();
        double iMaximumSupport = std::numeric_limits<double>::min();
        for (size_t j = 0; j < numLabels; ++j)
        {
            support(i,j) = 0.0;
            for (size_t k = 0; k < numObjects; ++k)
            {
                for (size_t l = 0; l < numLabels; ++l)
                {
                    double ijklSupport = strength(k,l)*compatibility(i,j,k,l);
                    support(i,j) += ijklSupport;
                    if (ijklSupport < iMinimumSupport)
                    {
                        iMinimumSupport = ijklSupport;
                    }
                    if (ijklSupport > iMaximumSupport)
                    {
                        iMaximumSupport = ijklSupport;
                    }
                }
            }
            normalizeSupport(i, iMinimumSupport, iMaximumSupport);
        }
        if (verbose > 1)
        {
            std::cout << "support for object " << i << std::endl << support.row(i) << std::endl;
        }
    }
}

void RelaxationLabeling::normalizeSupport(size_t i, double iMinimumSupport, double iMaximumSupport)
{
    iMaximumSupport -= iMinimumSupport;
    for (size_t j = 0; j < numLabels; ++j)
    {
        support(i,j) = (support(i,j) - iMinimumSupport)/iMaximumSupport;
    }
}

void RelaxationLabeling::updateStrength()
{
    for (size_t i = 0; i < numObjects; ++i)
    {
        double sumStrengths = 0;
        for (size_t j = 0; j < numLabels; ++j)
        {
            strength(i,j) += support(i,j)*supportFactor;
            sumStrengths += strength(i,j);
        }
        normalizeStrength(i, sumStrengths);
        if (verbose > 1)
        {
            std::cout << "strength for object " << i << std::endl << strength.row(i) << std::endl;
        }
    }
}

void RelaxationLabeling::normalizeStrength(size_t i, double iSumStrengths)
{
    for (size_t j = 0; j < numLabels; ++j)
    {
        strength(i, j) /= iSumStrengths;
    }
}

void RelaxationLabeling::assign()
{
    
}
