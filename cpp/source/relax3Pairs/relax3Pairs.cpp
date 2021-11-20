#include <ctime>
#include <iostream>
#include <limits>

#include "relax3Pairs.hpp"

RelaxationLabeling::RelaxationLabeling(const Eigen::Tensor<double, 6>& compatibility) :
    compatibility(compatibility),
    supportFactor(1.0),
    iteration(0),
    iterationLimit(70),
    save(true),
    verbose(0)
{
    numObjects = compatibility.dimension(0);
    numLabels = compatibility.dimension(1);
    
    double initialStrength = 1.0/numLabels;
    strength = Eigen::MatrixXd::Constant(numObjects, numLabels, initialStrength);
    support = Eigen::MatrixXd::Constant(numObjects, numLabels, 0.0);


    std::clock_t start = std::clock();
    while (iteration < iterationLimit)
    {
        iterate();
    }
    std::clock_t stop = std::clock();
    for (size_t i = 0; i < numObjects; ++i)
    {
        std::cout << "final strength for object " << i << std::endl << strength.row(i) << std::endl;
    }
    for (size_t i = 0; i < numObjects; ++i)
    {
        int index;
        double val = strength.row(i).maxCoeff(&index);
        std::cout << "best strength for object #" << i << " is " << val << " with label " << index << std::endl;
    }
    std::cout << iterationLimit << " loops for " << numObjects << " objects and " << numLabels << " labels: " << (stop - start)/1000.0 << " millisec.  Hertz: " << 1.0/((stop-start)/1000000.0) << std::endl;

    int numP = 1;
    for (int i=0; i<numObjects; i++) {
        numP *= (i+1);
    }
    std::cout << "Number permutations: " << numP << std::endl;
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

    std::array<long, 6> extent = {1,1,1,1,numObjects,numLabels};  // Size of array to slice from starting point

    for (size_t i = 0; i < numObjects; ++i)
    {
        for (size_t j = 0; j < numLabels; ++j)
        {
            for (size_t k = 0; k < numObjects; ++k)
            {
                for (size_t l = 0; l < numLabels; ++l)
                {
                    /*
                    std::array<long, 6> offset = {long(i),long(j),long(k),long(l),0,0};
                    Eigen::Tensor<double, 6> compatSliceTensor = compatibility.slice(offset, extent);

                    Eigen::MatrixXd compatSlice = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> (compatSliceTensor.data(), numObjects, numLabels);

                    support(i,j) += (compatSlice.array()*strength.array()).sum();
                    */

                       for (size_t m = 0; m < numObjects; ++m)
                       {
                       for (size_t n = 0; n < numLabels; ++n)
                       {
                       support(i,j) += compatibility(i,j,k,l,m,n)*(strength(k,l)*strength(m,n));
                       }
                       }
                }
            }
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
