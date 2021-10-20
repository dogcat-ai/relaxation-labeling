#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <sstream>
#include <string>

#include "compatibility_dummy.hpp"

CompatibilityDummy::CompatibilityDummy(size_t numObjects, size_t numLabels) :
    Compatibility(numObjects, numLabels)
{
    calculate();
}

void CompatibilityDummy::calculate()
{
    if (verbose > 1)
    {
        std::cout << "About to calculate compatibility:" << std::endl;
    }
    
    // output files of compatibility figures if desired
    std::vector<std::vector<std::unique_ptr<std::ofstream> > > compatibilityFiles(numObjects);
    if (save)
    {
        std::ofstream summaryFile("structure.csv", std::ofstream::out);
        summaryFile << "num objects,num labels" << std::endl;
        summaryFile << numObjects << "," << numLabels << std::endl;
        summaryFile.close();

        for (int i = 0; i < numObjects; ++i)
        {
            compatibilityFiles.reserve(numLabels);
            for (int j = 0; j < numLabels; ++j)
            {
                std::stringstream fileName;
                fileName << "compatibility_" << i << "_" << j << ".csv";
                compatibilityFiles[i].emplace_back(std::make_unique<std::ofstream>());
                compatibilityFiles[i][j]->open(fileName.str().c_str());
                for (int l = 0; l < numLabels; ++l)
                {
                    *compatibilityFiles[i][j] << ",label " << l;
                }
            }
        }
    }

    for (size_t i = 0; i < numObjects; ++i)
    {
        if (verbose > 1)
        {
            std::cout << "i = " << i << std::endl;
        }
        for (size_t j = 0; j < numLabels; ++j)
        {
            if (verbose > 1)
            {
                std::cout << "j = " << j << std::endl;
            }
            for (size_t k = 0; k < numObjects; ++k)
            {
                if (verbose > 1)
                {
                    std::cout << "k = " << k << std::endl;
                }
                if (save)
                {
                    *compatibilityFiles[i][j] << std::endl << "object " << k;
                }
                for (size_t l = 0; l < numLabels; ++l)
                {
                    if (i == j && k == l)
                    {
                        compatibility(i, j, k, l) = 1.0;
                    }
                    else
                    {
                        compatibility(i, j, k, l) = -1.0;
                    }
                    if (verbose > 1)
                    {
                        std::cout << "l = " << l << std::endl;
                        std::cout << '\t' << "compatibility = " << compatibility(i, j, k, l) << std::endl;
                    }
                    if (save)
                    {
                        *compatibilityFiles[i][j] << "," << compatibility(i,j,k,l);
                    }
                }
            }
            compatibilityFiles[i][j]->close();
        }
    }
}
