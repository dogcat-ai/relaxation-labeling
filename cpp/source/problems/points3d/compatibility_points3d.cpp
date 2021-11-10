#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <sstream>
#include <string>

#include "compatibility_points3d.hpp"

CompatibilityPoints3d::CompatibilityPoints3d(size_t numObjects, size_t numLabels, const std::vector<Eigen::Vector3d>& objects, const std::vector<Eigen::Vector3d>& labels) :
    Compatibility2Pairs(numObjects, numLabels),
    objects(objects),
    labels(labels)
{
    calculate();
}

void CompatibilityPoints3d::calculate()
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
            compatibilityFiles[i].reserve(numLabels);
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

    verbose = 2;
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
                    Eigen::Vector3d Vo = objects[k] - objects[i];
                    Eigen::Vector3d Vl = labels[l] - labels[j];

                    int offset = 0;
                    if (i == k && j == l) {
                        compatibility(i, j, k, l) = offset + 0.0;
                    }
                    else if (i == k || j == l)
                    {
                        compatibility(i, j, k, l) = offset - 1.0;
                    }
                    else
                    {
                        compatibility(i, j, k, l) = offset + Vo.dot(Vl)/(Vo.norm()*Vl.norm());
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
    //exit(0);
}
