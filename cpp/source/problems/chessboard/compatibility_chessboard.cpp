#include <iostream>
#include <fstream>
#include <memory>

#include "compatibility_chessboard.hpp"

CompatibilityChessboard::CompatibilityChessboard(size_t numObjects, size_t numLabels) :
    Compatibility3Pairs(numObjects, numLabels)
{
    calculate();
}

void CompatibilityChessboard::defineObjectsAndLabels()
{
}

void CompatibilityChessboard::calculate(FindChessboardCorners findChessboardCorners )
{
    for (size_t r=0; r<numRows; ++r)
    {
        for (size_t c=0; c<numColumns; ++c)
        {
            calculateOneCorner(r,c,numRows,numColumns);
        }
    }
}

void CompatibilityChessboard::calculateOneCorner(size_t r,size_t c,size_t numRows,size_t numColumns)
{
    double compatibilityValue = 0;
    int numDirections = 0;

    if (c+2 < numColumns)
    {
        compatibilityValue += calculateOneDirection(Direction::RIGHT, r, c, r, c+1, r, c+2);
        numDirections++;
    }
    if (c-2 < numColumns)
    {
        compatibilityValue += calculateOneDirection(Direction::LEFT, r, c, r, c-1, r, c-2);
        numDirections++;
    }
    if (r-2 < numColumns)
    {
        compatibilityValue += calculateOneDirection(Direction::UP, r, c, r-1, c, r-2, c);
        numDirections++;
    }
    if (r+2 < numColumns)
    {
        compatibilityValue += calculateOneDirection(Direction::DOWN, r, c, r+1, c, r+2, c);
        numDirections++;
    }
}

double CompatibilityChessboard::calculateOneDirection(Direction direction, size_t rowOrigin, size_t columnOrigin, size_t rowNear, size_t columnNear, size_t rowFar, size_t columnFar)
{
    Eigen::Vector2d origin;
    origin[0] = findChessboardCorners.corners[rowOrigin][columnOrigin][0];
    origin[1] = findChessboardCorners.corners[rowOrigin][columnOrigin][1];

    Eigen::Vector2d near;
    near[0] = findChessboardCorners.corners[rowNear][columnNear][0];
    near[1] = findChessboardCorners.corners[rowNear][columnNear][1];

    Eigen::Vector2d far;
    far[0] = findChessboardCorners.corners[rowFar][columnFar][0];
    far[1] = findChessboardCorners.corners[rowFar][columnFar][1];

    Eigen::Vector2d originToNear = near - origin;
    Eigen::Vector2d nearToFar = far - near;

    double compatibilityValue = originToNear.dot(nearToFar);

    return compatibilityValue;
}

void CompatibilityChessboard::calculate()
{
    if (verbose > 1)
    {
        std::cout << "About to calculate compatibility:" << std::endl;
    }
    
    std::vector<std::vector<std::vector<std::vector<std::shared_ptr<std::ofstream>>>>> compatibilityFiles(numObjects,
            std::vector<std::vector<std::vector<std::shared_ptr<std::ofstream>>>>(numLabels, 
                std::vector<std::vector<std::shared_ptr<std::ofstream>>>(numObjects,
                    std::vector<std::shared_ptr<std::ofstream>>(numLabels))));

    if (save)
    {
        std::ofstream summaryFile("structure.csv", std::ofstream::out);
        summaryFile << "num pairs,num objects,num labels" << std::endl;
        summaryFile << "3," << numObjects << "," << numLabels << std::endl;
        summaryFile.close();

        for (int i = 0; i < numObjects; ++i)
        {
            compatibilityFiles.reserve(numLabels);
            for (int j = 0; j < numLabels; ++j)
            {
                compatibilityFiles[j].reserve(numObjects);
                for (int k = 0; k < numObjects; ++k)
                {
                    compatibilityFiles[j].reserve(numLabels);
                    for (int l = 0; l < numLabels; ++l)
                    {
                        std::stringstream fileName;
                        fileName << "compatibility_" << i << "_" << j << "_" << k << "_" << l << ".csv";
                        compatibilityFiles[i][j][k][l]->open(fileName.str().c_str());
                        for (int n = 0; n < numLabels; ++n)
                        {
                            *compatibilityFiles[i][j][k][l] << ",label " << n;
                        }
                    }
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
                for (size_t l = 0; l < numLabels; ++l)
                {
                    if (verbose > 1)
                    {
                        std::cout << "l = " << l << std::endl;
                    }
                    for (size_t m = 0; m < numObjects; ++m)
                    {
                        if (save)
                        {
                            *compatibilityFiles[i][j][k][l] << std::endl << "object " << m;
                        }
                        if (verbose > 1)
                        {
                            std::cout << "m = " << m << std::endl;
                        }
                        for (size_t n = 0; n < numLabels; ++n)
                        {
                            compatibility(i, j, k, l, m, n) = 1.0;
                            if (verbose > 1)
                            {
                                std::cout << "n = " << n << std::endl;
                                std::cout << '\t' << "compatibility = " << compatibility(i, j, k, l, m, n) << std::endl;
                            }
                            if (save)
                            {
                                *compatibilityFiles[i][j][k][l] << "," << compatibility(i,j,k,l,m,n);
                            }
                        }
                    }
                    compatibilityFiles[i][j][k][l]->close();
                }
            }
        }
    }
}
