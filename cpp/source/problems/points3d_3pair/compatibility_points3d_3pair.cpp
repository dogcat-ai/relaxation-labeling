#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <sstream>
#include <string>

#include "compatibility_points3d_3pair.hpp"

CompatibilityPoints3d3pair::CompatibilityPoints3d3pair(size_t numObjects, size_t numLabels, const std::vector<Eigen::Vector3d>& objects, const std::vector<Eigen::Vector3d>& labels) :
    Compatibility3Pairs(numObjects, numLabels),
    objects(objects),
    labels(labels)
{
    calculate();
}

void CompatibilityPoints3d::calculate()
{
    for (size_t i = 0; i < numObjects; ++i)
    {
        for (size_t j = 0; j < numLabels; ++j)
        {
            for (size_t k = 0; k < numObjects; ++k)
            {
                for (size_t l = 0; l < numLabels; ++l)
                {
                    for (size_t m = 0; m < numObjects; ++m)
                    {
                        for (size_t n = 0; n < numLabels; ++n)
                        {
                            Eigen::Vector3d Vo1 = objects[k] - objects[i];
                            Eigen::Vector3d Vo2 = objects[m] - objects[i];

                            Eigen::Vector3d Vl1 = labels[l] - labels[j];
                            Eigen::Vector3d Vl2 = labels[n] - labels[j];


                            if (i == k && i == mm && j == l && j == n) {
                                compatibility(i, j, k, l, m, n) = 0.0;
                            }
                            // TODO Determine if these cases are correct wrt to setting to negative one.
                            else if (i == k || i == mm || mm == k || j == l || j == n || l == n)
                            {
                                compatibility(i, j, k, l, m, n) = -1.0;
                            }
                            else
                            {
                                V01.normalize();
                                V02.normalize();
                                Vl1.normalize();
                                Vl2.normalize();
                                compatibility(i, j, k, l, m, n) = V01.dot(V02) - Vl1.dot(Vl2);
                            }
                        }
                    }
                }
            }
        }
    }
}
