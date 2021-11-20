#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "../problems/points3d/compatibility_points3d.hpp"
#include "relax.hpp"


int main()
{
    std::cout << "relax_points3d top" << std::endl;
    const int n = 20;
    const float scale = 10.0;
    std::vector<Eigen::Vector3d> objects(n);
    for (int i=0; i<n; i++)
    {
        objects[i] = scale*Eigen::Vector3d::Random(3);
    }

    const float noiseScale = .02*scale;
    const int m = n-1;
    std::vector<Eigen::Vector3d> labels(m);
    for (int i=0; i<m; i++)
    {
        labels[i] = noiseScale*Eigen::Vector3d::Random(3) + objects[i];
    }

    std::cout << "Start --> " << std::endl;

    CompatibilityPoints3d compatibilityPoints3d(objects.size(),labels.size(), objects, labels);
    RelaxationLabeling relaxationLabeling(compatibilityPoints3d.compatibility);
}
