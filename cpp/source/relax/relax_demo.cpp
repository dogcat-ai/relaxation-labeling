#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "../problems/points3d/compatibility_points3d.hpp"
#include "relax.hpp"


int main()
{
    std::cout << "relax_points3d top" << std::endl;
    std::vector<Eigen::Vector3d> objects(4);
    objects[0] = Eigen::Vector3d(0,0,0);
    objects[1] = Eigen::Vector3d(1,2,1);
    objects[2] = Eigen::Vector3d(5,-2,2);
    objects[3] = Eigen::Vector3d(15,-6,5);

    std::vector<Eigen::Vector3d> labels(4);
    labels[0] = Eigen::Vector3d(0,0,0);
    labels[1] = Eigen::Vector3d(1,2,1);
    labels[2] = Eigen::Vector3d(5,-2,2);
    labels[3] = Eigen::Vector3d(15,-6,5);

    CompatibilityPoints3d compatibilityPoints3d(objects.size(),labels.size(), objects, labels);
    RelaxationLabeling relaxationLabeling(compatibilityPoints3d.compatibility);
}
