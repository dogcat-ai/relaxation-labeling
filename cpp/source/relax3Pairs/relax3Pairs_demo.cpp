#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "../problems/points3d_3pairs/compatibility_points3d_3pairs.hpp"
#include "relax3Pairs.hpp"


int main()
{
    const int runoff = 37 ;
    Eigen::Vector3d tmp;
    for (int i=0; i<runoff; i++) {
        Eigen::Vector3d tmp = Eigen::Vector3d::Random(3);
    }

    std::cout << "relax_points3d_3pairs top" << std::endl;
    const int n = 10;
    const float scale = 10.0;
    std::vector<Eigen::Vector3d> objects(n);
    for (int i=0; i<n; i++)
    {
        objects[i] = scale*Eigen::Vector3d::Random(3);
    }

    const float noiseScale = .005*scale;
    const int m = n;
    const float labelScale = 0.1;
    const Eigen::Vector3d labelOffset = Eigen::Vector3d(10,200,3000);
    Eigen::MatrixXd labelRotx = Eigen::MatrixXd::Constant(3, 3, 0.0);
    float alpha = .4;
    labelRotx(0,0) = 1.0;
    labelRotx(1,1) = std::cos(alpha);
    labelRotx(1,2) = std::sin(alpha);
    labelRotx(2,1) = -std::sin(alpha);
    labelRotx(2,2) = std::cos(alpha);

    Eigen::MatrixXd labelRoty = Eigen::MatrixXd::Constant(3, 3, 0.0);
    float alphay = .5;
    labelRoty(1,1) = 1.0;
    labelRoty(0,0) = std::cos(alphay);
    labelRoty(0,2) = std::sin(alphay);
    labelRoty(2,0) = -std::sin(alphay);
    labelRoty(2,2) = std::cos(alphay);

    Eigen::MatrixXd labelRotz = Eigen::MatrixXd::Constant(3, 3, 0.0);
    float alphaz = .6;
    labelRotz(2,2) = 1.0;
    labelRotz(0,0) = std::cos(alphaz);
    labelRotz(0,1) = std::sin(alphaz);
    labelRotz(1,0) = -std::sin(alphaz);
    labelRotz(1,1) = std::cos(alphaz);
    std::vector<Eigen::Vector3d> labels(m);
    for (int i=0; i<m; i++)
    {
        labels[m-1-i] = noiseScale*Eigen::Vector3d::Random(3) + labelOffset + labelRotz*labelRoty*labelRotx*labelScale*objects[i];
    }


    for (int i=0; i<m; i++)
    {
        std::cout << "#" << i << objects[i] << "  " << labels[i] << std::endl;
    }

    //objects.erase(objects.begin()+2);
    //labels.erase(labels.begin());
    labels[0] += Eigen::Vector3d(0,0,0);


    std::cout << "Start --> " << std::endl;

    CompatibilityPoints3d3pairs compatibilityPoints3d3pairs(objects.size(),labels.size(), objects, labels);
    RelaxationLabeling relaxationLabeling(compatibilityPoints3d3pairs.compatibility);
}
