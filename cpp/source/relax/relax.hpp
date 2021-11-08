#ifndef _RELAX_HPP
#define _RELAX_HPP

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

struct RelaxationLabeling {
    const Eigen::Tensor<double, 4>& compatibility;
    int numObjects;
    int numLabels;
    Eigen::MatrixXd strength; // can be thought of as the "current probability" of object i being label j
    double supportFactor;
    Eigen::MatrixXd support; // can be thought of as the "current marginal probability" of object i being label j.  That is, it is calculated based on the current strength and the compatibility, and will be used on this iteration to push strengths up and down.
    int iteration;
    int iterationLimit;
    bool save;
    int verbose;
    
    RelaxationLabeling(const Eigen::Tensor<double, 4>& compatibility);
    void iterate();
    void updateSupport();
    void normalizeSupport(size_t i, double iMinimumSupport, double iMaximumSupport);
    void updateStrength();
    void normalizeStrength(size_t i, double iSumStrengths);
    void assign();
};

#endif // _RELAX_HPP
