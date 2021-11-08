#include <iostream>

#include "../problems/points3d/compatibility_points3d.hpp"
#include "relax.hpp"

int main()
{
    std::cout << "relax_demo top" << std::endl;
    CompatibilityPoints3d compatibilityPoints3d(2,2);
    RelaxationLabeling relaxationLabeling(compatibilityPoints3d.compatibility);
}
