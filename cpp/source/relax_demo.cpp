#include <iostream>

#include "compatibility_dummy.hpp"
#include "relax.hpp"

int main()
{
    std::cout << "relax_demo top" << std::endl;
    CompatibilityDummy compatibilityDummy(2,2);
    RelaxationLabeling relaxationLabeling(compatibilityDummy.compatibility);
}
