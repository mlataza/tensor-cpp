#include <tensor.hpp>
#include <iostream>

using namespace tc;

int main()
{
    tensor<3> pos{1.0, 2.0, 3.0};

    std::cout << "pos = {" 
              << pos(0) << ", " 
              << pos(1) << ", " 
              << pos(2) << "}\n";

    return 0;
}