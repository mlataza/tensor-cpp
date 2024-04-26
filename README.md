# Tensor Library
A header-only library implementing Tensor template class. 

Requires C++ 20 standard.

Basic usage (see `apps/basic.cpp`):
```cpp
#include <Tensor.hpp>
#include <iostream>

using namespace tc;

int main()
{
    Tensor<double, 3> pos{1.0, 2.0, 3.0};

    std::cout << "pos = {" 
              << pos(0) << ", " 
              << pos(1) << ", " 
              << pos(2) << "}\n";

    return 0;
}
```

Output should be as follows:
```bash
$ apps/basic
pos = {1, 2, 3}
```