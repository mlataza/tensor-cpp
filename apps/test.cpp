#include <Tensor.hpp>
#include <iostream>

using namespace tc;

int main()
{
    Tensor<int> v0;                 // SCALAR
    Tensor<int, 3> v1;              // VECTOR
    Tensor<int, 3, 3> v2;           // MATRIX
    Tensor<int, 3, 3, 3> v3;        // 3D MATRIX

    std::cout << "RANKS: \n";
    std::cout << "v0: " << v0.rank() << '\n';
    std::cout << "v1: " << v1.rank() << '\n';
    std::cout << "v2: " << v2.rank() << '\n';
    std::cout << "v3: " << v3.rank() << '\n';

    std::cout << "\nSIZES: \n";
    std::cout << "v0: " << v0.size() << '\n';
    std::cout << "v1: " << v1.size() << '\n';
    std::cout << "v2: " << v2.size() << '\n';
    std::cout << "v3: " << v3.size() << '\n';

    std::cout << "\nINDICES: \n";
    v0() = 1;
    std::cout << "v0() = " << v0() << '\n';

    v1(0) = 1, v1(1) = 2, v1(2) = 3;
    std::cout << "\nv1(0) = " << v1(0) << '\n';
    std::cout << "v1(1) = " << v1(1) << '\n';
    std::cout << "v1(2) = " << v1(2) << '\n';

    v2(0, 0) = 1, v2(0, 1) = 2, v2(0, 2) = 3;
    v2(1, 0) = 4, v2(1, 1) = 5, v2(1, 2) = 6;
    v2(2, 0) = 7, v2(2, 1) = 8, v2(2, 2) = 9;
    std::cout << "\nv2(0, 0) = " << v2(0, 0) << '\n';
    std::cout << "v2(0, 1) = " << v2(0, 1) << '\n';
    std::cout << "v2(0, 2) = " << v2(0, 2) << '\n';
    std::cout << "v2(1, 0) = " << v2(1, 0) << '\n';
    std::cout << "v2(1, 1) = " << v2(1, 1) << '\n';
    std::cout << "v2(1, 2) = " << v2(1, 2) << '\n';
    std::cout << "v2(2, 0) = " << v2(2, 0) << '\n';
    std::cout << "v2(2, 1) = " << v2(2, 1) << '\n';
    std::cout << "v2(2, 2) = " << v2(2, 2) << '\n';

    v3(0, 0, 0) = v3(1, 1, 1) = v3(2, 2, 2) = 3;
    std::cout << "\nv3(0, 0, 0) = " << v3(0, 0, 0) << '\n';
    std::cout << "v3(1, 1, 1) = " << v3(1, 1, 1) << '\n';
    std::cout << "v3(2, 2, 2) = " << v3(2, 2, 2) << '\n';
    std::cout << "v3(0, 1, 0) = " << v3(0, 1, 0) << '\n';
    std::cout << "v3(1, 0, 0) = " << v3(1, 0, 0) << '\n';
    std::cout << "v3(0, 0, 1) = " << v3(0, 0, 1) << '\n';

    Tensor<int, 3> t1(1, 2, 3);


    return 0;
}