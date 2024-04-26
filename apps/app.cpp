#include <Tensor.hpp>
#include <iostream>

using namespace tc;

int main()
{

    Tensor<int> v0;          // SCALAR
    Tensor<int, 3> v1;       // VECTOR
    Tensor<int, 3, 3> v2;    // MATRIX
    Tensor<int, 3, 3, 3> v3; // 3D MATRIX

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

    std::cout << "\nCONSTRUCTOR:\n";
    Tensor<int, 3> t1(4, 5, 6);
    std::cout << "t1(0) = " << t1(0) << '\n';
    std::cout << "t1(1) = " << t1(1) << '\n';
    std::cout << "t1(2) = " << t1(2) << '\n';

    Tensor<int, 3, 3> t2(4, 5, 6, 4, 5, 6, 4, 5, 6);
    std::cout << "t2(0, 0) = " << t2(0, 0) << '\n';
    std::cout << "t2(0, 1) = " << t2(0, 1) << '\n';
    std::cout << "t2(0, 2) = " << t2(0, 2) << '\n';
    std::cout << "t2(1, 0) = " << t2(1, 0) << '\n';
    std::cout << "t2(1, 1) = " << t2(1, 1) << '\n';
    std::cout << "t2(1, 2) = " << t2(1, 2) << '\n';
    std::cout << "t2(2, 0) = " << t2(2, 0) << '\n';
    std::cout << "t2(2, 1) = " << t2(2, 1) << '\n';
    std::cout << "t2(2, 2) = " << t2(2, 2) << '\n';

    Tensor<int, 3> tt1(1.0f);
    std::cout << "tt1(0) = " << tt1(0) << '\n';
    std::cout << "tt1(1) = " << tt1(1) << '\n';
    std::cout << "tt1(2) = " << tt1(2) << '\n';

    std::cout << "\nSUM:\n";
    Tensor sum1 = t1 + v1;
    std::cout << "sum1(0) = " << sum1(0) << '\n';
    std::cout << "sum1(1) = " << sum1(1) << '\n';
    std::cout << "sum1(2) = " << sum1(2) << '\n';

    Tensor sum2 = t2 + v2;
    std::cout << "sum2(0, 0) = " << sum2(0, 0) << '\n';
    std::cout << "sum2(0, 1) = " << sum2(0, 1) << '\n';
    std::cout << "sum2(0, 2) = " << sum2(0, 2) << '\n';
    std::cout << "sum2(1, 0) = " << sum2(1, 0) << '\n';
    std::cout << "sum2(1, 1) = " << sum2(1, 1) << '\n';
    std::cout << "sum2(1, 2) = " << sum2(1, 2) << '\n';
    std::cout << "sum2(2, 0) = " << sum2(2, 0) << '\n';
    std::cout << "sum2(2, 1) = " << sum2(2, 1) << '\n';
    std::cout << "sum2(2, 2) = " << sum2(2, 2) << '\n';

    std::cout << "\nDIFFERENCE:\n";
    Tensor diff1 = t1 - v1;
    std::cout << "diff1(0) = " << diff1(0) << '\n';
    std::cout << "diff1(1) = " << diff1(1) << '\n';
    std::cout << "diff1(2) = " << diff1(2) << '\n';

    Tensor diff2 = t2 - v2;
    std::cout << "diff2(0, 0) = " << diff2(0, 0) << '\n';
    std::cout << "diff2(0, 1) = " << diff2(0, 1) << '\n';
    std::cout << "diff2(0, 2) = " << diff2(0, 2) << '\n';
    std::cout << "diff2(1, 0) = " << diff2(1, 0) << '\n';
    std::cout << "diff2(1, 1) = " << diff2(1, 1) << '\n';
    std::cout << "diff2(1, 2) = " << diff2(1, 2) << '\n';
    std::cout << "diff2(2, 0) = " << diff2(2, 0) << '\n';
    std::cout << "diff2(2, 1) = " << diff2(2, 1) << '\n';
    std::cout << "diff2(2, 2) = " << diff2(2, 2) << '\n';

    std::cout << "\nSCALED:\n";
    Tensor sc1 = t1 * 0.5;
    std::cout << "sc1(0) = " << sc1(0) << '\n';
    std::cout << "sc1(1) = " << sc1(1) << '\n';
    std::cout << "sc1(2) = " << sc1(2) << '\n';

    Tensor sc2 = 2.5 * t2;
    std::cout << "sc2(0, 0) = " << sc2(0, 0) << '\n';
    std::cout << "sc2(1, 1) = " << sc2(1, 1) << '\n';
    std::cout << "sc2(2, 2) = " << sc2(2, 2) << '\n';

    std::cout << "\nITERATOR:\n";
    for (auto v : v1)
    {
        std::cout << v << '\n';
    }

    for (const auto& v : v1)
    {
        std::cout << v << '\n';
    }

    // Modify using the iterator
    for (auto& v : v1)
    {
        v = 3;
    }

    for (const auto v : v1)
    {
        std::cout << v << '\n';
    }
    


    return 0;
}