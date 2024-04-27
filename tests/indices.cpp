#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<int> v0;          // SCALAR
    tensor<int, 3> v1;       // VECTOR
    tensor<int, 3, 3> v2;    // MATRIX
    tensor<int, 3, 3, 3> v3; // 3D MATRIX

    v0() = 1;
    assert(v0() == 1);

    v1(0) = 1, v1(1) = 2, v1(2) = 3;
    assert(v1(0) == 1);
    assert(v1(1) == 2);
    assert(v1(2) == 3);

    v2(0, 0) = 1, v2(0, 1) = 2, v2(0, 2) = 3;
    v2(1, 0) = 4, v2(1, 1) = 5, v2(1, 2) = 6;
    v2(2, 0) = 7, v2(2, 1) = 8, v2(2, 2) = 9;
    assert(v2(0, 0) == 1);
    assert(v2(0, 1) == 2);
    assert(v2(0, 2) == 3);
    assert(v2(1, 0) == 4);
    assert(v2(1, 1) == 5);
    assert(v2(1, 2) == 6);
    assert(v2(2, 0) == 7);
    assert(v2(2, 1) == 8);
    assert(v2(2, 2) == 9);

    v3(0, 0, 0) = v3(1, 1, 1) = v3(2, 2, 2) = 3;
    assert(v3(0, 0, 0) == 3);
    assert(v3(1, 1, 1) == 3);
    assert(v3(2, 2, 2) == 3);
    assert(v3(1, 0, 0) == 0);
    assert(v3(0, 0, 1) == 0);

    return 0;
}