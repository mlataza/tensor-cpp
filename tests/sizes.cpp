#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<int> v0;          // SCALAR
    tensor<int, 3> v1;       // VECTOR
    tensor<int, 3, 3> v2;    // MATRIX
    tensor<int, 3, 3, 3> v3; // 3D MATRIX

    assert(v0.size() == 1);
    assert(v1.size() == 3);
    assert(v2.size() == 9);
    assert(v3.size() == 27);

    return 0;
}