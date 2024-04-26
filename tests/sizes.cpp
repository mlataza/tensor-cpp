#include <Tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    Tensor<int> v0;          // SCALAR
    Tensor<int, 3> v1;       // VECTOR
    Tensor<int, 3, 3> v2;    // MATRIX
    Tensor<int, 3, 3, 3> v3; // 3D MATRIX

    assert(v0.size() == 1);
    assert(v1.size() == 3);
    assert(v2.size() == 9);
    assert(v3.size() == 27);

    return 0;
}