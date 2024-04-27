#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<int> v0;          // SCALAR
    tensor<int, 3> v1;       // VECTOR
    tensor<int, 3, 2> v2;    // MATRIX
    tensor<int, 3, 2, 1> v3; // 3D MATRIX

    assert(v1.shape(0) == 3);
    assert(v2.shape(0) == 3);
    assert(v2.shape(1) == 2);
    assert(v3.shape(0) == 3);
    assert(v3.shape(1) == 2);
    assert(v3.shape(2) == 1);

    auto s = v3.shapes();

    assert(s[0] == 3);
    assert(s[1] == 2);
    assert(s[2] == 1);

    return 0;
}