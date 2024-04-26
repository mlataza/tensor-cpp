#include <Tensor.hpp>
#include <cassert>

using namespace tc;

int main()
{
    Tensor<int> v0;          // SCALAR
    Tensor<int, 3> v1;       // VECTOR
    Tensor<int, 3, 3> v2;    // MATRIX
    Tensor<int, 3, 3, 3> v3; // 3D MATRIX

    assert(v0.rank() == 0);
    assert(v1.rank() == 1);
    assert(v2.rank() == 2);
    assert(v3.rank() == 3);

    return 0;
}