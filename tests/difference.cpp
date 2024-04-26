#include <Tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    Tensor<int> v0;          
    Tensor<int, 3> v1(1, 2, 3);       
    Tensor<int, 3, 3> v2(1, 2, 3, 4, 5, 6, 7, 8, 9);

    Tensor<int, 3> t1(4, 5, 6);
    Tensor<int, 3, 3> t2(4, 5, 6, 4, 5, 6, 4, 5, 6);

    Tensor diff1 = t1 - v1;
    assert(diff1(0) == 3);
    assert(diff1(1) == 3);
    assert(diff1(2) == 3);

    Tensor diff2 = t2 - v2;
    assert(diff2(0, 0) == 3);
    assert(diff2(0, 1) == 3);
    assert(diff2(0, 2) == 3);
    assert(diff2(1, 0) == 0);
    assert(diff2(1, 1) == 0);
    assert(diff2(1, 2) == 0);
    assert(diff2(2, 0) == -3);
    assert(diff2(2, 1) == -3);
    assert(diff2(2, 2) == -3);

    return 0;
}