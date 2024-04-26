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

    Tensor sum1 = t1 + v1;
    assert(sum1(0) == 5);
    assert(sum1(1) == 7);
    assert(sum1(2) == 9);

    Tensor sum2 = t2 + v2;
    assert(sum2(0, 0) == 5);
    assert(sum2(0, 1) == 7);
    assert(sum2(0, 2) == 9);
    assert(sum2(1, 0) == 8);
    assert(sum2(1, 1) == 10);
    assert(sum2(1, 2) == 12);
    assert(sum2(2, 0) == 11);
    assert(sum2(2, 1) == 13);
    assert(sum2(2, 2) == 15);

    return 0;
}