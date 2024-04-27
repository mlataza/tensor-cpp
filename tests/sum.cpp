#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<int> v0;          
    tensor<int, 3> v1(1, 2, 3);       
    tensor<int, 3, 3> v2(1, 2, 3, 4, 5, 6, 7, 8, 9);

    tensor<int, 3> t1(4, 5, 6);
    tensor<int, 3, 3> t2(4, 5, 6, 4, 5, 6, 4, 5, 6);

    tensor sum1 = t1 + v1;
    assert(sum1(0) == 5);
    assert(sum1(1) == 7);
    assert(sum1(2) == 9);

    tensor sum2 = t2 + v2;
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