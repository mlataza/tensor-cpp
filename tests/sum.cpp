#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<> v0;          
    tensor<3> v1{1.0, 2.0, 3.0};       
    tensor<3, 3> v2{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    tensor<3> t1{4.0, 5.0, 6.0};
    tensor<3, 3> t2{4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0};

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