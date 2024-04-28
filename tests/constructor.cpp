#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<3> t1(4.0, 5.0, 6.0);
    assert(t1(0) == 4);
    assert(t1(1) == 5);
    assert(t1(2) == 6);

    tensor<3, 3> t2(4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0);
    assert(t2(0, 0) == 4);
    assert(t2(0, 1) == 5);
    assert(t2(0, 2) == 6);
    assert(t2(1, 0) == 4);
    assert(t2(1, 1) == 5);
    assert(t2(1, 2) == 6);
    assert(t2(2, 0) == 4);
    assert(t2(2, 1) == 5);
    assert(t2(2, 2) == 6);

    tensor<3> tt1(1.5f);
    assert(tt1(0) == 1.5);
    assert(tt1(1) == 1.5);
    assert(tt1(2) == 1.5);

    return 0;
}