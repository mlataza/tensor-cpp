#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<int, 3> t1(4, 5, 6);
    assert(t1(0) == 4);
    assert(t1(1) == 5);
    assert(t1(2) == 6);

    tensor<int, 3, 3> t2(4, 5, 6, 4, 5, 6, 4, 5, 6);
    assert(t2(0, 0) == 4);
    assert(t2(0, 1) == 5);
    assert(t2(0, 2) == 6);
    assert(t2(1, 0) == 4);
    assert(t2(1, 1) == 5);
    assert(t2(1, 2) == 6);
    assert(t2(2, 0) == 4);
    assert(t2(2, 1) == 5);
    assert(t2(2, 2) == 6);

    tensor<int, 3> tt1(1.5f);
    assert(tt1(0) == 1);
    assert(tt1(1) == 1);
    assert(tt1(2) == 1);

    return 0;
}