#include <Tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    Tensor<int, 3> t1(4, 5, 6);
    Tensor<int, 3, 3> t2(4, 5, 6, 4, 5, 6, 4, 5, 6);

    Tensor sc1 = t1 * 0.5;
    assert(sc1(0) == 0);
    assert(sc1(1) == 0);
    assert(sc1(2) == 0);

    Tensor sc2 = 2.5 * t2;
    assert(sc2(0, 0) == 8);
    assert(sc2(0, 1) == 10);
    assert(sc2(0, 2) == 12);
    assert(sc2(1, 0) == 8);
    assert(sc2(1, 1) == 10);
    assert(sc2(1, 2) == 12);
    assert(sc2(2, 0) == 8);
    assert(sc2(2, 1) == 10);
    assert(sc2(2, 2) == 12);

    Tensor sc3 = 3 * t2;
    assert(sc3(0, 0) == 12);
    assert(sc3(0, 1) == 15);
    assert(sc3(0, 2) == 18);
    assert(sc3(1, 0) == 12);
    assert(sc3(1, 1) == 15);
    assert(sc3(1, 2) == 18);
    assert(sc3(2, 0) == 12);
    assert(sc3(2, 1) == 15);
    assert(sc3(2, 2) == 18);

    return 0;
}