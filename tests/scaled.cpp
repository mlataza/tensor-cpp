#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<3> t1(4.0, 5.0, 6.0);
    tensor<3, 3> t2(4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0);

    tensor sc1 = t1 * 0.5;
    assert(sc1(0) == 2.0);
    assert(sc1(1) == 2.5);
    assert(sc1(2) == 3.0);

    tensor sc2 = 2.5 * t2;
    assert(sc2(0, 0) == 10.0);
    assert(sc2(0, 1) == 12.5);
    assert(sc2(0, 2) == 15.0);
    assert(sc2(1, 0) == 10.0);
    assert(sc2(1, 1) == 12.5);
    assert(sc2(1, 2) == 15.0);
    assert(sc2(2, 0) == 10.0);
    assert(sc2(2, 1) == 12.5);
    assert(sc2(2, 2) == 15.0);

    tensor sc3 = 3 * t2;
    assert(sc3(0, 0) == 12);
    assert(sc3(0, 1) == 15);
    assert(sc3(0, 2) == 18);
    assert(sc3(1, 0) == 12);
    assert(sc3(1, 1) == 15);
    assert(sc3(1, 2) == 18);
    assert(sc3(2, 0) == 12);
    assert(sc3(2, 1) == 15);
    assert(sc3(2, 2) == 18);

    tensor<3> t3(3.0, 4.0, 5.0);

    tensor sc4 = t3 / 2.0;
    assert(sc4(0) == 1.5);
    assert(sc4(1) == 2.0);
    assert(sc4(2) == 2.5);

    return 0;
}