#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<int, 1, 3> v1(1, 2, 3);
    tensor<int, 3, 1> t1(2, 4, 6);

    tensor<int, 4> q1(2, 4, 6, 7);
    tensor<int, 4> r1(2, 1, 2, 1);

    tensor dot = v1 * t1;
    assert(dot.rank() == 2);
    assert(dot.shape(0) == 1);
    assert(dot.shape(1) == 1);
    assert(dot(0, 0) == 28);

    tensor dot2 = q1 * r1;
    assert(dot2.rank() == 0);
    assert(dot2() == 27);

    tensor<int, 3, 3> m1(1, 2, 3, 4, 5, 6, 7, 8, 9);
    tensor<int, 3, 3> m2(1, 0, 0, 0, 1, 0, 0, 0, 1);

    tensor mul = m1 * m2;
    assert(mul.rank() == 2);
    assert(mul.shape(0) == m1.shape(0));
    assert(mul.shape(1) == m2.shape(1));
    for (auto i = 0; i < mul.size(); i++)
    {
        assert(mul[i] == (i + 1));
    }

    return 0;
}