#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<1, 3> v1{1.0, 2.0, 3.0};
    tensor<3, 1> t1{2.0, 4.0, 6.0};

    tensor<4> q1{2.0, 4.0, 6.0, 7.0};
    tensor<4> r1{2.0, 1.0, 2.0, 1.0};

    tensor dot = v1 * t1;
    assert(dot.rank() == 2);
    assert(dot.shape(0) == 1);
    assert(dot.shape(1) == 1);
    assert(dot(0, 0) == 28);

    tensor dot2 = q1 * r1;
    assert(dot2.rank() == 0);
    assert(dot2() == 27);

    tensor<3, 3> m1{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    tensor<3, 3> m2{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

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