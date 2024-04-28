#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<3> t1(3.0);
    tensor<3> t2(4.0);

    // Test += operator
    t1 += t2;
    for (auto v : t1)
    {
        assert(v == 7.0);
    }

    // Test -= operator
    t2 -= t1;
    for (auto v : t2)
    {
        assert(v == -3.0);
    }

    // Test *= operator
    t1 *= 2.0;
    for (auto v : t1)
    {
        assert(v == 14.0);
    }

    // Test /= operator
    t2 /= 0.5;
    for (auto v : t2)
    {
        assert(v == -6.0);
    }
}