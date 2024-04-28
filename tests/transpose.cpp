#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    // 2d tensor transpose
    tensor<2, 3> m{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    auto m_p = transpose<0, 1>(m);

    assert(m.shape(0) == m_p.shape(1));
    assert(m.shape(1) == m_p.shape(0));

    for (auto i = 0; i < m.shape(0); i++)
    {
        for (auto j = 0; j < m.shape(1); j++)
        {
            assert(m(i, j) == m_p(j, i));
        }
    }

    auto m_pp = transpose<0, 1>(m_p);

    assert(m_pp.shape(0) == m.shape(0));
    assert(m_pp.shape(1) == m.shape(1));

    for (auto i = 0; i < m.shape(0); i++)
    {
        for (auto j = 0; j < m.shape(1); j++)
        {
            assert(m(i, j) == m_pp(i, j));
        }
    }

    // 3d tensor transpose
    tensor<2, 3, 2> t{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    auto t1 = transpose<0, 1>(t);

    assert(t.shape(0) == t1.shape(1));
    assert(t.shape(1) == t1.shape(0));

    for (auto i = 0; i < t.shape(0); i++)
    {
        for (auto j = 0; j < t.shape(1); j++)
        {
            for (auto k = 0; k < t.shape(2); k++)
            {
                assert(t(i, j, k) == t1(j, i, k));
            }
        }
    }

    auto t2 = transpose<1, 2>(t);

    assert(t.shape(1) == t2.shape(2));
    assert(t.shape(2) == t2.shape(1));

    for (auto i = 0; i < t.shape(0); i++)
    {
        for (auto j = 0; j < t.shape(1); j++)
        {
            for (auto k = 0; k < t.shape(2); k++)
            {
                assert(t(i, j, k) == t2(i, k, j));
            }
        }
    }

    auto t3 = transpose<0, 2>(t);

    assert(t.shape(0) == t3.shape(2));
    assert(t.shape(2) == t3.shape(0));

    for (auto i = 0; i < t.shape(0); i++)
    {
        for (auto j = 0; j < t.shape(1); j++)
        {
            for (auto k = 0; k < t.shape(2); k++)
            {
                assert(t(i, j, k) == t3(k, j, i));
            }
        }
    }

    return 0;
}