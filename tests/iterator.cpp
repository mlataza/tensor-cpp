#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<int, 3> v1(3);

    for (auto v : v1)
    {
        assert(v == 3);
    }

    tensor<int, 3, 3> v2(6);

    for (auto v : v2)
    {
        assert(v == 6);
    }

    return 0;
}