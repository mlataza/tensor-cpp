#include <tensor.hpp>

#include <cassert>

using namespace tc;

int main()
{
    tensor<3> v1(3);

    for (auto v : v1)
    {
        assert(v == 3);
    }
    
    for (auto it = std::rbegin(v1); it != std::rend(v1); ++it)
    {
        assert(*it == 3);
    }

    tensor<3, 3> v2(6);

    for (auto v : v2)
    {
        assert(v == 6);
    }

    for (auto it = std::rbegin(v2); it != std::rend(v2); ++it)
    {
        assert(*it == 6);
    }

    return 0;
}