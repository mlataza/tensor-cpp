#ifndef DEFINE_TENSOR_HPP_
#define DEFINE_TENSOR_HPP_

#include <array>
#include <initializer_list>
#include <vector>
#include <concepts>
#include <sstream>

namespace tc
{
    template <typename ValueType>
    concept is_arithmetic = std::is_arithmetic_v<ValueType>;

    template <is_arithmetic ValueType, std::size_t... Shapes>
    class Tensor
    {
        static constexpr auto _size = (Shapes * ... * 1);
        static constexpr auto _rank = sizeof...(Shapes);

        std::array<ValueType, _size> _values;

        // Flattened index = (...((0 x n1 + i1) x n2 + i2) x n3 + i3 ...) x nk + ik (row major)
        auto _flatten(std::convertible_to<std::size_t> auto... indices)
            requires(sizeof...(indices) == _rank)
        {
            static constexpr auto _shapes = std::array<std::size_t, _rank>{Shapes...};
            const auto _indices = std::array<std::size_t, _rank>{static_cast<std::size_t>(indices)...};

            std::size_t index = 0;

            for (std::size_t k = 0; k < _rank; k++)
            {
                auto i = _indices[k];
                auto n = _shapes[k];

                if (i >= n)
                {
                    throw std::out_of_range("Index exceeds dimension");
                }

                index = (index * n) + i;
            }

            return index;
        }

    public:
        Tensor() : _values{} {}

        Tensor(std::same_as<ValueType> auto... values)
            requires(sizeof...(values) == _size)
            : _values{values...}
        {
        }

        constexpr auto rank() const noexcept
        {
            return _rank;
        }

        constexpr auto size() const noexcept
        {
            return _size;
        }

        constexpr auto &operator()(std::convertible_to<std::size_t> auto... indices)
            requires(sizeof...(indices) == _rank)
        {
            return _values[_flatten(indices...)];
        }

        constexpr const auto &operator()(std::convertible_to<std::size_t> auto... indices) const
            requires(sizeof...(indices) == _rank)
        {
            return _values[_flatten(indices...)];
        }
    };

}

#endif