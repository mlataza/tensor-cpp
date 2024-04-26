#ifndef DEFINE_TENSOR_HPP_
#define DEFINE_TENSOR_HPP_

#include <array>
#include <initializer_list>
#include <vector>
#include <concepts>
#include <iostream>

namespace tc
{
    template <typename ValueType>
    concept is_arithmetic = std::is_arithmetic_v<ValueType>;

    template <is_arithmetic ValueType, std::size_t... Shapes>
    class Tensor
    {
        static constexpr auto _size = (Shapes * ... * 1);
        static constexpr auto _rank = sizeof...(Shapes);
        static constexpr auto _shapes = std::array<std::size_t, _rank>{Shapes...};

        std::array<ValueType, _size> _values;

        // Flattened index = (...((0 x n1 + i1) x n2 + i2) x n3 + i3 ...) x nk + ik (row major order)
        constexpr auto _flatten(std::convertible_to<std::size_t> auto... indices) const
            requires(sizeof...(indices) == _rank)
        {
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
        explicit Tensor() : _values{} {}

        explicit Tensor(std::same_as<ValueType> auto... values)
            requires(sizeof...(values) == _size)
            : _values{values...}
        {
        }

        explicit Tensor(ValueType value)
        {
            for (auto &val : _values)
            {
                val = value;
            }
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

        auto begin() noexcept
        {
            return _values.begin();
        }

        auto end() noexcept
        {
            return _values.end();
        }

        auto cbegin() const noexcept
        {
            return _values.begin();
        }

        auto cend() const noexcept
        {
            return _values.end();
        }

        auto operator+(const Tensor<ValueType, Shapes...> &rTensor) const
        {
            Tensor<ValueType, Shapes...> sum;

            for (std::size_t i = 0; i < size(); i++)
            {
                sum._values[i] = _values[i] + rTensor._values[i];
            }

            return sum;
        }

        auto operator-(const Tensor<ValueType, Shapes...> &rTensor) const
        {
            Tensor<ValueType, Shapes...> sum;

            for (std::size_t i = 0; i < size(); i++)
            {
                sum._values[i] = _values[i] - rTensor._values[i];
            }

            return sum;
        }

        auto operator*(ValueType rScalar) const
        {
            Tensor<ValueType, Shapes...> scaled;

            for (std::size_t i = 0; i < size(); i++)
            {
                scaled._values[i] = _values[i] * rScalar;
            }

            return scaled;
        }

        friend auto operator*(ValueType lScalar, const Tensor<ValueType, Shapes...> &rTensor)
        {
            Tensor<ValueType, Shapes...> scaled;

            for (std::size_t i = 0; i < rTensor.size(); i++)
            {
                scaled._values[i] = lScalar * rTensor._values[i];
            }

            return scaled;
        }
    };

}

#endif