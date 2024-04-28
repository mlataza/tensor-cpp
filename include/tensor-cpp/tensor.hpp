#ifndef DEFINE_TENSOR_HPP_
#define DEFINE_TENSOR_HPP_

#include <array>
#include <concepts>
#include <iostream>
#include <ranges>
#include <algorithm>

#include "sequence.hpp"

namespace tc
{
    template <typename ValueType, std::size_t... Shapes>
        requires(std::is_arithmetic_v<ValueType>)
    class basic_tensor
    {
        std::array<ValueType, (Shapes * ... * 1)> _values;

    public:
        using type = basic_tensor<ValueType, Shapes...>;
        using index_type = std::size_t;
        using shapes_sequence_type = std::index_sequence<Shapes...>;

        template <typename ShapesRange, typename IndicesRange>
        static auto flatten(const ShapesRange &shapes, IndicesRange &&indices)
        {
            index_type index = 0;

            for (auto it = shapes.cbegin(), iit = indices.begin(); it != shapes.cend(); ++it, ++iit)
            {
                auto i = *iit;
                auto n = *it;

                if (i >= n)
                {
                    throw std::out_of_range("Index exceeds dimension");
                }

                index = (index * n) + i;
            }

            return index;
        }

        template <typename ShapesRange, typename IndicesRange>
        static auto expand(const ShapesRange &shapes, index_type index, IndicesRange &&indices)
        {
            for (auto rit = shapes.crbegin(), riit = indices.rbegin(); rit != shapes.crend(); ++rit, ++riit)
            {
                auto shape = *rit;

                *riit = index % shape;
                index /= shape;
            }
        }

        constexpr explicit basic_tensor() : _values{} {}

        constexpr explicit basic_tensor(std::initializer_list<ValueType> list) : basic_tensor()
        {
            std::copy(list.begin(), list.end(), begin());
        }

        constexpr explicit basic_tensor(ValueType value) : basic_tensor()
        {
            std::fill(begin(), end(), value);
        }

        constexpr auto rank() const
        {
            return shapes_sequence_type::size();
        }

        constexpr auto size() const
        {
            return _values.size();
        }

        constexpr auto &operator()(std::convertible_to<index_type> auto... indices)
            requires(sizeof...(indices) == shapes_sequence_type::size())
        {
            return _values[flatten(shapes(), std::array<index_type, shapes_sequence_type::size()>{static_cast<index_type>(indices)...})];
        }

        constexpr const auto &operator()(std::convertible_to<index_type> auto... indices) const
            requires(sizeof...(indices) == shapes_sequence_type::size())
        {
            return _values[flatten(shapes(), std::array<index_type, shapes_sequence_type::size()>{static_cast<index_type>(indices)...})];
        }

        constexpr auto begin()
        {
            return _values.begin();
        }

        constexpr auto end()
        {
            return _values.end();
        }

        constexpr auto begin() const
        {
            return _values.begin();
        }

        constexpr auto end() const
        {
            return _values.end();
        }

        constexpr auto cbegin() const
        {
            return _values.cbegin();
        }

        constexpr auto cend() const
        {
            return _values.cend();
        }

        constexpr auto rbegin()
        {
            return _values.rbegin();
        }

        constexpr auto rend()
        {
            return _values.rend();
        }

        constexpr auto rbegin() const
        {
            return _values.rbegin();
        }

        constexpr auto rend() const
        {
            return _values.rend();
        }

        constexpr auto crbegin() const
        {
            return _values.crbegin();
        }

        constexpr auto crend() const
        {
            return _values.crend();
        }

        constexpr auto operator+(const type &rhs) const
        {
            type out;

            std::transform(cbegin(), cend(), rhs.cbegin(), out.begin(), std::plus{});

            return out;
        }

        constexpr auto operator-(const type &rhs) const
        {
            type out;

            std::transform(cbegin(), cend(), rhs.cbegin(), out.begin(), std::minus{});

            return out;
        }

        constexpr auto operator*(ValueType rhs) const
        {
            type out;

            std::transform(cbegin(), cend(), out.begin(), [rhs](auto val)
                           { return val * rhs; });

            return out;
        }

        constexpr friend auto operator*(ValueType lhs, const type &rhs)
        {
            type out;

            std::transform(rhs.cbegin(), rhs.cend(), out.begin(), [lhs](auto val)
                           { return lhs * val; });

            return out;
        }

        constexpr auto operator/(ValueType rhs) const
        {
            type out;

            std::transform(cbegin(), cend(), out.begin(), [rhs](auto val)
                           { return val / rhs; });

            return out;
        }

        constexpr auto shape(index_type dimension) const
        {
            return shapes()[dimension];
        }

        constexpr const auto &shapes() const
        {
            return sequence::to_array<shapes_sequence_type>::value;
        }
    };

    template <std::size_t... Shapes>
    using tensor = basic_tensor<double, Shapes...>;

    template <std::size_t... Shapes>
    using tensor_f = basic_tensor<float, Shapes...>;

    template <std::size_t... Shapes>
    using tensor_i = basic_tensor<int, Shapes...>;

    template <std::size_t... Shapes>
    using tensor_l = basic_tensor<long, Shapes...>;

    template <typename ValueType, typename Seq>
    struct tensor_sequence;

    template <typename ValueType, std::size_t... N>
    struct tensor_sequence<ValueType, std::index_sequence<N...>>
    {
        using type = basic_tensor<ValueType, N...>;
    };

    template <std::size_t Dim0, std::size_t Dim1, typename ValueType, std::size_t... Shapes>
    auto transpose(const basic_tensor<ValueType, Shapes...> &t)
    {
        using shapes_sequence_type = std::index_sequence<Shapes...>;
        using shapes_index_sequence_type = std::make_index_sequence<shapes_sequence_type::size()>;
        using transposed_shapes_index_sequence_type = sequence::transpose<Dim0, Dim1, shapes_index_sequence_type>::type;
        using out_shapes_sequence_type = sequence::get<shapes_sequence_type, transposed_shapes_index_sequence_type>::type;
        using out_type = tensor_sequence<ValueType, out_shapes_sequence_type>::type;

        out_type out;
        std::array<std::size_t, shapes_sequence_type::size()> indices;

        // Perform the transposition
        for (std::size_t i = 0; i < t.size(); i++)
        {
            // Get the indices from index
            t.expand(t.shapes(), i, indices);

            // Swap the indices of Dim0 and Dim1
            std::swap(indices[Dim0], indices[Dim1]);

            // Compute the new index for the result
            auto j = t.flatten(out.shapes(), indices);

            // Copy value to result
            *(out.begin() + j) = *(t.cbegin() + i);
        }

        return out;
    }

    template <typename ValueType, std::size_t... LShapes, std::size_t... RShapes>
    auto operator*(const basic_tensor<ValueType, LShapes...> &lhs, const basic_tensor<ValueType, RShapes...> &rhs)
    {
        using lhs_shapes_sequence_type = std::index_sequence<LShapes...>;
        using rhs_shapes_sequence_type = std::index_sequence<RShapes...>;

        static_assert(sequence::last<lhs_shapes_sequence_type>::value == sequence::first<rhs_shapes_sequence_type>::value, "LHS last shape matches RHS first shape.");

        using lhs_trim_shapes_sequence_type = sequence::trim_last<lhs_shapes_sequence_type>::type;
        using rhs_trim_shapes_sequence_type = sequence::trim_first<rhs_shapes_sequence_type>::type;

        using out_shapes_sequence_type = sequence::cat<lhs_trim_shapes_sequence_type, rhs_trim_shapes_sequence_type>::type;
        using out_type = tensor_sequence<ValueType, out_shapes_sequence_type>::type;

        const auto common_shape = (LShapes, ...);

        const auto lhs_shapes = sequence::to_array<lhs_trim_shapes_sequence_type>::value;
        const auto rhs_shapes = sequence::to_array<rhs_trim_shapes_sequence_type>::value;

        const auto lhs_size = lhs.size() / common_shape;
        const auto rhs_size = rhs.size() / common_shape;

        out_type out;

        std::array<std::size_t, out_shapes_sequence_type::size()> out_indices;

        std::array<std::size_t, sizeof...(LShapes)> lhs_indices;
        std::array<std::size_t, sizeof...(RShapes)> rhs_indices;

        std::array<std::size_t, lhs_trim_shapes_sequence_type::size()> lhs_trim_indices;
        std::array<std::size_t, rhs_trim_shapes_sequence_type::size()> rhs_trim_indices;

        // Perform tensor multiplication
        for (std::size_t i = 0; i < lhs_size; i++)
        {
            lhs.expand(lhs_shapes, i, lhs_trim_indices);
            std::copy(lhs_trim_indices.cbegin(), lhs_trim_indices.cend(), lhs_indices.begin());
            std::copy(lhs_trim_indices.cbegin(), lhs_trim_indices.cend(), out_indices.begin());

            for (std::size_t j = 0; j < rhs_size; j++)
            {
                lhs.expand(rhs_shapes, j, rhs_trim_indices);
                std::copy(rhs_trim_indices.cbegin(), rhs_trim_indices.cend(), rhs_indices.begin() + 1);
                std::copy(rhs_trim_indices.cbegin(), rhs_trim_indices.cend(), out_indices.begin() + lhs_trim_indices.size());

                ValueType value = 0;

                for (std::size_t k = 0; k < common_shape; k++)
                {
                    *lhs_indices.rbegin() = k;
                    *rhs_indices.begin() = k;

                    auto ii = lhs.flatten(lhs.shapes(), lhs_indices);
                    auto jj = rhs.flatten(rhs.shapes(), rhs_indices);

                    value += *(lhs.cbegin() + ii) * *(rhs.cbegin() + jj);
                }

                auto index = out.flatten(out.shapes(), out_indices);
                *(out.begin() + index) = value;
            }
        }

        return out;
    }

}

#endif