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
    struct basic_tensor
    {
        static constexpr auto container_size = (Shapes * ... * 1);

        using type = basic_tensor<ValueType, Shapes...>;
        using value_type = ValueType;
        using index_type = std::size_t;
        using shapes_sequence_type = std::index_sequence<Shapes...>;
        using indices_type = std::array<index_type, shapes_sequence_type::size()>;
        using container_type = std::array<value_type, container_size>;

        static constexpr auto shapes_array = sequence::to_array<shapes_sequence_type>::value;

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

        explicit basic_tensor() : _values{} {}

        explicit basic_tensor(std::initializer_list<value_type> list) : basic_tensor()
        {
            std::copy(list.begin(), list.end(), begin());
        }

        explicit basic_tensor(value_type value): basic_tensor()
        {
            std::fill(begin(), end(), value);
        }

        constexpr auto rank() const noexcept
        {
            return shapes_sequence_type::size();
        }

        constexpr auto size() const noexcept
        {
            return container_size;
        }

        constexpr auto &operator()(std::convertible_to<index_type> auto... indices)
            requires(sizeof...(indices) == shapes_sequence_type::size())
        {
            return _values[flatten(shapes(), indices_type{static_cast<index_type>(indices)...})];
        }

        constexpr const auto &operator()(std::convertible_to<index_type> auto... indices) const
            requires(sizeof...(indices) == shapes_sequence_type::size())
        {
            return *this(indices...);
        }

        constexpr auto &operator[](index_type index)
        {
            return _values[index];
        }

        constexpr const auto &operator[](index_type index) const
        {
            return _values[index];
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
            return _values.cbegin();
        }

        auto cend() const noexcept
        {
            return _values.cend();
        }

        auto rbegin() noexcept
        {
            return _values.rbegin();
        }

        auto rend() noexcept
        {
            return _values.rend();
        }

        auto crbegin() const noexcept
        {
            return _values.crbegin();
        }

        auto crend() const noexcept
        {
            return _values.crend();
        }

        auto operator+(const type &rhs) const
        {
            type out;

            std::transform(cbegin(), cend(), rhs.cbegin(), out.begin(), std::plus{});

            return out;
        }

        auto operator-(const type &rhs) const
        {
            type out;

            std::transform(cbegin(), cend(), rhs.cbegin(), out.begin(), std::minus{});

            return out;
        }

        auto operator*(value_type rhs) const
        {
            type out;

            std::transform(cbegin(), cend(), out.begin(), [rhs](auto val)
                           { return val * rhs; });

            return out;
        }

        friend auto operator*(value_type lhs, const type &rhs)
        {
            type out;

            std::transform(rhs.cbegin(), rhs.cend(), out.begin(), [lhs](auto val)
                           { return lhs * val; });

            return out;
        }

        auto operator/(value_type rhs) const
        {
            type out;

            std::transform(cbegin(), cend(), out.begin(), [rhs](auto val)
                           { return val / rhs; });

            return out;
        }

        constexpr auto shape(index_type dimension) const noexcept
        {
            return shapes()[dimension];
        }

        constexpr const auto &shapes() const noexcept
        {
            return shapes_array;
        }

    private:
        container_type _values;
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
    auto transpose(const basic_tensor<ValueType, Shapes...> &t) noexcept
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
            out[j] = t[i];
        }

        return out;
    }

    template <typename ValueType, std::size_t... LShapes, std::size_t... RShapes>
    auto operator*(const basic_tensor<ValueType, LShapes...> &lhs, const basic_tensor<ValueType, RShapes...> &rhs)
    {
        using lhs_shapes_sequence_type = std::index_sequence<LShapes...>;
        using rhs_shapes_sequence_type = std::index_sequence<RShapes...>;

        static_assert(sequence::last<lhs_shapes_sequence_type>::value == sequence::first<rhs_shapes_sequence_type>::value, "LHS last shape doesn't match RHS first shape.");

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

                    value += lhs[ii] * rhs[jj];
                }

                auto index = out.flatten(out.shapes(), out_indices);
                out[index] = value;
            }
        }

        return out;
    }

}

#endif