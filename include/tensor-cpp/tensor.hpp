#ifndef DEFINE_TENSOR_HPP_
#define DEFINE_TENSOR_HPP_

#include <array>
#include <concepts>
#include <iostream>
#include <ranges>

#include "sequence.hpp"

namespace tc
{
    template <typename ValueType, std::size_t... Shapes>
        requires(std::is_arithmetic_v<ValueType>)
    struct tensor
    {
        static constexpr auto container_size = (Shapes * ... * 1);

        using value_type = ValueType;
        using index_type = std::size_t;
        using shapes_sequence_type = std::index_sequence<Shapes...>;
        using indices_type = std::array<index_type, shapes_sequence_type::size()>;
        using container_type = std::array<value_type, container_size>;

        static constexpr auto shapes_array = sequence::to_array<shapes_sequence_type>::value;

        template <index_type... N>
        static constexpr auto from_sequence(std::index_sequence<N...>)
        {
            return tensor<value_type, N...>{};
        }

        template <typename ShapesRange, typename IndicesRange>
        static auto flatten(const ShapesRange &shapes, IndicesRange &&indices)
        {
            index_type index = 0;

            for (auto it = std::ranges::cbegin(shapes), iit = std::ranges::begin(indices); it != std::ranges::cend(shapes); ++it, ++iit)
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
            for (auto rit = std::ranges::crbegin(shapes), riit = std::ranges::rbegin(indices); rit != std::ranges::crend(shapes); ++rit, ++riit)
            {
                auto shape = *rit;

                *riit = index % shape;
                index /= shape;
            }
        }

        explicit tensor() : _values{} {}

        explicit tensor(std::same_as<value_type> auto... values)
            requires(sizeof...(values) == container_size)
            : _values{values...}
        {
        }

        explicit tensor(value_type value)
        {
            std::fill(_values.begin(), _values.end(), value);
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

        auto operator+(const tensor<value_type, Shapes...> &rhs) const
        {
            tensor<value_type, Shapes...> out;

            for (auto ot = out.begin(), lt = cbegin(), rt = rhs.cbegin(); ot != out.end(); ot++, lt++, rt++)
            {
                *ot = *lt + *rt;
            }

            return out;
        }

        auto operator-(const tensor<value_type, Shapes...> &rhs) const
        {
            tensor<value_type, Shapes...> out;

            for (auto ot = out.begin(), lt = cbegin(), rt = rhs.cbegin(); ot != out.end(); ot++, lt++, rt++)
            {
                *ot = *lt - *rt;
            }

            return out;
        }

        auto operator*(value_type rhs) const
        {
            tensor<value_type, Shapes...> out;

            for (auto ot = out.begin(), lt = cbegin(); ot != out.end(); ot++, lt++)
            {
                *ot = *lt * rhs;
            }

            return out;
        }

        friend auto operator*(value_type lhs, const tensor<value_type, Shapes...> &rhs)
        {
            tensor<value_type, Shapes...> out;

            for (auto ot = out.begin(), rt = rhs.cbegin(); ot != out.end(); ot++, rt++)
            {
                *ot = lhs * *rt;
            }

            return out;
        }

        auto operator/(value_type rhs) const
        {
            tensor<value_type, Shapes...> out;

            for (auto ot = out.begin(), lt = cbegin(); ot != out.end(); ot++, lt++)
            {
                *ot = *lt / rhs;
            }

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

        template <index_type Dim0, index_type Dim1,
                  typename shapes_index_sequence_type = std::make_index_sequence<shapes_sequence_type::size()>,
                  typename transposed_shapes_index_sequence_type = sequence::transpose<Dim0, Dim1, shapes_index_sequence_type>::type,
                  typename out_shapes_sequence_type = sequence::get<shapes_sequence_type, transposed_shapes_index_sequence_type>::type>
        auto transpose() const noexcept
        {
            auto out = from_sequence(out_shapes_sequence_type{});
            indices_type indices;

            // Perform the transposition
            for (index_type i = 0; i < size(); i++)
            {
                // Get the indices from index
                expand(shapes(), i, indices);

                // Swap the indices of Dim0 and Dim1
                std::swap(indices[Dim0], indices[Dim1]);

                // Compute the new index for the result
                auto j = flatten(out.shapes(), indices);

                // Copy value to result
                out[j] = _values[i];
            }

            return out;
        }

        template <index_type... RShapes,
                  typename rhs_shapes_sequence_type = std::index_sequence<RShapes...>,
                  typename lhs_trim_shapes_sequence_type = sequence::trim_last<shapes_sequence_type>::type,
                  typename rhs_trim_shapes_sequence_type = sequence::trim_first<rhs_shapes_sequence_type>::type,
                  typename out_shapes_sequence_type = sequence::cat<lhs_trim_shapes_sequence_type, rhs_trim_shapes_sequence_type>::type>
            requires((Shapes, ...) == sequence::first<rhs_shapes_sequence_type>::value)
        auto operator*(const tensor<value_type, RShapes...> &rhs) const noexcept
        {
            const auto common_shape = (Shapes, ...);
            const auto lhs_shapes = sequence::to_array<lhs_trim_shapes_sequence_type>::value;
            const auto rhs_shapes = sequence::to_array<rhs_trim_shapes_sequence_type>::value;
            const auto lhs_size = size() / common_shape;
            const auto rhs_size = rhs.size() / common_shape;

            auto out = from_sequence(out_shapes_sequence_type{});
            std::array<index_type, out_shapes_sequence_type::size()> out_indices;
            indices_type lhs_indices;
            std::array<index_type, sizeof...(RShapes)> rhs_indices;
            std::array<index_type, lhs_trim_shapes_sequence_type::size()> lhs_trim_indices;
            std::array<index_type, rhs_trim_shapes_sequence_type::size()> rhs_trim_indices;

            // Perform tensor multiplication
            for (index_type i = 0; i < lhs_size; i++)
            {
                expand(lhs_shapes, i, lhs_trim_indices);

                for (index_type j = 0; j < rhs_size; j++)
                {
                    expand(rhs_shapes, j, rhs_trim_indices);

                    value_type value{};

                    for (index_type k = 0; k < common_shape; k++)
                    {
                        std::copy(lhs_trim_indices.begin(), lhs_trim_indices.end(), lhs_indices.begin());
                        *std::ranges::rbegin(lhs_indices) = k;

                        std::copy(rhs_trim_indices.begin(), rhs_trim_indices.end(), rhs_indices.begin() + 1);
                        *std::ranges::begin(rhs_indices) = k;

                        auto ii = flatten(shapes(), lhs_indices);
                        auto jj = flatten(rhs.shapes(), rhs_indices);

                        value += _values[ii] * rhs[jj];
                    }

                    std::copy(lhs_trim_indices.begin(), lhs_trim_indices.end(), out_indices.begin());
                    std::copy(rhs_trim_indices.begin(), rhs_trim_indices.end(), out_indices.begin() + lhs_trim_indices.size());

                    auto index = flatten(out.shapes(), out_indices);
                    out[index] = value;
                }
            }

            return out;
        }

    private:
        container_type _values;
    };

}

#endif