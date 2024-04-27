#ifndef DEFINE_SEQUENCE_HPP_

#include <utility>
#include <array>

// Credits to https://devblogs.microsoft.com/oldnewthing/20200625-00/?p=103903 for hints.

namespace tc::sequence
{
    template <std::size_t N>
    using index = std::make_index_sequence<N>;

    template <typename Seq>
    struct to_array;

    template <std::size_t... N>
    struct to_array<std::index_sequence<N...>>
    {
        static constexpr std::array<std::size_t, sizeof...(N)> value{N...};
    };

    template <std::size_t Offset, typename Seq>
    struct offset;

    template <std::size_t Offset, std::size_t... N>
    struct offset<Offset, std::index_sequence<N...>>
    {
        using type = std::index_sequence<(N + Offset)...>;
    };

    template <typename Seq1, typename Seq2>
    struct cat;

    template <std::size_t... N1, std::size_t... N2>
    struct cat<std::index_sequence<N1...>, std::index_sequence<N2...>>
    {
        using type = std::index_sequence<N1..., N2...>;
    };

    template <typename Seq1, typename Seq2>
    struct get;

    template <typename Seq1, std::size_t... Index>
    struct get<Seq1, std::index_sequence<Index...>>
    {
        using type = std::index_sequence<std::get<Index>(to_array<Seq1>::value)...>;
    };

    template <std::size_t Dim0, std::size_t Dim1, typename Seq>
    struct transpose;

    template <std::size_t Dim0, std::size_t Dim1, std::size_t... N>
        requires(Dim0 < Dim1 && Dim1 < sizeof...(N))
    struct transpose<Dim0, Dim1, std::index_sequence<N...>>
    {
        // Build the sub-sequencies
        using seq0 = index<Dim0>;
        using seq1 = std::index_sequence<Dim1>;
        using seq2 = offset<Dim0 + 1, index<Dim1 - Dim0 - 1>>::type;
        using seq3 = std::index_sequence<Dim0>;
        using seq4 = offset<Dim1 + 1, index<sizeof...(N) - Dim1 - 1>>::type;

        // Concatenate all the sub-sequences pair by pair
        using seq01 = cat<seq0, seq1>::type;
        using seq012 = cat<seq01, seq2>::type;
        using seq0123 = cat<seq012, seq3>::type;
        using type = cat<seq0123, seq4>::type;
    };
}

#endif