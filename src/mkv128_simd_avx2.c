#include "mkv128_simd_avx2.h"

#if defined(__GNUC__) && defined(__x86_64__)

#pragma GCC push_options
#pragma GCC target("arch=x86-64-v3,tune=znver4")

#include <immintrin.h>

#include "PrecomputedTable128.h"

#define MKV128_AVX2_TARGET
#define MKV128_AVX2_INLINE __attribute__((always_inline)) inline

/*
 * Transpose four ordinary, consecutive MKV blocks into four vectors whose
 * elements belong to different blocks.  Combining two such transpositions
 * gives eight independent 32-bit lanes for every MKV state word.
 */
static MKV128_AVX2_INLINE void transpose_four_blocks(
    const uint8_t *input,
    __m128i words[4])
{
    const __m128i block0 = _mm_loadu_si128((const __m128i *)(input + 0u));
    const __m128i block1 = _mm_loadu_si128((const __m128i *)(input + 16u));
    const __m128i block2 = _mm_loadu_si128((const __m128i *)(input + 32u));
    const __m128i block3 = _mm_loadu_si128((const __m128i *)(input + 48u));
    const __m128i low01 = _mm_unpacklo_epi32(block0, block1);
    const __m128i high01 = _mm_unpackhi_epi32(block0, block1);
    const __m128i low23 = _mm_unpacklo_epi32(block2, block3);
    const __m128i high23 = _mm_unpackhi_epi32(block2, block3);

    words[0] = _mm_unpacklo_epi64(low01, low23);
    words[1] = _mm_unpackhi_epi64(low01, low23);
    words[2] = _mm_unpacklo_epi64(high01, high23);
    words[3] = _mm_unpackhi_epi64(high01, high23);
}

static MKV128_AVX2_INLINE void store_four_blocks(
    uint8_t *output,
    __m128i word0,
    __m128i word1,
    __m128i word2,
    __m128i word3)
{
    const __m128i low01 = _mm_unpacklo_epi32(word0, word1);
    const __m128i high01 = _mm_unpackhi_epi32(word0, word1);
    const __m128i low23 = _mm_unpacklo_epi32(word2, word3);
    const __m128i high23 = _mm_unpackhi_epi32(word2, word3);

    _mm_storeu_si128((__m128i *)(output + 0u),
                     _mm_unpacklo_epi64(low01, low23));
    _mm_storeu_si128((__m128i *)(output + 16u),
                     _mm_unpackhi_epi64(low01, low23));
    _mm_storeu_si128((__m128i *)(output + 32u),
                     _mm_unpacklo_epi64(high01, high23));
    _mm_storeu_si128((__m128i *)(output + 48u),
                     _mm_unpackhi_epi64(high01, high23));
}

static MKV128_AVX2_INLINE __m256i sub_mix_words_avx2(
    __m256i words,
    __m256i byte_mask)
{
    const __m256i index0 = _mm256_srli_epi32(words, 24);
    const __m256i index1 = _mm256_and_si256(
        _mm256_srli_epi32(words, 16), byte_mask);
    const __m256i index2 = _mm256_and_si256(
        _mm256_srli_epi32(words, 8), byte_mask);
    const __m256i index3 = _mm256_and_si256(words, byte_mask);
    const __m256i value0 = _mm256_i32gather_epi32(
        (const int *)L0, index0, 4);
    const __m256i value1 = _mm256_i32gather_epi32(
        (const int *)L1, index1, 4);
    const __m256i value2 = _mm256_i32gather_epi32(
        (const int *)L2, index2, 4);
    const __m256i value3 = _mm256_i32gather_epi32(
        (const int *)L3, index3, 4);

    return _mm256_xor_si256(_mm256_xor_si256(value0, value1),
                            _mm256_xor_si256(value2, value3));
}

/*
 * MKV constructs its byte S-box from two nibbles in GF(16).  AVX2's byte
 * shuffle naturally implements sixteen-entry GF(16) log/antilog tables,
 * avoiding four indexed 32-bit gathers for this second substitution layer.
 * The primitive element is 2 and the field polynomial is x^4 + x + 1.
 */
static MKV128_AVX2_INLINE __m256i sub_words_shuffle_avx2(__m256i words)
{
    const __m256i nibble_mask = _mm256_set1_epi8(15);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i fourteen = _mm256_set1_epi8(14);
    const __m256i log_table = _mm256_setr_epi8(
        0, 0, 1, 4, 2, 8, 5, 10, 3, 14, 9, 7, 6, 13, 11, 12,
        0, 0, 1, 4, 2, 8, 5, 10, 3, 14, 9, 7, 6, 13, 11, 12);
    const __m256i log_power11 = _mm256_setr_epi8(
        0, 0, 11, 14, 7, 13, 10, 5, 3, 4, 9, 2, 6, 8, 1, 12,
        0, 0, 11, 14, 7, 13, 10, 5, 3, 4, 9, 2, 6, 8, 1, 12);
    const __m256i log_power8 = _mm256_setr_epi8(
        0, 0, 8, 2, 1, 4, 10, 5, 9, 7, 12, 11, 3, 14, 13, 6,
        0, 0, 8, 2, 1, 4, 10, 5, 9, 7, 12, 11, 3, 14, 13, 6);
    const __m256i exponent_table = _mm256_setr_epi8(
        1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9, 1,
        1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9, 1);
    const __m256i inverse_table = _mm256_setr_epi8(
        0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8,
        0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8);
    const __m256i left = _mm256_and_si256(
        _mm256_srli_epi16(words, 4), nibble_mask);
    const __m256i right = _mm256_and_si256(words, nibble_mask);
    const __m256i left_zero = _mm256_cmpeq_epi8(left, zero);
    const __m256i right_zero = _mm256_cmpeq_epi8(right, zero);
    __m256i right_exponent = _mm256_add_epi8(
        _mm256_shuffle_epi8(log_power11, left),
        _mm256_shuffle_epi8(log_table, right));
    __m256i left_exponent = _mm256_add_epi8(
        _mm256_shuffle_epi8(log_power8, left),
        _mm256_shuffle_epi8(log_power11, right));
    __m256i output_right;
    __m256i output_left;

    /* Subtracting the all-ones comparison mask adds one modulo 16. */
    right_exponent = _mm256_sub_epi8(
        right_exponent, _mm256_cmpgt_epi8(right_exponent, fourteen));
    left_exponent = _mm256_sub_epi8(
        left_exponent, _mm256_cmpgt_epi8(left_exponent, fourteen));

    output_right = _mm256_shuffle_epi8(exponent_table, right_exponent);
    output_right = _mm256_andnot_si256(left_zero, output_right);
    output_right = _mm256_blendv_epi8(
        output_right, _mm256_shuffle_epi8(inverse_table, left), right_zero);

    output_left = _mm256_shuffle_epi8(exponent_table, left_exponent);
    output_left = _mm256_blendv_epi8(
        output_left, _mm256_shuffle_epi8(inverse_table, right), left_zero);
    output_left = _mm256_andnot_si256(right_zero, output_left);

    return _mm256_xor_si256(
        _mm256_or_si256(_mm256_slli_epi16(output_left, 4), output_right),
        _mm256_set1_epi8(1));
}

static MKV128_AVX2_INLINE void encrypt_eight_blocks_avx2(
    const mkv128_ctx *ctx,
    const uint8_t *input,
    uint8_t *output)
{
    const __m256i byte_mask = _mm256_set1_epi32(0xff);
    const __m256i byte_swap = _mm256_setr_epi8(
        3, 2, 1, 0, 7, 6, 5, 4,
        11, 10, 9, 8, 15, 14, 13, 12,
        3, 2, 1, 0, 7, 6, 5, 4,
        11, 10, 9, 8, 15, 14, 13, 12);
    __m128i first_four[4];
    __m128i second_four[4];
    __m256i state0;
    __m256i state1;
    __m256i state2;
    __m256i state3;
    unsigned round;

    transpose_four_blocks(input, first_four);
    transpose_four_blocks(input + 4u * MKV128_BLOCK_BYTES, second_four);

    state0 = _mm256_shuffle_epi8(
        _mm256_set_m128i(second_four[0], first_four[0]), byte_swap);
    state1 = _mm256_shuffle_epi8(
        _mm256_set_m128i(second_four[1], first_four[1]), byte_swap);
    state2 = _mm256_shuffle_epi8(
        _mm256_set_m128i(second_four[2], first_four[2]), byte_swap);
    state3 = _mm256_shuffle_epi8(
        _mm256_set_m128i(second_four[3], first_four[3]), byte_swap);

    for (round = 0u; round < MKV128_ROUNDS; ++round) {
        const uint32_t *round_key = ctx->round_keys + 8u * round;
        const __m256i word0 = sub_words_shuffle_avx2(
            _mm256_xor_si256(
                sub_mix_words_avx2(
                    _mm256_xor_si256(state0,
                                     _mm256_set1_epi32((int)round_key[0])),
                    byte_mask),
                _mm256_set1_epi32((int)round_key[4])));
        const __m256i word1 = sub_words_shuffle_avx2(
            _mm256_xor_si256(
                sub_mix_words_avx2(
                    _mm256_xor_si256(state1,
                                     _mm256_set1_epi32((int)round_key[1])),
                    byte_mask),
                _mm256_set1_epi32((int)round_key[5])));
        const __m256i word2 = sub_words_shuffle_avx2(
            _mm256_xor_si256(
                sub_mix_words_avx2(
                    _mm256_xor_si256(state2,
                                     _mm256_set1_epi32((int)round_key[2])),
                    byte_mask),
                _mm256_set1_epi32((int)round_key[6])));
        const __m256i word3 = sub_words_shuffle_avx2(
            _mm256_xor_si256(
                sub_mix_words_avx2(
                    _mm256_xor_si256(state3,
                                     _mm256_set1_epi32((int)round_key[3])),
                    byte_mask),
                _mm256_set1_epi32((int)round_key[7])));
        const __m256i all_words = _mm256_xor_si256(
            _mm256_xor_si256(word0, word1),
            _mm256_xor_si256(word2, word3));

        state0 = _mm256_xor_si256(all_words, word0);
        state1 = _mm256_xor_si256(all_words, word1);
        state2 = _mm256_xor_si256(all_words, word2);
        state3 = _mm256_xor_si256(all_words, word3);
    }

    state0 = _mm256_shuffle_epi8(
        _mm256_xor_si256(state0,
                         _mm256_set1_epi32((int)ctx->round_keys[
                             8u * MKV128_ROUNDS])),
        byte_swap);
    state1 = _mm256_shuffle_epi8(
        _mm256_xor_si256(state1,
                         _mm256_set1_epi32((int)ctx->round_keys[
                             8u * MKV128_ROUNDS + 1u])),
        byte_swap);
    state2 = _mm256_shuffle_epi8(
        _mm256_xor_si256(state2,
                         _mm256_set1_epi32((int)ctx->round_keys[
                             8u * MKV128_ROUNDS + 2u])),
        byte_swap);
    state3 = _mm256_shuffle_epi8(
        _mm256_xor_si256(state3,
                         _mm256_set1_epi32((int)ctx->round_keys[
                             8u * MKV128_ROUNDS + 3u])),
        byte_swap);

    store_four_blocks(output,
                      _mm256_castsi256_si128(state0),
                      _mm256_castsi256_si128(state1),
                      _mm256_castsi256_si128(state2),
                      _mm256_castsi256_si128(state3));
    store_four_blocks(output + 4u * MKV128_BLOCK_BYTES,
                      _mm256_extracti128_si256(state0, 1),
                      _mm256_extracti128_si256(state1, 1),
                      _mm256_extracti128_si256(state2, 1),
                      _mm256_extracti128_si256(state3, 1));
}

static MKV128_AVX2_TARGET void encrypt_complete_groups_avx2(
    const mkv128_ctx *ctx,
    const uint8_t *input,
    uint8_t *output,
    size_t group_count)
{
    while (group_count-- != 0u) {
        encrypt_eight_blocks_avx2(ctx, input, output);
        input += 8u * MKV128_BLOCK_BYTES;
        output += 8u * MKV128_BLOCK_BYTES;
    }
}

#pragma GCC pop_options

int mkv128_avx2_available(void)
{
    return __builtin_cpu_supports("avx2") != 0;
}

void mkv128_encrypt_blocks_avx2(const mkv128_ctx *ctx,
                                const uint8_t *input,
                                uint8_t *output,
                                size_t block_count)
{
    if (block_count >= 8u && mkv128_avx2_available()) {
        const size_t group_count = block_count / 8u;
        const size_t vector_blocks = group_count * 8u;

        encrypt_complete_groups_avx2(ctx, input, output, group_count);
        input += vector_blocks * MKV128_BLOCK_BYTES;
        output += vector_blocks * MKV128_BLOCK_BYTES;
        block_count -= vector_blocks;
    }

    while (block_count-- != 0u) {
        mkv128_encrypt_block(ctx, input, output);
        input += MKV128_BLOCK_BYTES;
        output += MKV128_BLOCK_BYTES;
    }
}

#else

int mkv128_avx2_available(void)
{
    return 0;
}

void mkv128_encrypt_blocks_avx2(const mkv128_ctx *ctx,
                                const uint8_t *input,
                                uint8_t *output,
                                size_t block_count)
{
    while (block_count-- != 0u) {
        mkv128_encrypt_block(ctx, input, output);
        input += MKV128_BLOCK_BYTES;
        output += MKV128_BLOCK_BYTES;
    }
}

#endif
