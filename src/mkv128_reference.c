#include "mkv128_reference.h"

#include <stddef.h>

/* A^4, derived from the MKV companion matrix over GF(2^8). */
static const uint8_t MIXWORDS[4][4] = {
    {0x01u, 0x02u, 0x01u, 0x03u},
    {0x03u, 0x07u, 0x01u, 0x04u},
    {0x04u, 0x0Bu, 0x03u, 0x0Du},
    {0x0Du, 0x1Eu, 0x06u, 0x14u}
};

static inline uint32_t reference_load_be32(const uint8_t input[4])
{
    return ((uint32_t)input[0] << 24u)
         | ((uint32_t)input[1] << 16u)
         | ((uint32_t)input[2] << 8u)
         | (uint32_t)input[3];
}

static inline void reference_store_be32(uint8_t output[4], uint32_t value)
{
    output[0] = (uint8_t)(value >> 24u);
    output[1] = (uint8_t)(value >> 16u);
    output[2] = (uint8_t)(value >> 8u);
    output[3] = (uint8_t)value;
}

static uint8_t reference_gf16_multiply(uint8_t left, uint8_t right)
{
    uint8_t result = 0u;
    unsigned bit;

    for (bit = 0u; bit < 4u; ++bit) {
        if ((right & 1u) != 0u) {
            result ^= left;
        }
        left = (uint8_t)((left << 1u)
             ^ (((left & 0x08u) != 0u) ? 0x13u : 0u));
        left &= 0x0Fu;
        right >>= 1u;
    }
    return result;
}

static uint8_t reference_gf16_power(uint8_t value, unsigned exponent)
{
    uint8_t result = 1u;

    if (value == 0u) {
        return 0u;
    }

    while (exponent != 0u) {
        if ((exponent & 1u) != 0u) {
            result = reference_gf16_multiply(result, value);
        }
        value = reference_gf16_multiply(value, value);
        exponent >>= 1u;
    }
    return result;
}

static uint8_t reference_construct_sbox_entry(uint8_t input)
{
    const uint8_t left = (uint8_t)(input >> 4u);
    const uint8_t right = (uint8_t)(input & 0x0Fu);
    uint8_t output_right;
    uint8_t output_left;

    if (right != 0u) {
        output_right = reference_gf16_multiply(
            reference_gf16_power(left, 11u), right);
    } else {
        output_right = reference_gf16_power(left, 14u);
    }

    if (output_right != 0u) {
        output_left = reference_gf16_power(
            reference_gf16_multiply(right, output_right), 13u);
    } else {
        output_left = reference_gf16_power(right, 14u);
    }

    return (uint8_t)((((unsigned)output_left << 4u) | output_right) ^ 0x01u);
}

static inline uint8_t reference_gf256_multiply(uint8_t left, uint8_t right)
{
    uint8_t result = 0u;
    unsigned bit;

    for (bit = 0u; bit < 8u; ++bit) {
        if ((right & 1u) != 0u) {
            result ^= left;
        }
        left = (uint8_t)((left << 1u)
             ^ (((left & 0x80u) != 0u) ? 0x2Bu : 0u));
        right >>= 1u;
    }
    return result;
}

static inline uint32_t reference_sub_word(
    const mkv128_reference_ctx *ctx,
    uint32_t word)
{
    return ((uint32_t)ctx->sbox[word >> 24u] << 24u)
         | ((uint32_t)ctx->sbox[(word >> 16u) & 0xffu] << 16u)
         | ((uint32_t)ctx->sbox[(word >> 8u) & 0xffu] << 8u)
         | (uint32_t)ctx->sbox[word & 0xffu];
}

static inline uint32_t reference_sub_mix_word(
    const mkv128_reference_ctx *ctx,
    uint32_t word)
{
    const uint8_t substituted[4] = {
        ctx->sbox[word >> 24u],
        ctx->sbox[(word >> 16u) & 0xffu],
        ctx->sbox[(word >> 8u) & 0xffu],
        ctx->sbox[word & 0xffu]
    };
    uint32_t packed = 0u;
    unsigned row;
    unsigned column;

    for (row = 0u; row < 4u; ++row) {
        uint8_t value = 0u;

        for (column = 0u; column < 4u; ++column) {
            value ^= reference_gf256_multiply(
                MIXWORDS[row][column], substituted[column]);
        }
        packed = (packed << 8u) | value;
    }

    return packed;
}

static inline void reference_apply_unkeyed_round(
    const mkv128_reference_ctx *ctx,
    uint32_t state[4])
{
    const uint32_t word0 = reference_sub_word(
        ctx, reference_sub_mix_word(ctx, state[0]));
    const uint32_t word1 = reference_sub_word(
        ctx, reference_sub_mix_word(ctx, state[1]));
    const uint32_t word2 = reference_sub_word(
        ctx, reference_sub_mix_word(ctx, state[2]));
    const uint32_t word3 = reference_sub_word(
        ctx, reference_sub_mix_word(ctx, state[3]));

    state[0] = word1 ^ word2 ^ word3;
    state[1] = word0 ^ word2 ^ word3;
    state[2] = word0 ^ word1 ^ word3;
    state[3] = word0 ^ word1 ^ word2;
}

int mkv128_reference_set_encrypt_key(
    mkv128_reference_ctx *ctx,
    const uint8_t key[MKV128_KEY_BYTES])
{
    uint32_t left[4];
    uint32_t right[4];
    unsigned index;
    unsigned round;
    unsigned word;

    if (ctx == NULL || key == NULL) {
        return 0;
    }

    for (index = 0u; index < 256u; ++index) {
        ctx->sbox[index] = reference_construct_sbox_entry((uint8_t)index);
    }

    for (word = 0u; word < 4u; ++word) {
        left[word] = reference_load_be32(key + 4u * word);
        right[word] = left[word] ^ UINT32_MAX;
        ctx->round_keys[word] = left[word];
    }

    for (round = 0u; round < MKV128_ROUNDS; ++round) {
        left[3] ^= 2u * round + 1u;
        reference_apply_unkeyed_round(ctx, left);
        reference_apply_unkeyed_round(ctx, left);

        right[3] ^= 2u * round + 2u;
        reference_apply_unkeyed_round(ctx, right);
        reference_apply_unkeyed_round(ctx, right);

        for (word = 0u; word < 4u; ++word) {
            const uint32_t next_left = right[word];
            const uint32_t next_right = right[word] ^ left[word];

            ctx->round_keys[4u * (2u * round + 1u) + word] = next_left;
            ctx->round_keys[4u * (2u * round + 2u) + word] = next_right;
            left[word] = next_left;
            right[word] = next_right;
        }
    }

    return 1;
}

void mkv128_reference_encrypt_block(
    const mkv128_reference_ctx *ctx,
    const uint8_t input[MKV128_BLOCK_BYTES],
    uint8_t output[MKV128_BLOCK_BYTES])
{
    uint32_t state0 = reference_load_be32(input);
    uint32_t state1 = reference_load_be32(input + 4u);
    uint32_t state2 = reference_load_be32(input + 8u);
    uint32_t state3 = reference_load_be32(input + 12u);
    unsigned round;

    for (round = 0u; round < MKV128_ROUNDS; ++round) {
        const uint32_t *round_key = ctx->round_keys + 8u * round;
        const uint32_t word0 = reference_sub_word(
            ctx, reference_sub_mix_word(ctx, state0 ^ round_key[0]) ^ round_key[4]);
        const uint32_t word1 = reference_sub_word(
            ctx, reference_sub_mix_word(ctx, state1 ^ round_key[1]) ^ round_key[5]);
        const uint32_t word2 = reference_sub_word(
            ctx, reference_sub_mix_word(ctx, state2 ^ round_key[2]) ^ round_key[6]);
        const uint32_t word3 = reference_sub_word(
            ctx, reference_sub_mix_word(ctx, state3 ^ round_key[3]) ^ round_key[7]);

        state0 = word1 ^ word2 ^ word3;
        state1 = word0 ^ word2 ^ word3;
        state2 = word0 ^ word1 ^ word3;
        state3 = word0 ^ word1 ^ word2;
    }

    reference_store_be32(output, state0 ^ ctx->round_keys[8u * MKV128_ROUNDS]);
    reference_store_be32(output + 4u, state1 ^ ctx->round_keys[8u * MKV128_ROUNDS + 1u]);
    reference_store_be32(output + 8u, state2 ^ ctx->round_keys[8u * MKV128_ROUNDS + 2u]);
    reference_store_be32(output + 12u, state3 ^ ctx->round_keys[8u * MKV128_ROUNDS + 3u]);
}
