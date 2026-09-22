#include "mkv128_core.h"

#include <stddef.h>
#include <string.h>

#include "PrecomputedTable128.h"

static inline uint32_t load_be32(const uint8_t input[4])
{
    return ((uint32_t)input[0] << 24)
         | ((uint32_t)input[1] << 16)
         | ((uint32_t)input[2] << 8)
         | (uint32_t)input[3];
}

static inline void store_be32(uint8_t output[4], uint32_t value)
{
    output[0] = (uint8_t)(value >> 24);
    output[1] = (uint8_t)(value >> 16);
    output[2] = (uint8_t)(value >> 8);
    output[3] = (uint8_t)value;
}

/* One lookup per byte evaluates MixWords(SubCells(word)). */
static inline uint32_t sub_mix_word(uint32_t word)
{
    return L0[word >> 24]
         ^ L1[(word >> 16) & 0xffu]
         ^ L2[(word >> 8) & 0xffu]
         ^ L3[word & 0xffu];
}

/* L4 keeps the S-box output in the most significant byte. */
static inline uint32_t sub_word(uint32_t word)
{
    return L4[word >> 24]
         ^ (L4[(word >> 16) & 0xffu] >> 8)
         ^ (L4[(word >> 8) & 0xffu] >> 16)
         ^ (L4[word & 0xffu] >> 24);
}

/* One lookup per byte evaluates invMixWords(invSubCells(word)). */
static inline uint32_t inverse_sub_mix_word(uint32_t word)
{
    return iL0[word >> 24]
         ^ iL1[(word >> 16) & 0xffu]
         ^ iL2[(word >> 8) & 0xffu]
         ^ iL3[word & 0xffu];
}

static inline uint32_t inverse_sub_word(uint32_t word)
{
    return iL4[word >> 24]
         ^ (iL4[(word >> 16) & 0xffu] >> 8)
         ^ (iL4[(word >> 8) & 0xffu] >> 16)
         ^ (iL4[word & 0xffu] >> 24);
}

static inline void apply_unkeyed_round(uint32_t state[4])
{
    const uint32_t word0 = sub_word(sub_mix_word(state[0]));
    const uint32_t word1 = sub_word(sub_mix_word(state[1]));
    const uint32_t word2 = sub_word(sub_mix_word(state[2]));
    const uint32_t word3 = sub_word(sub_mix_word(state[3]));

    state[0] = word1 ^ word2 ^ word3;
    state[1] = word0 ^ word2 ^ word3;
    state[2] = word0 ^ word1 ^ word3;
    state[3] = word0 ^ word1 ^ word2;
}

int mkv128_set_encrypt_key(mkv128_ctx *ctx,
                           const uint8_t key[MKV128_KEY_BYTES])
{
    uint32_t left[4];
    uint32_t right[4];
    unsigned round;
    unsigned word;

    if (ctx == NULL || key == NULL) {
        return 0;
    }

    for (word = 0; word < 4u; ++word) {
        left[word] = load_be32(key + 4u * word);
        right[word] = left[word] ^ UINT32_MAX;
        ctx->round_keys[word] = left[word];
    }

    for (round = 0; round < MKV128_ROUNDS; ++round) {
        left[3] ^= 2u * round + 1u;
        apply_unkeyed_round(left);
        apply_unkeyed_round(left);

        right[3] ^= 2u * round + 2u;
        apply_unkeyed_round(right);
        apply_unkeyed_round(right);

        for (word = 0; word < 4u; ++word) {
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

int mkv128_set_decrypt_key(mkv128_ctx *ctx,
                           const uint8_t key[MKV128_KEY_BYTES])
{
    unsigned round;
    unsigned word;

    if (!mkv128_set_encrypt_key(ctx, key)) {
        return 0;
    }

    /* Move each second subkey through the inverse linear transformation. */
    for (round = 0; round < MKV128_ROUNDS; ++round) {
        const unsigned offset = 4u * (2u * round + 1u);

        for (word = 0; word < 4u; ++word) {
            ctx->round_keys[offset + word] =
                inverse_sub_mix_word(sub_word(ctx->round_keys[offset + word]));
        }
    }

    return 1;
}

void mkv128_encrypt_block(const mkv128_ctx *ctx,
                          const uint8_t input[MKV128_BLOCK_BYTES],
                          uint8_t output[MKV128_BLOCK_BYTES])
{
    uint32_t state0 = load_be32(input);
    uint32_t state1 = load_be32(input + 4u);
    uint32_t state2 = load_be32(input + 8u);
    uint32_t state3 = load_be32(input + 12u);
    unsigned round;

    for (round = 0; round < MKV128_ROUNDS; ++round) {
        const uint32_t *round_key = ctx->round_keys + 8u * round;
        const uint32_t word0 =
            sub_word(sub_mix_word(state0 ^ round_key[0]) ^ round_key[4]);
        const uint32_t word1 =
            sub_word(sub_mix_word(state1 ^ round_key[1]) ^ round_key[5]);
        const uint32_t word2 =
            sub_word(sub_mix_word(state2 ^ round_key[2]) ^ round_key[6]);
        const uint32_t word3 =
            sub_word(sub_mix_word(state3 ^ round_key[3]) ^ round_key[7]);

        state0 = word1 ^ word2 ^ word3;
        state1 = word0 ^ word2 ^ word3;
        state2 = word0 ^ word1 ^ word3;
        state3 = word0 ^ word1 ^ word2;
    }

    store_be32(output, state0 ^ ctx->round_keys[8u * MKV128_ROUNDS]);
    store_be32(output + 4u, state1 ^ ctx->round_keys[8u * MKV128_ROUNDS + 1u]);
    store_be32(output + 8u, state2 ^ ctx->round_keys[8u * MKV128_ROUNDS + 2u]);
    store_be32(output + 12u, state3 ^ ctx->round_keys[8u * MKV128_ROUNDS + 3u]);
}

void mkv128_decrypt_block(const mkv128_ctx *ctx,
                          const uint8_t input[MKV128_BLOCK_BYTES],
                          uint8_t output[MKV128_BLOCK_BYTES])
{
    uint32_t state0 = load_be32(input)
                    ^ ctx->round_keys[8u * MKV128_ROUNDS];
    uint32_t state1 = load_be32(input + 4u)
                    ^ ctx->round_keys[8u * MKV128_ROUNDS + 1u];
    uint32_t state2 = load_be32(input + 8u)
                    ^ ctx->round_keys[8u * MKV128_ROUNDS + 2u];
    uint32_t state3 = load_be32(input + 12u)
                    ^ ctx->round_keys[8u * MKV128_ROUNDS + 3u];
    unsigned round = MKV128_ROUNDS;

    while (round-- != 0u) {
        const uint32_t *round_key = ctx->round_keys + 8u * round;
        const uint32_t word0 = state1 ^ state2 ^ state3;
        const uint32_t word1 = state0 ^ state2 ^ state3;
        const uint32_t word2 = state0 ^ state1 ^ state3;
        const uint32_t word3 = state0 ^ state1 ^ state2;

        state0 = inverse_sub_word(inverse_sub_mix_word(word0) ^ round_key[4])
               ^ round_key[0];
        state1 = inverse_sub_word(inverse_sub_mix_word(word1) ^ round_key[5])
               ^ round_key[1];
        state2 = inverse_sub_word(inverse_sub_mix_word(word2) ^ round_key[6])
               ^ round_key[2];
        state3 = inverse_sub_word(inverse_sub_mix_word(word3) ^ round_key[7])
               ^ round_key[3];
    }

    store_be32(output, state0);
    store_be32(output + 4u, state1);
    store_be32(output + 8u, state2);
    store_be32(output + 12u, state3);
}
