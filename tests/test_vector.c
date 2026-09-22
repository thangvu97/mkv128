/*
 * test_vector.c - ba phuong an ma hoa MKV-128 trong mot tep duy nhat.
 *
 * Ba ham trien khai duoc dung trong main:
 *   1) mkv128_encrypt_direct : S-box va MixWords tinh truc tiep.
 *   2) mkv128_encrypt_ttable : S-box va MixWords da gop vao bang tra T.
 *   3) mkv128_encrypt_avx512 : 16 khoi song song bang AVX-512 gather.
 *
 * Ma tran, S-box va bang T duoc sinh khi chuong trinh khoi dong; khong can
 * them PrecomputedTable128.h hay bat ky tep nguon nao khac de bien dich.
 * Nhanh thu ba la AVX-512 T-table: no van dung cung bang T, nhung doc 16
 * khoi va thuc hien cac phep XOR tren 16 lanh vector trong mot vong.
 * Neu CPU khong co AVX-512, ham tu dong dung lai phuong an bang T de van
 * cho ra ket qua dung.
 *
 * Bien dich tren Windows/MSYS2:
 *   gcc -std=c11 -O3 -march=znver4 tests/test_vector.c -o test_vector.exe
 */

#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define MKV128_CAN_COMPILE_AVX512 1
#include <immintrin.h>
#else
#define MKV128_CAN_COMPILE_AVX512 0
#endif

#define MKV128_BLOCK_BYTES 16u
#define MKV128_KEY_BYTES 16u
#define MKV128_ROUNDS 6u
#define MKV128_KEY_WORDS (4u * (2u * MKV128_ROUNDS + 1u))

typedef struct mkv128_tables {
    uint8_t sbox[256];
    uint32_t l0[256];
    uint32_t l1[256];
    uint32_t l2[256];
    uint32_t l3[256];
    uint32_t l4[256];
} mkv128_tables;

static const uint8_t MIXWORDS[4][4] = {
    {0x01u, 0x02u, 0x01u, 0x03u},
    {0x03u, 0x07u, 0x01u, 0x04u},
    {0x04u, 0x0Bu, 0x03u, 0x0Du},
    {0x0Du, 0x1Eu, 0x06u, 0x14u}
};

static uint8_t gf16_mul(uint8_t left, uint8_t right)
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

static uint8_t gf16_pow(uint8_t value, unsigned exponent)
{
    uint8_t result = 1u;

    if (value == 0u) {
        return 0u;
    }
    while (exponent != 0u) {
        if ((exponent & 1u) != 0u) {
            result = gf16_mul(result, value);
        }
        value = gf16_mul(value, value);
        exponent >>= 1u;
    }
    return result;
}

static uint8_t construct_sbox_entry(uint8_t input)
{
    const uint8_t left = (uint8_t)(input >> 4u);
    const uint8_t right = (uint8_t)(input & 0x0Fu);
    uint8_t output_right;
    uint8_t output_left;

    if (right != 0u) {
        output_right = gf16_mul(gf16_pow(left, 11u), right);
    } else {
        output_right = gf16_pow(left, 14u);
    }

    if (output_right != 0u) {
        output_left = gf16_pow(gf16_mul(right, output_right), 13u);
    } else {
        output_left = gf16_pow(right, 14u);
    }

    return (uint8_t)((((unsigned)output_left << 4u) | output_right)
                     ^ 0x01u);
}

static uint8_t gf256_mul(uint8_t left, uint8_t right)
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

static uint32_t load_be32(const uint8_t input[4])
{
    return ((uint32_t)input[0] << 24u)
         | ((uint32_t)input[1] << 16u)
         | ((uint32_t)input[2] << 8u)
         | (uint32_t)input[3];
}

static void store_be32(uint8_t output[4], uint32_t value)
{
    output[0] = (uint8_t)(value >> 24u);
    output[1] = (uint8_t)(value >> 16u);
    output[2] = (uint8_t)(value >> 8u);
    output[3] = (uint8_t)value;
}

static uint32_t direct_sub_word(const mkv128_tables *tables, uint32_t word)
{
    return ((uint32_t)tables->sbox[word >> 24u] << 24u)
         | ((uint32_t)tables->sbox[(word >> 16u) & 0xffu] << 16u)
         | ((uint32_t)tables->sbox[(word >> 8u) & 0xffu] << 8u)
         | (uint32_t)tables->sbox[word & 0xffu];
}

static uint32_t direct_sub_mix_word(const mkv128_tables *tables,
                                    uint32_t word)
{
    const uint8_t substituted[4] = {
        tables->sbox[word >> 24u],
        tables->sbox[(word >> 16u) & 0xffu],
        tables->sbox[(word >> 8u) & 0xffu],
        tables->sbox[word & 0xffu]
    };
    uint32_t packed = 0u;
    unsigned row;

    for (row = 0u; row < 4u; ++row) {
        uint8_t value = 0u;
        unsigned column;

        for (column = 0u; column < 4u; ++column) {
            value ^= gf256_mul(MIXWORDS[row][column], substituted[column]);
        }
        packed = (packed << 8u) | value;
    }
    return packed;
}

static uint32_t table_sub_mix_word(const mkv128_tables *tables,
                                   uint32_t word)
{
    return tables->l0[word >> 24u]
         ^ tables->l1[(word >> 16u) & 0xffu]
         ^ tables->l2[(word >> 8u) & 0xffu]
         ^ tables->l3[word & 0xffu];
}

static uint32_t table_sub_word(const mkv128_tables *tables, uint32_t word)
{
    return tables->l4[word >> 24u]
         ^ (tables->l4[(word >> 16u) & 0xffu] >> 8u)
         ^ (tables->l4[(word >> 8u) & 0xffu] >> 16u)
         ^ (tables->l4[word & 0xffu] >> 24u);
}

static void initialize_tables(mkv128_tables *tables)
{
    unsigned input;

    memset(tables, 0, sizeof(*tables));
    for (input = 0u; input < 256u; ++input) {
        const uint8_t s = construct_sbox_entry((uint8_t)input);
        unsigned column;

        tables->sbox[input] = s;
        tables->l4[input] = (uint32_t)s << 24u;
        for (column = 0u; column < 4u; ++column) {
            const uint32_t packed =
                ((uint32_t)gf256_mul(MIXWORDS[0][column], s) << 24u)
              | ((uint32_t)gf256_mul(MIXWORDS[1][column], s) << 16u)
              | ((uint32_t)gf256_mul(MIXWORDS[2][column], s) << 8u)
              | (uint32_t)gf256_mul(MIXWORDS[3][column], s);

            if (column == 0u) tables->l0[input] = packed;
            if (column == 1u) tables->l1[input] = packed;
            if (column == 2u) tables->l2[input] = packed;
            if (column == 3u) tables->l3[input] = packed;
        }
    }
}

static void apply_unkeyed_round_direct(const mkv128_tables *tables,
                                       uint32_t state[4])
{
    const uint32_t a = direct_sub_word(
        tables, direct_sub_mix_word(tables, state[0]));
    const uint32_t b = direct_sub_word(
        tables, direct_sub_mix_word(tables, state[1]));
    const uint32_t c = direct_sub_word(
        tables, direct_sub_mix_word(tables, state[2]));
    const uint32_t d = direct_sub_word(
        tables, direct_sub_mix_word(tables, state[3]));

    state[0] = b ^ c ^ d;
    state[1] = a ^ c ^ d;
    state[2] = a ^ b ^ d;
    state[3] = a ^ b ^ c;
}

static void apply_unkeyed_round_table(const mkv128_tables *tables,
                                      uint32_t state[4])
{
    const uint32_t a = table_sub_word(
        tables, table_sub_mix_word(tables, state[0]));
    const uint32_t b = table_sub_word(
        tables, table_sub_mix_word(tables, state[1]));
    const uint32_t c = table_sub_word(
        tables, table_sub_mix_word(tables, state[2]));
    const uint32_t d = table_sub_word(
        tables, table_sub_mix_word(tables, state[3]));

    state[0] = b ^ c ^ d;
    state[1] = a ^ c ^ d;
    state[2] = a ^ b ^ d;
    state[3] = a ^ b ^ c;
}

static void expand_key_direct(const mkv128_tables *tables,
                              const uint8_t key[MKV128_KEY_BYTES],
                              uint32_t round_keys[MKV128_KEY_WORDS])
{
    uint32_t left[4];
    uint32_t right[4];
    unsigned round;
    unsigned word;

    for (word = 0u; word < 4u; ++word) {
        left[word] = load_be32(key + 4u * word);
        right[word] = left[word] ^ UINT32_MAX;
        round_keys[word] = left[word];
    }
    for (round = 0u; round < MKV128_ROUNDS; ++round) {
        left[3] ^= 2u * round + 1u;
        apply_unkeyed_round_direct(tables, left);
        apply_unkeyed_round_direct(tables, left);
        right[3] ^= 2u * round + 2u;
        apply_unkeyed_round_direct(tables, right);
        apply_unkeyed_round_direct(tables, right);

        for (word = 0u; word < 4u; ++word) {
            const uint32_t next_left = right[word];
            const uint32_t next_right = right[word] ^ left[word];

            round_keys[4u * (2u * round + 1u) + word] = next_left;
            round_keys[4u * (2u * round + 2u) + word] = next_right;
            left[word] = next_left;
            right[word] = next_right;
        }
    }
}

static void expand_key_table(const mkv128_tables *tables,
                             const uint8_t key[MKV128_KEY_BYTES],
                             uint32_t round_keys[MKV128_KEY_WORDS])
{
    uint32_t left[4];
    uint32_t right[4];
    unsigned round;
    unsigned word;

    for (word = 0u; word < 4u; ++word) {
        left[word] = load_be32(key + 4u * word);
        right[word] = left[word] ^ UINT32_MAX;
        round_keys[word] = left[word];
    }
    for (round = 0u; round < MKV128_ROUNDS; ++round) {
        left[3] ^= 2u * round + 1u;
        apply_unkeyed_round_table(tables, left);
        apply_unkeyed_round_table(tables, left);
        right[3] ^= 2u * round + 2u;
        apply_unkeyed_round_table(tables, right);
        apply_unkeyed_round_table(tables, right);

        for (word = 0u; word < 4u; ++word) {
            const uint32_t next_left = right[word];
            const uint32_t next_right = right[word] ^ left[word];

            round_keys[4u * (2u * round + 1u) + word] = next_left;
            round_keys[4u * (2u * round + 2u) + word] = next_right;
            left[word] = next_left;
            right[word] = next_right;
        }
    }
}

static void encrypt_with_direct_rounds(
    const mkv128_tables *tables,
    const uint32_t round_keys[MKV128_KEY_WORDS],
    const uint8_t input[MKV128_BLOCK_BYTES],
    uint8_t output[MKV128_BLOCK_BYTES])
{
    uint32_t state[4] = {
        load_be32(input), load_be32(input + 4u),
        load_be32(input + 8u), load_be32(input + 12u)
    };
    unsigned round;

    for (round = 0u; round < MKV128_ROUNDS; ++round) {
        const uint32_t *rk = round_keys + 8u * round;
        const uint32_t a = direct_sub_word(
            tables, direct_sub_mix_word(tables, state[0] ^ rk[0]) ^ rk[4]);
        const uint32_t b = direct_sub_word(
            tables, direct_sub_mix_word(tables, state[1] ^ rk[1]) ^ rk[5]);
        const uint32_t c = direct_sub_word(
            tables, direct_sub_mix_word(tables, state[2] ^ rk[2]) ^ rk[6]);
        const uint32_t d = direct_sub_word(
            tables, direct_sub_mix_word(tables, state[3] ^ rk[3]) ^ rk[7]);

        state[0] = b ^ c ^ d;
        state[1] = a ^ c ^ d;
        state[2] = a ^ b ^ d;
        state[3] = a ^ b ^ c;
    }
    state[0] ^= round_keys[48u];
    state[1] ^= round_keys[49u];
    state[2] ^= round_keys[50u];
    state[3] ^= round_keys[51u];
    store_be32(output, state[0]);
    store_be32(output + 4u, state[1]);
    store_be32(output + 8u, state[2]);
    store_be32(output + 12u, state[3]);
}

static void encrypt_with_table_rounds(
    const mkv128_tables *tables,
    const uint32_t round_keys[MKV128_KEY_WORDS],
    const uint8_t input[MKV128_BLOCK_BYTES],
    uint8_t output[MKV128_BLOCK_BYTES])
{
    uint32_t state[4] = {
        load_be32(input), load_be32(input + 4u),
        load_be32(input + 8u), load_be32(input + 12u)
    };
    unsigned round;

    for (round = 0u; round < MKV128_ROUNDS; ++round) {
        const uint32_t *rk = round_keys + 8u * round;
        const uint32_t a = table_sub_word(
            tables, table_sub_mix_word(tables, state[0] ^ rk[0]) ^ rk[4]);
        const uint32_t b = table_sub_word(
            tables, table_sub_mix_word(tables, state[1] ^ rk[1]) ^ rk[5]);
        const uint32_t c = table_sub_word(
            tables, table_sub_mix_word(tables, state[2] ^ rk[2]) ^ rk[6]);
        const uint32_t d = table_sub_word(
            tables, table_sub_mix_word(tables, state[3] ^ rk[3]) ^ rk[7]);

        state[0] = b ^ c ^ d;
        state[1] = a ^ c ^ d;
        state[2] = a ^ b ^ d;
        state[3] = a ^ b ^ c;
    }
    state[0] ^= round_keys[48u];
    state[1] ^= round_keys[49u];
    state[2] ^= round_keys[50u];
    state[3] ^= round_keys[51u];
    store_be32(output, state[0]);
    store_be32(output + 4u, state[1]);
    store_be32(output + 8u, state[2]);
    store_be32(output + 12u, state[3]);
}

/* Phuong an 1: tinh S-box va MixWords truc tiep trong moi vong. */
static void mkv128_encrypt_direct(const mkv128_tables *tables,
                                  const uint8_t key[MKV128_KEY_BYTES],
                                  const uint8_t input[MKV128_BLOCK_BYTES],
                                  uint8_t output[MKV128_BLOCK_BYTES])
{
    uint32_t round_keys[MKV128_KEY_WORDS];

    expand_key_direct(tables, key, round_keys);
    encrypt_with_direct_rounds(tables, round_keys, input, output);
}

/* Phuong an 2: gop S-box va MixWords thanh bon bang L0-L3. */
static void mkv128_encrypt_ttable(const mkv128_tables *tables,
                                  const uint8_t key[MKV128_KEY_BYTES],
                                  const uint8_t input[MKV128_BLOCK_BYTES],
                                  uint8_t output[MKV128_BLOCK_BYTES])
{
    uint32_t round_keys[MKV128_KEY_WORDS];

    expand_key_table(tables, key, round_keys);
    encrypt_with_table_rounds(tables, round_keys, input, output);
}

#if MKV128_CAN_COMPILE_AVX512
static int avx512_available(void)
{
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw");
}

__attribute__((target("avx512f,avx512bw")))
static inline __m512i avx512_sub_mix(const mkv128_tables *tables,
                                     __m512i word)
{
    const __m512i mask = _mm512_set1_epi32(0xff);
    const __m512i p0 = _mm512_i32gather_epi32(
        _mm512_srli_epi32(word, 24), tables->l0, 4);
    const __m512i p1 = _mm512_i32gather_epi32(
        _mm512_and_si512(_mm512_srli_epi32(word, 16), mask), tables->l1, 4);
    const __m512i p2 = _mm512_i32gather_epi32(
        _mm512_and_si512(_mm512_srli_epi32(word, 8), mask), tables->l2, 4);
    const __m512i p3 = _mm512_i32gather_epi32(
        _mm512_and_si512(word, mask), tables->l3, 4);

    return _mm512_ternarylogic_epi32(
        _mm512_ternarylogic_epi32(p0, p1, p2, 0x96), p3,
        _mm512_setzero_si512(), 0x96);
}

__attribute__((target("avx512f,avx512bw")))
static inline __m512i avx512_sub_word(const mkv128_tables *tables,
                                       __m512i word)
{
    const __m512i mask = _mm512_set1_epi32(0xff);
    const __m512i p0 = _mm512_i32gather_epi32(
        _mm512_srli_epi32(word, 24), tables->l4, 4);
    const __m512i p1 = _mm512_i32gather_epi32(
        _mm512_and_si512(_mm512_srli_epi32(word, 16), mask), tables->l4, 4);
    const __m512i p2 = _mm512_i32gather_epi32(
        _mm512_and_si512(_mm512_srli_epi32(word, 8), mask), tables->l4, 4);
    const __m512i p3 = _mm512_i32gather_epi32(
        _mm512_and_si512(word, mask), tables->l4, 4);

    return _mm512_xor_si512(
        _mm512_xor_si512(p0, _mm512_srli_epi32(p1, 8)),
        _mm512_xor_si512(_mm512_srli_epi32(p2, 16),
                        _mm512_srli_epi32(p3, 24)));
}

__attribute__((target("avx512f,avx512bw")))
static void encrypt_avx512_batch16(
    const mkv128_tables *tables,
    const uint32_t round_keys[MKV128_KEY_WORDS],
    const uint8_t input[16u * MKV128_BLOCK_BYTES],
    uint8_t output[16u * MKV128_BLOCK_BYTES])
{
    const __m512i block_offsets = _mm512_setr_epi32(
        0, 16, 32, 48, 64, 80, 96, 112,
        128, 144, 160, 176, 192, 208, 224, 240);
    const __m512i byte_swap = _mm512_setr_epi32(
        0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F,
        0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F,
        0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F,
        0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F);
    __m512i state[4];
    unsigned round;

    state[0] = _mm512_shuffle_epi8(
        _mm512_i32gather_epi32(block_offsets, input + 0u, 1), byte_swap);
    state[1] = _mm512_shuffle_epi8(
        _mm512_i32gather_epi32(block_offsets, input + 4u, 1), byte_swap);
    state[2] = _mm512_shuffle_epi8(
        _mm512_i32gather_epi32(block_offsets, input + 8u, 1), byte_swap);
    state[3] = _mm512_shuffle_epi8(
        _mm512_i32gather_epi32(block_offsets, input + 12u, 1), byte_swap);

    for (round = 0u; round < MKV128_ROUNDS; ++round) {
        const uint32_t *rk = round_keys + 8u * round;
        __m512i a = avx512_sub_word(tables,
            _mm512_xor_si512(
                avx512_sub_mix(tables, _mm512_xor_si512(
                    state[0], _mm512_set1_epi32((int)rk[0]))),
                _mm512_set1_epi32((int)rk[4])));
        __m512i b = avx512_sub_word(tables,
            _mm512_xor_si512(
                avx512_sub_mix(tables, _mm512_xor_si512(
                    state[1], _mm512_set1_epi32((int)rk[1]))),
                _mm512_set1_epi32((int)rk[5])));
        __m512i c = avx512_sub_word(tables,
            _mm512_xor_si512(
                avx512_sub_mix(tables, _mm512_xor_si512(
                    state[2], _mm512_set1_epi32((int)rk[2]))),
                _mm512_set1_epi32((int)rk[6])));
        __m512i d = avx512_sub_word(tables,
            _mm512_xor_si512(
                avx512_sub_mix(tables, _mm512_xor_si512(
                    state[3], _mm512_set1_epi32((int)rk[3]))),
                _mm512_set1_epi32((int)rk[7])));
        state[0] = _mm512_xor_si512(b, _mm512_xor_si512(c, d));
        state[1] = _mm512_xor_si512(a, _mm512_xor_si512(c, d));
        state[2] = _mm512_xor_si512(a, _mm512_xor_si512(b, d));
        state[3] = _mm512_xor_si512(a, _mm512_xor_si512(b, c));
    }

    state[0] = _mm512_xor_si512(state[0],
                                _mm512_set1_epi32((int)round_keys[48u]));
    state[1] = _mm512_xor_si512(state[1],
                                _mm512_set1_epi32((int)round_keys[49u]));
    state[2] = _mm512_xor_si512(state[2],
                                _mm512_set1_epi32((int)round_keys[50u]));
    state[3] = _mm512_xor_si512(state[3],
                                _mm512_set1_epi32((int)round_keys[51u]));

    _mm512_i32scatter_epi32(output + 0u, block_offsets,
        _mm512_shuffle_epi8(state[0], byte_swap), 1);
    _mm512_i32scatter_epi32(output + 4u, block_offsets,
        _mm512_shuffle_epi8(state[1], byte_swap), 1);
    _mm512_i32scatter_epi32(output + 8u, block_offsets,
        _mm512_shuffle_epi8(state[2], byte_swap), 1);
    _mm512_i32scatter_epi32(output + 12u, block_offsets,
        _mm512_shuffle_epi8(state[3], byte_swap), 1);
}
#else
static int avx512_available(void) { return 0; }
#endif

/* Phuong an 3: AVX-512 xu ly 16 khoi; fallback bang T neu CPU thieu AVX-512. */
static void mkv128_encrypt_avx512(const mkv128_tables *tables,
                                  const uint8_t key[MKV128_KEY_BYTES],
                                  const uint8_t input[MKV128_BLOCK_BYTES],
                                  uint8_t output[MKV128_BLOCK_BYTES])
{
    uint32_t round_keys[MKV128_KEY_WORDS];

    expand_key_table(tables, key, round_keys);
#if MKV128_CAN_COMPILE_AVX512
    if (avx512_available()) {
        uint8_t batch_input[16u * MKV128_BLOCK_BYTES];
        uint8_t batch_output[16u * MKV128_BLOCK_BYTES];
        unsigned block;

        for (block = 0u; block < 16u; ++block) {
            memcpy(batch_input + block * MKV128_BLOCK_BYTES,
                   input, MKV128_BLOCK_BYTES);
        }
        encrypt_avx512_batch16(tables, round_keys, batch_input, batch_output);
        memcpy(output, batch_output, MKV128_BLOCK_BYTES);
        return;
    }
#endif
    encrypt_with_table_rounds(tables, round_keys, input, output);
}

static void print_hex(const uint8_t value[MKV128_BLOCK_BYTES])
{
    unsigned index;

    for (index = 0u; index < MKV128_BLOCK_BYTES; ++index) {
        printf("%02X", (unsigned)value[index]);
    }
}

typedef void (*encrypt_function)(const mkv128_tables *,
                                 const uint8_t [MKV128_KEY_BYTES],
                                 const uint8_t [MKV128_BLOCK_BYTES],
                                 uint8_t [MKV128_BLOCK_BYTES]);

typedef struct test_vector {
    const char *name;
    uint8_t key[MKV128_KEY_BYTES];
    uint8_t plaintext[MKV128_BLOCK_BYTES];
    uint8_t expected[MKV128_BLOCK_BYTES];
} test_vector;

static int hex_value(int character)
{
    if (character >= '0' && character <= '9') {
        return character - '0';
    }
    if (character >= 'A' && character <= 'F') {
        return character - 'A' + 10;
    }
    if (character >= 'a' && character <= 'f') {
        return character - 'a' + 10;
    }
    return -1;
}

/* Doc 32 chu so hex; cho phep chen khoang trang, dau ':' hoac '-' cho de doc. */
static int parse_hex_block(const char *text,
                           uint8_t output[MKV128_BLOCK_BYTES])
{
    unsigned digits = 0u;
    const unsigned char *cursor = (const unsigned char *)text;

    while (*cursor != '\0' && *cursor != '\n' && *cursor != '\r') {
        const int value = hex_value(*cursor);

        if (isspace(*cursor) || *cursor == ':' || *cursor == '-') {
            ++cursor;
            continue;
        }
        if (value < 0 || digits >= 32u) {
            return 0;
        }
        if ((digits & 1u) == 0u) {
            output[digits / 2u] = (uint8_t)(value << 4u);
        } else {
            output[digits / 2u] |= (uint8_t)value;
        }
        ++digits;
        ++cursor;
    }
    return digits == 32u;
}

static int line_is_blank(const char *text)
{
    const unsigned char *cursor = (const unsigned char *)text;

    while (*cursor != '\0') {
        if (!isspace(*cursor)) {
            return 0;
        }
        ++cursor;
    }
    return 1;
}

static void run_interactive_vector(const mkv128_tables *tables,
                                   const encrypt_function methods[3],
                                   const char *method_names[3])
{
    uint8_t key[MKV128_KEY_BYTES];
    uint8_t input[MKV128_BLOCK_BYTES];
    char line[256];
    unsigned method_index;

    puts("Nhap mot vec-to tuy chon (Enter de bo qua):");
    fputs("  Khoa 128 bit, 32 ky tu hex: ", stdout);
    fflush(stdout);
    if (fgets(line, sizeof(line), stdin) == NULL || line_is_blank(line)) {
        puts("Khong co vec-to tuy chon.");
        return;
    }
    if (!parse_hex_block(line, key)) {
        puts("Khoa khong hop le: can dung dung 32 chu so hex.");
        return;
    }

    fputs("  Input 128 bit, 32 ky tu hex: ", stdout);
    fflush(stdout);
    if (fgets(line, sizeof(line), stdin) == NULL || line_is_blank(line)) {
        puts("Thieu input; bo qua vec-to tuy chon.");
        return;
    }
    if (!parse_hex_block(line, input)) {
        puts("Input khong hop le: can dung dung 32 chu so hex.");
        return;
    }

    puts("\nKet qua vec-to tu nhap:");
    printf("  Khoa : "); print_hex(key); puts("");
    printf("  Input: "); print_hex(input); puts("");
    for (method_index = 0u; method_index < 3u; ++method_index) {
        uint8_t output[MKV128_BLOCK_BYTES];

        methods[method_index](tables, key, input, output);
        printf("  %-28s: ", method_names[method_index]);
        print_hex(output);
        puts("");
    }
}

int main(void)
{
    test_vector vectors[3] = {
        {
            "Vec-to cong bo trong Phu luc A.1.1",
            {0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
             0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x11},
            {0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
             0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,0x00},
            {0xB3,0x31,0x22,0x83,0x34,0xC3,0xF8,0x1A,
             0x37,0x20,0x65,0x91,0x49,0x87,0x56,0xA1}
        },
        {
            "Khoa 0, ban ro 0 (vec-to bo sung doi chieu)",
            {0},
            {0},
            {0xF5,0xBE,0x3D,0x46,0xF5,0xBE,0x3D,0x46,
             0xF5,0xBE,0x3D,0x46,0x7E,0xBE,0x58,0xD7}
        },
        {
            "Khoa FF, ban ro FF (vec-to bo sung doi chieu)",
            {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
             0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF},
            {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
             0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF},
            {0xD5,0x28,0xA5,0x45,0xD5,0x28,0xA5,0x45,
             0xD5,0x28,0xA5,0x45,0x63,0x00,0xCB,0x1E}
        }
    };
    const char *method_names[3] = {
        "Truc tiep theo byte",
        "Bang tra T",
        "AVX-512 gather (16 khoi)"
    };
    const encrypt_function methods[3] = {
        mkv128_encrypt_direct, mkv128_encrypt_ttable, mkv128_encrypt_avx512
    };
    mkv128_tables tables;
    unsigned vector_index;
    int all_passed = 1;

    initialize_tables(&tables);

    puts("MKV-128: ba phuong an, ba test vector");
    puts("(Mot vector cong bo; hai vector bo sung co expected co dinh.)\n");
    printf("AVX-512: %s\n\n", avx512_available() ? "AVAILABLE" : "UNAVAILABLE (fallback T-table)");

    for (vector_index = 0u; vector_index < 3u; ++vector_index) {
        const unsigned method_index = vector_index;
        uint8_t output[MKV128_BLOCK_BYTES];
        int passed;

        printf("[%u] %s\n", vector_index + 1u, vectors[vector_index].name);
        printf("  Key      : "); print_hex(vectors[vector_index].key); puts("");
        printf("  Plaintext: "); print_hex(vectors[vector_index].plaintext); puts("");
        printf("  Expected : "); print_hex(vectors[vector_index].expected); puts("");

        methods[method_index](&tables,
                              vectors[vector_index].key,
                              vectors[vector_index].plaintext, output);
        passed = memcmp(output, vectors[vector_index].expected,
                        MKV128_BLOCK_BYTES) == 0;
        printf("  %-28s: ", method_names[method_index]);
        print_hex(output);
        printf("  %s\n", passed ? "PASS" : "FAIL");
        all_passed &= passed;
        puts("");
    }

    puts(all_passed ? "OVERALL: PASS" : "OVERALL: FAIL");
    if (all_passed) {
        run_interactive_vector(&tables, methods, method_names);
    }
    return all_passed ? 0 : 1;
}
