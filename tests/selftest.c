/*
 * MKV-128 block-cipher self-test.
 *
 * This program validates the independently written direct implementation
 * against the optimized T-table implementation.  It exercises only the
 * 16-byte block primitive and does not depend on a mode of operation.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mkv128_core.h"
#include "mkv128_reference.h"

#define INDEPENDENT_CASES 64u
#define EXPECTED_SBOX_FNV1A UINT32_C(0x1759E98D)

static const uint8_t published_key[MKV128_KEY_BYTES] = {
    0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, 0x08u,
    0x09u, 0x0Au, 0x0Bu, 0x0Cu, 0x0Du, 0x0Eu, 0x0Fu, 0x11u
};

static const uint8_t published_plaintext[MKV128_BLOCK_BYTES] = {
    0x11u, 0x22u, 0x33u, 0x44u, 0x55u, 0x66u, 0x77u, 0x88u,
    0x99u, 0xAAu, 0xBBu, 0xCCu, 0xDDu, 0xEEu, 0xFFu, 0x00u
};

static const uint8_t published_ciphertext[MKV128_BLOCK_BYTES] = {
    0xB3u, 0x31u, 0x22u, 0x83u, 0x34u, 0xC3u, 0xF8u, 0x1Au,
    0x37u, 0x20u, 0x65u, 0x91u, 0x49u, 0x87u, 0x56u, 0xA1u
};

static void print_hex(const uint8_t *value, size_t length)
{
    size_t index;

    for (index = 0u; index < length; ++index) {
        printf("%02X", (unsigned)value[index]);
    }
}

static int report_check(int condition, const char *description)
{
    printf("%-63s %s\n", description, condition ? "PASS" : "FAIL");
    return condition;
}

static void fill_pattern(uint8_t *buffer, size_t length, uint32_t seed)
{
    size_t index;

    for (index = 0u; index < length; ++index) {
        seed ^= seed << 13u;
        seed ^= seed >> 17u;
        seed ^= seed << 5u;
        buffer[index] = (uint8_t)seed;
    }
}

static uint32_t sbox_fingerprint(const uint8_t sbox[256])
{
    uint32_t hash = UINT32_C(2166136261);
    unsigned index;

    for (index = 0u; index < 256u; ++index) {
        hash ^= sbox[index];
        hash *= UINT32_C(16777619);
    }
    return hash;
}

static int sbox_is_permutation(const uint8_t sbox[256])
{
    uint8_t seen[256] = {0u};
    unsigned index;

    for (index = 0u; index < 256u; ++index) {
        if (seen[sbox[index]] != 0u) {
            return 0;
        }
        seen[sbox[index]] = 1u;
    }
    return 1;
}

static int initialize_contexts(
    const uint8_t key[MKV128_KEY_BYTES],
    mkv128_ctx *encrypt_context,
    mkv128_ctx *decrypt_context,
    mkv128_reference_ctx *direct_context)
{
    return mkv128_set_encrypt_key(encrypt_context, key) != 0 &&
           mkv128_set_decrypt_key(decrypt_context, key) != 0 &&
           mkv128_reference_set_encrypt_key(direct_context, key) != 0;
}

static int test_published_vector(void)
{
    mkv128_ctx encrypt_context;
    mkv128_ctx decrypt_context;
    mkv128_reference_ctx direct_context;
    uint8_t direct_output[MKV128_BLOCK_BYTES];
    uint8_t table_output[MKV128_BLOCK_BYTES];
    uint8_t recovered[MKV128_BLOCK_BYTES];
    int success = 1;

    if (!initialize_contexts(published_key, &encrypt_context,
                             &decrypt_context, &direct_context)) {
        fputs("Failed to initialize the published key.\n", stderr);
        return 0;
    }

    mkv128_reference_encrypt_block(&direct_context, published_plaintext,
                                   direct_output);
    mkv128_encrypt_block(&encrypt_context, published_plaintext, table_output);
    mkv128_decrypt_block(&decrypt_context, published_ciphertext, recovered);

    puts("\n===== PUBLISHED MKV-128 BLOCK-CIPHER VECTOR =====");
    printf("Key                 : ");
    print_hex(published_key, sizeof(published_key));
    printf("\nPlaintext           : ");
    print_hex(published_plaintext, sizeof(published_plaintext));
    printf("\nExpected ciphertext : ");
    print_hex(published_ciphertext, sizeof(published_ciphertext));
    printf("\nDirect output       : ");
    print_hex(direct_output, sizeof(direct_output));
    printf("\nT-table output      : ");
    print_hex(table_output, sizeof(table_output));
    printf("\nRecovered plaintext : ");
    print_hex(recovered, sizeof(recovered));
    puts("\n");

    success &= report_check(
        memcmp(direct_output, published_ciphertext,
               sizeof(direct_output)) == 0,
        "Published vector: direct implementation without T-tables");
    success &= report_check(
        memcmp(table_output, published_ciphertext,
               sizeof(table_output)) == 0,
        "Published vector: T-table encryption");
    success &= report_check(
        memcmp(recovered, published_plaintext, sizeof(recovered)) == 0,
        "Published vector: T-table decryption");
    success &= report_check(
        memcmp(encrypt_context.round_keys, direct_context.round_keys,
               sizeof(encrypt_context.round_keys)) == 0,
        "All encryption round keys match the direct implementation");
    success &= report_check(
        sbox_is_permutation(direct_context.sbox) &&
        direct_context.sbox[0x00u] == 0x01u &&
        direct_context.sbox[0x10u] == 0x00u &&
        direct_context.sbox[0x20u] == 0x08u &&
        direct_context.sbox[0xFFu] == 0x8Bu &&
        sbox_fingerprint(direct_context.sbox) == EXPECTED_SBOX_FNV1A,
        "Generated S-box is the expected 256-byte permutation");

    return success;
}

static int test_in_place_operations(void)
{
    mkv128_ctx encrypt_context;
    mkv128_ctx decrypt_context;
    mkv128_reference_ctx direct_context;
    uint8_t table_buffer[MKV128_BLOCK_BYTES];
    uint8_t direct_buffer[MKV128_BLOCK_BYTES];
    int success = 1;

    if (!initialize_contexts(published_key, &encrypt_context,
                             &decrypt_context, &direct_context)) {
        return 0;
    }

    memcpy(table_buffer, published_plaintext, sizeof(table_buffer));
    mkv128_encrypt_block(&encrypt_context, table_buffer, table_buffer);
    success &= report_check(
        memcmp(table_buffer, published_ciphertext,
               sizeof(table_buffer)) == 0,
        "In-place T-table encryption");

    mkv128_decrypt_block(&decrypt_context, table_buffer, table_buffer);
    success &= report_check(
        memcmp(table_buffer, published_plaintext,
               sizeof(table_buffer)) == 0,
        "In-place T-table decryption");

    memcpy(direct_buffer, published_plaintext, sizeof(direct_buffer));
    mkv128_reference_encrypt_block(&direct_context, direct_buffer,
                                   direct_buffer);
    success &= report_check(
        memcmp(direct_buffer, published_ciphertext,
               sizeof(direct_buffer)) == 0,
        "In-place direct encryption without T-tables");

    return success;
}

static int test_independent_inputs(void)
{
    unsigned case_index;

    for (case_index = 0u; case_index < INDEPENDENT_CASES; ++case_index) {
        mkv128_ctx encrypt_context;
        mkv128_ctx decrypt_context;
        mkv128_reference_ctx direct_context;
        uint8_t key[MKV128_KEY_BYTES];
        uint8_t plaintext[MKV128_BLOCK_BYTES];
        uint8_t direct_output[MKV128_BLOCK_BYTES];
        uint8_t table_output[MKV128_BLOCK_BYTES];
        uint8_t recovered[MKV128_BLOCK_BYTES];
        uint8_t in_place[MKV128_BLOCK_BYTES];

        fill_pattern(key, sizeof(key),
                     UINT32_C(0xACE50001) + case_index);
        fill_pattern(plaintext, sizeof(plaintext),
                     UINT32_C(0x51920001) + case_index);

        if (!initialize_contexts(key, &encrypt_context, &decrypt_context,
                                 &direct_context)) {
            fprintf(stderr, "Key setup failed for case %u.\n", case_index);
            return 0;
        }

        mkv128_reference_encrypt_block(&direct_context, plaintext,
                                       direct_output);
        mkv128_encrypt_block(&encrypt_context, plaintext, table_output);
        mkv128_decrypt_block(&decrypt_context, table_output, recovered);

        memcpy(in_place, plaintext, sizeof(in_place));
        mkv128_encrypt_block(&encrypt_context, in_place, in_place);
        mkv128_decrypt_block(&decrypt_context, in_place, in_place);

        if (memcmp(encrypt_context.round_keys, direct_context.round_keys,
                   sizeof(encrypt_context.round_keys)) != 0 ||
            memcmp(table_output, direct_output, sizeof(table_output)) != 0 ||
            memcmp(recovered, plaintext, sizeof(recovered)) != 0 ||
            memcmp(in_place, plaintext, sizeof(in_place)) != 0 ||
            !sbox_is_permutation(direct_context.sbox) ||
            sbox_fingerprint(direct_context.sbox) != EXPECTED_SBOX_FNV1A) {
            fprintf(stderr, "Independent block test failed for case %u.\n",
                    case_index);
            return 0;
        }
    }

    return report_check(
        1,
        "64 keys and plaintexts: keys, outputs and round trips match");
}

int main(void)
{
    int success = 1;

    success &= test_published_vector();

    puts("\n===== IN-PLACE BLOCK OPERATIONS =====");
    success &= test_in_place_operations();

    puts("\n===== INDEPENDENT KEY AND BLOCK TESTS =====");
    success &= test_independent_inputs();

    printf("\nOVERALL MKV-128 BLOCK SELF-TEST: %s\n",
           success ? "PASS" : "FAIL");
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
