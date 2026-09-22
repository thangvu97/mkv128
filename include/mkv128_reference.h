#ifndef MKV128_REFERENCE_H
#define MKV128_REFERENCE_H

#include <stdint.h>

#include "mkv128_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Direct implementation used as a fair no-T-table control.  The ordinary
 * 256-byte substitution box is generated from the MKV A-construction once;
 * MixWords is evaluated directly in GF(2^8) for every encrypted block.
 */
typedef struct mkv128_reference_ctx {
    uint32_t round_keys[MKV128_KEY_WORDS];
    uint8_t sbox[256];
} mkv128_reference_ctx;

int mkv128_reference_set_encrypt_key(
    mkv128_reference_ctx *ctx,
    const uint8_t key[MKV128_KEY_BYTES]);

void mkv128_reference_encrypt_block(
    const mkv128_reference_ctx *ctx,
    const uint8_t input[MKV128_BLOCK_BYTES],
    uint8_t output[MKV128_BLOCK_BYTES]);

#ifdef __cplusplus
}
#endif

#endif /* MKV128_REFERENCE_H */
