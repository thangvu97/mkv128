#ifndef MKV128_CORE_H
#define MKV128_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MKV128_BLOCK_BYTES 16u
#define MKV128_KEY_BYTES 16u


#define MKV128_ROUNDS 6u
#define MKV128_KEY_WORDS (4u * (2u * MKV128_ROUNDS + 1u))

typedef struct mkv128_ctx {
    uint32_t round_keys[MKV128_KEY_WORDS];
} mkv128_ctx;

int mkv128_set_encrypt_key(mkv128_ctx *ctx,
                           const uint8_t key[MKV128_KEY_BYTES]);
int mkv128_set_decrypt_key(mkv128_ctx *ctx,
                           const uint8_t key[MKV128_KEY_BYTES]);

void mkv128_encrypt_block(const mkv128_ctx *ctx,
                          const uint8_t input[MKV128_BLOCK_BYTES],
                          uint8_t output[MKV128_BLOCK_BYTES]);
void mkv128_decrypt_block(const mkv128_ctx *ctx,
                          const uint8_t input[MKV128_BLOCK_BYTES],
                          uint8_t output[MKV128_BLOCK_BYTES]);

#ifdef __cplusplus
}
#endif

#endif
