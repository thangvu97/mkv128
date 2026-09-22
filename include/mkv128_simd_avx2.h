#ifndef MKV128_SIMD_AVX2_H
#define MKV128_SIMD_AVX2_H

#include <stddef.h>
#include <stdint.h>

#include "mkv128_core.h"

#ifdef __cplusplus
extern "C" {
#endif

int mkv128_avx2_available(void);


void mkv128_encrypt_blocks_avx2(const mkv128_ctx *ctx,
                                const uint8_t *input,
                                uint8_t *output,
                                size_t block_count);

#ifdef __cplusplus
}
#endif

#endif /* MKV128_SIMD_AVX2_H */
