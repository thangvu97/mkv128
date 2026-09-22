#ifndef MKV128_SIMD_AVX2_H
#define MKV128_SIMD_AVX2_H

#include <stddef.h>
#include <stdint.h>

#include "mkv128_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Includes the operating-system checks performed by the compiler runtime. */
int mkv128_avx2_available(void);

/*
 * Encrypt eight independent blocks per AVX2 iteration.  A scalar fallback
 * handles machines without AVX2 and any final group of fewer than eight.
 * The input and output buffers may be identical.
 */
void mkv128_encrypt_blocks_avx2(const mkv128_ctx *ctx,
                                const uint8_t *input,
                                uint8_t *output,
                                size_t block_count);

#ifdef __cplusplus
}
#endif

#endif /* MKV128_SIMD_AVX2_H */
