#ifndef MKV128_SIMD_AVX512_H
#define MKV128_SIMD_AVX512_H

#include <stddef.h>
#include <stdint.h>

#include "mkv128_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* AVX-512F, AVX-512BW and AVX-512VBMI must all be usable by the OS. */
int mkv128_avx512_available(void);

/* True only when the no-gather GFNI/VBMI fast path is executable. */
int mkv128_avx512_gfni_available(void);

/*
 * Encrypt sixteen independent blocks per iteration.  Scalar handles the
 * final partial group and CPUs without the required AVX-512 extensions.
 */
void mkv128_encrypt_blocks_avx512(const mkv128_ctx *ctx,
                                  const uint8_t *input,
                                  uint8_t *output,
                                  size_t block_count);

#ifdef __cplusplus
}
#endif

#endif /* MKV128_SIMD_AVX512_H */
