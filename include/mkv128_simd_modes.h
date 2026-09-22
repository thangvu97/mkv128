#ifndef MKV128_SIMD_MODES_H
#define MKV128_SIMD_MODES_H

#include <stddef.h>
#include <stdint.h>

#include "mkv128_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MKV128_SIMD_MAX_WORKERS 32u

typedef enum mkv128_backend {
    MKV128_BACKEND_SCALAR = 0,
    MKV128_BACKEND_AVX2 = 1,
    MKV128_BACKEND_AVX512 = 2
} mkv128_backend;

const char *mkv128_backend_name(mkv128_backend backend);
int mkv128_backend_available(mkv128_backend backend);

/* Independent blocks may be encrypted in place; partial overlap is rejected. */
int mkv128_ecb_encrypt_parallel_simd(
    const mkv128_ctx *context,
    const uint8_t *input,
    uint8_t *output,
    size_t blocks,
    unsigned workers,
    mkv128_backend backend);

#ifdef __cplusplus
}
#endif

#endif /* MKV128_SIMD_MODES_H */
