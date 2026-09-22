#ifndef _WIN32
#error "The MKV SIMD dispatcher requires Windows worker threads."
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>

#include "mkv128_simd_avx2.h"
#include "mkv128_simd_avx512.h"
#include "mkv128_simd_modes.h"

typedef struct simd_worker_job {
    const mkv128_ctx *context;
    const uint8_t *input;
    uint8_t *output;
    size_t first_block;
    size_t last_block;
    mkv128_backend backend;
} simd_worker_job;

static int valid_buffers(const uint8_t *input, uint8_t *output, size_t bytes)
{
    uintptr_t left;
    uintptr_t right;

    if (bytes == 0u) {
        return 1;
    }
    if (input == NULL || output == NULL) {
        return 0;
    }
    if (input == output) {
        return 1;
    }

    left = (uintptr_t)input;
    right = (uintptr_t)output;
    return (left > right ? left - right : right - left) >= bytes;
}

const char *mkv128_backend_name(mkv128_backend backend)
{
    switch (backend) {
    case MKV128_BACKEND_SCALAR:
        return "T-table";
    case MKV128_BACKEND_AVX2:
        return "AVX2 + T-table";
    case MKV128_BACKEND_AVX512:
        return mkv128_avx512_gfni_available()
            ? "AVX-512/VBMI/GFNI"
            : "AVX-512/VBMI + T-table";
    default:
        return "unsupported";
    }
}

int mkv128_backend_available(mkv128_backend backend)
{
    switch (backend) {
    case MKV128_BACKEND_SCALAR:
        return 1;
    case MKV128_BACKEND_AVX2:
        return mkv128_avx2_available();
    case MKV128_BACKEND_AVX512:
        return mkv128_avx512_available();
    default:
        return 0;
    }
}

static void encrypt_scalar_blocks(
    const mkv128_ctx *context,
    const uint8_t *input,
    uint8_t *output,
    size_t blocks)
{
    size_t block;

    for (block = 0u; block < blocks; ++block) {
        const size_t offset = block * (size_t)MKV128_BLOCK_BYTES;

        mkv128_encrypt_block(context, input + offset, output + offset);
    }
}

static void process_ecb_range(const simd_worker_job *job)
{
    const size_t offset = job->first_block * (size_t)MKV128_BLOCK_BYTES;
    const size_t blocks = job->last_block - job->first_block;

    switch (job->backend) {
    case MKV128_BACKEND_AVX2:
        mkv128_encrypt_blocks_avx2(
            job->context, job->input + offset, job->output + offset, blocks);
        break;
    case MKV128_BACKEND_AVX512:
        mkv128_encrypt_blocks_avx512(
            job->context, job->input + offset, job->output + offset, blocks);
        break;
    case MKV128_BACKEND_SCALAR:
    default:
        encrypt_scalar_blocks(
            job->context, job->input + offset, job->output + offset, blocks);
        break;
    }
}

static DWORD WINAPI simd_worker_entry(LPVOID parameter)
{
    process_ecb_range((const simd_worker_job *)parameter);
    return 0u;
}

static int dispatch_jobs(
    const mkv128_ctx *context,
    const uint8_t *input,
    uint8_t *output,
    size_t blocks,
    unsigned requested_workers,
    mkv128_backend backend)
{
    simd_worker_job jobs[MKV128_SIMD_MAX_WORKERS];
    HANDLE handles[MKV128_SIMD_MAX_WORKERS - 1u];
    unsigned workers;
    unsigned worker;
    unsigned launched = 0u;
    int success = 1;

    if (blocks == 0u) {
        return 1;
    }

    workers = requested_workers > MKV128_SIMD_MAX_WORKERS
        ? MKV128_SIMD_MAX_WORKERS
        : requested_workers;
    if ((size_t)workers > blocks) {
        workers = (unsigned)blocks;
    }

    for (worker = 0u; worker < workers; ++worker) {
        jobs[worker].context = context;
        jobs[worker].input = input;
        jobs[worker].output = output;
        jobs[worker].first_block =
            (blocks / workers) * worker
            + ((blocks % workers) * worker) / workers;
        jobs[worker].last_block =
            (blocks / workers) * (worker + 1u)
            + ((blocks % workers) * (worker + 1u)) / workers;
        jobs[worker].backend = backend;
    }

    for (worker = 1u; worker < workers; ++worker) {
        HANDLE thread = CreateThread(
            NULL, 0u, simd_worker_entry, &jobs[worker], 0u, NULL);

        if (thread == NULL) {
            process_ecb_range(&jobs[worker]);
        } else {
            handles[launched++] = thread;
        }
    }

    process_ecb_range(&jobs[0]);

    if (launched != 0u) {
        const DWORD wait_result = WaitForMultipleObjects(
            (DWORD)launched, handles, TRUE, INFINITE);

        if (wait_result == WAIT_FAILED) {
            for (worker = 0u; worker < launched; ++worker) {
                if (WaitForSingleObject(handles[worker], INFINITE)
                    != WAIT_OBJECT_0) {
                    success = 0;
                }
            }
            success = 0;
        }

        for (worker = 0u; worker < launched; ++worker) {
            CloseHandle(handles[worker]);
        }
    }

    return success;
}

int mkv128_ecb_encrypt_parallel_simd(
    const mkv128_ctx *context,
    const uint8_t *input,
    uint8_t *output,
    size_t blocks,
    unsigned workers,
    mkv128_backend backend)
{
    size_t bytes;

    if (context == NULL || workers == 0u
        || blocks > SIZE_MAX / (size_t)MKV128_BLOCK_BYTES
        || !mkv128_backend_available(backend)) {
        return 0;
    }

    bytes = blocks * (size_t)MKV128_BLOCK_BYTES;
    if (!valid_buffers(input, output, bytes)) {
        return 0;
    }

    return dispatch_jobs(
        context, input, output, blocks, workers, backend);
}
