

#ifndef _WIN32
#error "This benchmark requires Windows timers and worker threads."
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mkv128_core.h"
#include "mkv128_reference.h"
#include "mkv128_simd_avx2.h"
#include "mkv128_simd_avx512.h"
#include "mkv128_simd_modes.h"

#define DEFAULT_BLOCKS 20000000u
#define DEFAULT_RUNS 5u
#define MAX_RUNS 9u
#define MAX_SCHEDULE 5u
#define MAX_BENCH_WORKERS 16u
#define TEST_BLOCKS 257u
#define WARMUP_BLOCKS 262144u
#define DIRECT_FULL_WARMUP_PASSES 2u
#define MAX_MEDIAN_DEVIATION 0.20

static const uint8_t STANDARD_KEY[MKV128_KEY_BYTES] = {
    0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, 0x08u,
    0x09u, 0x0Au, 0x0Bu, 0x0Cu, 0x0Du, 0x0Eu, 0x0Fu, 0x11u
};

static const uint8_t STANDARD_INPUT[MKV128_BLOCK_BYTES] = {
    0x11u, 0x22u, 0x33u, 0x44u, 0x55u, 0x66u, 0x77u, 0x88u,
    0x99u, 0xAAu, 0xBBu, 0xCCu, 0xDDu, 0xEEu, 0xFFu, 0x00u
};

static const uint8_t STANDARD_OUTPUT[MKV128_BLOCK_BYTES] = {
    0xB3u, 0x31u, 0x22u, 0x83u, 0x34u, 0xC3u, 0xF8u, 0x1Au,
    0x37u, 0x20u, 0x65u, 0x91u, 0x49u, 0x87u, 0x56u, 0xA1u
};

static int report_check(int condition, const char *description)
{
    if (!condition) {
        printf("%-28s FAIL\n", description);
    }
    return condition;
}

static int report_power_source(int require_ac)
{
    SYSTEM_POWER_STATUS power;

    if (!GetSystemPowerStatus(&power)) {
        printf("Power: UNKNOWN (GetSystemPowerStatus failed)\n");
        return !require_ac;
    }

    printf("Power: %s",
           power.ACLineStatus == 1u ? "AC ONLINE"
         : power.ACLineStatus == 0u ? "BATTERY"
                                    : "UNKNOWN");
    if (power.BatteryLifePercent != 255u) {
        printf(", battery %u%%", (unsigned)power.BatteryLifePercent);
    }
    printf("\n");

    if (require_ac && power.ACLineStatus != 1u) {
        fprintf(stderr,
                "Official benchmark aborted: an online AC source is required.\n");
        return 0;
    }
    return 1;
}

static int configure_benchmark_priority(void)
{
    if (!SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)) {
        fprintf(stderr, "Cannot set Above normal process priority.\n");
        return 0;
    }
    return 1;
}

static void fill_pattern(uint8_t *buffer, size_t bytes, uint64_t seed)
{
    size_t index;

    for (index = 0u; index < bytes; ++index) {
        seed ^= seed << 13u;
        seed ^= seed >> 7u;
        seed ^= seed << 17u;
        buffer[index] = (uint8_t)seed;
    }
}

static int test_published_vectors(
    const mkv128_ctx *context,
    const mkv128_ctx *decrypt_context,
    const mkv128_reference_ctx *reference)
{
    uint8_t input[16u * MKV128_BLOCK_BYTES];
    uint8_t output[16u * MKV128_BLOCK_BYTES];
    uint8_t independent[MKV128_BLOCK_BYTES];
    size_t lane;
    int passed = 1;

    mkv128_reference_encrypt_block(reference, STANDARD_INPUT, independent);
    passed &= report_check(
        memcmp(independent, STANDARD_OUTPUT, sizeof(independent)) == 0,
        "Vector: direct control");

    mkv128_encrypt_block(context, STANDARD_INPUT, independent);
    passed &= report_check(
        memcmp(independent, STANDARD_OUTPUT, sizeof(independent)) == 0,
        "Vector: T-table encrypt");

    mkv128_decrypt_block(decrypt_context, STANDARD_OUTPUT, independent);
    passed &= report_check(
        memcmp(independent, STANDARD_INPUT, sizeof(independent)) == 0,
        "Vector: T-table decrypt");

    for (lane = 0u; lane < 16u; ++lane) {
        memcpy(input + lane * MKV128_BLOCK_BYTES, STANDARD_INPUT,
               MKV128_BLOCK_BYTES);
    }

    if (mkv128_avx2_available()) {
        int same = 1;

        memset(output, 0u, sizeof(output));
        mkv128_encrypt_blocks_avx2(context, input, output, 8u);
        for (lane = 0u; lane < 8u; ++lane) {
            same &= memcmp(output + lane * MKV128_BLOCK_BYTES,
                           STANDARD_OUTPUT, MKV128_BLOCK_BYTES) == 0;
        }
        passed &= report_check(
            same, "Vector: AVX2 x8");
    }

    if (mkv128_avx512_available()) {
        int same = 1;

        memset(output, 0u, sizeof(output));
        mkv128_encrypt_blocks_avx512(context, input, output, 16u);
        for (lane = 0u; lane < 16u; ++lane) {
            same &= memcmp(output + lane * MKV128_BLOCK_BYTES,
                           STANDARD_OUTPUT, MKV128_BLOCK_BYTES) == 0;
        }
        passed &= report_check(
            same, "Vector: AVX-512 x16");
    }

    return passed;
}

static int test_ecb_differential(const mkv128_ctx *context)
{
    static const unsigned workers[] = {1u, 2u, 3u, 7u, 8u, 16u, 32u};
    static const size_t block_counts[] = {
        0u, 1u, 2u, 7u, 8u, 9u, 15u, 16u, 17u, 23u, 24u, 25u,
        31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u
    };
    uint8_t input[TEST_BLOCKS * MKV128_BLOCK_BYTES];
    uint8_t expected[sizeof(input)];
    uint8_t actual[sizeof(input)];
    unsigned backend_index;
    size_t block;

    fill_pattern(input, sizeof(input), UINT64_C(0xBB67AE8584CAA73B));
    for (block = 0u; block < TEST_BLOCKS; ++block) {
        const size_t offset = block * (size_t)MKV128_BLOCK_BYTES;

        mkv128_encrypt_block(context, input + offset, expected + offset);
    }

    for (backend_index = 0u; backend_index < 3u; ++backend_index) {
        const mkv128_backend backend = (mkv128_backend)backend_index;
        size_t count_index;

        if (!mkv128_backend_available(backend)) {
            continue;
        }

        for (count_index = 0u;
             count_index < sizeof(block_counts) / sizeof(block_counts[0]);
             ++count_index) {
            const size_t blocks = block_counts[count_index];
            const size_t bytes = blocks * (size_t)MKV128_BLOCK_BYTES;
            size_t worker_index;

            for (worker_index = 0u;
                 worker_index < sizeof(workers) / sizeof(workers[0]);
                 ++worker_index) {
                if (!mkv128_ecb_encrypt_parallel_simd(
                        context, input, actual, blocks,
                        workers[worker_index], backend)
                    || memcmp(actual, expected, bytes) != 0) {
                    fprintf(stderr,
                            "ECB mismatch: backend=%s blocks=%llu workers=%u\n",
                            mkv128_backend_name(backend),
                            (unsigned long long)blocks, workers[worker_index]);
                    return 0;
                }
            }

            memcpy(actual, input, bytes);
            if (!mkv128_ecb_encrypt_parallel_simd(
                    context, actual, actual, blocks, 3u, backend)
                || memcmp(actual, expected, bytes) != 0) {
                fprintf(stderr, "In-place ECB mismatch: backend=%s blocks=%llu\n",
                        mkv128_backend_name(backend), (unsigned long long)blocks);
                return 0;
            }
        }
    }

    return 1;
}

static int test_independent_keys(void)
{
    uint8_t key[MKV128_KEY_BYTES];
    uint8_t input[32u * MKV128_BLOCK_BYTES];
    uint8_t expected[sizeof(input)];
    uint8_t actual[sizeof(input)];
    unsigned key_number;

    for (key_number = 0u; key_number < 32u; ++key_number) {
        mkv128_ctx context;
        mkv128_reference_ctx reference;
        size_t block;
        unsigned backend_index;

        fill_pattern(key, sizeof(key),
                     UINT64_C(0xA54FF53A5F1D36F1) + key_number);
        fill_pattern(input, sizeof(input),
                     UINT64_C(0x510E527FADE682D1) + key_number);
        if (!mkv128_set_encrypt_key(&context, key)
            || !mkv128_reference_set_encrypt_key(&reference, key)
            || memcmp(context.round_keys, reference.round_keys,
                      sizeof(context.round_keys)) != 0) {
            return 0;
        }

        for (block = 0u; block < 32u; ++block) {
            const size_t offset = block * (size_t)MKV128_BLOCK_BYTES;

            mkv128_reference_encrypt_block(
                &reference, input + offset, expected + offset);
        }

        for (backend_index = 0u; backend_index < 3u; ++backend_index) {
            const mkv128_backend backend = (mkv128_backend)backend_index;

            if (!mkv128_backend_available(backend)) {
                continue;
            }
            if (!mkv128_ecb_encrypt_parallel_simd(
                    &context, input, actual, 32u, 1u, backend)
                || memcmp(actual, expected, sizeof(actual)) != 0) {
                fprintf(stderr, "Independent key %u failed: %s\n",
                        key_number, mkv128_backend_name(backend));
                return 0;
            }
        }
    }

    return 1;
}

static int run_selftests(
    const mkv128_ctx *context,
    const mkv128_ctx *decrypt_context,
    const mkv128_reference_ctx *reference)
{
    int passed;

    passed = test_published_vectors(context, decrypt_context, reference);
    passed &= report_check(test_ecb_differential(context),
                           "ECB boundaries and in-place");
    passed &= report_check(test_independent_keys(),
                           "32 independent keys");

    if (passed) {
        printf("Self-test: PASS\n");
    }
    return passed;
}

static int compare_doubles(const void *left, const void *right)
{
    const double first = *(const double *)left;
    const double second = *(const double *)right;

    return (first > second) - (first < second);
}

static double median(const double *values, unsigned count);

static unsigned samples_near_median(
    const double *values,
    unsigned count,
    double middle)
{
    unsigned accepted = 0u;
    unsigned index;

    for (index = 0u; index < count; ++index) {
        const double distance = values[index] > middle
            ? values[index] - middle : middle - values[index];

        if (middle > 0.0 && distance / middle <= MAX_MEDIAN_DEVIATION) {
            ++accepted;
        }
    }
    return accepted;
}

static void encrypt_reference_blocks(
    const mkv128_reference_ctx *reference,
    const uint8_t *input,
    uint8_t *output,
    size_t blocks)
{
    size_t block;

    for (block = 0u; block < blocks; ++block) {
        const size_t offset = block * (size_t)MKV128_BLOCK_BYTES;

        mkv128_reference_encrypt_block(
            reference, input + offset, output + offset);
    }
}

static int measure_reference(
    const mkv128_reference_ctx *reference,
    const uint8_t *input,
    const uint8_t *expected,
    uint8_t *actual,
    size_t blocks,
    unsigned runs,
    LARGE_INTEGER frequency)
{
    const size_t bytes = blocks * (size_t)MKV128_BLOCK_BYTES;
    const size_t warmup_blocks = blocks < WARMUP_BLOCKS
        ? blocks : (size_t)WARMUP_BLOCKS;
    const size_t warmup_bytes = warmup_blocks * (size_t)MKV128_BLOCK_BYTES;
    double samples[MAX_RUNS];
    double result;
    unsigned accepted;
    unsigned required;
    unsigned warmup_pass;
    unsigned run;

    encrypt_reference_blocks(reference, input, actual, warmup_blocks);
    if (memcmp(actual, expected, warmup_bytes) != 0) {
        fprintf(stderr, "Direct-control warm-up mismatch.\n");
        return 0;
    }

    for (warmup_pass = 0u;
         warmup_pass < DIRECT_FULL_WARMUP_PASSES;
         ++warmup_pass) {
        encrypt_reference_blocks(reference, input, actual, blocks);
        if (memcmp(actual, expected, bytes) != 0) {
            fprintf(stderr, "Direct-control full warm-up mismatch.\n");
            return 0;
        }
    }

    for (run = 0u; run < runs; ++run) {
        LARGE_INTEGER start;
        LARGE_INTEGER finish;
        double seconds;

        if (!QueryPerformanceCounter(&start)) {
            return 0;
        }
        encrypt_reference_blocks(reference, input, actual, blocks);
        if (!QueryPerformanceCounter(&finish)) {
            return 0;
        }
        seconds = (double)(finish.QuadPart - start.QuadPart)
                / (double)frequency.QuadPart;
        if (seconds <= 0.0 || memcmp(actual, expected, bytes) != 0) {
            fprintf(stderr, "Direct-control full ciphertext mismatch.\n");
            return 0;
        }
        samples[run] = (double)bytes / (seconds * 1000000.0);
    }

    result = median(samples, runs);
    accepted = samples_near_median(samples, runs, result);
    required = runs >= 3u ? runs - 1u : runs;
    printf("%-28s %6u %12.2f %8s\n",
           "Direct control", 1u, result, "-");
    fflush(stdout);
    if (accepted < required) {
        fprintf(stderr,
                "Direct-control samples are unstable: only %u/%u are within %.2f%% of the median.\n",
                accepted, runs, 100.0 * MAX_MEDIAN_DEVIATION);
        return 0;
    }
    return 1;
}

static double median(const double *values, unsigned count)
{
    double ordered[MAX_RUNS];

    memcpy(ordered, values, count * sizeof(ordered[0]));
    qsort(ordered, count, sizeof(ordered[0]), compare_doubles);
    if ((count & 1u) != 0u) {
        return ordered[count / 2u];
    }
    return (ordered[count / 2u - 1u] + ordered[count / 2u]) / 2.0;
}

static unsigned build_worker_schedule(unsigned maximum, unsigned *schedule)
{
    static const unsigned candidates[] = {1u, 2u, 4u, 8u, 16u};
    unsigned count = 0u;
    unsigned index;

    for (index = 0u; index < sizeof(candidates) / sizeof(candidates[0]); ++index) {
        if (candidates[index] <= maximum) {
            schedule[count++] = candidates[index];
        }
    }
    return count;
}

static int run_configuration(
    const mkv128_ctx *context,
    mkv128_backend backend,
    const uint8_t *input,
    uint8_t *output,
    size_t blocks,
    unsigned workers)
{
    return mkv128_ecb_encrypt_parallel_simd(
        context, input, output, blocks, workers, backend);
}

static int measure_ecb(
    const mkv128_ctx *context,
    const uint8_t *input,
    const uint8_t *expected,
    uint8_t *actual,
    size_t blocks,
    unsigned max_workers,
    unsigned runs,
    LARGE_INTEGER frequency)
{
    const size_t bytes = blocks * (size_t)MKV128_BLOCK_BYTES;
    unsigned schedule[MAX_SCHEDULE];
    double scalar_results[MAX_SCHEDULE] = {0.0};
    const unsigned schedule_count = build_worker_schedule(max_workers, schedule);
    unsigned backend_index;

    if (schedule_count == 0u) {
        fprintf(stderr, "The worker schedule is empty.\n");
        return 0;
    }

    for (backend_index = 0u; backend_index < 3u; ++backend_index) {
        const mkv128_backend backend = (mkv128_backend)backend_index;
        unsigned schedule_index;

        if (!mkv128_backend_available(backend)) {
            continue;
        }

        for (schedule_index = 0u; schedule_index < schedule_count;
             ++schedule_index) {
            const unsigned workers = schedule[schedule_index];
            const size_t warmup_blocks = blocks < WARMUP_BLOCKS
                ? blocks : (size_t)WARMUP_BLOCKS;
            const size_t warmup_bytes =
                warmup_blocks * (size_t)MKV128_BLOCK_BYTES;
            double samples[MAX_RUNS];
            double result;
            unsigned accepted;
            unsigned required;
            unsigned run;

            if (!run_configuration(context, backend, input, actual,
                                   warmup_blocks, workers)
                || memcmp(actual, expected, warmup_bytes) != 0) {
                fprintf(stderr, "Warm-up failed: %s / %u workers.\n",
                        mkv128_backend_name(backend), workers);
                return 0;
            }

            for (run = 0u; run < runs; ++run) {
                LARGE_INTEGER start;
                LARGE_INTEGER finish;
                double seconds;

                if (!QueryPerformanceCounter(&start)
                    || !run_configuration(context, backend,
                                          input, actual, blocks, workers)
                    || !QueryPerformanceCounter(&finish)) {
                    fprintf(stderr, "Timed ECB run failed: %s / %u workers.\n",
                            mkv128_backend_name(backend), workers);
                    return 0;
                }

                seconds = (double)(finish.QuadPart - start.QuadPart)
                        / (double)frequency.QuadPart;
                if (seconds <= 0.0) {
                    return 0;
                }
                samples[run] = (double)bytes / (seconds * 1000000.0);

                if (memcmp(actual, expected, bytes) != 0) {
                    fprintf(stderr,
                            "Full ECB ciphertext mismatch: %s / %u workers / run %u.\n",
                            mkv128_backend_name(backend), workers, run + 1u);
                    return 0;
                }
            }

            result = median(samples, runs);
            accepted = samples_near_median(samples, runs, result);
            required = runs >= 3u ? runs - 1u : runs;
            if (backend == MKV128_BACKEND_SCALAR) {
                scalar_results[schedule_index] = result;
            }
            if (scalar_results[schedule_index] <= 0.0) {
                fprintf(stderr, "The required scalar baseline was not measured.\n");
                return 0;
            }
            printf("%-28s %6u %12.2f %8.2fx\n",
                   mkv128_backend_name(backend), workers, result,
                   result / scalar_results[schedule_index]);
            if (accepted < required) {
                fprintf(stderr,
                        "Unstable samples: %s / %u workers / only %u/%u are within %.2f%% of the median.\n",
                        mkv128_backend_name(backend), workers,
                        accepted, runs, 100.0 * MAX_MEDIAN_DEVIATION);
                return 0;
            }
            fflush(stdout);
        }
    }

    return 1;
}

static int run_benchmarks(
    const mkv128_ctx *context,
    const mkv128_reference_ctx *reference,
    size_t blocks,
    unsigned workers,
    unsigned runs)
{
    const size_t bytes = blocks * (size_t)MKV128_BLOCK_BYTES;
    uint8_t *input = (uint8_t *)malloc(bytes);
    uint8_t *expected = (uint8_t *)malloc(bytes);
    uint8_t *actual = (uint8_t *)malloc(bytes);
    LARGE_INTEGER frequency;
    int passed = 0;

    if (input == NULL || expected == NULL || actual == NULL
        || !QueryPerformanceFrequency(&frequency)
        || frequency.QuadPart <= 0) {
        fprintf(stderr, "Unable to allocate three %llu-byte buffers or start timer.\n",
                (unsigned long long)bytes);
        goto cleanup;
    }

    fill_pattern(input, bytes, UINT64_C(0x243F6A8885A308D3));
    encrypt_reference_blocks(reference, input, expected, blocks);
    printf("\nECB | %llu blocks | %u runs | up to %u threads\n",
           (unsigned long long)blocks, runs, workers);
    printf("Implementation                 Threads       MB/s    Speedup\n");
    fflush(stdout);

    passed = measure_reference(reference, input, expected, actual,
                               blocks, runs, frequency)
          && measure_ecb(context, input, expected, actual,
                         blocks, workers, runs, frequency);
    if (passed) {
        passed = report_power_source(1);
    }

cleanup:
    free(actual);
    free(expected);
    free(input);
    return passed;
}

static int parse_number(const char *text, unsigned long long *result)
{
    char *tail = NULL;
    unsigned long long parsed;

    if (text == NULL || *text == '\0' || *text == '-') {
        return 0;
    }
    errno = 0;
    parsed = strtoull(text, &tail, 10);
    if (errno != 0 || tail == text || *tail != '\0') {
        return 0;
    }
    *result = parsed;
    return 1;
}

int main(int argc, char **argv)
{
    mkv128_ctx context;
    mkv128_ctx decrypt_context;
    mkv128_reference_ctx reference;
    SYSTEM_INFO system;
    unsigned workers;
    size_t blocks = DEFAULT_BLOCKS;
    unsigned runs = DEFAULT_RUNS;
    unsigned long long value;
    int only_selftest = 0;

    GetSystemInfo(&system);
    workers = system.dwNumberOfProcessors;
    if (workers == 0u) {
        workers = 1u;
    }
    if (workers > MAX_BENCH_WORKERS) {
        workers = MAX_BENCH_WORKERS;
    }
    if (workers > MKV128_SIMD_MAX_WORKERS) {
        workers = MKV128_SIMD_MAX_WORKERS;
    }

    if (argc > 5) {
        goto invalid_arguments;
    }
    if (argc > 1) {
        if (!parse_number(argv[1], &value)
            || value == 0u
            || value > MAX_BENCH_WORKERS
            || value > MKV128_SIMD_MAX_WORKERS) {
            goto invalid_arguments;
        }
        workers = (unsigned)value;
    }
    if (argc > 2) {
        if (!parse_number(argv[2], &value)
            || value == 0u
            || value > SIZE_MAX / (size_t)MKV128_BLOCK_BYTES
            || value > UINT32_MAX) {
            goto invalid_arguments;
        }
        blocks = (size_t)value;
    }
    if (argc > 3) {
        if (!parse_number(argv[3], &value)
            || value == 0u || value > MAX_RUNS) {
            goto invalid_arguments;
        }
        runs = (unsigned)value;
    }
    if (argc > 4) {
        if (strcmp(argv[4], "selftest") != 0) {
            goto invalid_arguments;
        }
        only_selftest = 1;
    }

    if (!report_power_source(!only_selftest)
        || (!only_selftest && !configure_benchmark_priority())
        || !mkv128_set_encrypt_key(&context, STANDARD_KEY)
        || !mkv128_set_decrypt_key(&decrypt_context, STANDARD_KEY)
        || !mkv128_reference_set_encrypt_key(&reference, STANDARD_KEY)
        || !run_selftests(&context, &decrypt_context, &reference)) {
        return EXIT_FAILURE;
    }
    if (only_selftest) {
        return EXIT_SUCCESS;
    }
    return run_benchmarks(&context, &reference, blocks, workers, runs)
        ? EXIT_SUCCESS
        : EXIT_FAILURE;

invalid_arguments:
    fprintf(stderr,
            "Usage: %s [max-workers:1..16] [blocks] [runs:1..9] [selftest]\n",
            argv[0]);
    return EXIT_FAILURE;
}
