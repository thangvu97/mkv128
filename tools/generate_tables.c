/*
 * Reproducible MKV-128 T-table generator.
 *
 * The S-box is derived directly from the A-construction over GF(2^4), and
 * MixWords is calculated as the fourth power of its companion matrix over
 * GF(2^8).  Neither the S-box nor the resulting MixWords matrices are copied
 * from an existing implementation.
 *
 * Construction: Nguyen et al., "MKV: a new block cipher for the post-quantum
 * cryptography transition", Mathematical Aspects of Cryptography 16(2),
 * 113-138, 2025. DOI: https://doi.org/10.4213/mvk497
 *
 * Build: gcc -std=c11 -O2 -Wall -Wextra -pedantic tools/generate_tables.c -o gen
 * Run:   ./gen [PrecomputedTable128.h]
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GF16_POLYNOMIAL  UINT16_C(0x013)
#define GF256_POLYNOMIAL UINT16_C(0x12B)
#define MATRIX_SIZE      4u
#define TABLE_SIZE       256u
#define TABLE_COUNT      5u
#define DEFAULT_OUTPUT   "PrecomputedTable128.h"

typedef struct {
    uint8_t value[MATRIX_SIZE][MATRIX_SIZE];
} matrix4_t;

typedef struct {
    uint32_t forward[TABLE_COUNT][TABLE_SIZE];
    uint32_t inverse[TABLE_COUNT][TABLE_SIZE];
} mkv_tables_t;

static uint8_t gf_multiply(uint8_t left, uint8_t right,
                           unsigned degree, uint16_t polynomial)
{
    const unsigned mask = (1u << degree) - 1u;
    const unsigned reduction = polynomial & mask;
    unsigned result = 0u;
    unsigned multiplicand = left;
    unsigned multiplier = right;
    unsigned bit;

    for (bit = 0u; bit < degree; ++bit) {
        if ((multiplier & 1u) != 0u) {
            result ^= multiplicand;
        }

        if ((multiplicand & (1u << (degree - 1u))) != 0u) {
            multiplicand = ((multiplicand << 1u) & mask) ^ reduction;
        } else {
            multiplicand = (multiplicand << 1u) & mask;
        }
        multiplier >>= 1u;
    }

    return (uint8_t)result;
}

static uint8_t gf_power(uint8_t value, unsigned exponent,
                        unsigned degree, uint16_t polynomial)
{
    uint8_t result = 1u;

    /* The MKV A-construction extends negative powers by setting 0^(-k) = 0. */
    if (value == 0u) {
        return 0u;
    }

    while (exponent != 0u) {
        if ((exponent & 1u) != 0u) {
            result = gf_multiply(result, value, degree, polynomial);
        }
        value = gf_multiply(value, value, degree, polynomial);
        exponent >>= 1u;
    }

    return result;
}

static uint8_t gf16_multiply(uint8_t left, uint8_t right)
{
    return gf_multiply(left, right, 4u, GF16_POLYNOMIAL);
}

static uint8_t gf16_power(uint8_t value, unsigned exponent)
{
    return gf_power(value, exponent, 4u, GF16_POLYNOMIAL);
}

static uint8_t gf256_multiply(uint8_t left, uint8_t right)
{
    return gf_multiply(left, right, 8u, GF256_POLYNOMIAL);
}

static uint8_t gf256_inverse(uint8_t value)
{
    /* Every nonzero element satisfies a^255 = 1 in GF(2^8). */
    return gf_power(value, 254u, 8u, GF256_POLYNOMIAL);
}

static uint8_t construct_sbox_entry(uint8_t input)
{
    const uint8_t left = (uint8_t)(input >> 4u);
    const uint8_t right = (uint8_t)(input & 0x0Fu);
    uint8_t output_right;
    uint8_t output_left;

    /* In GF(16), x^-4 = x^11, x^-2 = x^13 and x^-1 = x^14. */
    if (right != 0u) {
        output_right = gf16_multiply(gf16_power(left, 11u), right);
    } else {
        output_right = gf16_power(left, 14u);
    }

    if (output_right != 0u) {
        output_left = gf16_power(gf16_multiply(right, output_right), 13u);
    } else {
        output_left = gf16_power(right, 14u);
    }

    /* The XOR with one belongs to the completed byte, not to y_r beforehand. */
    return (uint8_t)(((unsigned)output_left << 4u) | output_right) ^ 0x01u;
}

static int construct_sboxes(uint8_t sbox[TABLE_SIZE],
                            uint8_t inverse_sbox[TABLE_SIZE])
{
    uint8_t seen[TABLE_SIZE] = {0u};
    unsigned input;

    for (input = 0u; input < TABLE_SIZE; ++input) {
        const uint8_t output = construct_sbox_entry((uint8_t)input);

        if (seen[output] != 0u) {
            fprintf(stderr, "Generated S-box is not bijective at input 0x%02X.\n", input);
            return 0;
        }
        seen[output] = 1u;
        sbox[input] = output;
        inverse_sbox[output] = (uint8_t)input;
    }

    return 1;
}

static matrix4_t matrix_identity(void)
{
    matrix4_t result = {{{0u}}};
    unsigned index;

    for (index = 0u; index < MATRIX_SIZE; ++index) {
        result.value[index][index] = 1u;
    }

    return result;
}

static matrix4_t matrix_multiply(const matrix4_t *left, const matrix4_t *right)
{
    matrix4_t result = {{{0u}}};
    unsigned row;
    unsigned column;
    unsigned index;

    for (row = 0u; row < MATRIX_SIZE; ++row) {
        for (column = 0u; column < MATRIX_SIZE; ++column) {
            for (index = 0u; index < MATRIX_SIZE; ++index) {
                result.value[row][column] ^= gf256_multiply(
                    left->value[row][index], right->value[index][column]);
            }
        }
    }

    return result;
}

static matrix4_t matrix_power(const matrix4_t *matrix, unsigned exponent)
{
    matrix4_t result = matrix_identity();
    matrix4_t factor = *matrix;

    while (exponent != 0u) {
        if ((exponent & 1u) != 0u) {
            result = matrix_multiply(&result, &factor);
        }
        factor = matrix_multiply(&factor, &factor);
        exponent >>= 1u;
    }

    return result;
}

static int matrix_inverse(const matrix4_t *matrix, matrix4_t *inverse)
{
    uint8_t augmented[MATRIX_SIZE][2u * MATRIX_SIZE] = {{0u}};
    unsigned row;
    unsigned column;

    for (row = 0u; row < MATRIX_SIZE; ++row) {
        for (column = 0u; column < MATRIX_SIZE; ++column) {
            augmented[row][column] = matrix->value[row][column];
        }
        augmented[row][MATRIX_SIZE + row] = 1u;
    }

    for (column = 0u; column < MATRIX_SIZE; ++column) {
        unsigned pivot = column;
        unsigned position;
        uint8_t scale;

        while (pivot < MATRIX_SIZE && augmented[pivot][column] == 0u) {
            ++pivot;
        }
        if (pivot == MATRIX_SIZE) {
            fprintf(stderr, "MixWords matrix is singular at column %u.\n", column);
            return 0;
        }

        if (pivot != column) {
            for (position = 0u; position < 2u * MATRIX_SIZE; ++position) {
                const uint8_t temporary = augmented[column][position];
                augmented[column][position] = augmented[pivot][position];
                augmented[pivot][position] = temporary;
            }
        }

        scale = gf256_inverse(augmented[column][column]);
        for (position = 0u; position < 2u * MATRIX_SIZE; ++position) {
            augmented[column][position] = gf256_multiply(
                augmented[column][position], scale);
        }

        for (row = 0u; row < MATRIX_SIZE; ++row) {
            const uint8_t coefficient = augmented[row][column];

            if (row == column || coefficient == 0u) {
                continue;
            }
            for (position = 0u; position < 2u * MATRIX_SIZE; ++position) {
                augmented[row][position] ^= gf256_multiply(
                    coefficient, augmented[column][position]);
            }
        }
    }

    for (row = 0u; row < MATRIX_SIZE; ++row) {
        for (column = 0u; column < MATRIX_SIZE; ++column) {
            inverse->value[row][column] = augmented[row][MATRIX_SIZE + column];
        }
    }

    return 1;
}

static int matrix_is_identity(const matrix4_t *matrix)
{
    unsigned row;
    unsigned column;

    for (row = 0u; row < MATRIX_SIZE; ++row) {
        for (column = 0u; column < MATRIX_SIZE; ++column) {
            if (matrix->value[row][column] != (uint8_t)(row == column)) {
                return 0;
            }
        }
    }

    return 1;
}

static uint32_t pack_matrix_column(const matrix4_t *matrix,
                                   unsigned column, uint8_t value)
{
    uint32_t packed = 0u;
    unsigned row;

    for (row = 0u; row < MATRIX_SIZE; ++row) {
        packed = (packed << 8u) | gf256_multiply(matrix->value[row][column], value);
    }

    return packed;
}

static void construct_tables(mkv_tables_t *tables,
                             const uint8_t sbox[TABLE_SIZE],
                             const uint8_t inverse_sbox[TABLE_SIZE],
                             const matrix4_t *mixwords,
                             const matrix4_t *inverse_mixwords)
{
    unsigned input;
    unsigned column;

    for (input = 0u; input < TABLE_SIZE; ++input) {
        for (column = 0u; column < MATRIX_SIZE; ++column) {
            tables->forward[column][input] = pack_matrix_column(
                mixwords, column, sbox[input]);
            tables->inverse[column][input] = pack_matrix_column(
                inverse_mixwords, column, inverse_sbox[input]);
        }

        tables->forward[4u][input] = (uint32_t)sbox[input] << 24u;
        tables->inverse[4u][input] = (uint32_t)inverse_sbox[input] << 24u;
    }
}

static void print_matrix(const char *name, const matrix4_t *matrix)
{
    unsigned row;
    unsigned column;

    printf("%s:\n", name);
    for (row = 0u; row < MATRIX_SIZE; ++row) {
        printf("  [");
        for (column = 0u; column < MATRIX_SIZE; ++column) {
            printf("%s%02X", column == 0u ? "" : " ", matrix->value[row][column]);
        }
        printf("]\n");
    }
}

static int write_array(FILE *output, const char *name,
                       const uint32_t values[TABLE_SIZE])
{
    unsigned index;

    if (fprintf(output, "static const uint32_t %s[256] = {\n", name) < 0) {
        return 0;
    }

    for (index = 0u; index < TABLE_SIZE; ++index) {
        if (index % 8u == 0u && fputs("    ", output) == EOF) {
            return 0;
        }

        if (fprintf(output, "0x%08Xu%s", (unsigned)values[index],
                    index + 1u == TABLE_SIZE ? "" : ",") < 0) {
            return 0;
        }

        if (index % 8u == 7u) {
            if (fputc('\n', output) == EOF) {
                return 0;
            }
        } else if (fputc(' ', output) == EOF) {
            return 0;
        }
    }

    return fputs("};\n\n", output) != EOF;
}

static int write_header(const char *path, const mkv_tables_t *tables)
{
    static const char *const forward_names[TABLE_COUNT] = {
        "L0", "L1", "L2", "L3", "L4"
    };
    static const char *const inverse_names[TABLE_COUNT] = {
        "iL0", "iL1", "iL2", "iL3", "iL4"
    };
    FILE *output = fopen(path, "w");
    unsigned table;
    int success = 1;

    if (output == NULL) {
        perror(path);
        return 0;
    }

    if (fputs("/* Generated by tools/generate_tables.c; do not edit by hand. */\n"
              "#ifndef MKV128_PRECOMPUTED_TABLE128_H\n"
              "#define MKV128_PRECOMPUTED_TABLE128_H\n\n"
              "#include <stdint.h>\n\n", output) == EOF) {
        success = 0;
    }

    for (table = 0u; success != 0 && table < TABLE_COUNT; ++table) {
        success = write_array(output, forward_names[table], tables->forward[table]);
    }
    for (table = 0u; success != 0 && table < TABLE_COUNT; ++table) {
        success = write_array(output, inverse_names[table], tables->inverse[table]);
    }

    if (success != 0 && fputs("#endif /* MKV128_PRECOMPUTED_TABLE128_H */\n", output) == EOF) {
        success = 0;
    }
    if (fclose(output) != 0) {
        success = 0;
    }

    if (success == 0) {
        fprintf(stderr, "Failed while writing %s.\n", path);
    }
    return success;
}

int main(int argc, char **argv)
{
    static const matrix4_t companion = {{
        {0x00u, 0x01u, 0x00u, 0x00u},
        {0x00u, 0x00u, 0x01u, 0x00u},
        {0x00u, 0x00u, 0x00u, 0x01u},
        {0x01u, 0x02u, 0x01u, 0x03u}
    }};
    uint8_t sbox[TABLE_SIZE];
    uint8_t inverse_sbox[TABLE_SIZE];
    mkv_tables_t tables;
    matrix4_t mixwords;
    matrix4_t inverse_mixwords;
    matrix4_t identity_check;
    const char *output_path;

    if (argc > 2) {
        fprintf(stderr, "Usage: %s [output-header-path]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 2 &&
        (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("Usage: %s [output-header-path]\n", argv[0]);
        printf("Default output: %s\n", DEFAULT_OUTPUT);
        return EXIT_SUCCESS;
    }
    output_path = argc == 2 ? argv[1] : DEFAULT_OUTPUT;

    if (construct_sboxes(sbox, inverse_sbox) == 0) {
        return EXIT_FAILURE;
    }

    mixwords = matrix_power(&companion, 4u);
    if (matrix_inverse(&mixwords, &inverse_mixwords) == 0) {
        return EXIT_FAILURE;
    }

    identity_check = matrix_multiply(&mixwords, &inverse_mixwords);
    if (matrix_is_identity(&identity_check) == 0) {
        fprintf(stderr, "Verification failed: M * M^-1 is not the identity.\n");
        return EXIT_FAILURE;
    }

    construct_tables(&tables, sbox, inverse_sbox, &mixwords, &inverse_mixwords);

    if (sbox[0u] != 0x01u || tables.forward[0u][0u] != UINT32_C(0x0103040D)) {
        fprintf(stderr, "Sanity check failed for S(0x00) or L0[0x00].\n");
        return EXIT_FAILURE;
    }

    if (write_header(output_path, &tables) == 0) {
        return EXIT_FAILURE;
    }

    printf("GF(16) polynomial: 0x%02X\n", GF16_POLYNOMIAL);
    printf("GF(256) polynomial: 0x%03X\n", GF256_POLYNOMIAL);
    printf("S(0x00) = 0x%02X; S(0x01) = 0x%02X; S(0x02) = 0x%02X\n",
           sbox[0u], sbox[1u], sbox[2u]);
    print_matrix("MixWords M = A^4", &mixwords);
    print_matrix("Inverse MixWords M^-1", &inverse_mixwords);
    printf("Verification: M * M^-1 = I\n");
    printf("L0[0x00] = 0x%08X; iL0[0x00] = 0x%08X\n",
           (unsigned)tables.forward[0u][0u], (unsigned)tables.inverse[0u][0u]);
    printf("Generated %u tables, %u entries: %s\n",
           2u * TABLE_COUNT, 2u * TABLE_COUNT * TABLE_SIZE, output_path);

    return EXIT_SUCCESS;
}
