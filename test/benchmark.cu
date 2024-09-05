// #define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../api.h"
#include "../common.h"
#include "../fors.h"
#include "../hash.h"
#include "../params.h"
#include "../rng.h"
#include "../wots.h"

#if (defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F))
#define WARM_NTESTS 10
#define NTESTS 10
#else // if (defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F))
#define WARM_NTESTS 2
#define NTESTS 2
#endif // if (defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F))

#include "all_option.h"

#define MEASURE(TEXT, MUL, FNCALL)                                                                 \
    printf(TEXT);                                                                                  \
    g_result = 0;                                                                                  \
    g_count = 0;                                                                                   \
    for (int i = 0; i < MUL; i++)                                                                  \
        FNCALL;                                                                                    \
    printf("%11.2lf ms (%2.2lf sec)\n", g_result / MUL / 1e3, g_result / MUL / 1e6);

void test_spx();
void test_sha();
void test_wots();
void test_xmss();
void test_fors();

int main() {
    show_para();

    printf("Warming up %d iterations.\n", WARM_NTESTS);
    printf("Running %d iterations.\n", NTESTS);

    test_spx();

    test_wots();

    return 0;
} // main

void test_spx() {
    setbuf(stdout, NULL);

    unsigned char *pk, *sk, *m, *sm, *mout;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES * 4));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES * 4));
    CHECK(cudaMallocHost(&m, SPX_MLEN * 4));
    CHECK(cudaMallocHost(&sm, (SPX_BYTES + SPX_MLEN) * 4));
    CHECK(cudaMallocHost(&mout, (SPX_BYTES + SPX_MLEN) * 4));

    unsigned char addr[SPX_ADDR_BYTES];
    unsigned long long smlen, mlen;

    randombytes(m, SPX_MLEN);
    randombytes(addr, SPX_ADDR_BYTES);

    // warm up
    for (int i = 0; i < WARM_NTESTS; i++) {
        face_crypto_sign_keypair(pk, sk);
        face_crypto_sign(sm, &smlen, m, SPX_MLEN, sk);
        face_crypto_sign_open(mout, &mlen, sm, smlen, pk);
    }

    // printf("---------- GPU serial version ----------\n");
    MEASURE("Generating keypair.. ", NTESTS, face_crypto_sign_keypair(pk, sk));
    double result_kg = g_result;
    MEASURE("Signing..            ", NTESTS, face_crypto_sign(sm, &smlen, m, SPX_MLEN, sk));
    double result_sign = g_result;
    MEASURE("Verifying..          ", NTESTS, face_crypto_sign_open(mout, &mlen, sm, smlen, pk));
    double result_verify = g_result;
    printf("Time (ms) for kg, sign, verify\n");
    printf("%.2lf %.2lf %.2lf\n", result_kg / NTESTS / 1e3, result_sign / NTESTS / 1e3,
           result_verify / NTESTS / 1e3);

    printf("Signature size: %d (%.2f KiB)\n", SPX_BYTES, SPX_BYTES / 1024.0);
    printf("Public key size: %d (%.2f KiB)\n", SPX_PK_BYTES, SPX_PK_BYTES / 1024.0);
    printf("Secret key size: %d (%.2f KiB)\n", SPX_SK_BYTES, SPX_SK_BYTES / 1024.0);

    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(sm));
    CHECK(cudaFreeHost(mout));
}

void test_wots() {
    setbuf(stdout, NULL);

    unsigned char *sk_seed, *pub_seed, *pk1, *pk2, *sig, *m;
    uint32_t* addr;

    CHECK(cudaMallocHost(&sk_seed, SPX_N));
    CHECK(cudaMallocHost(&pub_seed, SPX_N));
    CHECK(cudaMallocHost(&pk1, SPX_WOTS_PK_BYTES));
    CHECK(cudaMallocHost(&pk2, SPX_WOTS_PK_BYTES));
    CHECK(cudaMallocHost(&sig, SPX_WOTS_BYTES));
    CHECK(cudaMallocHost(&m, SPX_N));
    CHECK(cudaMallocHost(&addr, 8 * sizeof(uint32_t)));

    randombytes(sk_seed, SPX_N);
    randombytes(pub_seed, SPX_N);
    randombytes(m, SPX_N);
    randombytes((unsigned char*) addr, 8 * sizeof(uint32_t));

    // warm up
    face_ap_wots_gen_pk(pk1, sk_seed, pub_seed, addr, WARM_NTESTS);
    face_wots_sign(sig, m, sk_seed, pub_seed, addr, WARM_NTESTS);
    face_wots_pk_from_sig(pk2, sig, m, pub_seed, addr, WARM_NTESTS);

    printf("\n");
    MEASURE("Generating keypair.. ", NTESTS, face_wots_gen_pk(pk1, sk_seed, pub_seed, addr, 1));
    double result_kg = g_result;
    MEASURE("Signing..            ", NTESTS, face_wots_sign(sig, m, sk_seed, pub_seed, addr, 1));
    double result_sign = g_result;
    MEASURE("Verifying..          ", NTESTS, face_wots_pk_from_sig(pk2, sig, m, pub_seed, addr, 1));
    double result_verify = g_result;
    printf("Time (ms) for kg, sign, verify\n");
    printf("%.2lf %.2lf %.2lf\n", result_kg / NTESTS / 1e3, result_sign / NTESTS / 1e3,
           result_verify / NTESTS / 1e3);

    CHECK(cudaFreeHost(sk_seed));
    CHECK(cudaFreeHost(pub_seed));
    CHECK(cudaFreeHost(pk1));
    CHECK(cudaFreeHost(pk2));
    CHECK(cudaFreeHost(sig));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(addr));
}

void test_fors() {
    setbuf(stdout, NULL);

    unsigned char *sk_seed, *pub_seed, *pk1, *pk2, *sig, *m;
    uint32_t* addr;

    CHECK(cudaMallocHost(&sk_seed, SPX_N));
    CHECK(cudaMallocHost(&pub_seed, SPX_N));
    CHECK(cudaMallocHost(&pk1, SPX_FORS_PK_BYTES));
    CHECK(cudaMallocHost(&pk2, SPX_FORS_PK_BYTES));
    CHECK(cudaMallocHost(&sig, SPX_FORS_BYTES));
    CHECK(cudaMallocHost(&m, SPX_FORS_MSG_BYTES));
    CHECK(cudaMallocHost(&addr, 8 * sizeof(uint32_t)));

    randombytes(sk_seed, SPX_N);
    randombytes(pub_seed, SPX_N);
    randombytes(m, SPX_FORS_MSG_BYTES);
    randombytes((unsigned char*) addr, 8 * sizeof(uint32_t));

    face_initialize_hash_function(pub_seed, sk_seed);

    // warm up
    fors_sign(sig, pk1, m, sk_seed, pub_seed, addr);
    fors_pk_from_sig(pk2, sig, m, pub_seed, addr);

    printf("\n");
    MEASURE("Signing..            ", NTESTS,
            face_fors_sign(sig, pk1, m, sk_seed, pub_seed, addr, 1));
    double result_sign = g_result;
    MEASURE("Verifying..          ", NTESTS, face_fors_pk_from_sig(pk2, sig, m, pub_seed, addr, 1));
    double result_verify = g_result;
    printf("Time (us) for sign, verify\n");
    printf("%.3lf %.3lf\n", result_sign / NTESTS, result_verify / 2);

    CHECK(cudaFreeHost(sk_seed));
    CHECK(cudaFreeHost(pub_seed));
    CHECK(cudaFreeHost(pk1));
    CHECK(cudaFreeHost(pk2));
    CHECK(cudaFreeHost(sig));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(addr));
}
