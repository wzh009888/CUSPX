// #define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../api.h"
#include "../common.h"
#include "../fors.h"
#include "../params.h"
#include "../rng.h"
#include "../wots.h"

#define SPX_MLEN 32
#if defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F)
#define WARM_NTESTS 30
#define NTESTS 5
#else // if defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F)
#define WARM_NTESTS 2
#define NTESTS 2
#endif // if defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F)

#include "all_option.h"
#include <iostream>
using namespace std;

#define MEASURE(TEXT, MUL, FNCALL)                                                                 \
    printf(TEXT);                                                                                  \
    g_result = 0;                                                                                  \
    g_count = 0;                                                                                   \
    for (int i = 0; i < MUL; i++)                                                                  \
        FNCALL;                                                                                    \
    printf("%11.3lf ms (%2.3lf sec)\n", g_result / MUL / 1e3, g_result / MUL / 1e6);

void test_all();
void test_nsight();
void test_compare();


int main(int argc, char** argv) {
    // test_nsight();
    // test_all();
    test_compare();
}

// 128f测试
void test_nsight() {
    /* Make stdout buffer more responsive. */
    setbuf(stdout, NULL);

    int num = 512;

    u8 *pk, *sk;
    u8 *m, *sm, *mout;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES * num));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES * num));
    CHECK(cudaMallocHost(&m, SPX_MLEN * num));
    CHECK(cudaMallocHost(&sm, (SPX_BYTES + SPX_MLEN) * num));
    CHECK(cudaMallocHost(&mout, (SPX_BYTES + SPX_MLEN) * num));

    unsigned long long smlen;
    unsigned long long mlen;

    randombytes(m, SPX_MLEN * num);
    show_para();

    face_mhp_crypto_sign_keypair_scheme2(pk, sk, 512, 40);
    face_mhp_crypto_sign_scheme2(sm, &smlen, m, SPX_MLEN, sk, 512, 40);
    face_mhp_crypto_sign_open(mout, &mlen, sm, smlen, pk, 512, 18);

    CHECK(cudaFreeHost(pk));
    CHECK(cudaFreeHost(sk));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(sm));
    CHECK(cudaFreeHost(mout));
}


void test_all() {
    /* Make stdout buffer more responsive. */
    setbuf(stdout, NULL);

    int num = 32768;

    u8 *pk, *sk;
    u8 *m, *sm, *mout;
    double time1, time2, time3;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES * num));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES * num));
    CHECK(cudaMallocHost(&m, SPX_MLEN * num));
    CHECK(cudaMallocHost(&sm, (SPX_BYTES + SPX_MLEN) * num));
    CHECK(cudaMallocHost(&mout, (SPX_BYTES + SPX_MLEN) * num));

    unsigned long long smlen;
    unsigned long long mlen;
    // struct timespec start, stop;
    // double result;

    randombytes(m, SPX_MLEN * num);
    show_para();

    printf("warming up\n");
    printf("Running %d iterations.\n", WARM_NTESTS);
    for (int i = 0; i < WARM_NTESTS; i++) {
        face_mhp_crypto_sign_keypair_scheme2(pk, sk, 32, 8);
        face_mhp_crypto_sign_scheme2(sm, &smlen, m, SPX_MLEN, sk, 32, 8);
        face_mhp_crypto_sign_open(mout, &mlen, sm, smlen, pk, 32, 8);
    }

    printf("scheme 2 hybrid parallelism benchmark\n");
    printf("Running %d iterations.\n", NTESTS);

    int test[] = {32, 64, 128, 256, 512, 1024, 2048}; // f/s right
    // int test[] = {512}; // s max to 128

    for (int k = 0; k < 8; k++) {
        int i = 1;
        g_result = 0;
        for (int j = 0; j < NTESTS; j++)
            face_mhp_crypto_sign_keypair_scheme2(pk, sk, test[k], i);
        time1 = g_result / NTESTS;
        g_result = 0;
        for (int j = 0; j < NTESTS; j++)
            face_mhp_crypto_sign_scheme2(sm, &smlen, m, SPX_MLEN, sk, test[k], i);
        time2 = g_result / NTESTS;
        g_result = 0;
        for (int j = 0; j < NTESTS; j++)
            face_mhp_crypto_sign_open(mout, &mlen, sm, smlen, pk, test[k], i);
        time3 = g_result / NTESTS;
        printf("%d\t%d\t%.2f %.2f %.2f \n", test[k], i * test[k], test[k] * 1e6 / time1,
               test[k] * 1e6 / time2, test[k] * 1e6 / time3);
        printf("%d\t%d\t%.2f %.2f %.2f \n", test[k], i * test[k], time1 / 1e3,
               time2 / 1e3, time3 / 1e3);
    }
    printf("\n");

    for (int k = 0; k < 8; k++) {
        int begin = 1024 / test[k];
        int end = 32768 / test[k];
        int step = (end - begin) / 31;
        if (step == 0) step = 1;
        if (begin == 0) begin = 1;
        for (int i = begin; i < end; i += step) {
            // for (int i = begin; i <= begin; i += step) {
            g_result = 0;
            for (int j = 0; j < NTESTS; j++)
                face_mhp_crypto_sign_keypair_scheme2(pk, sk, test[k], i);
            time1 = g_result / NTESTS;
            g_result = 0;
            for (int j = 0; j < NTESTS; j++)
                face_mhp_crypto_sign_scheme2(sm, &smlen, m, SPX_MLEN, sk, test[k], i);
            time2 = g_result / NTESTS;
            g_result = 0;
            for (int j = 0; j < NTESTS; j++)
                face_mhp_crypto_sign_open(mout, &mlen, sm, smlen, pk, test[k], i);
            time3 = g_result / NTESTS;
            printf("%d\t%d\t%.2f %.2f %.2f \n", test[k], i * test[k], test[k] * 1e6 / time1,
                   test[k] * 1e6 / time2, test[k] * 1e6 / time3);
        }
        printf("\n");
    }

    CHECK(cudaFreeHost(pk));
    CHECK(cudaFreeHost(sk));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(sm));
    CHECK(cudaFreeHost(mout));

    // 之前的代码见version_4.1
} // main


void test_compare() {
    /* Make stdout buffer more responsive. */
    setbuf(stdout, NULL);

    int num = 32768;

    u8 *pk, *sk;
    u8 *m, *sm, *mout;
    double time1, time2, time3;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES * num));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES * num));
    CHECK(cudaMallocHost(&m, SPX_MLEN * num));
    CHECK(cudaMallocHost(&sm, (SPX_BYTES + SPX_MLEN) * num));
    CHECK(cudaMallocHost(&mout, (SPX_BYTES + SPX_MLEN) * num));

    unsigned long long smlen;
    unsigned long long mlen;
    // struct timespec start, stop;
    // double result;

    randombytes(m, SPX_MLEN * num);
    show_para();

    printf("warming up\n");
    printf("Running %d iterations.\n", WARM_NTESTS);
    for (int i = 0; i < WARM_NTESTS; i++) {
        face_mhp_crypto_sign_keypair_scheme2(pk, sk, 32, 8);
        face_mhp_crypto_sign_scheme2(sm, &smlen, m, SPX_MLEN, sk, 32, 8);
        face_mhp_crypto_sign_open(mout, &mlen, sm, smlen, pk, 32, 8);
    }

    printf("scheme 2 hybrid parallelism benchmark\n");
    printf("Running %d iterations.\n", NTESTS);

    int test[] = {512}; // s max to 128

    for (int k = 0; k < 1; k++) {
        int i = 1;
        g_result = 0;
        for (int j = 0; j < NTESTS; j++)
            face_mhp_crypto_sign_keypair_scheme2(pk, sk, test[k], i);
        time1 = g_result / NTESTS;
        g_result = 0;
        for (int j = 0; j < NTESTS; j++)
            face_mhp_crypto_sign_scheme2_compare(sm, &smlen, m, SPX_MLEN, sk, test[k], i);
        time2 = g_result / NTESTS;
        g_result = 0;
        for (int j = 0; j < NTESTS; j++)
            face_mhp_crypto_sign_open_compare(mout, &mlen, sm, smlen, pk, test[k], i);
        time3 = g_result / NTESTS;
        printf("%d\t%d\t%.2f %.2f %.2f \n", test[k], i * test[k], test[k] * 1e6 / time1,
               test[k] * 1e6 / time2, test[k] * 1e6 / time3);
        printf("%d\t%d\t%.2f %.2f %.2f \n", test[k], i * test[k], time1 / 1e3,
               time2 / 1e3, time3 / 1e3);
    }
    printf("\n");

    for (int k = 0; k < 1; k++) {
        int begin = 1024 / test[k];
        int end = 51200 / test[k];
        int step = (end - begin) / 49;
        if (step == 0) step = 1;
        if (begin == 0) begin = 1;
        for (int i = begin; i < end; i += step) {
            // for (int i = begin; i <= begin; i += step) {
            g_result = 0;
            for (int j = 0; j < NTESTS; j++)
                face_mhp_crypto_sign_keypair_scheme2(pk, sk, test[k], i);
            time1 = g_result / NTESTS;
            g_result = 0;
            for (int j = 0; j < NTESTS; j++)
                face_mhp_crypto_sign_scheme2_compare(sm, &smlen, m, SPX_MLEN, sk, test[k], i);
            time2 = g_result / NTESTS;
            g_result = 0;
            for (int j = 0; j < NTESTS; j++)
                face_mhp_crypto_sign_open_compare(mout, &mlen, sm, smlen, pk, test[k], i);
            time3 = g_result / NTESTS;
            printf("%d\t%d\t%.2f %.2f %.2f \n", test[k], i * test[k], test[k] * 1e6 / time1,
                   test[k] * 1e6 / time2, test[k] * 1e6 / time3);
        }
        printf("\n");
    }

    CHECK(cudaFreeHost(pk));
    CHECK(cudaFreeHost(sk));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(sm));
    CHECK(cudaFreeHost(mout));

    // 之前的代码见version_4.1
} // main
