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
#define WARM_NTESTS 40
#define NTESTS 200
#else // if (defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F))
#define WARM_NTESTS 20
#define NTESTS 100
#endif // if (defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F))

#include "all_option.h"

#define MEASURE(TEXT, MUL, FNCALL)                                                                 \
    printf(TEXT);                                                                                  \
    g_result = 0;                                                                                  \
    g_count = 0;                                                                                   \
    for (int i = 0; i < MUL; i++)                                                                  \
        FNCALL;                                                                                    \
    printf("%11.3lf ms (%2.3lf sec)\n", g_result / MUL / 1e3, g_result / MUL / 1e6);

#define INNER_MEASURE(TEXT, MUL, FNCALL)                                                           \
    printf(TEXT);                                                                                  \
    g_result = 0;                                                                                  \
    g_count = 0;                                                                                   \
    FNCALL;                                                                                        \
    printf("%11.3lf us (%2.3lf ms)\n", g_result / MUL, g_result / MUL / 1e3);

void test_spx();
void test_hash();
void test_wots();
void test_wots_cc();
void test_xmss();
void test_fors();
void test_ht();

int main() {
    show_para();

    printf("Warming up %d iterations.\n", WARM_NTESTS);
    printf("Running %d iterations.\n", NTESTS);

    test_spx();
    test_hash();
    test_ht();
    test_wots();
    test_xmss();
    test_fors();

    test_wots_cc();

    return 0;
} // main

// 整理成三级，便于测试

void test_spx() {
    /* Make stdout buffer more responsive. */
    setbuf(stdout, NULL);

    unsigned char *pk, *sk, *m, *sm, *mout, *addr;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES));
    CHECK(cudaMallocHost(&m, SPX_MLEN));
    CHECK(cudaMallocHost(&sm, SPX_BYTES + SPX_MLEN));
    CHECK(cudaMallocHost(&mout, SPX_BYTES + SPX_MLEN));
    CHECK(cudaMallocHost(&addr, SPX_ADDR_BYTES));

    unsigned long long smlen, mlen;

    randombytes(m, SPX_MLEN);
    randombytes(addr, SPX_ADDR_BYTES);

    // warm up
    for (int i = 0; i < WARM_NTESTS; i++) {
        face_ap_crypto_sign_keypair_2(pk, sk);
        face_ap_crypto_sign_keypair_23(pk, sk);
        face_ap_crypto_sign(sm, &smlen, m, SPX_MLEN, sk);
        face_ap_crypto_sign_open(mout, &mlen, sm, smlen, pk);
    }

    printf("\n");
    MEASURE("PKGEN 0 ..           ", 2, face_crypto_sign_keypair(pk, sk));
    MEASURE("PKGEN 2 ..           ", NTESTS, face_ap_crypto_sign_keypair_2(pk, sk));
    MEASURE("PKGEN 2+3 ..         ", NTESTS, face_ap_crypto_sign_keypair_23(pk, sk));

    MEASURE("Signing 0 ..         ", 2, face_crypto_sign(sm, &smlen, m, SPX_MLEN, sk));
    MEASURE("Signing 1 ..         ", 10, face_ap_crypto_sign_1(sm, &smlen, m, SPX_MLEN, sk));
    MEASURE("Signing 1+2 ..       ", NTESTS, face_ap_crypto_sign_12(sm, &smlen, m, SPX_MLEN, sk));
    MEASURE("Signing 1+2+3 ..     ", NTESTS, face_ap_crypto_sign_123(sm, &smlen, m, SPX_MLEN, sk));

    MEASURE("Verifying 0 ..       ", NTESTS, face_crypto_sign_open(mout, &mlen, sm, smlen, pk));
    MEASURE("Verifying 2+3 ..     ", NTESTS, face_ap_crypto_sign_open(mout, &mlen, sm, smlen, pk));

    CHECK(cudaFreeHost(pk));
    CHECK(cudaFreeHost(sk));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(sm));
    CHECK(cudaFreeHost(mout));
    CHECK(cudaFreeHost(addr));
}

void test_hash() {
    // test the performance of hash function

    // T_l: dev_thash: l
    // H: dev_thash: 2
    // F: dev_thash: 1
    // PRF: dev_prf_addr
    // PRF_msg: dev_gen_message_random
    // H_msg: dev_hash_message

    face_h(10000); // warm up

    printf("\n");
    INNER_MEASURE("T_l wots ..     ", 10000, face_tl(SPX_WOTS_LEN, 10000));
    INNER_MEASURE("T_l fors ..     ", 10000, face_tl(SPX_FORS_TREES, 10000));
    INNER_MEASURE("H ..            ", 10000, face_h(10000));
    INNER_MEASURE("F ..            ", 10000, face_f(10000));
    INNER_MEASURE("PRF ..          ", 10000, face_prf(10000));
    INNER_MEASURE("PRF_msg ..      ", 10000, face_prf_msg(10000));
    INNER_MEASURE("H_msg ..        ", 10000, face_h_msg(10000));
}

void test_wots() {
    setbuf(stdout, NULL);

    unsigned char *sk_seed, *pub_seed, *pk1, *ht_pk;
    //  *pk2, *sig, *m;
    uint32_t* addr;

    CHECK(cudaMallocHost(&sk_seed, SPX_N));
    CHECK(cudaMallocHost(&pub_seed, SPX_N));
    CHECK(cudaMallocHost(&pk1, SPX_WOTS_PK_BYTES));
    CHECK(cudaMallocHost(&ht_pk, SPX_D * SPX_WOTS_PK_BYTES));

    CHECK(cudaMallocHost(&addr, 8 * sizeof(uint32_t)));

    randombytes(sk_seed, SPX_N);
    randombytes(pub_seed, SPX_N);
    // randombytes(m, SPX_N);
    randombytes((unsigned char*) addr, 8 * sizeof(uint32_t));

    // 三个函数差不多，因此仅测试pkgen

    // warm up
    face_ap_wots_gen_pk(pk1, sk_seed, pub_seed, addr, WARM_NTESTS);

    printf("\n");

    INNER_MEASURE("WOTS KG..            ", NTESTS,
                  face_wots_gen_pk(pk1, sk_seed, pub_seed, addr, NTESTS));
    // double result = g_result;

    INNER_MEASURE("WOTS AP KG..         ", NTESTS,
                  face_ap_wots_gen_pk(pk1, sk_seed, pub_seed, addr, NTESTS));
    // result = g_result;

    // 多树版本
    INNER_MEASURE("HT WOTS KG..         ", NTESTS,
                  face_ht_wots_gen_pk(ht_pk, sk_seed, pub_seed, addr, NTESTS));
    // result = g_result;

    INNER_MEASURE("HT WOTS KG..         ", NTESTS,
                  face_ap_ht_wots_gen_pk_1(ht_pk, sk_seed, pub_seed, addr, NTESTS));
    // result = g_result;

    INNER_MEASURE("HT WOTS KG..         ", NTESTS,
                  face_ap_ht_wots_gen_pk_12(ht_pk, sk_seed, pub_seed, addr, NTESTS));
    // result = g_result;

    CHECK(cudaFreeHost(sk_seed));
    CHECK(cudaFreeHost(pub_seed));
    CHECK(cudaFreeHost(pk1));
    CHECK(cudaFreeHost(ht_pk));
    CHECK(cudaFreeHost(addr));

    // CHECK(cudaMallocHost(&pk2, SPX_WOTS_PK_BYTES));
    // CHECK(cudaMallocHost(&sig, SPX_WOTS_BYTES));
    // CHECK(cudaMallocHost(&m, SPX_N));

    // INNER_MEASURE("WOTS Signing..       ", NTESTS,
    //               face_ap_wots_sign(sig, m, sk_seed, pub_seed, addr, NTESTS));
    // INNER_MEASURE("WOTS Verifying..     ", NTESTS,
    //               face_ap_wots_pk_from_sig(pk2, sig, m, pub_seed, addr, NTESTS));

    // CHECK(cudaFreeHost(pk2));
    // CHECK(cudaFreeHost(sig));
    // CHECK(cudaFreeHost(m));
}

// coalescented access
void test_wots_cc() {
    setbuf(stdout, NULL);

    unsigned char *sk_seed, *pub_seed, *pk1;
    uint32_t* addr;
    int num = 32;

    CHECK(cudaMallocHost(&sk_seed, num * SPX_N));
    CHECK(cudaMallocHost(&pub_seed, num * SPX_N));
    CHECK(cudaMallocHost(&pk1, num * SPX_WOTS_PK_BYTES));
    CHECK(cudaMallocHost(&addr, num * 8 * sizeof(uint32_t)));

    randombytes(sk_seed, num * SPX_N);
    randombytes(pub_seed, num * SPX_N);
    randombytes((unsigned char*) addr, num * 8 * sizeof(uint32_t));

    printf("Test for face_ap_wots_gen_pk_cc\n");
    face_ap_wots_gen_pk_cc(pk1, sk_seed, pub_seed, addr, NTESTS);

    CHECK(cudaFreeHost(sk_seed));
    CHECK(cudaFreeHost(pub_seed));
    CHECK(cudaFreeHost(pk1));
    CHECK(cudaFreeHost(addr));
}

void test_fors() {
    setbuf(stdout, NULL);

    unsigned char *sk_seed, *pub_seed, *pk1, *pk2, *sig, *m;
    uint32_t* addr;

    CHECK(cudaMallocHost(&sk_seed, SPX_N));
    CHECK(cudaMallocHost(&pub_seed, SPX_N));
    CHECK(cudaMallocHost(&pk1, SPX_FORS_PK_BYTES));

    CHECK(cudaMallocHost(&sig, SPX_FORS_BYTES));
    CHECK(cudaMallocHost(&m, SPX_FORS_MSG_BYTES));
    CHECK(cudaMallocHost(&addr, 8 * sizeof(uint32_t)));

    randombytes(sk_seed, SPX_N);
    randombytes(pub_seed, SPX_N);
    randombytes(m, SPX_FORS_MSG_BYTES);
    randombytes((unsigned char*) addr, 8 * sizeof(uint32_t));

    face_initialize_hash_function(pub_seed, sk_seed);

    // warm up
    face_ap_fors_sign(sig, pk1, m, sk_seed, pub_seed, addr, WARM_NTESTS);

    printf("\n");

    INNER_MEASURE("FORS Signing..            ", NTESTS,
                  face_fors_sign(sig, pk1, m, sk_seed, pub_seed, addr, NTESTS));
    INNER_MEASURE("FORS level 1 Signing..    ", NTESTS,
                  face_ap_fors_sign_1(sig, pk1, m, sk_seed, pub_seed, addr, NTESTS));
    // 并行效率下降原因：
    // 1. 全局同步；
    // 2. 全局变量增加；
    // 3. Merkle树构造过程中并行度下降；
    // 4. T_l函数不可并行；
    // 5. memcpy并行度较低
    INNER_MEASURE("FORS level 1+2 Signing..  ", NTESTS,
                  face_ap_fors_sign_12(sig, pk1, m, sk_seed, pub_seed, addr, NTESTS));
    // double result_sign = g_result;

    CHECK(cudaFreeHost(sk_seed));
    CHECK(cudaFreeHost(pub_seed));
    CHECK(cudaFreeHost(pk1));
    CHECK(cudaFreeHost(sig));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(addr));

    CHECK(cudaMallocHost(&pk2, SPX_FORS_PK_BYTES));
    // INNER_MEASURE("FORS Verifying..     ", NTESTS,
    //               face_ap_fors_pk_from_sig(pk2, sig, m, pub_seed, addr, NTESTS));
    // double result_verify = g_result;
    // CHECK(cudaFreeHost(pk2));
}

void test_xmss() {
    /* Make stdout buffer more responsive. */
    setbuf(stdout, NULL);

    // warm up
    printf("\n");
    face_ap_treehash_wots_23(WARM_NTESTS, 0, 0, 0);

    INNER_MEASURE("Treehash_wots.. ", NTESTS, face_treehash_wots(NTESTS, 0, 0, 0));
    INNER_MEASURE("Treehash_wots.. ", NTESTS, face_ap_treehash_wots_2(NTESTS, 0, 0, 0));
    INNER_MEASURE("Treehash_wots.. ", NTESTS, face_ap_treehash_wots_23(NTESTS, 0, 0, 0));
}

void test_ht() {
    /* Make stdout buffer more responsive. */
    setbuf(stdout, NULL);

    unsigned char *pk, *sk, *m, *sm;
    // , *addr;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES));
    CHECK(cudaMallocHost(&m, SPX_MLEN));
    CHECK(cudaMallocHost(&sm, SPX_BYTES + SPX_MLEN));

    unsigned long long smlen;
    //  mlen;

    randombytes(m, SPX_MLEN);
    randombytes(sk, SPX_SK_BYTES);

    // warm up
    // for (int i = 0; i < WARM_NTESTS; i++) {
    // face_ap_ht_1(sm, &smlen, m, SPX_MLEN, sk, WARM_NTESTS);
    // }

    printf("\n");
    INNER_MEASURE("ht level 1 ..             ", (NTESTS / 20),
                  face_ht(sm, &smlen, m, SPX_MLEN, sk, NTESTS / 20));
    INNER_MEASURE("ht level 1 ..             ", (NTESTS / 4),
                  face_ap_ht_1(sm, &smlen, m, SPX_MLEN, sk, NTESTS / 4));
    INNER_MEASURE("ht level 1+2 ..           ", NTESTS,
                  face_ap_ht_12(sm, &smlen, m, SPX_MLEN, sk, NTESTS));
    INNER_MEASURE("ht level 1+2+3 ..         ", NTESTS,
                  face_ap_ht_123(sm, &smlen, m, SPX_MLEN, sk, NTESTS));

    CHECK(cudaFreeHost(pk));
    CHECK(cudaFreeHost(sk));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(sm));
}