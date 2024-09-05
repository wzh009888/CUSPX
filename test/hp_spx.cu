/*
 * Correctness testing of multi-keypair data parallelism
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../api.h"
#include "../common.h"
#include "../params.h"
#include "../rng.h"

#define SPX_MLEN 32

// multi-keypair is tested only in paper

// 只使用方案2，这个更为灵活
// #define single_stream_multi_key_hybrid_parallelism
// #define single_stream_multi_key_hybrid_parallelism_scheme1
#define single_stream_multi_key_hybrid_parallelism_scheme2
// #define single_stream_multi_key_hybrid_parallelism_seperate
// #define multi_stream_multi_key_hybrid_parallelism
// #define single_stream_single_key_hybrid_parallelism
// #define multi_stream_single_key_hybrid_parallelism

#if defined(single_stream_multi_key_hybrid_parallelism)
#define GPU_KEYGEN face_mhp_crypto_sign_keypair // implemented
#define GPU_SIGN face_mhp_crypto_sign
#define GPU_VERIFY face_mhp_crypto_sign_open
#elif defined(single_stream_multi_key_hybrid_parallelism_scheme1)
#define GPU_KEYGEN face_mhp_crypto_sign_keypair_1
#define GPU_SIGN face_mhp_crypto_sign_1
#define GPU_VERIFY face_mhp_crypto_sign_open
#elif defined(single_stream_multi_key_hybrid_parallelism_scheme2)
#define GPU_KEYGEN face_mhp_crypto_sign_keypair_scheme2
#define GPU_SIGN face_mhp_crypto_sign_scheme2
#define GPU_VERIFY face_mhp_crypto_sign_open
#elif defined(single_stream_multi_key_hybrid_parallelism_seperate)
// #define GPU_KEYGEN face_mhp_sign_keypair_seperate
// #define GPU_SIGN face_mhp_crypto_sign
// #define GPU_VERIFY face_mhp_sign_open_seperate
#elif defined(multi_stream_multi_key_hybrid_parallelism)
// #define GPU_KEYGEN face_ms_mhp_crypto_sign_keypair
// #define GPU_SIGN face_mhp_crypto_sign
// #define GPU_VERIFY face_mhp_sign_open_seperate
#elif defined(single_stream_single_key_hybrid_parallelism)
// #define GPU_KEYGEN face_mdp_crypto_sign_keypair
// #define GPU_SIGN face_mdp_crypto_sign
// #define GPU_VERIFY face_mdp_crypto_sign_open
#elif defined(multi_stream_single_key_hybrid_parallelism)
// #define GPU_KEYGEN face_mdp_crypto_sign_keypair
// #define GPU_SIGN face_mdp_crypto_sign
// #define GPU_VERIFY face_mdp_crypto_sign_open
#endif // if defined(single_stream_multi_key_data_parallelism)

#include "all_option.h"

int main() {
    int ret = 0;
    int wrong = 0;
    u32 num = 512;     // number of tasks 1312 * 8 = 10496
    u32 intra_para = 8; // intra-group parallelism

#if defined(single_stream_multi_key_hybrid_parallelism)
    printf("single-stream multi-keypair hybrid parallelism\n");
#elif defined(multi_stream_multi_key_hybrid_parallelism)
    printf("multi-stream multi-keypair hybrid parallelism\n");
#elif defined(single_stream_single_key_hybrid_parallelism)
    printf("single-stream single-keypair hybrid parallelism\n");
#elif defined(multi_stream_single_key_hybrid_parallelism)
    printf("multi-stream multi-keypair hybrid parallelism\n");
#endif // if defined(single_stream_multi_key_data_parallelism)

    /* Make stdout buffer more responsive. */
    setbuf(stdout, NULL);

    // big num used with link-time optimization, otherwise error

    u8 *pk, *sk, *test_pk, *test_sk;
    u8 *m, *sm, *mout, *test_sm;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES * num));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES * num));
    CHECK(cudaMallocHost(&test_pk, SPX_PK_BYTES));
    CHECK(cudaMallocHost(&test_sk, SPX_SK_BYTES));
    CHECK(cudaMallocHost(&m, SPX_MLEN * num));
    CHECK(cudaMallocHost(&sm, (SPX_BYTES + SPX_MLEN) * num));
    CHECK(cudaMallocHost(&test_sm, (SPX_BYTES + SPX_MLEN) * num));
    CHECK(cudaMallocHost(&mout, (SPX_BYTES + SPX_MLEN) * num));
    unsigned long long smlen;
    unsigned long long mlen;

    randombytes(m, SPX_MLEN * num);
    show_para();
    printf("num = %d\n", num);

    printf("----- Test gpu keypair generation -----\n");

    GPU_KEYGEN(pk, sk, num, intra_para);

#if defined(DEBUG_MODE) && defined(SAME_CHECK)
    if (crypto_sign_keypair(test_pk, test_sk)) {
        printf("failed!\n");
        return -1;
    }

    // same seed is used
    int loc = -1;
    for (int j = 0; j < num; j++) {
        for (int i = 0; i < SPX_PK_BYTES; i++) {
            if (test_pk[i] != pk[i + SPX_PK_BYTES * j]) {
                printf("wrong pk, j = %d\n", j);
                wrong = 1;
                loc = j;
                break;
            }
        }
        for (int i = 0; i < SPX_SK_BYTES; i++) {
            if (test_sk[i] != sk[i + SPX_SK_BYTES * j]) {
                printf("wrong sk, j = %d\n", j);
                wrong = 1;
                loc = j;
                break;
            }
        }
        if (wrong == 1) break;
    }
    if (wrong == 0) {
        printf("    pk & sk are all same.\n");
    } else {
        printf("  X pk & sk are not same.\n");

        for (int i = 0; i < SPX_PK_BYTES; i++) {
            printf("%02x", test_pk[i]);
        }
        printf("\n");

        for (int i = 0; i < SPX_PK_BYTES; i++) {
            printf("%02x", pk[loc * SPX_SK_BYTES + i]);
        }
        printf("\n");
    }

#endif // ifdef SAME_CHECK

    // memcpy(pk, test_pk, SPX_PK_BYTES * num);
    // memcpy(sk, test_sk, SPX_SK_BYTES * num);

    // return 0;
    //--------------------------------------//
    //--------------------------------------//
    //--------------------------------------//
    //--------------------------------------//
    printf("----- Test gpu signature generation -----\n");

    GPU_SIGN(sm, &smlen, m, SPX_MLEN, sk, num, 8);

#if defined(DEBUG_MODE) && defined(SAME_CHECK)
    for (int i = 0; i < num; i++) {
        if (i % 512 == 0) printf("    main loop iter: %d\n", i);
        face_ap_crypto_sign(test_sm + i * SM_BYTES, &smlen, m + i * SPX_MLEN, SPX_MLEN,
                            sk + i * SPX_SK_BYTES);
    }

    // same seed is used
    for (int j = 0; j < num; j++) {
        for (int i = 0; i < SPX_SM_BYTES; i++) {
            if (test_sm[i + SPX_SM_BYTES * j] != sm[i + SPX_SM_BYTES * j]) {
                printf("wrong sm, j = %d, i = %d\n", j, i);
                wrong = 1;
                break;
            }
        }
        if (wrong == 1) break;
    }
    if (wrong == 0)
        printf("    sig are all same !!.\n");
    else
        printf("  X sig are not same !!.\n");
#endif // ifdef SAME_CHECK

    if (smlen != SPX_BYTES + SPX_MLEN) {
        printf("  X smlen incorrect [%llu != %u]!\n", smlen, SPX_BYTES);
        ret = -1;
    } else {
        printf("    smlen as expected [%llu].\n", smlen);
    }
    // memcpy(sm + SM_BYTES, sm, SM_BYTES);

    /* Test if signature is valid. */
    ret = 0;
    wrong = 0;
    for (int i = 0; i < num; i++) {
        if (crypto_sign_open(mout, &mlen, sm + i * SM_BYTES, smlen, pk + i * SPX_PK_BYTES)) {
            printf("  X verification failed!\n");
            printf("i = %d\n", i);
            ret = -1;
            wrong++;
            break;
        }
        /* Test if the correct message was recovered. */
        if (mlen != SPX_MLEN) {
            printf("  X mlen incorrect [%llu != %u]!\n", mlen, SPX_MLEN);
            ret = -1;
            wrong++;
            break;
        }
        if (memcmp(m + i * SPX_MLEN, mout, SPX_MLEN)) {
            printf("  X output message incorrect!\n");
            ret = -1;
            wrong++;
            break;
        }
    }
    if (wrong == 0) {
        printf("    verification succeeded.\n");
        printf("    mlen as expected [%llu].\n", mlen);
        printf("    output message as expected.\n");
    }

    /* Test if signature is valid when validating in-place. */
    ret = 0;
    wrong = 0;
    for (int i = 0; i < num; i++) {
        if (crypto_sign_open(sm + i * SM_BYTES, &mlen, sm + i * SM_BYTES, smlen,
                             pk + i * SPX_PK_BYTES)) {
            printf("  X in-place verification failed!\n");
            ret = -1;
            wrong++;
            break;
        }
    }
    if (wrong == 0) printf("    in-place verification succeeded.\n");

    /* Test if flipping bits invalidates the signature (it should). */

    /* Flip the first bit of the message. Should invalidate. */
    ret = 0;
    wrong = 0;
    for (int i = 0; i < num; i++) {
        sm[smlen - 1] ^= 1;
        if (!crypto_sign_open(mout, &mlen, sm, smlen, pk)) {
            printf("  X flipping a bit of m DID NOT invalidate signature!\n");
            ret = -1;
            wrong++;
            break;
        }
        sm[smlen - 1] ^= 1;
    }
    if (wrong == 0) printf("    flipping a bit of m invalidates signature.\n");

#ifdef SPX_TEST_INVALIDSIG
    int j;
    /* Flip one bit per hash; the signature is entirely hashes. */
    for (j = 0; j < (int) (smlen - SPX_MLEN); j += SPX_N) {
        sm[j] ^= 1;
        if (!crypto_sign_open(mout, &mlen, sm, smlen, pk)) {
            printf("  X flipping bit %d DID NOT invalidate sig + m!\n", j);
            sm[j] ^= 1;
            ret = -1;
            break;
        }
        sm[j] ^= 1;
    }
    if (j >= (int) (smlen - SPX_MLEN)) {
        printf("    changing any signature hash invalidates signature.\n");
    }
#endif // ifdef SPX_TEST_INVALIDSIG

    //--------------------------------------//
    //--------------------------------------//
    //--------------------------------------//
    //--------------------------------------//

    // return 0;

    printf("----- Test gpu verification -----\n");

#ifdef SAME_CHECK

#else
    for (int i = 0; i < num; i++) {
        if (i % 512 == 0) printf("i = %d\n", i);
        face_ap_crypto_sign(test_sm + i * SM_BYTES, &smlen, m + i * SPX_MLEN, SPX_MLEN,
                            sk + i * SPX_SK_BYTES);
    }
#endif

    if (smlen != SPX_BYTES + SPX_MLEN) {
        printf("  X smlen incorrect [%llu != %u]!\n", smlen, SPX_BYTES);
        ret = -1;
    } else {
        printf("    smlen as expected [%llu].\n", smlen);
    }

    // printf("signature finished\n");

    /* Test if signature is valid. */
    // if (crypto_sign_open(mout, &mlen, sm, smlen, pk)) {
    if (GPU_VERIFY(mout, &mlen, test_sm, smlen, pk, num, intra_para)) {
        printf("  X verification failed!\n");
        ret = -1;
    } else {
        printf("    verification succeeded.\n");
    }

    /* Test if the correct message was recovered. */
    if (mlen != SPX_MLEN) {
        printf("  X mlen incorrect [%llu != %u]!\n", mlen, SPX_MLEN);
        ret = -1;
    } else {
        printf("    mlen as expected [%llu].\n", mlen);
    }
    for (int i = 0; i < num; i++) {
        if (memcmp(m + i * SPX_MLEN, mout + i * SPX_MLEN, SPX_MLEN)) {
            printf("  X output message incorrect! %d\n", i);
            ret = -1;
            wrong = 1;
            break;
        }
    }
    if (wrong == 0) {
        printf("    output message as expected.\n");
    }

    /* Test if signature is valid when validating in-place. */
    if (GPU_VERIFY(test_sm, &mlen, test_sm, smlen, pk, num, intra_para)) {
        printf("  X in-place verification failed!\n");
        ret = -1;
    } else {
        printf("    in-place verification succeeded.\n");
    }

    /* Test if flipping bits invalidates the signature (it should). */

    /* Flip the first bit of the message. Should invalidate. */
    for (int i = 0; i < num; i++)
        test_sm[i * SM_BYTES + smlen - 1] ^= 1;
    if (!GPU_VERIFY(mout, &mlen, test_sm, smlen, pk, num, intra_para)) {
        printf("  X flipping a bit of m DID NOT invalidate signature!\n");
        ret = -1;
    } else {
        printf("    flipping a bit of m invalidates signature.\n");
    }
    for (int i = 0; i < num; i++)
        test_sm[i * SM_BYTES + smlen - 1] ^= 1;

#ifdef SPX_TEST_INVALIDSIG
    /* Flip one bit per hash; the signature is entirely hashes. */
    for (int j = 0; j < (int) (smlen - SPX_MLEN); j += SPX_N) {
        test_sm[j] ^= 1;
        if (!GPU_VERIFY(mout, &mlen, test_sm, smlen, pk, num)) {
            printf("  X flipping bit %d DID NOT invalidate sig + m!\n", j);
            test_sm[j] ^= 1;
            ret = -1;
            break;
        }
        test_sm[j] ^= 1;
    }
    if (j >= (int) (smlen - SPX_MLEN)) {
        printf("    changing any signature hash invalidates signature.\n");
    }
#endif // ifdef SPX_TEST_INVALIDSIG

    CHECK(cudaFreeHost(pk));
    CHECK(cudaFreeHost(sk));
    CHECK(cudaFreeHost(test_pk));
    CHECK(cudaFreeHost(test_sk));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(sm));
    CHECK(cudaFreeHost(test_sm));
    CHECK(cudaFreeHost(mout));

    return ret;
} // main
