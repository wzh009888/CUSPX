#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../api.h"
#include "../hash.h"
#include "../rng.h"
#include "../wots.h"
#include "all_option.h"

#include "../common.h"

int main() {
    /* Make stdout buffer more responsive. */
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

    printf("Testing WOTS signature and PK derivation.. \n");

    initialize_hash_function(pub_seed, sk_seed);

    face_initialize_hash_function(pub_seed, sk_seed);

    wots_gen_pk(pk1, sk_seed, pub_seed, addr);
    wots_sign(sig, m, sk_seed, pub_seed, addr);
    wots_pk_from_sig(pk2, sig, m, pub_seed, addr);

    if (memcmp(pk1, pk2, SPX_WOTS_PK_BYTES)) {
        printf("CPU code test failed!\n");
        // return -1;
    } else {
        printf("CPU code test succeeded.\n");
    }

    face_wots_gen_pk(pk1, sk_seed, pub_seed, addr, 1);
    wots_sign(sig, m, sk_seed, pub_seed, addr);
    wots_pk_from_sig(pk2, sig, m, pub_seed, addr);

    if (memcmp(pk1, pk2, SPX_WOTS_PK_BYTES)) {
        printf("face_wots_gen_pk failed!\n");
        return -1;
    } else {
        printf("face_wots_gen_pk test succeeded.\n");
    }

    wots_gen_pk(pk1, sk_seed, pub_seed, addr);
    face_wots_sign(sig, m, sk_seed, pub_seed, addr, 1);
    face_wots_pk_from_sig(pk2, sig, m, pub_seed, addr, 1);

    if (memcmp(pk1, pk2, SPX_WOTS_PK_BYTES)) {
        printf("face_wots_sign and face_wots_pk_from_sig failed!\n");
        // return -1;
    } else {
        printf("face_wots_sign and face_wots_pk_from_sig succeeded.\n");
    }
    //
    face_wots_gen_pk(pk1, sk_seed, pub_seed, addr, 1);
    face_wots_sign(sig, m, sk_seed, pub_seed, addr, 1);
    face_wots_pk_from_sig(pk2, sig, m, pub_seed, addr, 1);

    if (memcmp(pk1, pk2, SPX_WOTS_PK_BYTES)) {
        printf("GPU test failed!\n");
        // return -1;
    } else {
        printf("GPU code test succeeded.\n");
    }

    CHECK(cudaFreeHost(sk_seed));
    CHECK(cudaFreeHost(pub_seed));
    CHECK(cudaFreeHost(pk1));
    CHECK(cudaFreeHost(pk2));
    CHECK(cudaFreeHost(sig));
    CHECK(cudaFreeHost(m));
    CHECK(cudaFreeHost(addr));

    return 0;
} // main
