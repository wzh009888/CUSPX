#include <iostream>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
using namespace std;

#include "all_option.h"

#include "address.h"
#include "api.h"
#include "fors.h"
#include "hash.h"
#include "params.h"
#include "rng.h"
#include "thash.h"
#include "utils.h"
#include "wots.h"

#include <cooperative_groups.h>

__device__ u8 one_root[SPX_N];
__device__ u8 dev_ap_root[SPX_N * SPX_D + SPX_N]; // +N为了避免内存溢出
__device__ u8 one_wots_pk[SPX_WOTS_BYTES];

// provided the max number of tasks is 16384
#define MAX_TASKS_HP (8192)

__device__ u8 hp_roots[MAX_TASKS_HP * SPX_FORS_TREES * SPX_N];
__device__ u8 hp_worktid[MAX_TASKS_HP * SPX_WOTS_LEN];

// provided the max parallelism is 128
__device__ u8 hp_xmss_roots[MAX_TASKS_HP * SPX_N * 128];
__device__ u8 hp_auth_path[MAX_TASKS_HP * SPX_N * SPX_TREE_HEIGHT];
__device__ u32 hp_indices[MAX_TASKS_HP * SPX_FORS_TREES];
__device__ u32 hp_length[MAX_TASKS_HP * SPX_WOTS_LEN];
__device__ u32 hp_fors_tree_addr[MAX_TASKS_HP * 8];

__device__ u8 dev_leaf[SPX_N * 1024 * 1024 * 44]; // 存放xmss fors的叶节点
__device__ u8 dev_ap[SPX_N * 20 * 22];            // 最大长度20，最大22层
__device__ u8 dev_wpk[512 * 512 * SPX_WOTS_BYTES];

__device__ u8 dev_mhash[8192 * SPX_FORS_MSG_BYTES]; // 最大16384个任务
__device__ u8 dev_root[8192 * 22 * SPX_N];          // 最大16384个任务，最大要存22层
__device__ uint64_t dev_o_tree[8192];               // 最大16384个任务
__device__ uint32_t dev_o_idx_leaf[8192];           // 最大16384个任务
__device__ uint64_t dev_tree[8192];
__device__ uint32_t dev_idx_leaf[8192];

inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[]
        = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128},
           {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},
           {0x75, 64},  {0x80, 64},  {0x86, 128}, {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
} // _ConvertSMVer2Cores

/**
 * Computes the leaf at a given address. First generates the WOTS key pair,
 * then computes leaf by hashing horizontally.
 */
static void wots_gen_leaf(u8* leaf, const u8* sk_seed, const u8* pub_seed, uint32_t addr_idx,
                          const uint32_t tree_addr[8]) {
    u8 pk[SPX_WOTS_BYTES];
    uint32_t wots_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);

    copy_subtree_addr(wots_addr, tree_addr);
    set_keypair_addr(wots_addr, addr_idx);
    wots_gen_pk(pk, sk_seed, pub_seed, wots_addr);

    copy_keypair_addr(wots_pk_addr, wots_addr);
    thash(leaf, pk, SPX_WOTS_LEN, pub_seed, wots_pk_addr);
} // wots_gen_leaf

__device__ void dev_wots_gen_leaf(u8* leaf, const u8* sk_seed, const u8* pub_seed,
                                  uint32_t addr_idx, const uint32_t tree_addr[8]) {
    u8 pk[SPX_WOTS_BYTES];
    uint32_t wots_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);

    dev_copy_subtree_addr(wots_addr, tree_addr);
    dev_set_keypair_addr(wots_addr, addr_idx);
    dev_wots_gen_pk(pk, sk_seed, pub_seed, wots_addr);

    dev_copy_keypair_addr(wots_pk_addr, wots_addr);
    dev_thash(leaf, pk, SPX_WOTS_LEN, pub_seed, wots_pk_addr);
}

/*
 * Returns the length of a secret key, in bytes
 */
u64 crypto_sign_secretkeybytes(void) {
    return CRYPTO_SECRETKEYBYTES;
} // crypto_sign_secretkeybytes

/*
 * Returns the length of a public key, in bytes
 */
u64 crypto_sign_publickeybytes(void) {
    return CRYPTO_PUBLICKEYBYTES;
} // crypto_sign_publickeybytes

/*
 * Returns the length of a signature, in bytes
 */
u64 crypto_sign_bytes(void) {
    return CRYPTO_BYTES;
} // crypto_sign_bytes

/*
 * Returns the length of the seed required to generate a key pair, in bytes
 */
u64 crypto_sign_seedbytes(void) {
    return CRYPTO_SEEDBYTES;
} // crypto_sign_seedbytes

/*
 * Generates an SPX key pair given a seed of length
 * Format sk: [SK_SEED || SK_PRF || PUB_SEED || root]
 * Format pk: [PUB_SEED || root]
 */
int crypto_sign_seed_keypair(u8* pk, u8* sk, const u8* seed) {
    /* We do not need the auth path in key generation, but it simplifies the
       code to have just one treehash routine that computes both root and path
       in one function. */
    u8 auth_path[SPX_TREE_HEIGHT * SPX_N];
    uint32_t top_tree_addr[8] = {0};

    memset(top_tree_addr, 0, 8 * sizeof(uint32_t));

    set_layer_addr(top_tree_addr, SPX_D - 1);
    set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Initialize SK_SEED, SK_PRF and PUB_SEED from seed. */
    memcpy(sk, seed, CRYPTO_SEEDBYTES);

    memcpy(pk, sk + 2 * SPX_N, SPX_N);

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    initialize_hash_function(pk, sk);

    /* Compute root node of the top-most subtree. */
    treehash(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT, wots_gen_leaf,
             top_tree_addr);

    memcpy(pk + SPX_N, sk + 3 * SPX_N, SPX_N);

    return 0;
} // crypto_sign_seed_keypair

__device__ int dev_crypto_sign_seed_keypair(u8* pk, u8* sk, const u8* seed) {
    /* We do not need the auth path in key generation, but it simplifies the
       code to have just one treehash routine that computes both root and path
       in one function. */
    u8 auth_path[SPX_TREE_HEIGHT * SPX_N];
    uint32_t top_tree_addr[8] = {0};

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Initialize SK_SEED, SK_PRF and PUB_SEED from seed. */
    memcpy(sk, seed, CRYPTO_SEEDBYTES);

    memcpy(pk, sk + 2 * SPX_N, SPX_N);

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    dev_initialize_hash_function(pk, sk);

    /* Compute root node of the top-most subtree. */
    dev_treehash(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT,
                 dev_wots_gen_leaf, top_tree_addr);

    memcpy(pk + SPX_N, sk + 3 * SPX_N, SPX_N);

    return 0;
}

__device__ int dev_ap_crypto_sign_seed_keypair(u8* pk, u8* sk, const u8* seed) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    /* We do not need the auth path in key generation, but it simplifies the
       code to have just one treehash routine that computes both root and path
       in one function. */
    u8 auth_path[SPX_TREE_HEIGHT * SPX_N];
    uint32_t top_tree_addr[8] = {0};

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Initialize SK_SEED, SK_PRF and PUB_SEED from seed. */
    if (tid == 0) memcpy(sk, seed, CRYPTO_SEEDBYTES);

    if (tid == 0) memcpy(pk, sk + 2 * SPX_N, SPX_N);

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pk, sk);
    g.sync();

    /* Compute root node of the top-most subtree. */
#ifdef USING_PARALLEL_PKGEN_TREEHASH_XMSS
    dev_ap_treehash_wots(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT,
                         dev_wots_gen_leaf, top_tree_addr);
#else  // ifdef USING_PARALLEL_PKGEN_TREEHASH_XMSS
    if (tid == 0)
        dev_treehash(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT,
                     dev_wots_gen_leaf, top_tree_addr);
#endif // ifdef USING_PARALLEL_PKGEN_TREEHASH_XMSS

    if (tid == 0) memcpy(pk + SPX_N, sk + 3 * SPX_N, SPX_N);

    return 0;
}

__device__ int dev_ap_crypto_sign_seed_keypair_2(u8* pk, u8* sk, const u8* seed) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    /* We do not need the auth path in key generation, but it simplifies the
       code to have just one treehash routine that computes both root and path
       in one function. */
    u8 auth_path[SPX_TREE_HEIGHT * SPX_N];
    uint32_t top_tree_addr[8] = {0};

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Initialize SK_SEED, SK_PRF and PUB_SEED from seed. */
    if (tid == 0) memcpy(sk, seed, CRYPTO_SEEDBYTES);

    if (tid == 0) memcpy(pk, sk + 2 * SPX_N, SPX_N);

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pk, sk);
    g.sync();

    /* Compute root node of the top-most subtree. */
    dev_ap_treehash_wots_2(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT,
                           dev_wots_gen_leaf, top_tree_addr);

    if (tid == 0) memcpy(pk + SPX_N, sk + 3 * SPX_N, SPX_N);

    return 0;
}

__device__ int dev_ap_crypto_sign_seed_keypair_23(u8* pk, u8* sk, const u8* seed) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    /* We do not need the auth path in key generation, but it simplifies the
       code to have just one treehash routine that computes both root and path
       in one function. */
    u8 auth_path[SPX_TREE_HEIGHT * SPX_N];
    uint32_t top_tree_addr[8] = {0};

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Initialize SK_SEED, SK_PRF and PUB_SEED from seed. */
    if (tid == 0) memcpy(sk, seed, CRYPTO_SEEDBYTES);

    if (tid == 0) memcpy(pk, sk + 2 * SPX_N, SPX_N);

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pk, sk);
    g.sync();

    /* Compute root node of the top-most subtree. */
    dev_ap_treehash_wots_23(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT,
                            dev_wots_gen_leaf, top_tree_addr);

    if (tid == 0) memcpy(pk + SPX_N, sk + 3 * SPX_N, SPX_N);

    return 0;
}

__device__ int dev_seed_treehash_wots(u8* pk, u8* sk, const u8* seed) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    u8 auth_path[SPX_TREE_HEIGHT * SPX_N];
    uint32_t top_tree_addr[8] = {0};

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Initialize SK_SEED, SK_PRF and PUB_SEED from seed. */
    if (tid == 0) memcpy(sk, seed, CRYPTO_SEEDBYTES);
    if (tid == 0) memcpy(pk, sk + 2 * SPX_N, SPX_N);

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pk, sk);
    g.sync();

    dev_treehash(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT,
                 dev_wots_gen_leaf, top_tree_addr);

    if (tid == 0) memcpy(pk + SPX_N, sk + 3 * SPX_N, SPX_N);

    return 0;
} // dev_crypto_sign_seed_keypair

/*
 * Generates an SPX key pair.
 * Format sk: [SK_SEED || SK_PRF || PUB_SEED || root]
 * Format pk: [PUB_SEED || root]
 */
int crypto_sign_keypair(u8* pk, u8* sk) {
    u8 seed[CRYPTO_SEEDBYTES];

    randombytes(seed, CRYPTO_SEEDBYTES);
#ifdef DEBUG_MODE
    for (int i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;
#endif // ifdef DEBUG_MODE
    crypto_sign_seed_keypair(pk, sk, seed);

    return 0;
} // crypto_sign_keypair

__device__ int dev_ap_seed_treehash_wots_2(u8* pk, u8* sk, const u8* seed) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    u8 auth_path[SPX_TREE_HEIGHT * SPX_N];
    uint32_t top_tree_addr[8] = {0};

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Initialize SK_SEED, SK_PRF and PUB_SEED from seed. */
    if (tid == 0) memcpy(sk, seed, CRYPTO_SEEDBYTES);
    if (tid == 0) memcpy(pk, sk + 2 * SPX_N, SPX_N);

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pk, sk);
    g.sync();

    dev_ap_treehash_wots_2(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT,
                           dev_wots_gen_leaf, top_tree_addr);

    if (tid == 0) memcpy(pk + SPX_N, sk + 3 * SPX_N, SPX_N);

    return 0;
}

__device__ int dev_ap_seed_treehash_wots_23(u8* pk, u8* sk, const u8* seed) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    u8 auth_path[SPX_TREE_HEIGHT * SPX_N];
    uint32_t top_tree_addr[8] = {0};

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Initialize SK_SEED, SK_PRF and PUB_SEED from seed. */
    if (tid == 0) memcpy(sk, seed, CRYPTO_SEEDBYTES);
    if (tid == 0) memcpy(pk, sk + 2 * SPX_N, SPX_N);

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pk, sk);
    g.sync();

    dev_ap_treehash_wots_23(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT,
                            dev_wots_gen_leaf, top_tree_addr);

    if (tid == 0) memcpy(pk + SPX_N, sk + 3 * SPX_N, SPX_N);

    return 0;
}

/*
 * Generates an SPX key pair.
 * Format sk: [SK_SEED || SK_PRF || PUB_SEED || root]
 * Format pk: [PUB_SEED || root]
 */
// int crypto_sign_keypair(u8* pk, u8* sk) {
//     u8 seed[CRYPTO_SEEDBYTES];

//     randombytes(seed, CRYPTO_SEEDBYTES);
// #ifdef DEBUG_MODE
//     for (int i = 0; i < CRYPTO_SEEDBYTES; i++)
//         seed[i] = i;
// #endif // ifdef DEBUG_MODE
//     crypto_sign_seed_keypair(pk, sk, seed);

//     return 0;
// } // crypto_sign_keypair

__device__ int dev_crypto_sign_keypair(u8* pk, u8* sk) {
    u8 seed[CRYPTO_SEEDBYTES];

#ifdef DEBUG_MODE
    for (u32 i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;
#else
    dev_randombytes(seed, CRYPTO_SEEDBYTES);
#endif // ifdef DEBUG_MODE
    dev_crypto_sign_seed_keypair(pk, sk, seed);

    return 0;
} // dev_crypto_sign_keypair

__global__ void global_crypto_sign_keypair(u8* pk, u8* sk) {
    dev_crypto_sign_keypair(pk, sk);

} // global_crypto_sign_keypair

__device__ int dev_ap_crypto_sign_keypair(u8* pk, u8* sk) {
    u8 seed[CRYPTO_SEEDBYTES];

#ifdef DEBUG_MODE
    for (int i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;
#else
    dev_randombytes(seed, CRYPTO_SEEDBYTES);
#endif // ifdef DEBUG_MODE

    dev_ap_crypto_sign_seed_keypair(pk, sk, seed);

    return 0;
} // dev_crypto_sign_keypair

__device__ int dev_ap_crypto_sign_keypair_2(u8* pk, u8* sk) {
    u8 seed[CRYPTO_SEEDBYTES];

#ifdef DEBUG_MODE
    for (int i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;
#else
    dev_randombytes(seed, CRYPTO_SEEDBYTES);
#endif // ifdef DEBUG_MODE

    dev_ap_crypto_sign_seed_keypair_2(pk, sk, seed);

    return 0;
}

__device__ int dev_ap_crypto_sign_keypair_23(u8* pk, u8* sk) {
    u8 seed[CRYPTO_SEEDBYTES];

#ifdef DEBUG_MODE
    for (int i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;
#else
    dev_randombytes(seed, CRYPTO_SEEDBYTES);
#endif // ifdef DEBUG_MODE

    dev_ap_crypto_sign_seed_keypair_23(pk, sk, seed);

    return 0;
}

__device__ int dev_treehash_wots(u8* pk, u8* sk) {
    u8 seed[CRYPTO_SEEDBYTES];

    for (int i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;

    dev_seed_treehash_wots(pk, sk, seed);

    return 0;
}

__device__ int dev_ap_treehash_wots_2(u8* pk, u8* sk) {
    u8 seed[CRYPTO_SEEDBYTES];

    for (int i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;

    dev_ap_seed_treehash_wots_2(pk, sk, seed);

    return 0;
}

__device__ int dev_ap_treehash_wots_23(u8* pk, u8* sk) {
    u8 seed[CRYPTO_SEEDBYTES];

    for (int i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;

    dev_ap_seed_treehash_wots_23(pk, sk, seed);

    return 0;
}

__global__ void global_ap_crypto_sign_keypair(u8* keypair) {
    u8* pk = keypair;
    u8* sk = keypair + SPX_PK_BYTES;
    dev_ap_crypto_sign_keypair(pk, sk);
}

__global__ void global_ap_crypto_sign_keypair_2(u8* keypair) {
    u8* pk = keypair;
    u8* sk = keypair + SPX_PK_BYTES;
    dev_ap_crypto_sign_keypair_2(pk, sk);
}

__global__ void global_ap_crypto_sign_keypair_23(u8* keypair) {
    u8* pk = keypair;
    u8* sk = keypair + SPX_PK_BYTES;
    dev_ap_crypto_sign_keypair_23(pk, sk);
}

__global__ void global_treehash_wots(u8* keypair, uint32_t loop_num) {
    u8* pk = keypair;
    u8* sk = keypair + SPX_PK_BYTES;
    for (int i = 0; i < loop_num; i++)
        dev_treehash_wots(pk, sk);
} // global_ap_crypto_sign_keypair

__global__ void global_ap_treehash_wots_2(u8* keypair, uint32_t loop_num) {
    u8* pk = keypair;
    u8* sk = keypair + SPX_PK_BYTES;
    for (int i = 0; i < loop_num; i++)
        dev_ap_treehash_wots_2(pk, sk);
} // global_ap_crypto_sign_keypair

__global__ void global_ap_treehash_wots_23(u8* keypair, uint32_t loop_num) {
    u8* pk = keypair;
    u8* sk = keypair + SPX_PK_BYTES;
    for (int i = 0; i < loop_num; i++)
        dev_ap_treehash_wots_23(pk, sk);
} // global_ap_crypto_sign_keypair

__global__ void global_dp_crypto_sign_keypair(u8* pk, u8* sk, u32 dp_num) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // printf("tid = %d\n", tid);
    if (tid < dp_num) dev_crypto_sign_keypair(pk + tid * SPX_PK_BYTES, sk + tid * SPX_SK_BYTES);

} // global_dp_crypto_sign_keypair

// scheme 1: Subtree-First
__global__ void global_mhp_crypto_sign_keypair_1(u8* pk, u8* sk, u32 dp_num, u32 intra_para) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    // u32 leaf_num = (1 << SPX_TREE_HEIGHT);
    u32 para = tnum / dp_num; // 组内并行度
    if (intra_para > 0) para = intra_para;

#ifdef PRINT_ALL
    if (tid == 0)
        printf("tnum = %d, para = %d, The leaf_num is %d, dp_num = %d\n", tnum, para, leaf_num,
               dp_num);
#endif
    u8 seed[CRYPTO_SEEDBYTES];

    uint32_t top_tree_addr[8] = {0};

#ifdef DEBUG_MODE
    for (u32 i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;
#else
    dev_randombytes(seed, CRYPTO_SEEDBYTES);
#endif // ifdef DEBUG_MODE

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    // 子树方案，在组内并行度小于叶节点数量时使用
    u32 id = tid % para;
    u32 ttid = tid / para;
    u8* t_pk = pk + ttid * SPX_PK_BYTES;
    u8* t_sk = sk + ttid * SPX_SK_BYTES;
    memcpy(t_sk, seed, CRYPTO_SEEDBYTES);
    memcpy(t_pk, t_sk + 2 * SPX_N, SPX_N);
    dev_initialize_hash_function(t_pk, t_sk);

    u8* sk_seed = t_sk;
    u8* pub_seed = t_sk + 2 * SPX_N;

    u32 tree_height = SPX_TREE_HEIGHT;
    u32 stleaf = (1 << SPX_TREE_HEIGHT) / para;
    u32 stNum = para;                                      // 子树的数量等于并行度
    u32 stheight = SPX_TREE_HEIGHT - log(para) / log(2.0); // 子树的树高
    u8* roots = hp_xmss_roots + ttid * stNum * SPX_N;

    if (tid < dp_num * para) {
        unsigned char stack[(SPX_TREE_HEIGHT + 1) * SPX_N];
        unsigned int heights[SPX_TREE_HEIGHT + 1];
        unsigned int offset = 0;

        for (u32 i = id * stleaf; i < (id + 1) * stleaf; i++) {
            dev_wots_gen_leaf(stack + offset * SPX_N, sk_seed, pub_seed, i, top_tree_addr);
            offset++;
            heights[offset - 1] = 0;

            /* While the top-most nodes are of equal height.. */
            while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
                u32 tree_idx = (i >> (heights[offset - 1] + 1));
                dev_set_tree_height(top_tree_addr, heights[offset - 1] + 1);
                dev_set_tree_index(top_tree_addr, tree_idx);
                dev_thash(stack + (offset - 2) * SPX_N, stack + (offset - 2) * SPX_N, 2, pub_seed,
                          top_tree_addr);

                offset--;
                heights[offset - 1]++;
            }
        }
        memcpy(roots + id * SPX_N, stack, SPX_N);
    }

    for (int i = 1, ii = 1; i <= tree_height - stheight; i++) {
        __syncthreads();
        dev_set_tree_height(top_tree_addr, i + stheight);
        if (id < para) {
            for (int j = id; j < (stNum >> i); j += para) {
                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(top_tree_addr, j);
                memcpy(roots + off + SPX_N, roots + off + ii * SPX_N, SPX_N);
                dev_thash(roots + off, roots + off, 2, pub_seed, top_tree_addr);
            }
        }
        ii *= 2;
    }

    __syncthreads();

    memcpy(t_sk + 3 * SPX_N, roots, SPX_N);
    memcpy(t_pk + SPX_N, t_sk + 3 * SPX_N, SPX_N);
}

__global__ void global_mhp_crypto_sign_keypair_scheme2(u8* pk, u8* sk, u32 dp_num) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    u32 leaf_num = (1 << SPX_TREE_HEIGHT);
    u8 seed[CRYPTO_SEEDBYTES];
    uint32_t top_tree_addr[8] = {0};

#ifdef DEBUG_MODE
    for (u32 i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;
#else
    dev_randombytes(seed, CRYPTO_SEEDBYTES);
#endif // ifdef DEBUG_MODE

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    // 分步骤方案，先构建叶节点，然后构建分支节点

    if (tid < dp_num) {
        u8* _pk = pk + tid * SPX_PK_BYTES;
        u8* _sk = sk + tid * SPX_SK_BYTES;
        memcpy(_sk, seed, CRYPTO_SEEDBYTES);
        memcpy(_pk, _sk + 2 * SPX_N, SPX_N);
        dev_initialize_hash_function(_pk, _sk);
    }

    uint32_t wots_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);
    dev_copy_subtree_addr(wots_addr, top_tree_addr);

    // 使用混合处理任务，需要读取其他线程的sk_seed和pub_seed
    // 一个叶子的wots是连着的，一个任务的多个叶子是连着的，多个任务连续存放，分配给tnum个线程
    g.sync();
    int wots_per_task = SPX_WOTS_LEN * leaf_num;
    for (int i = tid; i < wots_per_task * dp_num; i += tnum) {               // 35 * 4 * 8
        dev_set_keypair_addr(wots_addr, (i % wots_per_task) / SPX_WOTS_LEN); // 标记叶子位置
        dev_set_chain_addr(wots_addr, i % SPX_WOTS_LEN);                     // 标记wots位置
        dev_set_hash_addr(wots_addr, 0);
        u8* sk_seed = sk + (i / wots_per_task) * SPX_SK_BYTES;
        u8* pub_seed = sk_seed + 2 * SPX_N;
        dev_prf_addr(dev_wpk + i * SPX_N, sk_seed, wots_addr);
        dev_gen_chain(dev_wpk + i * SPX_N, dev_wpk + i * SPX_N, 0, SPX_WOTS_W - 1, pub_seed,
                      wots_addr);
    }

    g.sync();
    for (int i = tid; i < leaf_num * dp_num; i += tnum) { // 32
        u8* sk_seed = sk + (i / leaf_num) * SPX_SK_BYTES;
        u8* pub_seed = sk_seed + 2 * SPX_N;
        dev_copy_keypair_addr(wots_pk_addr, wots_addr);
        dev_set_keypair_addr(wots_pk_addr, i % leaf_num); // 标记叶子位置
        dev_thash(dev_leaf + i * SPX_N, dev_wpk + i * SPX_WOTS_BYTES, SPX_WOTS_LEN, pub_seed,
                  wots_pk_addr);
    }

    for (int i = 1, ii = 1; i <= SPX_TREE_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(top_tree_addr, i);
        int li = (leaf_num >> i);
        for (int j = tid; j < li * dp_num; j += tnum) {
            u32 tt = j / li; // which task
            u32 ll = j % li; // which leaf
            u8* pub_seed = pk + tt * SPX_SK_BYTES + 2 * SPX_N;

            int off = 2 * j * ii * SPX_N;
            dev_set_tree_index(top_tree_addr, ll);
            memcpy(dev_leaf + off + SPX_N, dev_leaf + off + ii * SPX_N, SPX_N);
            dev_thash(dev_leaf + off, dev_leaf + off, 2, pub_seed, top_tree_addr);
        }
        ii *= 2;
    }

    // g.sync();

    if (tid < dp_num) {
        memcpy(sk + tid * SPX_SK_BYTES + 3 * SPX_N, dev_leaf + tid * leaf_num * SPX_N, SPX_N);
        memcpy(pk + tid * SPX_PK_BYTES + SPX_N, sk + tid * SPX_SK_BYTES + 3 * SPX_N, SPX_N);
    }
}

__global__ void global_mhp_sign_keypair_seperate(u8* all_pk, u8* all_sk, u32 dp_num) {
    u8 seed[CRYPTO_SEEDBYTES];

#ifdef DEBUG_MODE
    for (int i = 0; i < CRYPTO_SEEDBYTES; i++)
        seed[i] = i;
#else
    dev_randombytes(seed, CRYPTO_SEEDBYTES);
#endif // ifdef DEBUG_MODE

    // dev_ap_crypto_sign_seed_keypair(pk, sk, seed);
    u8* pk = all_pk + blockIdx.x * SPX_PK_BYTES;
    u8* sk = all_sk + blockIdx.x * SPX_SK_BYTES;

    // cooperative_groups::grid_group g = cooperative_groups::this_grid();
    // const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    /* We do not need the auth path in key generation, but it simplifies the
       code to have just one treehash routine that computes both root and path
       in one function. */
    u8 auth_path[SPX_TREE_HEIGHT * SPX_N];
    uint32_t top_tree_addr[8] = {0};

    dev_set_layer_addr(top_tree_addr, SPX_D - 1);
    dev_set_type(top_tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Initialize SK_SEED, SK_PRF and PUB_SEED from seed. */
    if (threadIdx.x == 0) memcpy(sk, seed, CRYPTO_SEEDBYTES);

    if (threadIdx.x == 0) memcpy(pk, sk + 2 * SPX_N, SPX_N);

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (threadIdx.x == 0) dev_initialize_hash_function(pk, sk);
    // g.sync();
    __syncthreads();

    /* Compute root node of the top-most subtree. */
    // #ifdef USING_PARALLEL_PKGEN_TREEHASH_XMSS
    dev_ap_treehash_wots_shared(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0,
                                SPX_TREE_HEIGHT, dev_wots_gen_leaf, top_tree_addr);
    // #else  // ifdef USING_PARALLEL_PKGEN_TREEHASH_XMSS
    // if (threadIdx.x == 0)
    //     dev_treehash(sk + 3 * SPX_N, auth_path, sk, sk + 2 * SPX_N, 0, 0, SPX_TREE_HEIGHT,
    //                  dev_wots_gen_leaf, top_tree_addr);
    // #endif // ifdef USING_PARALLEL_PKGEN_TREEHASH_XMSS

    if (threadIdx.x == 0) memcpy(pk + SPX_N, sk + 3 * SPX_N, SPX_N);

    // return 0;
}

int face_crypto_sign_keypair(u8* pk, u8* sk) {
    u8 *dev_pk = NULL, *dev_sk = NULL;
    int device = DEVICE_USED;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));

    void* kernelArgs[] = {&dev_pk, &dev_sk};

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_crypto_sign_keypair, 1, 1, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(pk, dev_pk, SPX_PK_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(sk, dev_sk, SPX_SK_BYTES * sizeof(u8), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_pk);
    cudaFree(dev_sk);

    return 0;
} // face_crypto_sign_keypair

int face_ap_crypto_sign_keypair(u8* pk, u8* sk) {
    u8* dev_keypair = NULL;
    u8 keypair[SPX_SK_BYTES + SPX_PK_BYTES];
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    u32 threads = 32;
    // only treehash_wots need to be parallelized
    // max parallelism (1 << h) * len
    u32 blocks = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN / threads + 1;
    // blocks /= 2;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

#ifdef KEYGEN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef KEYGEN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_keypair,
                                                  threads, 0);
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef KEYGEN_SUITBLE_BLOCK

#ifdef KEYGEN_SUITBLE_BLOCK
    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

    if (threads * blocks > maxallthreads) printf("threads * blocks > maxallthreads\n");
#endif // ifdef KEYGEN_SUITBLE_BLOCK

    CHECK(cudaMalloc((void**) &dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES));

    struct timespec start, stop;

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;

    void* kernelArgs[] = {&dev_keypair};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_crypto_sign_keypair, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(keypair, dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES, D2H));
    memcpy(pk, keypair, SPX_PK_BYTES);
    memcpy(sk, &keypair[SPX_PK_BYTES], SPX_SK_BYTES);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_keypair);

    return 0;
}

int face_ap_crypto_sign_keypair_2(u8* pk, u8* sk) {
    u8* dev_keypair = NULL;
    u8 keypair[SPX_SK_BYTES + SPX_PK_BYTES];
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    u32 threads = 32;
    // only treehash_wots need to be parallelized
    // max parallelism (1 << h) * len
    u32 blocks = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN / threads + 1;
    // blocks /= 2;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

#ifdef KEYGEN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef KEYGEN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_keypair_2,
                                                  threads, 0);
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef KEYGEN_SUITBLE_BLOCK

#ifdef KEYGEN_SUITBLE_BLOCK
    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

    if (threads * blocks > maxallthreads) printf("threads * blocks > maxallthreads\n");
#endif // ifdef KEYGEN_SUITBLE_BLOCK

    CHECK(cudaMalloc((void**) &dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES));

    struct timespec start, stop;

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;

    void* kernelArgs[] = {&dev_keypair};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_crypto_sign_keypair_2, blocks, threads,
                                kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(keypair, dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES, D2H));
    memcpy(pk, keypair, SPX_PK_BYTES);
    memcpy(sk, &keypair[SPX_PK_BYTES], SPX_SK_BYTES);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_keypair);

    return 0;
}

int face_ap_crypto_sign_keypair_23(u8* pk, u8* sk) {
    u8* dev_keypair = NULL;
    u8 keypair[SPX_SK_BYTES + SPX_PK_BYTES];
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    u32 threads = 32;
    // only treehash_wots need to be parallelized
    // max parallelism (1 << h) * len
    u32 blocks = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN / threads + 1;
    // blocks /= 2;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

#ifdef KEYGEN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef KEYGEN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_keypair_23,
                                                  threads, 0);
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef KEYGEN_SUITBLE_BLOCK

#ifdef KEYGEN_SUITBLE_BLOCK
    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

    if (threads * blocks > maxallthreads) printf("threads * blocks > maxallthreads\n");

        // #if  defined(SPX_192S) || defined(SPX_256S)
        //     blocks = maxallthreads * 2 / threads;
        // #endif

#endif // ifdef KEYGEN_SUITBLE_BLOCK

    CHECK(cudaMalloc((void**) &dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES));

    struct timespec start, stop;

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;

    void* kernelArgs[] = {&dev_keypair};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_crypto_sign_keypair_23, blocks, threads,
                                kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(keypair, dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES, D2H));
    memcpy(pk, keypair, SPX_PK_BYTES);
    memcpy(sk, &keypair[SPX_PK_BYTES], SPX_SK_BYTES);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_keypair);

    return 0;
}

int face_treehash_wots(uint32_t loop_num, uint32_t blocks, uint32_t threads,
                       uint32_t maxallthreads) {
    unsigned char *pk, *sk;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES));

    u8* dev_keypair = NULL;
    u8 keypair[SPX_SK_BYTES + SPX_PK_BYTES];
    int device = DEVICE_USED;
    // int maxallthreads;
    cudaDeviceProp deviceProp;
    if (blocks == 0) {
        threads = 1;
        // only treehash_wots need to be parallelized
        blocks = 1;

        CHECK(cudaSetDevice(device));
        cudaGetDeviceProperties(&deviceProp, device);

        maxallthreads = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
            * deviceProp.multiProcessorCount;

#ifdef KEYGEN_SUITBLE_BLOCK
        if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

        if (threads * blocks > maxallthreads) printf("threads * blocks > maxallthreads\n");
#endif // ifdef KEYGEN_SUITBLE_BLOCK
    }

    CHECK(cudaMalloc((void**) &dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES));

    struct timespec start, stop;

    // if (g_count == 0)
    //     printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads *
    //     blocks,
    //            maxallthreads);
    g_count++;

    void* kernelArgs[] = {&dev_keypair, &loop_num};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_treehash_wots, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(keypair, dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES, D2H));
    memcpy(pk, keypair, SPX_PK_BYTES);
    memcpy(sk, &keypair[SPX_PK_BYTES], SPX_SK_BYTES);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaFree(dev_keypair));
    CHECK(cudaFreeHost(pk));
    CHECK(cudaFreeHost(sk));

    return 0;
}

int face_ap_treehash_wots_2(uint32_t loop_num, uint32_t blocks, uint32_t threads,
                            uint32_t maxallthreads) {
    unsigned char *pk, *sk;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES));

    u8* dev_keypair = NULL;
    u8 keypair[SPX_SK_BYTES + SPX_PK_BYTES];
    int device = DEVICE_USED;
    // int maxallthreads;
    cudaDeviceProp deviceProp;
    if (blocks == 0) {
        threads = 32;
        // only treehash_wots need to be parallelized
        blocks = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN / threads + 1;

        CHECK(cudaSetDevice(device));
        cudaGetDeviceProperties(&deviceProp, device);

        maxallthreads = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
            * deviceProp.multiProcessorCount;

#ifdef KEYGEN_SUITBLE_BLOCK
        if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

        if (threads * blocks > maxallthreads) printf("threads * blocks > maxallthreads\n");
#endif // ifdef KEYGEN_SUITBLE_BLOCK
    }

    CHECK(cudaMalloc((void**) &dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES));

    struct timespec start, stop;

    // if (g_count == 0)
    //     printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads *
    //     blocks,
    //            maxallthreads);
    g_count++;

    void* kernelArgs[] = {&dev_keypair, &loop_num};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_treehash_wots_2, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(keypair, dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES, D2H));
    memcpy(pk, keypair, SPX_PK_BYTES);
    memcpy(sk, &keypair[SPX_PK_BYTES], SPX_SK_BYTES);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaFree(dev_keypair));
    CHECK(cudaFreeHost(pk));
    CHECK(cudaFreeHost(sk));

    return 0;
}

int face_ap_treehash_wots_23(uint32_t loop_num, uint32_t blocks, uint32_t threads,
                             uint32_t maxallthreads) {
    unsigned char *pk, *sk;

    CHECK(cudaMallocHost(&pk, SPX_PK_BYTES));
    CHECK(cudaMallocHost(&sk, SPX_SK_BYTES));

    u8* dev_keypair = NULL;
    u8 keypair[SPX_SK_BYTES + SPX_PK_BYTES];
    int device = DEVICE_USED;
    // int maxallthreads;
    cudaDeviceProp deviceProp;
    if (blocks == 0) {
        threads = 32;
        // only treehash_wots need to be parallelized
        blocks = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN / threads + 1;

        CHECK(cudaSetDevice(device));
        cudaGetDeviceProperties(&deviceProp, device);

        maxallthreads = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
            * deviceProp.multiProcessorCount;

#ifdef KEYGEN_SUITBLE_BLOCK
        if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

        // #if defined(SPX_128S) || defined(SPX_192S) || defined(SPX_256S)
        //         blocks = maxallthreads * 2 / threads;
        // #endif

        if (threads * blocks > maxallthreads) printf("threads * blocks > maxallthreads\n");
#endif // ifdef KEYGEN_SUITBLE_BLOCK
    }

    CHECK(cudaMalloc((void**) &dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES));

    struct timespec start, stop;

    // if (g_count == 0)
    //     printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads *
    //     blocks,
    //            maxallthreads);
    g_count++;

    void* kernelArgs[] = {&dev_keypair, &loop_num};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_treehash_wots_23, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(keypair, dev_keypair, SPX_SK_BYTES + SPX_PK_BYTES, D2H));
    memcpy(pk, keypair, SPX_PK_BYTES);
    memcpy(sk, &keypair[SPX_PK_BYTES], SPX_SK_BYTES);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaFree(dev_keypair));
    CHECK(cudaFreeHost(pk));
    CHECK(cudaFreeHost(sk));

    return 0;
}

int face_mdp_crypto_sign_keypair(u8* pk, u8* sk, u32 num) {
    struct timespec start, stop;
    struct timespec b2, e2;
    double result;
    u8 *dev_pk = NULL, *dev_sk = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

#ifdef KEYGEN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    maxblocks = maxallthreads / threads;
    if (maxallthreads % threads != 0) printf("wrong in dp threads\n");
#else  // ifdef KEYGEN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_dp_crypto_sign_keypair,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef KEYGEN_SUITBLE_BLOCK

    if (num < maxallthreads)
        malloc_size = num / threads * threads + threads;
    else
        malloc_size = maxallthreads;

    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * malloc_size * sizeof(u8)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;

    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = num / loop;
        int r = num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1

#ifdef PRINT_ALL
        printf("mdp keypair: maxblocks, blocks, threads, s: %u %u %u %d\n", maxblocks, blocks,
               threads, s);
#endif
        void* Args[] = {&dev_pk, &dev_sk, &s};

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &b2);
        CHECK(cudaDeviceSynchronize());
        cudaLaunchCooperativeKernel((void*) global_dp_crypto_sign_keypair, blocks, threads, Args);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &e2);

        CHECK(cudaMemcpy(pk, dev_pk, s * SPX_PK_BYTES * sizeof(u8), D2H));
        CHECK(cudaMemcpy(sk, dev_sk, s * SPX_SK_BYTES * sizeof(u8), D2H));
        pk += s * SPX_PK_BYTES;
        sk += s * SPX_SK_BYTES;
        left -= s;

        result = (e2.tv_sec - b2.tv_sec) * 1e6 + (e2.tv_nsec - b2.tv_nsec) / 1e3;
        g_inner_result += result;
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    cudaFree(dev_pk);
    cudaFree(dev_sk);

    return 0;
} // face_mdp_crypto_sign_keypair

//
int face_mgpu_mdp_crypto_sign_keypair(u8* pk, u8* sk, u32 num) {
    struct timespec start, stop;
    struct timespec b2, e2;
    double result;
    int ngpu = 2;
    u8 *dev_pk[ngpu], *dev_sk[ngpu];
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxblocks, maxallthreads;
    cudaGetDeviceProperties(&deviceProp, 0);
    num /= ngpu;

#ifdef KEYGEN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    maxblocks = maxallthreads / threads;
    if (maxallthreads % threads != 0) printf("wrong in dp threads\n");
#else  // ifdef KEYGEN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_dp_crypto_sign_keypair,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef KEYGEN_SUITBLE_BLOCK

    if (num < maxallthreads)
        malloc_size = num / threads * threads + threads;
    else
        malloc_size = maxallthreads;

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));

        CHECK(cudaMalloc((void**) &dev_pk[j], SPX_PK_BYTES * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_sk[j], SPX_SK_BYTES * malloc_size * sizeof(u8)));
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;

    for (u32 iter = 0; iter < loop; iter++) {
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }

#ifdef PRINT_ALL
        printf("mdp keypair: maxblocks, blocks, threads, s: %u %u %u %d\n", maxblocks, blocks,
               threads, s);
#endif

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &b2);
        CHECK(cudaDeviceSynchronize());

        for (int j = 0; j < ngpu; j++) {
            void* Args[] = {&dev_pk[j], &dev_sk[j], &s};
            CHECK(cudaSetDevice(j));
            cudaLaunchCooperativeKernel((void*) global_dp_crypto_sign_keypair, blocks, threads,
                                        Args);
        }

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &e2);

        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpy(pk + j * s * SPX_PK_BYTES, dev_pk[j], s * SPX_PK_BYTES * sizeof(u8),
                             D2H));
            CHECK(cudaMemcpy(sk + j * s * SPX_SK_BYTES, dev_sk[j], s * SPX_SK_BYTES * sizeof(u8),
                             D2H));
        }
        pk += s * SPX_PK_BYTES * ngpu;
        sk += s * SPX_SK_BYTES * ngpu;
        left -= s;

        result = (e2.tv_sec - b2.tv_sec) * 1e6 + (e2.tv_nsec - b2.tv_nsec) / 1e3;
        g_inner_result += result;
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    for (int j = 0; j < 2; j++) {
        CHECK(cudaFree(dev_pk[j]));
        CHECK(cudaFree(dev_sk[j]));
    }

    return 0;
} // face_mdp_crypto_sign_keypair

int face_ms_mdp_crypto_sign_keypair(u8* pk, u8* sk, u32 num) {
    struct timespec start, stop;
    double result;
    u8 *dev_pk = NULL, *dev_sk = NULL;
    int device = DEVICE_USED;
    int threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

    malloc_size = num / threads * threads + (num % threads ? threads : 0);

    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * malloc_size * sizeof(u8)));

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef USING_STREAM_1
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_dp_crypto_sign_keypair,
                                                  threads, 0);
    maxallthreads = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
#endif // ifdef USING_STREAM_1

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    u32 loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;
    cudaStream_t stream[loop];

#ifdef PRINT_ALL
    printf("malloc_size = %d, loop = %d\n", malloc_size, loop);
#endif

    // 使用数组记录位置才可以稳定，不可以使用指针的方式来标记四个变量的偏移
    // 可能是因为流对指针的混用
    u8 *p_sk[loop], *p_pk[loop], *p_dev_sk[loop], *p_dev_pk[loop];
    for (u32 i = 0; i < loop; i++) {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    CHECK(cudaDeviceSynchronize());
    u32 sum = 0;
    u32 ss[loop], ssum[loop], bblocks[loop];
    for (u32 i = 0; i < loop; i++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxallthreads > left) {
            s = left;
            bblocks[i] = s / threads + (s % threads ? 1 : 0);
        } else {
            bblocks[i] = maxallthreads / threads;
            s = maxallthreads;
        }
        ssum[i] = sum;
        ss[i] = s;
        sum += s;
        left -= s;
#else  // if LARGE_SCHEME == 1
        u32 q = num / loop;
        u32 r = num % loop;
        u32 s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        printf("ms_mdp keypair: maxallthreads, blocks, threads, s %u %u %u %u\n", maxallthreads,
               blocks, threads, s);
#endif // ifdef PRINT_ALL

        p_pk[i] = pk + ssum[i] * SPX_PK_BYTES;
        p_sk[i] = sk + ssum[i] * SPX_SK_BYTES;
        p_dev_pk[i] = dev_pk + ssum[i] * SPX_PK_BYTES;
        p_dev_sk[i] = dev_sk + ssum[i] * SPX_SK_BYTES;
    }
    // CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < loop; i++) {
        void* Args[] = {&p_dev_pk[i], &p_dev_sk[i], &ss[i]};

        cudaLaunchCooperativeKernel((void*) global_dp_crypto_sign_keypair, bblocks[i], threads,
                                    Args, 0, stream[i]);
    }

    for (int i = 0; i < loop; i++) {
        CHECK(cudaMemcpyAsync(p_pk[i], p_dev_pk[i], ss[i] * SPX_PK_BYTES, D2H, stream[i]));
        CHECK(cudaMemcpyAsync(p_sk[i], p_dev_sk[i], ss[i] * SPX_SK_BYTES, D2H, stream[i]));
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;
    for (u32 i = 0; i < loop; i++) {
        cudaStreamDestroy(stream[i]);
    }

    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_sk));

    return 0;
} // face_ms_mdp_crypto_sign_keypair

int face_mgpu_ms_mdp_crypto_sign_keypair(u8* pk, u8* sk, u32 num) {
    struct timespec start, stop;
    double result;
    int ngpu = 2;
    u8 *dev_pk[2], *dev_sk[2];
    // int device = DEVICE_USED;
    int threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size, maxallthreads;
    num /= ngpu;

    cudaGetDeviceProperties(&deviceProp, 0);

    malloc_size = num / threads * threads + (num % threads ? threads : 0);

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef USING_STREAM_1
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_dp_crypto_sign_keypair,
                                                  threads, 0);
    maxallthreads = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
#endif // ifdef USING_STREAM_1

    for (int j = 0; j < 2; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaMalloc((void**) &dev_pk[j], SPX_PK_BYTES * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_sk[j], SPX_SK_BYTES * malloc_size * sizeof(u8)));
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    u32 loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;

#ifdef PRINT_ALL
    printf("malloc_size = %d, loop = %d\n", malloc_size, loop);
#endif

    // 使用数组记录位置才可以稳定，不可以使用指针的方式来标记四个变量的偏移
    // 可能是因为流对指针的混用
    // 必须声明二位数组的指针，否则多gpu无法寻址
    cudaStream_t stream[ngpu][loop];
    u8 *p_sk[ngpu][loop], *p_pk[ngpu][loop], *p_dev_sk[ngpu][loop], *p_dev_pk[ngpu][loop];

    for (u32 i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaStreamCreate(&stream[j][i]));
        }
    }
    u32 sum = 0;
    u32 ss[loop], ssum[loop], bblocks[loop]; // 这些数组所有GPU相同

    CHECK(cudaDeviceSynchronize());
    for (u32 i = 0; i < loop; i++) {
        u32 s;
        if (maxallthreads > left) {
            s = left;
            bblocks[i] = s / threads + (s % threads ? 1 : 0);
        } else {
            bblocks[i] = maxallthreads / threads;
            s = maxallthreads;
        }
        ssum[i] = sum;
        ss[i] = s;
        sum += s;
        left -= s;
#ifdef PRINT_ALL
        printf("ms_mdp keypair: maxallthreads, blocks, threads, s %u %u %u %u\n", maxallthreads,
               blocks, threads, s);
#endif // ifdef PRINT_ALL

        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            p_pk[j][i] = pk + j * s * SPX_PK_BYTES + ngpu * ssum[i] * SPX_PK_BYTES;
            p_sk[j][i] = sk + j * s * SPX_SK_BYTES + ngpu * ssum[i] * SPX_SK_BYTES;
            p_dev_pk[j][i] = dev_pk[j] + ssum[i] * SPX_PK_BYTES;
            p_dev_sk[j][i] = dev_sk[j] + ssum[i] * SPX_SK_BYTES;
        }
    }

    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            void* Args[] = {&p_dev_pk[j][i], &p_dev_sk[j][i], &ss[i]};
            CHECK(cudaSetDevice(j));
            cudaLaunchCooperativeKernel((void*) global_dp_crypto_sign_keypair, bblocks[i], threads,
                                        Args, 0, stream[j][i]);
        }
    }

    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpyAsync(p_pk[j][i], p_dev_pk[j][i], ss[i] * SPX_PK_BYTES, D2H,
                                  stream[j][i]));
            CHECK(cudaMemcpyAsync(p_sk[j][i], p_dev_sk[j][i], ss[i] * SPX_SK_BYTES, D2H,
                                  stream[j][i]));
        }
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;
    for (u32 i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            cudaStreamDestroy(stream[j][i]);
        }
    }

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaFree(dev_pk[j]));
        CHECK(cudaFree(dev_sk[j]));
    }

    return 0;
}

int face_mhp_sign_keypair_seperate(u8* pk, u8* sk, u32 num) {
    // Cases with more than 10496 tasks are not processed
    struct timespec start, stop;
    double result;
    u8 *dev_pk = NULL, *dev_sk = NULL;
    int device = DEVICE_USED;
    int blocks = num, threads = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN;
    // threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    // int malloc_size;
    // int maxblocks;

    // int maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mhp_sign_keypair_seperate,
                                                  threads, 0);
    // maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    // maxallthreads = maxblocks * threads;
    // if (num < maxallthreads) malloc_size = num / threads * threads + threads;
    // else malloc_size = maxallthreads;
    u32 s = num;

#ifdef PRINT_ALL
    printf("malloc_size = %d\n", s);
#endif // ifdef PRINT_ALL

    CHECK(cudaMalloc((void**) &dev_pk, s * SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk, s * SPX_SK_BYTES * sizeof(u8)));

    blocks = num;
    // #ifdef PRINT_ALL
    // int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    int max = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
    int need_threads = threads * num;
    printf("seperate hp kg: max, need_threads, blocks, threads, this: %d %u %u %u %u\n", max,
           need_threads, blocks, threads, blocks * threads);
    // #endif // ifdef PRINT_ALL

    void* Args[] = {&dev_pk, &dev_sk, &s};

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_mhp_sign_keypair_seperate, blocks, threads, Args);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(pk, dev_pk, s * SPX_PK_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(sk, dev_sk, s * SPX_SK_BYTES * sizeof(u8), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    cudaFree(dev_pk);
    cudaFree(dev_sk);

    return 0;
}

int face_mhp_crypto_sign_keypair(u8* pk, u8* sk, u32 num, u32 intra_para) {
    face_mhp_crypto_sign_keypair_scheme2(pk, sk, num, intra_para);
    return 0;
}

// 不考虑超过maxallthreads的情况
int face_mhp_crypto_sign_keypair_1(u8* pk, u8* sk, u32 num, u32 intra_para) {
    struct timespec start, stop;
    double result;
    u8 *dev_pk = NULL, *dev_sk = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mhp_crypto_sign_keypair_1,
                                                  threads, 0);
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    int maxallthreads = maxblocks * threads;

    CHECK(cudaMalloc((void**) &dev_pk, num * SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk, num * SPX_SK_BYTES * sizeof(u8)));

    blocks = num * intra_para / threads;
    if (blocks == 0) blocks = 1;

    if (blocks * threads > maxallthreads) printf("error number\n");
#ifdef PRINT_ALL
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    printf("mhp keypair--maxblocks, blocks, threads, num: %u %u %u %d\n", maxblocks, blocks,
           threads, num);
#endif // ifdef PRINT_ALL

    void* Args[] = {&dev_pk, &dev_sk, &num, &intra_para};

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_mhp_crypto_sign_keypair_1, blocks, threads, Args);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(pk, dev_pk, num * SPX_PK_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(sk, dev_sk, num * SPX_SK_BYTES * sizeof(u8), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_sk));

    return 0;
}

int face_mhp_crypto_sign_keypair_scheme2(u8* pk, u8* sk, u32 num, u32 intra_para) {
    struct timespec start, stop;
    double result;
    u8 *dev_pk = NULL, *dev_sk = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, global_mhp_crypto_sign_keypair_scheme2, threads, 0);
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    int maxallthreads = maxblocks * threads;

    CHECK(cudaMalloc((void**) &dev_pk, num * SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk, num * SPX_SK_BYTES * sizeof(u8)));

    blocks = num * intra_para / threads;
    if (blocks == 0) blocks = 1;
    if (blocks * threads > maxallthreads) blocks = maxallthreads / threads;

#ifdef PRINT_ALL
    printf("mhp keypair--maxblocks, blocks, threads: %u %u %u\n", maxblocks, blocks, threads);
#endif // ifdef PRINT_ALL

    void* Args[] = {&dev_pk, &dev_sk, &num};

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_mhp_crypto_sign_keypair_scheme2, blocks, threads,
                                Args);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(pk, dev_pk, num * SPX_PK_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(sk, dev_sk, num * SPX_SK_BYTES * sizeof(u8), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_sk));

    return 0;
}

// 不再依据任务数来确定流的数量，而是依据：任务数*并行度
int face_ms_mhp_crypto_sign_keypair(u8* pk, u8* sk, u32 num) {
    struct timespec start, stop;
    double result;
    u8 *dev_pk = NULL, *dev_sk = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int malloc_size, maxallthreads;
    int para = num * HP_PARALLELISM;

    malloc_size = num;

    CHECK(cudaMalloc((void**) &dev_pk, malloc_size * SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk, malloc_size * SPX_SK_BYTES * sizeof(u8)));

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, global_mhp_crypto_sign_keypair_scheme2, threads, 0);

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef USING_STREAM_1

    maxallthreads = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
#endif // ifdef USING_STREAM_1

    // maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    // maxallthreads = maxblocks * threads;
    // u32 s = num;

    blocks = num * HP_PARALLELISM / threads;
    if (blocks == 0) blocks = 1;

    u32 loop = para / maxallthreads + (para % maxallthreads ? 1 : 0);
    u32 left = para;

    // printf("loop = %d, para = %d, maxallthreads = %d\n", loop, para, maxallthreads);

    cudaStream_t stream[loop];
    u8 *p_sk[loop], *p_pk[loop], *p_dev_sk[loop], *p_dev_pk[loop];
    for (u32 i = 0; i < loop; i++) {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    u32 sum = 0;
    u32 ss[loop], ssum[loop], bblocks[loop];

    for (u32 i = 0; i < loop; i++) {
        u32 s;
        if (maxallthreads > left) {
            s = left;
            bblocks[i] = s / threads + (s % threads ? 1 : 0);

        } else {
            bblocks[i] = maxallthreads / threads;
            s = maxallthreads;
        }
        ssum[i] = sum;
        ss[i] = s / HP_PARALLELISM;
        sum += s / HP_PARALLELISM;
        left -= s;

        p_pk[i] = pk + ssum[i] * SPX_PK_BYTES;
        p_sk[i] = sk + ssum[i] * SPX_SK_BYTES;
        p_dev_pk[i] = dev_pk + ssum[i] * SPX_PK_BYTES;
        p_dev_sk[i] = dev_sk + ssum[i] * SPX_SK_BYTES;

#ifdef PRINT_ALL
        int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
        printf(
            "ms mhp keypair--HP_PARALLELISM = %d, maxblocks = %d, allthreads = %d, blocks = %u, "
            "threads = %u, ssum[i] = %d, s = %d\n",
            HP_PARALLELISM, maxblocks, bblocks[i] * threads, bblocks[i], threads, ssum[i], ss[i]);
#endif // ifdef PRINT_ALL
    }

    // 不同流使用了相同的静态内存，导致出错，先不处理，对性能没有影响，数据并行没有这个问题
    for (u32 i = 0; i < loop; i++) {
        void* Args[] = {&p_dev_pk[i], &p_dev_sk[i]};
        cudaLaunchCooperativeKernel((void*) global_mhp_crypto_sign_keypair_scheme2, bblocks[i],
                                    threads, Args, 0, stream[i]);
        // CHECK(cudaDeviceSynchronize()); // 使用同步来确实是否正确
    }

    for (int i = 0; i < loop; i++) {
        CHECK(cudaMemcpyAsync(p_pk[i], p_dev_pk[i], ss[i] * SPX_PK_BYTES, D2H, stream[i]));
        CHECK(cudaMemcpyAsync(p_sk[i], p_dev_sk[i], ss[i] * SPX_SK_BYTES, D2H, stream[i]));
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;
    for (u32 i = 0; i < loop; i++) {
        cudaStreamDestroy(stream[i]);
    }

    cudaFree(dev_pk);
    cudaFree(dev_sk);

    return 0;
}

/**
 * Returns an array containing a detached signature.
 */
int crypto_sign_signature(uint8_t* sig, size_t* siglen, const uint8_t* m, size_t mlen,
                          const uint8_t* sk) {
    const u8* sk_seed = sk;
    const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;

    u8 optrand[SPX_N];
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    u64 i;
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    initialize_hash_function(pub_seed, sk_seed);

    set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Optionally, signing can be made non-deterministic using optrand.
       This can help counter side-channel attacks that would benefit from
       getting a large number of traces when the signer uses the same nodes. */
    randombytes(optrand, SPX_N);
#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#endif // ifdef DEBUG_MODE

    /* Compute the digest randomization value. */
    gen_message_random(sig, sk_prf, optrand, m, mlen);

    /* Derive the message digest and leaf index from R, PK and M. */
    hash_message(mhash, &tree, &idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    set_tree_addr(wots_addr, tree);
    set_keypair_addr(wots_addr, idx_leaf);

    /* Sign the message hash using FORS. */
    fors_sign(sig, root, mhash, sk_seed, pub_seed, wots_addr);
    sig += SPX_FORS_BYTES;

    for (i = 0; i < SPX_D; i++) {
        set_layer_addr(tree_addr, i);
        set_tree_addr(tree_addr, tree);

        copy_subtree_addr(wots_addr, tree_addr);
        set_keypair_addr(wots_addr, idx_leaf);

        /* Compute a WOTS signature. */
        wots_sign(sig, root, sk_seed, pub_seed, wots_addr);
        sig += SPX_WOTS_BYTES;

        /* Compute the authentication path for the used WOTS leaf. */
        treehash(root, sig, sk_seed, pub_seed, idx_leaf, 0, SPX_TREE_HEIGHT, wots_gen_leaf,
                 tree_addr);
        sig += SPX_TREE_HEIGHT * SPX_N;

        /* Update the indices for the next layer. */
        idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
        tree = tree >> SPX_TREE_HEIGHT;
    }

    *siglen = SPX_BYTES;

    return 0;
} // crypto_sign_signature

__device__ int dev_crypto_sign_signature(uint8_t* sig, size_t* siglen, const uint8_t* m,
                                         size_t mlen, const uint8_t* sk) {
    const u8* sk_seed = sk;
    const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;

    u8 optrand[SPX_N];
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    u64 i;
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Optionally, signing can be made non-deterministic using optrand.
       This can help counter side-channel attacks that would benefit from
       getting a large number of traces when the signer uses the same nodes. */

#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#else
    dev_randombytes(optrand, SPX_N);
#endif // ifdef DEBUG_MODE
    /* Compute the digest randomization value. */
    dev_gen_message_random(sig, sk_prf, optrand, m, mlen);

    /* Derive the message digest and leaf index from R, PK and M. */
    dev_hash_message(mhash, &tree, &idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    dev_set_tree_addr(wots_addr, tree);
    dev_set_keypair_addr(wots_addr, idx_leaf);

    /* Sign the message hash using FORS. */
    dev_fors_sign(sig, root, mhash, sk_seed, pub_seed, wots_addr);
    sig += SPX_FORS_BYTES;

    for (i = 0; i < SPX_D; i++) {
        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        /* Compute a WOTS signature. */
        dev_wots_sign(sig, root, sk_seed, pub_seed, wots_addr);
        sig += SPX_WOTS_BYTES;

        /* Compute the authentication path for the used WOTS leaf. */
        dev_treehash(root, sig, sk_seed, pub_seed, idx_leaf, 0, SPX_TREE_HEIGHT, dev_wots_gen_leaf,
                     tree_addr);

        sig += SPX_TREE_HEIGHT * SPX_N;

        /* Update the indices for the next layer. */
        idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
        tree = tree >> SPX_TREE_HEIGHT;
    }

    *siglen = SPX_BYTES;

    return 0;
}

__device__ int dev_ap_crypto_sign_signature_23(uint8_t* sig, size_t* siglen, const uint8_t* m,
                                               size_t mlen, const uint8_t* sk) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const u8* sk_seed = sk;
    const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;

    u8 optrand[SPX_N];
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    uint64_t tree, t_tree;
    uint32_t idx_leaf, t_idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Optionally, signing can be made non-deterministic using optrand.
       This can help counter side-channel attacks that would benefit from
       getting a large number of traces when the signer uses the same nodes. */

#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#else
    dev_randombytes(optrand, SPX_N);
#endif // ifdef DEBUG_MODE

    /* Compute the digest randomization value. */
    if (tid == 0) dev_gen_message_random(sig, sk_prf, optrand, m, mlen);
    g.sync();

    /* Derive the message digest and leaf index from R, PK and M. */
    dev_hash_message(mhash, &t_tree, &t_idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    dev_set_tree_addr(wots_addr, t_tree);
    dev_set_keypair_addr(wots_addr, t_idx_leaf);
    /* Sign the message hash using FORS. */
#ifdef USING_PARALLEL_FORS_SIGN
    dev_ap_fors_sign(sig, root, mhash, sk_seed, pub_seed, wots_addr);
    g.sync();
#else  // ifdef USING_PARALLEL_FORS_SIGN
    if (tid == 0) dev_fors_sign(sig, root, mhash, sk_seed, pub_seed, wots_addr);
#endif // ifdef USING_PARALLEL_FORS_SIGN
    sig += SPX_FORS_BYTES;
    // if (tid == 0) printf("123123 level 2 + 3\n");

    /* Compute xmssTreeHash */
    u8* t_sig; // = sig;

    for (int i = 0; i < SPX_D; i++) {
        tree = t_tree >> (i * SPX_TREE_HEIGHT);
        if (i == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((i - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }
        t_sig = sig + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        if (tid == 0) memcpy(dev_ap_root + i * SPX_N, root, SPX_N);

        /* Compute the authentication path for the used WOTS leaf. */
        dev_ap_treehash_wots(root, t_sig, sk_seed, pub_seed, idx_leaf, 0, SPX_TREE_HEIGHT,
                             dev_wots_gen_leaf, tree_addr);
    }
    g.sync();

    /* Compute a WOTS signature. */
    for (int i = 0; i < SPX_D; i++) {
        tree = t_tree >> (i * SPX_TREE_HEIGHT);
        if (i == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((i - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }
        t_sig = sig + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_ap_wots_sign(t_sig, dev_ap_root + i * SPX_N, sk_seed, pub_seed, wots_addr, 0);
    }

    if (tid == 0) *siglen = SPX_BYTES;

    return 0;
}

__device__ int dev_ap_crypto_sign_signature_1(uint8_t* sig, size_t* siglen, const uint8_t* m,
                                              size_t mlen, const uint8_t* sk) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tnum = gridDim.x * blockDim.x;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const u8* sk_seed = sk;
    const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;
    u8* t_sig;

    u8 optrand[SPX_N];
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    uint64_t tree, t_tree;
    uint32_t idx_leaf, t_idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Optionally, signing can be made non-deterministic using optrand.
       This can help counter side-channel attacks that would benefit from
       getting a large number of traces when the signer uses the same nodes. */

#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#else
    dev_randombytes(optrand, SPX_N);
#endif // ifdef DEBUG_MODE

    /* Compute the digest randomization value. */
    if (tid == 0) dev_gen_message_random(sig, sk_prf, optrand, m, mlen);
    g.sync();

    /* Derive the message digest and leaf index from R, PK and M. */
    dev_hash_message(mhash, &t_tree, &t_idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    dev_set_tree_addr(wots_addr, t_tree);
    dev_set_keypair_addr(wots_addr, t_idx_leaf);
    /* Sign the message hash using FORS. */

    dev_ap_fors_sign_1(sig, root, mhash, sk_seed, pub_seed, wots_addr);
    g.sync();
    if (tid == 0) memcpy(dev_ap_root, root, SPX_N);
    sig += SPX_FORS_BYTES;
    // if (tid == 0) printf("123123 level 1\n");

    /* Compute xmssTreeHash */

    for (int i = tid; i < SPX_D; i += tnum) {
        tree = t_tree >> (i * SPX_TREE_HEIGHT);
        if (i == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((i - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }
        t_sig = sig + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_treehash(root, t_sig, sk_seed, pub_seed, idx_leaf, 0, SPX_TREE_HEIGHT,
                     dev_wots_gen_leaf, tree_addr);

        memcpy(dev_ap_root + i * SPX_N + SPX_N, root, SPX_N);
    }
    // g.sync();
    __syncthreads();

    /* Compute a WOTS signature. */
    for (int i = tid; i < SPX_D; i += tnum) {
        tree = t_tree >> (i * SPX_TREE_HEIGHT);
        if (i == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((i - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }
        t_sig = sig + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_wots_sign(t_sig, dev_ap_root + i * SPX_N, sk_seed, pub_seed, wots_addr);
    }

    if (tid == 0) *siglen = SPX_BYTES;

    return 0;
}

__device__ int dev_ap_crypto_sign_signature_12(uint8_t* sig, size_t* siglen, const uint8_t* m,
                                               size_t mlen, const uint8_t* sk) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tnum = gridDim.x * blockDim.x;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const u8* sk_seed = sk;
    const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;
    u8* t_sig;

    u8 optrand[SPX_N];
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    uint64_t tree, t_tree;
    uint32_t idx_leaf, t_idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Optionally, signing can be made non-deterministic using optrand.
       This can help counter side-channel attacks that would benefit from
       getting a large number of traces when the signer uses the same nodes. */

#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#else
    dev_randombytes(optrand, SPX_N);
#endif // ifdef DEBUG_MODE

    /* Compute the digest randomization value. */
    if (tid == 0) dev_gen_message_random(sig, sk_prf, optrand, m, mlen);
    g.sync();

    /* Derive the message digest and leaf index from R, PK and M. */
    dev_hash_message(mhash, &t_tree, &t_idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    dev_set_tree_addr(wots_addr, t_tree);
    dev_set_keypair_addr(wots_addr, t_idx_leaf);
    /* Sign the message hash using FORS. */

    dev_ap_fors_sign_12(sig, root, mhash, sk_seed, pub_seed, wots_addr);
    g.sync();
    if (tid == 0) memcpy(dev_ap_root, root, SPX_N);
    sig += SPX_FORS_BYTES;
    // if (tid == 0) printf("**! level 1+2 !**\n");
    int leaf_num = (1 << SPX_TREE_HEIGHT);

    /* Compute xmssTreeHash */
    for (int iter = tid; iter < leaf_num * SPX_D; iter += tnum) {
        u32 i = iter / leaf_num % SPX_D; // which level
        u32 j = iter % leaf_num;         // which leaf
        tree = t_tree >> (i * SPX_TREE_HEIGHT);

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_wots_gen_leaf(dev_leaf + (i * leaf_num + j) * SPX_N, sk_seed, pub_seed, j, tree_addr);
    }

    g.sync();
    // 将这些树认为是一些连着的树，这些树间需要做区分，但是计算和单个任务的分支节点计算一样
    // 叶节点的认证路径需要单独处理
    for (int iter = 0; iter < SPX_D; iter++) {
        if (iter == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((iter - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }

        u8* leaf_node = dev_leaf + iter * leaf_num * SPX_N;

        if (tid == ((idx_leaf >> 0) ^ 0x1))
            memcpy(dev_ap + iter * SPX_TREE_HEIGHT * SPX_N, leaf_node + tid * SPX_N, SPX_N);
    }

    int branch_para = tnum; // parallel scale for processing branch node

    for (int i = 1, ii = 1; i <= SPX_TREE_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(tree_addr, i);
        if (tid < branch_para) {
            int li = (leaf_num >> i);
            for (int j = tid; j < SPX_D * li; j += branch_para) {
                int iter = j / li;
                tree = t_tree >> (iter * SPX_TREE_HEIGHT);
                if (iter == 0) {
                    idx_leaf = t_idx_leaf;
                } else {
                    u64 last_tree = t_tree >> ((iter - 1) * SPX_TREE_HEIGHT);
                    idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
                }
                dev_set_layer_addr(tree_addr, iter);
                dev_set_tree_addr(tree_addr, tree);

                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(tree_addr, j % li);
                u8 temp[SPX_N * 2];
                memcpy(temp, dev_leaf + off, SPX_N);
                memcpy(&temp[SPX_N], dev_leaf + off + ii * SPX_N, SPX_N);
                dev_thash(dev_leaf + off, temp, 2, pub_seed, tree_addr);
                if (j % li == ((idx_leaf >> i) ^ 0x1)) {
                    memcpy(dev_ap + iter * SPX_TREE_HEIGHT * SPX_N + i * SPX_N, dev_leaf + off,
                           SPX_N);
                }
            }
        }
        ii *= 2;
    }

    // para scale of last iter: SPX_D; thus __syncthreads
    __syncthreads();

    // 这里的memcpy可以优化
    if (tid < SPX_D) {
        t_sig = sig + tid * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;
        memcpy(t_sig, dev_ap + tid * SPX_TREE_HEIGHT * SPX_N, SPX_N * SPX_TREE_HEIGHT); // auth_path
        memcpy(dev_ap_root + tid * SPX_N + SPX_N, dev_leaf + tid * leaf_num * SPX_N, SPX_N); // root
    }

    g.sync();

    /* Compute a WOTS signature. */
    for (int i = tid; i < SPX_D; i += tnum) {
        tree = t_tree >> (i * SPX_TREE_HEIGHT);
        if (i == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((i - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }
        t_sig = sig + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_wots_sign(t_sig, dev_ap_root + i * SPX_N, sk_seed, pub_seed, wots_addr);
    }

    if (tid == 0) *siglen = SPX_BYTES;

    return 0;
}

__device__ void dev_ht(uint8_t* sig, size_t* siglen, const uint8_t* m, size_t mlen,
                       const uint8_t* sk) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    // const unsigned int tnum = gridDim.x * blockDim.x;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const u8* sk_seed = sk;
    // const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;
    u8* t_sig;

    // u8 optrand[SPX_N];
    // u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    uint64_t tree, t_tree;
    uint32_t idx_leaf, t_idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    sig += SPX_FORS_BYTES;

    /* Compute xmssTreeHash */

    if (tid == 0)
        for (int i = 0; i < SPX_D; i++) {
            tree = t_tree >> (i * SPX_TREE_HEIGHT);
            if (i == 0) {
                idx_leaf = t_idx_leaf;
            } else {
                u64 last_tree = t_tree >> ((i - 1) * SPX_TREE_HEIGHT);
                idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
            }
            t_sig = sig + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;

            dev_set_layer_addr(tree_addr, i);
            dev_set_tree_addr(tree_addr, tree);

            dev_copy_subtree_addr(wots_addr, tree_addr);
            dev_set_keypair_addr(wots_addr, idx_leaf);

            dev_treehash(root, t_sig, sk_seed, pub_seed, idx_leaf, 0, SPX_TREE_HEIGHT,
                         dev_wots_gen_leaf, tree_addr);

            memcpy(dev_ap_root + i * SPX_N + SPX_N, root, SPX_N);
        }
    __syncthreads();
}

__device__ void dev_ap_ht_1(uint8_t* sig, size_t* siglen, const uint8_t* m, size_t mlen,
                            const uint8_t* sk) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tnum = gridDim.x * blockDim.x;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const u8* sk_seed = sk;
    // const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;
    u8* t_sig;

    // u8 optrand[SPX_N];
    // u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    uint64_t tree, t_tree;
    uint32_t idx_leaf, t_idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    sig += SPX_FORS_BYTES;

    /* Compute xmssTreeHash */

    for (int i = tid; i < SPX_D; i += tnum) {
        tree = t_tree >> (i * SPX_TREE_HEIGHT);
        if (i == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((i - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }
        t_sig = sig + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_treehash(root, t_sig, sk_seed, pub_seed, idx_leaf, 0, SPX_TREE_HEIGHT,
                     dev_wots_gen_leaf, tree_addr);

        memcpy(dev_ap_root + i * SPX_N + SPX_N, root, SPX_N);
    }
    __syncthreads();
}

__device__ void dev_ap_ht_12(uint8_t* sig, size_t* siglen, const uint8_t* m, size_t mlen,
                             const uint8_t* sk) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tnum = gridDim.x * blockDim.x;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const u8* sk_seed = sk;
    // const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;
    u8* t_sig;

    // u8 optrand[SPX_N];
    // u8 mhash[SPX_FORS_MSG_BYTES];
    // u8 root[SPX_N];
    uint64_t tree, t_tree = 1;
    uint32_t idx_leaf, t_idx_leaf = 1;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Optionally, signing can be made non-deterministic using optrand.
       This can help counter side-channel attacks that would benefit from
       getting a large number of traces when the signer uses the same nodes. */

    dev_set_tree_addr(wots_addr, t_tree);
    dev_set_keypair_addr(wots_addr, t_idx_leaf);

    sig += SPX_FORS_BYTES;
    // if (tid == 0) printf("**! level 1+2 !**\n");
    int leaf_num = (1 << SPX_TREE_HEIGHT);

    /* Compute xmssTreeHash */
    for (int iter = tid; iter < leaf_num * SPX_D; iter += tnum) {
        u32 i = iter / leaf_num % SPX_D; // which level
        u32 j = iter % leaf_num;         // which leaf
        tree = t_tree >> (i * SPX_TREE_HEIGHT);

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_wots_gen_leaf(dev_leaf + (i * leaf_num + j) * SPX_N, sk_seed, pub_seed, j, tree_addr);
    }

    g.sync();
    // 将这些树认为是一些连着的树，这些树间需要做区分，但是计算和单个任务的分支节点计算一样
    // 叶节点的认证路径需要单独处理
    for (int iter = 0; iter < SPX_D; iter++) {
        if (iter == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((iter - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }

        u8* leaf_node = dev_leaf + iter * leaf_num * SPX_N;

        if (tid == ((idx_leaf >> 0) ^ 0x1))
            memcpy(dev_ap + iter * SPX_TREE_HEIGHT * SPX_N, leaf_node + tid * SPX_N, SPX_N);
    }

    int branch_para = tnum; // parallel scale for processing branch node

    for (int i = 1, ii = 1; i <= SPX_TREE_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(tree_addr, i);
        if (tid < branch_para) {
            int li = (leaf_num >> i);
            for (int j = tid; j < SPX_D * li; j += branch_para) {
                int iter = j / li;
                tree = t_tree >> (iter * SPX_TREE_HEIGHT);
                if (iter == 0) {
                    idx_leaf = t_idx_leaf;
                } else {
                    u64 last_tree = t_tree >> ((iter - 1) * SPX_TREE_HEIGHT);
                    idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
                }
                dev_set_layer_addr(tree_addr, iter);
                dev_set_tree_addr(tree_addr, tree);

                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(tree_addr, j % li);
                u8 temp[SPX_N * 2];
                memcpy(temp, dev_leaf + off, SPX_N);
                memcpy(&temp[SPX_N], dev_leaf + off + ii * SPX_N, SPX_N);
                dev_thash(dev_leaf + off, temp, 2, pub_seed, tree_addr);
                if (j % li == ((idx_leaf >> i) ^ 0x1)) {
                    memcpy(dev_ap + iter * SPX_TREE_HEIGHT * SPX_N + i * SPX_N, dev_leaf + off,
                           SPX_N);
                }
            }
        }
        ii *= 2;
    }

    // para scale of last iter: SPX_D; thus __syncthreads
    __syncthreads();

    if (tid < SPX_D) {
        t_sig = sig + tid * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;
        memcpy(t_sig, dev_ap + tid * SPX_TREE_HEIGHT * SPX_N, SPX_N * SPX_TREE_HEIGHT); // auth_path
        memcpy(dev_ap_root + tid * SPX_N + SPX_N, dev_leaf + tid * leaf_num * SPX_N, SPX_N); // root
    }
}

__device__ void dev_ap_ht_123(uint8_t* sig, size_t* siglen, const uint8_t* m, size_t mlen,
                              const uint8_t* sk) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tnum = gridDim.x * blockDim.x;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const u8* sk_seed = sk;
    const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;
    u8* t_sig;

    u8 optrand[SPX_N];
    u8 mhash[SPX_FORS_MSG_BYTES];
    // u8 root[SPX_N];
    uint64_t tree, t_tree = 1;
    uint32_t idx_leaf, t_idx_leaf = 1;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Optionally, signing can be made non-deterministic using optrand.
       This can help counter side-channel attacks that would benefit from
       getting a large number of traces when the signer uses the same nodes. */

#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#else
    dev_randombytes(optrand, SPX_N);
#endif // ifdef DEBUG_MODE

    /* Compute the digest randomization value. */
    if (tid == 0) dev_gen_message_random(sig, sk_prf, optrand, m, mlen);
    g.sync();

    /* Derive the message digest and leaf index from R, PK and M. */
    dev_hash_message(mhash, &t_tree, &t_idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    dev_set_tree_addr(wots_addr, t_tree);
    dev_set_keypair_addr(wots_addr, t_idx_leaf);
    /* Sign the message hash using FORS. */

    // dev_ap_fors_sign_12(sig, root, mhash, sk_seed, pub_seed, wots_addr);
    // g.sync();
    // if (tid == 0) memcpy(dev_ap_root, root, SPX_N);
    sig += SPX_FORS_BYTES;
    // if (tid == 0) printf("** level 1+2+3 **\n");
    int leaf_num = (1 << SPX_TREE_HEIGHT);

    /* Compute xmssTreeHash */
    for (int iter = tid; iter < leaf_num * SPX_D * SPX_WOTS_LEN; iter += tnum) {
        u32 i = iter / SPX_WOTS_LEN / leaf_num % SPX_D; // which level
        u32 j = iter / SPX_WOTS_LEN % leaf_num;         // which leaf
        u32 k = iter % SPX_WOTS_LEN;                    // which wots
        tree = t_tree >> (i * SPX_TREE_HEIGHT);

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);
        uint32_t w_addr[8] = {0};
        uint32_t wpk_addr[8] = {0};

        dev_set_type(w_addr, SPX_ADDR_TYPE_WOTS);
        dev_set_type(wpk_addr, SPX_ADDR_TYPE_WOTSPK);

        dev_copy_subtree_addr(w_addr, tree_addr);
        dev_set_keypair_addr(w_addr, j);

        dev_set_chain_addr(w_addr, k);
        dev_wots_gen_sk(dev_wpk + iter * SPX_N, sk_seed, w_addr);
        dev_gen_chain(dev_wpk + iter * SPX_N, dev_wpk + iter * SPX_N, 0, SPX_WOTS_W - 1, pub_seed,
                      w_addr);
    }

    g.sync();

    for (int iter = tid; iter < leaf_num * SPX_D; iter += tnum) {
        u32 i = iter / leaf_num % SPX_D; // which level
        u32 j = iter % leaf_num;         // which leaf
        tree = t_tree >> (i * SPX_TREE_HEIGHT);

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);
        uint32_t wpk_addr[8] = {0};

        dev_set_type(wpk_addr, SPX_ADDR_TYPE_WOTSPK);

        dev_copy_subtree_addr(wpk_addr, tree_addr);
        dev_set_keypair_addr(wpk_addr, j);
        dev_thash(dev_leaf + (i * leaf_num + j) * SPX_N, dev_wpk + iter * SPX_WOTS_BYTES,
                  SPX_WOTS_LEN, pub_seed, wpk_addr);
    }

    g.sync();
    // 将这些树认为是一些连着的树，这些树间需要做区分，但是计算和单个任务的分支节点计算一样
    // 叶节点的认证路径需要单独处理
    for (int iter = 0; iter < SPX_D; iter++) {
        if (iter == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((iter - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }

        u8* leaf_node = dev_leaf + iter * leaf_num * SPX_N;

        if (tid == ((idx_leaf >> 0) ^ 0x1))
            memcpy(dev_ap + iter * SPX_TREE_HEIGHT * SPX_N, leaf_node + tid * SPX_N, SPX_N);
    }

    int branch_para = tnum; // parallel scale for processing branch node

    for (int i = 1, ii = 1; i <= SPX_TREE_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(tree_addr, i);
        if (tid < branch_para) {
            int li = (leaf_num >> i);
            for (int j = tid; j < SPX_D * li; j += branch_para) {
                int iter = j / li;
                tree = t_tree >> (iter * SPX_TREE_HEIGHT);
                if (iter == 0) {
                    idx_leaf = t_idx_leaf;
                } else {
                    u64 last_tree = t_tree >> ((iter - 1) * SPX_TREE_HEIGHT);
                    idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
                }
                dev_set_layer_addr(tree_addr, iter);
                dev_set_tree_addr(tree_addr, tree);

                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(tree_addr, j % li);
                u8 temp[SPX_N * 2];
                memcpy(temp, dev_leaf + off, SPX_N);
                memcpy(&temp[SPX_N], dev_leaf + off + ii * SPX_N, SPX_N);
                dev_thash(dev_leaf + off, temp, 2, pub_seed, tree_addr);
                if (j % li == ((idx_leaf >> i) ^ 0x1)) {
                    memcpy(dev_ap + iter * SPX_TREE_HEIGHT * SPX_N + i * SPX_N, dev_leaf + off,
                           SPX_N);
                }
            }
        }
        ii *= 2;
    }

    // para scale of last iter: SPX_D; thus __syncthreads
    __syncthreads();

    if (tid < SPX_D) {
        t_sig = sig + tid * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;
        memcpy(t_sig, dev_ap + tid * SPX_TREE_HEIGHT * SPX_N, SPX_N * SPX_TREE_HEIGHT); // auth_path
        memcpy(dev_ap_root + tid * SPX_N + SPX_N, dev_leaf + tid * leaf_num * SPX_N, SPX_N); // root
    }

    g.sync();

    /* Compute a WOTS signature. */
    // for (int iter = tid; iter < SPX_D * SPX_WOTS_LEN; iter += tnum) {
    //     unsigned int lengths[SPX_WOTS_LEN];
    //     u32 i = iter / SPX_WOTS_LEN % SPX_D; // 哪一层
    //     u32 j = iter % SPX_WOTS_LEN;         // 哪一个wots
    //     tree = t_tree >> (i * SPX_TREE_HEIGHT);
    //     if (i == 0) {
    //         idx_leaf = t_idx_leaf;
    //     } else {
    //         u64 last_tree = t_tree >> ((i - 1) * SPX_TREE_HEIGHT);
    //         idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
    //     }
    //     t_sig = sig + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);

    //     dev_set_layer_addr(wots_addr, i);
    //     dev_set_tree_addr(wots_addr, tree);
    //     dev_set_keypair_addr(wots_addr, idx_leaf);

    //     dev_chain_lengths(lengths, dev_ap_root + i * SPX_N);

    //     dev_set_chain_addr(wots_addr, j);
    //     dev_wots_gen_sk(t_sig + j * SPX_N, sk_seed, wots_addr);
    //     dev_gen_chain(t_sig + j * SPX_N, t_sig + j * SPX_N, 0, lengths[j], pub_seed, wots_addr);
    // }

    if (tid == 0) *siglen = SPX_BYTES;
}

__device__ int dev_ap_crypto_sign_signature_123(uint8_t* sig, size_t* siglen, const uint8_t* m,
                                                size_t mlen, const uint8_t* sk) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tnum = gridDim.x * blockDim.x;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const u8* sk_seed = sk;
    const u8* sk_prf = sk + SPX_N;
    const u8* pk = sk + 2 * SPX_N;
    const u8* pub_seed = pk;
    u8* t_sig;

    u8 optrand[SPX_N];
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    uint64_t tree, t_tree;
    uint32_t idx_leaf, t_idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    if (tid == 0) dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Optionally, signing can be made non-deterministic using optrand.
       This can help counter side-channel attacks that would benefit from
       getting a large number of traces when the signer uses the same nodes. */

#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#else
    dev_randombytes(optrand, SPX_N);
#endif // ifdef DEBUG_MODE

    /* Compute the digest randomization value. */
    if (tid == 0) dev_gen_message_random(sig, sk_prf, optrand, m, mlen);
    g.sync();

    /* Derive the message digest and leaf index from R, PK and M. */
    dev_hash_message(mhash, &t_tree, &t_idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    dev_set_tree_addr(wots_addr, t_tree);
    dev_set_keypair_addr(wots_addr, t_idx_leaf);
    /* Sign the message hash using FORS. */

    dev_ap_fors_sign_12(sig, root, mhash, sk_seed, pub_seed, wots_addr);
    g.sync();
    if (tid == 0) memcpy(dev_ap_root, root, SPX_N);
    sig += SPX_FORS_BYTES;
    // if (tid == 0) printf("** level 1+2+3 **\n");
    int leaf_num = (1 << SPX_TREE_HEIGHT);

    /* Compute xmssTreeHash */
    for (int iter = tid; iter < leaf_num * SPX_D * SPX_WOTS_LEN; iter += tnum) {
        u32 i = iter / SPX_WOTS_LEN / leaf_num % SPX_D; // which level
        u32 j = iter / SPX_WOTS_LEN % leaf_num;         // which leaf
        u32 k = iter % SPX_WOTS_LEN;                    // which wots
        tree = t_tree >> (i * SPX_TREE_HEIGHT);

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);
        uint32_t w_addr[8] = {0};
        uint32_t wpk_addr[8] = {0};

        dev_set_type(w_addr, SPX_ADDR_TYPE_WOTS);
        dev_set_type(wpk_addr, SPX_ADDR_TYPE_WOTSPK);

        dev_copy_subtree_addr(w_addr, tree_addr);
        dev_set_keypair_addr(w_addr, j);

        dev_set_chain_addr(w_addr, k);
        dev_wots_gen_sk(dev_wpk + iter * SPX_N, sk_seed, w_addr);
        dev_gen_chain(dev_wpk + iter * SPX_N, dev_wpk + iter * SPX_N, 0, SPX_WOTS_W - 1, pub_seed,
                      w_addr);
    }

    g.sync();

    for (int iter = tid; iter < leaf_num * SPX_D; iter += tnum) {
        u32 i = iter / leaf_num % SPX_D; // which level
        u32 j = iter % leaf_num;         // which leaf
        tree = t_tree >> (i * SPX_TREE_HEIGHT);

        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);
        uint32_t wpk_addr[8] = {0};

        dev_set_type(wpk_addr, SPX_ADDR_TYPE_WOTSPK);

        dev_copy_subtree_addr(wpk_addr, tree_addr);
        dev_set_keypair_addr(wpk_addr, j);
        dev_thash(dev_leaf + (i * leaf_num + j) * SPX_N, dev_wpk + iter * SPX_WOTS_BYTES,
                  SPX_WOTS_LEN, pub_seed, wpk_addr);
    }

    g.sync();
    // 将这些树认为是一些连着的树，这些树间需要做区分，但是计算和单个任务的分支节点计算一样
    // 叶节点的认证路径需要单独处理
    for (int iter = 0; iter < SPX_D; iter++) {
        if (iter == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((iter - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }

        u8* leaf_node = dev_leaf + iter * leaf_num * SPX_N;

        if (tid == ((idx_leaf >> 0) ^ 0x1))
            memcpy(dev_ap + iter * SPX_TREE_HEIGHT * SPX_N, leaf_node + tid * SPX_N, SPX_N);
    }

    int branch_para = tnum; // parallel scale for processing branch node

    for (int i = 1, ii = 1; i <= SPX_TREE_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(tree_addr, i);
        if (tid < branch_para) {
            int li = (leaf_num >> i);
            for (int j = tid; j < SPX_D * li; j += branch_para) {
                int iter = j / li;
                tree = t_tree >> (iter * SPX_TREE_HEIGHT);
                if (iter == 0) {
                    idx_leaf = t_idx_leaf;
                } else {
                    u64 last_tree = t_tree >> ((iter - 1) * SPX_TREE_HEIGHT);
                    idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
                }
                dev_set_layer_addr(tree_addr, iter);
                dev_set_tree_addr(tree_addr, tree);

                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(tree_addr, j % li);
                u8 temp[SPX_N * 2];
                memcpy(temp, dev_leaf + off, SPX_N);
                memcpy(&temp[SPX_N], dev_leaf + off + ii * SPX_N, SPX_N);
                dev_thash(dev_leaf + off, temp, 2, pub_seed, tree_addr);
                if (j % li == ((idx_leaf >> i) ^ 0x1)) {
                    memcpy(dev_ap + iter * SPX_TREE_HEIGHT * SPX_N + i * SPX_N, dev_leaf + off,
                           SPX_N);
                }
            }
        }
        ii *= 2;
    }

    // para scale of last iter: SPX_D; thus __syncthreads
    __syncthreads();

    if (tid < SPX_D) {
        t_sig = sig + tid * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;
        memcpy(t_sig, dev_ap + tid * SPX_TREE_HEIGHT * SPX_N, SPX_N * SPX_TREE_HEIGHT); // auth_path
        memcpy(dev_ap_root + tid * SPX_N + SPX_N, dev_leaf + tid * leaf_num * SPX_N, SPX_N); // root
    }

    g.sync();

    /* Compute a WOTS signature. */
    for (int iter = tid; iter < SPX_D * SPX_WOTS_LEN; iter += tnum) {
        unsigned int lengths[SPX_WOTS_LEN];
        u32 i = iter / SPX_WOTS_LEN % SPX_D; // 哪一层
        u32 j = iter % SPX_WOTS_LEN;         // 哪一个wots
        tree = t_tree >> (i * SPX_TREE_HEIGHT);
        if (i == 0) {
            idx_leaf = t_idx_leaf;
        } else {
            u64 last_tree = t_tree >> ((i - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }
        t_sig = sig + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);

        dev_set_layer_addr(wots_addr, i);
        dev_set_tree_addr(wots_addr, tree);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_chain_lengths(lengths, dev_ap_root + i * SPX_N);

        dev_set_chain_addr(wots_addr, j);
        dev_wots_gen_sk(t_sig + j * SPX_N, sk_seed, wots_addr);
        dev_gen_chain(t_sig + j * SPX_N, t_sig + j * SPX_N, 0, lengths[j], pub_seed, wots_addr);
    }

    if (tid == 0) *siglen = SPX_BYTES;

    return 0;
}

__device__ int dev_ap_crypto_sign_signature(uint8_t* sig, size_t* siglen, const uint8_t* m,
                                            size_t mlen, const uint8_t* sk) {

    dev_ap_crypto_sign_signature_1(sig, siglen, m, mlen, sk);
    dev_ap_crypto_sign_signature_23(sig, siglen, m, mlen, sk);
}

/**
 * Verifies a detached signature and message under a given public key.
 */
int crypto_sign_verify(const uint8_t* sig, size_t siglen, const uint8_t* m, size_t mlen,
                       const uint8_t* pk) {
    const u8* pub_seed = pk;
    const u8* pub_root = pk + SPX_N;
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 wots_pk[SPX_WOTS_BYTES];
    u8 root[SPX_N];
    u8 leaf[SPX_N];
    unsigned int i;
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    if (siglen != SPX_BYTES) {
        return -1;
    }

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    initialize_hash_function(pub_seed, NULL);

    set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);
    set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);

    /* Derive the message digest and leaf index from R || PK || M. */
    /* The additional SPX_N is a result of the hash domain separator. */
    hash_message(mhash, &tree, &idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    /* Layer correctly defaults to 0, so no need to set_layer_addr */
    set_tree_addr(wots_addr, tree);
    set_keypair_addr(wots_addr, idx_leaf);

    fors_pk_from_sig(root, sig, mhash, pub_seed, wots_addr);
    sig += SPX_FORS_BYTES;

    /* For each subtree.. */
    for (i = 0; i < SPX_D; i++) {
        set_layer_addr(tree_addr, i);
        set_tree_addr(tree_addr, tree);

        copy_subtree_addr(wots_addr, tree_addr);
        set_keypair_addr(wots_addr, idx_leaf);

        copy_keypair_addr(wots_pk_addr, wots_addr);

        /* The WOTS public key is only correct if the signature was correct. */
        /* Initially, root is the FORS pk, but on subsequent iterations it is
           the root of the subtree below the currently processed subtree. */
        wots_pk_from_sig(wots_pk, sig, root, pub_seed, wots_addr);
        sig += SPX_WOTS_BYTES;

        /* Compute the leaf node using the WOTS public key. */
        thash(leaf, wots_pk, SPX_WOTS_LEN, pub_seed, wots_pk_addr);

        /* Compute the root node of this subtree. */
        compute_root(root, leaf, idx_leaf, 0, sig, SPX_TREE_HEIGHT, pub_seed, tree_addr);
        sig += SPX_TREE_HEIGHT * SPX_N;

        /* Update the indices for the next layer. */
        idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
        tree = tree >> SPX_TREE_HEIGHT;
    }

    /* Check if the root node equals the root node in the public key. */
    if (memcmp(root, pub_root, SPX_N)) {
        return -1;
    }

    return 0;
} // crypto_sign_verify

__device__ int dev_crypto_sign_verify(const uint8_t* sig, size_t siglen, const uint8_t* m,
                                      size_t mlen, const uint8_t* pk) {
    const u8* pub_seed = pk;
    const u8* pub_root = pk + SPX_N;
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 wots_pk[SPX_WOTS_BYTES];
    u8 root[SPX_N];
    u8 leaf[SPX_N];
    unsigned int i;
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    if (siglen != SPX_BYTES) {
        return -1;
    }

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    dev_initialize_hash_function(pub_seed, NULL);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);

    /* Derive the message digest and leaf index from R || PK || M. */
    /* The additional SPX_N is a result of the hash domain separator. */
    dev_hash_message(mhash, &tree, &idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    /* Layer correctly defaults to 0, so no need to set_layer_addr */
    dev_set_tree_addr(wots_addr, tree);
    dev_set_keypair_addr(wots_addr, idx_leaf);

    dev_fors_pk_from_sig(root, sig, mhash, pub_seed, wots_addr);
    sig += SPX_FORS_BYTES;

    /* For each subtree.. */
    for (i = 0; i < SPX_D; i++) {
        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_copy_keypair_addr(wots_pk_addr, wots_addr);

        /* The WOTS public key is only correct if the signature was correct. */
        /* Initially, root is the FORS pk, but on subsequent iterations it is
           the root of the subtree below the currently processed subtree. */
        dev_wots_pk_from_sig(wots_pk, sig, root, pub_seed, wots_addr);
        sig += SPX_WOTS_BYTES;

        /* Compute the leaf node using the WOTS public key. */
        dev_thash(leaf, wots_pk, SPX_WOTS_LEN, pub_seed, wots_pk_addr);

        /* Compute the root node of this subtree. */
        dev_compute_root(root, leaf, idx_leaf, 0, sig, SPX_TREE_HEIGHT, pub_seed, tree_addr);
        sig += SPX_TREE_HEIGHT * SPX_N;

        /* Update the indices for the next layer. */
        idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
        tree = tree >> SPX_TREE_HEIGHT;
    }

    /* Check if the root node equals the root node in the public key. */
    for (int i = 0; i < SPX_N; i++) {
        if (root[i] != pub_root[i]) {
            // printf("error\n");
            return -1;
        }
    }

    return 0;
} // dev_crypto_sign_verify

__device__ int dev_ap_crypto_sign_verify(const uint8_t* sig, size_t siglen, const uint8_t* m,
                                         size_t mlen, const uint8_t* pk) {
    // const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // cooperative_groups::grid_group g = cooperative_groups::this_grid();

    __shared__ u8 s_root[SPX_N];
    __shared__ u8 s_wots_pk[SPX_WOTS_BYTES];
    const u8* pub_seed = pk;
    const u8* pub_root = pk + SPX_N;
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    u8 leaf[SPX_N];
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    if (siglen != SPX_BYTES) {
        return -1;
    }

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    dev_initialize_hash_function(pub_seed, NULL);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);

    /* Derive the message digest and leaf index from R || PK || M. */
    /* The additional SPX_N is a result of the hash domain separator. */
    dev_hash_message(mhash, &tree, &idx_leaf, sig, pk, m, mlen);
    sig += SPX_N;

    /* Layer correctly defaults to 0, so no need to set_layer_addr */
    dev_set_tree_addr(wots_addr, tree);
    dev_set_keypair_addr(wots_addr, idx_leaf);

#ifdef USING_PARALLEL_FORS_PK_FROM_SIG
    dev_ap_fors_pk_from_sig(root, sig, mhash, pub_seed, wots_addr);
#else  // ifdef USING_PARALLEL_FORS_PK_FROM_SIG
    if (tid == 0) dev_fors_pk_from_sig(root, sig, mhash, pub_seed, wots_addr);
#endif // ifdef USING_PARALLEL_FORS_PK_FROM_SIG
    sig += SPX_FORS_BYTES;

    /* For each subtree.. */
    for (int i = 0; i < SPX_D; i++) {
        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_copy_keypair_addr(wots_pk_addr, wots_addr);

        /* The WOTS public key is only correct if the signature was correct. */
        /* Initially, root is the FORS pk, but on subsequent iterations it is
           the root of the subtree below the currently processed subtree. */
#ifdef USING_PARALLEL_WOTS_PK_FROM_SIG
        if (threadIdx.x == 0) memcpy(s_root, root, SPX_N);
        dev_ap_wots_pk_from_sig(s_wots_pk, sig, s_root, pub_seed, wots_addr);
#else  // ifdef USING_PARALLEL_WOTS_PK_FROM_SIG
        __syncthreads();
        if (threadIdx.x == 0) dev_wots_pk_from_sig(s_wots_pk, sig, root, pub_seed, wots_addr);
#endif // ifdef USING_PARALLEL_WOTS_PK_FROM_SIG
        sig += SPX_WOTS_BYTES;

        /* Compute the leaf node using the WOTS public key. */
        __syncthreads();
        if (threadIdx.x == 0) dev_thash(leaf, s_wots_pk, SPX_WOTS_LEN, pub_seed, wots_pk_addr);

        /* Compute the root node of this subtree. */
        if (threadIdx.x == 0)
            dev_compute_root(root, leaf, idx_leaf, 0, sig, SPX_TREE_HEIGHT, pub_seed, tree_addr);

        sig += SPX_TREE_HEIGHT * SPX_N;

        /* Update the indices for the next layer. */
        idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
        tree = tree >> SPX_TREE_HEIGHT;
    }

    /* Check if the root node equals the root node in the public key. */

    if (threadIdx.x == 0) {
        for (int i = 0; i < SPX_N; i++) {
            if (root[i] != pub_root[i]) {
                // printf("blockIdx.x = %d\n", blockIdx.x);
                // printf("error\n");
                return -1;
            }
        }
    }

    return 0;
}

/**
 * Returns an array containing the signature followed by the message.
 */
int crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;

    crypto_sign_signature(sm, &siglen, m, (size_t) mlen, sk);

    memmove(sm + SPX_BYTES, m, mlen);
    *smlen = siglen + mlen;

    return 0;
} // crypto_sign

__device__ int dev_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;

    dev_crypto_sign_signature(sm, &siglen, m, (size_t) mlen, sk);

    memcpy(sm + SPX_BYTES, m, mlen);
    *smlen = siglen + mlen;

    return 0;
} // dev_crypto_sign

__device__ int dev_ht_o(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;

    dev_ht(sm, &siglen, m, (size_t) mlen, sk);

    return 0;
}

__device__ int dev_ap_ht_o_1(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;

    dev_ap_ht_1(sm, &siglen, m, (size_t) mlen, sk);

    return 0;
}

__device__ int dev_ap_ht_o_12(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;

    dev_ap_ht_12(sm, &siglen, m, (size_t) mlen, sk);

    return 0;
}

__device__ int dev_ap_ht_o_123(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;

    dev_ap_ht_123(sm, &siglen, m, (size_t) mlen, sk);

    return 0;
}

__global__ void global_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    dev_crypto_sign(sm, smlen, m, mlen, sk);
} // global_crypto_sign

__device__ int dev_ap_crypto_sign_23(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    dev_ap_crypto_sign_signature_23(sm, &siglen, m, (size_t) mlen, sk);

    if (tid == 0) {
        memcpy(sm + SPX_BYTES, m, mlen);
        *smlen = siglen + mlen;
    }

    return 0;
}

__device__ int dev_ap_crypto_sign_1(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    dev_ap_crypto_sign_signature_1(sm, &siglen, m, (size_t) mlen, sk);

    if (tid == 0) {
        memcpy(sm + SPX_BYTES, m, mlen);
        *smlen = siglen + mlen;
    }

    return 0;
}

__device__ int dev_ap_crypto_sign_12(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    dev_ap_crypto_sign_signature_12(sm, &siglen, m, (size_t) mlen, sk);

    if (tid == 0) {
        memcpy(sm + SPX_BYTES, m, mlen);
        *smlen = siglen + mlen;
    }

    return 0;
}

__device__ int dev_ap_crypto_sign_123(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    size_t siglen;
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    dev_ap_crypto_sign_signature_123(sm, &siglen, m, (size_t) mlen, sk);

    if (tid == 0) {
        memcpy(sm + SPX_BYTES, m, mlen);
        *smlen = siglen + mlen;
    }

    return 0;
}

__global__ void
// __launch_bounds__(128,4)
global_ap_crypto_sign_1(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    u64 t_smlen[1];
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    dev_ap_crypto_sign_1(sm, t_smlen, m, mlen, sk);
    if (tid == 0) smlen[0] = t_smlen[0];
}

__global__ void
// __launch_bounds__(128,4)
global_ap_crypto_sign_23(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    u64 t_smlen[1];
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    dev_ap_crypto_sign_23(sm, t_smlen, m, mlen, sk);
    if (tid == 0) smlen[0] = t_smlen[0];
}

__global__ void
// __launch_bounds__(128,4)
global_ap_crypto_sign_12(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    u64 t_smlen[1];
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    dev_ap_crypto_sign_12(sm, t_smlen, m, mlen, sk);
    if (tid == 0) smlen[0] = t_smlen[0];
}

__global__ void global_ht(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, int loop_num) {
    u64 t_smlen[1];

    for (int i = 0; i < loop_num; i++)
        dev_ht_o(sm, t_smlen, m, mlen, sk);
}

__global__ void global_ap_ht_1(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk,
                               int loop_num) {
    u64 t_smlen[1];

    for (int i = 0; i < loop_num; i++)
        dev_ap_ht_o_1(sm, t_smlen, m, mlen, sk);
}

__global__ void global_ap_ht_12(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk,
                                int loop_num) {
    u64 t_smlen[1];
    for (int i = 0; i < loop_num; i++)
        dev_ap_ht_o_12(sm, t_smlen, m, mlen, sk);
}

__global__ void global_ap_ht_123(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk,
                                 int loop_num) {
    u64 t_smlen[1];
    for (int i = 0; i < loop_num; i++)
        dev_ap_ht_o_123(sm, t_smlen, m, mlen, sk);
}

__global__ void
// __launch_bounds__(128,4)
global_ap_crypto_sign_123(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    u64 t_smlen[1];
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    dev_ap_crypto_sign_123(sm, t_smlen, m, mlen, sk);
    if (tid == 0) smlen[0] = t_smlen[0];
}

__global__ void t_global_mdp_crypto_sign(u8* sm, const u8* m, const u8* sk, u32 dp_num) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    u64 smlen[1];
    u64 mlen = SPX_MLEN;

    if (tid < dp_num) {
        dev_crypto_sign(sm + tid * SPX_SM_BYTES, smlen, m + tid * SPX_MLEN, mlen,
                        sk + tid * SPX_SK_BYTES);
    }
}

__global__ void global_mdp_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk,
                                       u32 dp_num) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < dp_num) {
        dev_crypto_sign(sm + tid * SPX_SM_BYTES, smlen, m + tid * SPX_MLEN, mlen,
                        sk + tid * SPX_SK_BYTES);
    }
} // global_mdp_crypto_sign

__global__ void global_sdp_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk,
                                       u32 dp_num) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < dp_num) {
        dev_crypto_sign(sm + tid * SPX_SM_BYTES, smlen, m + tid * SPX_MLEN, mlen, sk);
    }
} // global_sdp_crypto_sign

// 每个部分的并行度不同，因此单独处理，中间使用全局同步，内部可以使用共享内存，横跨的部分使用全局内容
__global__ void global_mhp_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk,
                                       u32 dp_num) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    u32 para = tnum / dp_num;
    u32 id = tid % para;
    u32 ttid = tid / para;
    u8* t_sm = sm + (tid / para) * SPX_SM_BYTES;
    const u8* t_m = m + (tid / para) * SPX_MLEN;
    const u8* t_sk = sk + (tid / para) * SPX_SK_BYTES;

    u8* sig = t_sm;
    const u8* sk_seed = t_sk;
    const u8* sk_prf = t_sk + SPX_N;
    const u8* pk = t_sk + 2 * SPX_N;
    const u8* pub_seed = pk;

    u8 optrand[SPX_N];
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    dev_initialize_hash_function(pub_seed, sk_seed);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* Optionally, signing can be made non-deterministic using optrand.
       This can help counter side-channel attacks that would benefit from
       getting a large number of traces when the signer uses the same nodes. */
    dev_randombytes(optrand, SPX_N);
#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0;
#endif // ifdef DEBUG_MODE
    /* Compute the digest randomization value. */
    dev_gen_message_random(sig, sk_prf, optrand, t_m, mlen);

    /* Derive the message digest and leaf index from R, PK and M. */
    dev_hash_message(mhash, &tree, &idx_leaf, sig, pk, t_m, mlen);
    sig += SPX_N;

    dev_set_tree_addr(wots_addr, tree);
    dev_set_keypair_addr(wots_addr, idx_leaf);

    /* Sign the message hash using FORS. */
    // dev_fors_sign(sig, root, mhash, sk_seed, pub_seed, wots_addr);
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};

    dev_copy_keypair_addr(fors_tree_addr, wots_addr);
    dev_copy_keypair_addr(fors_pk_addr, wots_addr);

    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

#ifdef HYBRID_SIGN_FORS_LOAD_BALANCING
    if (id == 0) {
        memcpy(hp_fors_tree_addr + ttid * 8, fors_tree_addr, 8 * sizeof(int));
        dev_message_to_indices(hp_indices + ttid * SPX_FORS_TREES, mhash);
    }
    __syncthreads();

    if (tid < dp_num * para) {
        for (u32 i = tid; i < SPX_FORS_TREES * dp_num; i += (para * dp_num)) {
            u32 tree_idx = (i % SPX_FORS_TREES);
            u32 task_idx = (i / SPX_FORS_TREES);
            uint32_t t_fors_tree_addr[8] = {0};
            const u8* for_sk_seed = sk + task_idx * SPX_SK_BYTES;
            const u8* for_pub_seed = for_sk_seed + 2 * SPX_N;
            u8* t_sig = sm + task_idx * SPX_SM_BYTES + SPX_N;
            u8* fors_sig = t_sig + tree_idx * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
            u32 idx_offset = tree_idx * (1 << SPX_FORS_HEIGHT);
            memcpy(t_fors_tree_addr, hp_fors_tree_addr + task_idx * 8, 8 * sizeof(int));

            dev_set_tree_height(t_fors_tree_addr, 0);
            dev_set_tree_index(t_fors_tree_addr, hp_indices[i] + idx_offset);

            dev_fors_gen_sk(fors_sig, for_sk_seed, t_fors_tree_addr);
            fors_sig += SPX_N;

            /* Compute the authentication path for this leaf node. */
            dev_treehash(hp_roots + i * SPX_N, fors_sig, for_sk_seed, for_pub_seed, hp_indices[i],
                         idx_offset, SPX_FORS_HEIGHT, dev_fors_gen_leaf, t_fors_tree_addr);
        }
    }
#else  // ifdef HYBRID_SIGN_FORS_LOAD_BALANCING
    uint32_t indices[SPX_FORS_TREES];
    dev_message_to_indices(indices, mhash);

    if (tid < dp_num * para) {
        for (u32 i = id; i < SPX_FORS_TREES; i += para) {
            u8* fors_sig = sig + i * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
            u32 idx_offset = i * (1 << SPX_FORS_HEIGHT);

            dev_set_tree_height(fors_tree_addr, 0);
            dev_set_tree_index(fors_tree_addr, indices[i] + idx_offset);

            dev_fors_gen_sk(fors_sig, sk_seed, fors_tree_addr);
            fors_sig += SPX_N;

            /* Compute the authentication path for this leaf node. */
            dev_treehash(hp_roots + ttid * SPX_FORS_TREES * SPX_N + i * SPX_N, fors_sig, sk_seed,
                         pub_seed, indices[i], idx_offset, SPX_FORS_HEIGHT, dev_fors_gen_leaf,
                         fors_tree_addr);
        }
    }
#endif // ifdef HYBRID_SIGN_FORS_LOAD_BALANCING
    __syncthreads();

    if (tid < dp_num * para) {
        /* Hash horizontally across all tree roots to derive the public key. */
        dev_thash(root, hp_roots + ttid * SPX_FORS_TREES * SPX_N, SPX_FORS_TREES, pub_seed,
                  fors_pk_addr);
    }

    sig += SPX_FORS_BYTES;

    for (u32 i = 0; i < SPX_D; i++) {
        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        if (id == 0) memcpy(hp_fors_tree_addr + ttid * 8, wots_addr, 8 * sizeof(int));

            /* Compute a WOTS signature. */
            // dev_wots_sign(sig, root, sk_seed, pub_seed, wots_addr);

#ifdef WOTS_SIGN_LOAD_BALANCING
        unsigned int lengths[SPX_WOTS_LEN];

        dev_chain_lengths(lengths, root);
        if (id == 0) {
            u32 ll[SPX_WOTS_LEN];
            memset(ll, 0, sizeof(u32) * SPX_WOTS_LEN);
            for (int i = 0; i < SPX_WOTS_LEN; i++) {
                int min = 999999;
                int mintid = -1;
                for (int j = 0; j < para; j++) {
                    if (ll[j] < min) {
                        min = ll[j];
                        mintid = j;
                    }
                }
                hp_worktid[i] = mintid;
                ll[mintid] += lengths[i];
            }
        }
        g.sync();
        for (int i = 0; i < SPX_WOTS_LEN; i++) {
            if (hp_worktid[i] == ttid) {
                dev_set_chain_addr(wots_addr, i);
                dev_wots_gen_sk(sig + i * SPX_N, sk_seed, wots_addr);
                dev_gen_chain(sig + i * SPX_N, sig + i * SPX_N, 0, lengths[i], pub_seed, wots_addr);
            }
        }
#else  // ifdef WOTS_SIGN_LOAD_BALANCING
        if (tid < dp_num * para) {
            unsigned int lengths[SPX_WOTS_LEN];

            dev_chain_lengths(lengths, root);

            for (u32 i = id; i < SPX_WOTS_LEN; i += para) {
                dev_set_chain_addr(wots_addr, i);
                dev_wots_gen_sk(sig + i * SPX_N, sk_seed, wots_addr);
                dev_gen_chain(sig + i * SPX_N, sig + i * SPX_N, 0, lengths[i], pub_seed, wots_addr);
            }
        }
#endif // ifdef WOTS_SIGN_LOAD_BALANCING
        sig += SPX_WOTS_BYTES;
        __syncthreads();
        u32 leaf_num = (1 << SPX_TREE_HEIGHT);

        /* Compute the authentication path for the used WOTS leaf. */
        // dev_treehash(root, sig, sk_seed, pub_seed, idx_leaf, 0,
        // 	     SPX_TREE_HEIGHT, dev_wots_gen_leaf, wots_addr);
        if (para > leaf_num) {
            if (tid == 0) printf("Ensure parallelism larger than %d\n", leaf_num);
            para = leaf_num;
        }

        u32 stleaf = (1 << SPX_TREE_HEIGHT) / para;
        u32 stNum = para;                                      // 子树的数量等于并行度
        u32 stheight = SPX_TREE_HEIGHT - log(para) / log(2.0); // 子树的树高
        u8* auth_path = sig;
        u8* roots = hp_xmss_roots + ttid * stNum * SPX_N;
        u8* t_auth_path = hp_auth_path + ttid * SPX_TREE_HEIGHT * SPX_N;
        if (tid < dp_num * para) {
            unsigned char stack[(SPX_TREE_HEIGHT + 1) * SPX_N];
            unsigned int heights[SPX_TREE_HEIGHT + 1];
            unsigned int offset = 0;

            for (u32 i = id * stleaf; i < (id + 1) * stleaf; i++) {
                dev_wots_gen_leaf(stack + offset * SPX_N, sk_seed, pub_seed, i, tree_addr);
                offset++;
                heights[offset - 1] = 0;

                /* If this is a node we need for the auth path.. */
                if ((idx_leaf ^ 0x1) == i) memcpy(t_auth_path, stack + (offset - 1) * SPX_N, SPX_N);

                /* While the top-most nodes are of equal height.. */
                while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
                    u32 tree_idx = (i >> (heights[offset - 1] + 1));
                    dev_set_tree_height(tree_addr, heights[offset - 1] + 1);
                    dev_set_tree_index(tree_addr, tree_idx);
                    dev_thash(stack + (offset - 2) * SPX_N, stack + (offset - 2) * SPX_N, 2,
                              pub_seed, tree_addr);

                    offset--;
                    heights[offset - 1]++;

                    if (((idx_leaf >> heights[offset - 1]) ^ 0x1) == tree_idx) {
                        memcpy(t_auth_path + heights[offset - 1] * SPX_N,
                               stack + (offset - 1) * SPX_N, SPX_N);
                    }
                }
            }
            memcpy(roots + id * SPX_N, stack, SPX_N);
        }
        for (int i = 1, ii = 1; i <= SPX_TREE_HEIGHT - stheight; i++) {
            __syncthreads();
            dev_set_tree_height(tree_addr, i + stheight);
            if (tid < dp_num * para) {
                for (int j = id; j < (stNum >> i); j += para) {
                    int off = 2 * j * ii * SPX_N;
                    dev_set_tree_index(tree_addr, j);
                    memcpy(roots + off + SPX_N, roots + off + ii * SPX_N, SPX_N);
                    dev_thash(roots + off, roots + off, 2, pub_seed, tree_addr);
                    if (j == ((idx_leaf >> (i + stheight)) ^ 0x1)) {
                        memcpy(t_auth_path + (stheight + i) * SPX_N, roots + off, SPX_N);
                    }
                }
            }
            ii *= 2;
        }

        __syncthreads();
        memcpy(auth_path, t_auth_path, SPX_TREE_HEIGHT * SPX_N);
        memcpy(root, roots, SPX_N);

        sig += SPX_TREE_HEIGHT * SPX_N;

        idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
        tree = tree >> SPX_TREE_HEIGHT;
    }

    if (id == 0) {
        memcpy(t_sm + SPX_BYTES, t_m, mlen);
        *smlen = SPX_BYTES + mlen;
    }
}

// 这是仅第一级并行，不用于展示，用于存档
// 方案一使用_scheme1表示
__global__ void global_mhp_crypto_sign_1(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk,
                                         u32 dp_num) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;

    u8 optrand[SPX_N];
    // u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    uint64_t tree;
    //  origin_tree;
    uint32_t idx_leaf;
    //  origin_idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};
    if (tid == 0) printf("level 1 sign\n");

    u32 para, id, ttid;
    u8 *t_sm, *sig;
    const u8 *t_m, *t_sk, *sk_seed, *sk_prf, *pk, *pub_seed;

#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#else
    dev_randombytes(optrand, SPX_N);
#endif // ifdef DEBUG_MODE
    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* 第一部分，此部分并行度始终与任务数相同 */
    sig = sm + tid * SPX_SM_BYTES;
    t_m = m + tid * SPX_MLEN;
    t_sk = sk + tid * SPX_SK_BYTES;
    pk = t_sk + 2 * SPX_N;

    sk_seed = t_sk;
    sk_prf = t_sk + SPX_N;
    pub_seed = t_sk + 2 * SPX_N;

    if (tid < dp_num) dev_initialize_hash_function(pub_seed, sk_seed);
    if (tid < dp_num) dev_gen_message_random(sig, sk_prf, optrand, t_m, mlen);

    if (tid < dp_num)
        dev_hash_message(&dev_mhash[tid * SPX_FORS_MSG_BYTES], &dev_tree[tid], &dev_idx_leaf[tid],
                         sig, pk, t_m, mlen);

        /* Part 2, FORS */

#ifdef HYBRID_SIGN_FORS_LOAD_BALANCING

    uint32_t fors_tree_addr[8] = {0};

    dev_copy_keypair_addr(fors_tree_addr, wots_addr);
    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);

    if (tid < dp_num) {
        dev_message_to_indices(hp_indices + tid * SPX_FORS_TREES,
                               &dev_mhash[tid * SPX_FORS_MSG_BYTES]);
    }
    g.sync();

    for (u32 i = tid; i < SPX_FORS_TREES * dp_num; i += tnum) {
        u32 tree_idx = i % SPX_FORS_TREES;
        u32 task_idx = i / SPX_FORS_TREES;
        const u8* for_sk_seed = sk + task_idx * SPX_SK_BYTES;
        const u8* for_pub_seed = for_sk_seed + 2 * SPX_N;
        u8* t_sig = sm + task_idx * SPX_SM_BYTES + SPX_N;
        u8* fors_sig = t_sig + tree_idx * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
        u32 idx_offset = tree_idx * (1 << SPX_FORS_HEIGHT);
        dev_set_tree_addr(fors_tree_addr, dev_tree[task_idx]);
        dev_set_keypair_addr(fors_tree_addr, dev_idx_leaf[task_idx]);

        dev_set_tree_height(fors_tree_addr, 0);
        dev_set_tree_index(fors_tree_addr, hp_indices[i] + idx_offset);

        dev_fors_gen_sk(fors_sig, for_sk_seed, fors_tree_addr);
        fors_sig += SPX_N;

        dev_treehash(hp_roots + i * SPX_N, fors_sig, for_sk_seed, for_pub_seed, hp_indices[i],
                     idx_offset, SPX_FORS_HEIGHT, dev_fors_gen_leaf, fors_tree_addr);
    }
#else  // ifdef HYBRID_SIGN_FORS_LOAD_BALANCING

    g.sync();

    para = 8;
    id = tid % para;
    ttid = tid / para;
    t_sm = sm + (tid / para) * SPX_SM_BYTES;
    t_m = m + (tid / para) * SPX_MLEN;
    t_sk = sk + (tid / para) * SPX_SK_BYTES;

    sig = t_sm + SPX_N;
    sk_seed = t_sk;
    pk = t_sk + 2 * SPX_N;
    pub_seed = pk;

    dev_set_tree_addr(wots_addr, dev_tree[ttid]);
    dev_set_keypair_addr(wots_addr, dev_idx_leaf[ttid]);

    uint32_t fors_tree_addr[8] = {0};

    dev_copy_keypair_addr(fors_tree_addr, wots_addr);
    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);

    uint32_t indices[SPX_FORS_TREES];
    dev_message_to_indices(indices, &dev_mhash[ttid * SPX_FORS_MSG_BYTES]);

    if (tid < dp_num * para) {
        for (u32 i = id; i < SPX_FORS_TREES; i += para) {
            u8* fors_sig = sig + i * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
            u32 idx_offset = i * (1 << SPX_FORS_HEIGHT);

            dev_set_tree_height(fors_tree_addr, 0);
            dev_set_tree_index(fors_tree_addr, indices[i] + idx_offset);

            dev_fors_gen_sk(fors_sig, sk_seed, fors_tree_addr);
            fors_sig += SPX_N;

            dev_treehash(hp_roots + ttid * SPX_FORS_TREES * SPX_N + i * SPX_N, fors_sig, sk_seed,
                         pub_seed, indices[i], idx_offset, SPX_FORS_HEIGHT, dev_fors_gen_leaf,
                         fors_tree_addr);
        }
    }
#endif // ifdef HYBRID_SIGN_FORS_LOAD_BALANCING
    g.sync();

    if (tid < dp_num) {
        pub_seed = sk + tid * SPX_SK_BYTES + 2 * SPX_N;
        uint32_t fors_pk_addr[8] = {0};
        dev_set_tree_addr(wots_addr, dev_tree[tid]);
        dev_set_keypair_addr(wots_addr, dev_idx_leaf[tid]);

        dev_copy_keypair_addr(fors_pk_addr, wots_addr);
        dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

        dev_thash(&dev_root[tid * SPX_N], hp_roots + tid * SPX_FORS_TREES * SPX_N, SPX_FORS_TREES,
                  pub_seed, fors_pk_addr);
    }

    g.sync();
    /* 第三部分：XMSS_treehash */
    /* 仅第一级并行 */

    for (u32 iter = tid; iter < SPX_D * dp_num; iter += tnum) {
        id = iter % SPX_D;
        ttid = iter / SPX_D;

        sig = sm + ttid * SPX_SM_BYTES + SPX_N + SPX_FORS_BYTES;
        sk_seed = sk + ttid * SPX_SK_BYTES;
        pub_seed = t_sk + 2 * SPX_N;

        tree = dev_tree[ttid] >> (id * SPX_TREE_HEIGHT);
        if (id == 0) {
            idx_leaf = dev_idx_leaf[ttid];
        } else {
            u64 last_tree = dev_tree[ttid] >> ((id - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }

        dev_set_layer_addr(tree_addr, id);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);
        u8* t_sig = sig + id * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;

        /* 构树 */
        dev_treehash(root, t_sig, sk_seed, pub_seed, idx_leaf, 0, SPX_TREE_HEIGHT,
                     dev_wots_gen_leaf, tree_addr);

        memcpy(&dev_root[dp_num * (id + 1) * SPX_N + ttid * SPX_N], root, SPX_N);
    }

    g.sync();

    /* 第四部分：wots_sign */
    // 第一级并行没有完成
    para = 1;
    id = tid % para;
    ttid = tid / para;
    t_sm = sm + (tid / para) * SPX_SM_BYTES;
    t_m = m + (tid / para) * SPX_MLEN;
    t_sk = sk + (tid / para) * SPX_SK_BYTES;

    sig = t_sm + SPX_N + SPX_FORS_BYTES;
    sk_seed = t_sk;
    pk = t_sk + 2 * SPX_N;
    pub_seed = pk;

    if (tid == 0) printf("wots_sign level 0\n");

    tree = dev_tree[tid];
    idx_leaf = dev_idx_leaf[tid];

    for (u32 iter = 0; iter < SPX_D; iter++) {
        dev_set_layer_addr(tree_addr, iter);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        if (tid < dp_num) {
            unsigned int lengths[SPX_WOTS_LEN];

            dev_chain_lengths(lengths, &dev_root[dp_num * iter * SPX_N + ttid * SPX_N]);

            for (u32 i = 0; i < SPX_WOTS_LEN; i++) {
                dev_set_chain_addr(wots_addr, i);
                dev_wots_gen_sk(sig + i * SPX_N, sk_seed, wots_addr);
                dev_gen_chain(sig + i * SPX_N, sig + i * SPX_N, 0, lengths[i], pub_seed, wots_addr);
            }
        }

        sig = sig + SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N;

        idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
        tree = tree >> SPX_TREE_HEIGHT;
    }

    g.sync();
    // 第5部分，最大并行度为dp_num
    t_sm = sm + tid * SPX_SM_BYTES;
    t_m = m + tid * SPX_MLEN;

    if (tid < dp_num) {
        memcpy(t_sm + SPX_BYTES, t_m, mlen);
        *smlen = SPX_BYTES + mlen;
    }
}

__global__ void global_mhp_crypto_sign_scheme2(u8* sm, u64* smlen, const u8* m, u64 mlen,
                                               const u8* sk, u32 dp_num) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;

    u8 optrand[SPX_N];
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    u32 id, ttid;
    u8 *t_sm, *sig;
    const u8 *t_m, *t_sk, *sk_seed, *sk_prf, *pk, *pub_seed;

#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#else
    dev_randombytes(optrand, SPX_N);
#endif // ifdef DEBUG_MODE
    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* 第一部分，此部分并行度始终与任务数相同 */
    sig = sm + tid * SPX_SM_BYTES;
    t_m = m + tid * SPX_MLEN;
    t_sk = sk + tid * SPX_SK_BYTES;
    pk = t_sk + 2 * SPX_N;

    sk_seed = t_sk;
    sk_prf = t_sk + SPX_N;
    pub_seed = t_sk + 2 * SPX_N;

    uint32_t fors_tree_addr[8] = {0};

    dev_copy_keypair_addr(fors_tree_addr, wots_addr);
    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);

    if (tid < dp_num) {
        dev_initialize_hash_function(pub_seed, sk_seed);
        dev_gen_message_random(sig, sk_prf, optrand, t_m, mlen);
        dev_hash_message(&dev_mhash[tid * SPX_FORS_MSG_BYTES], &dev_tree[tid], &dev_idx_leaf[tid],
                         sig, pk, t_m, mlen);
        dev_message_to_indices(hp_indices + tid * SPX_FORS_TREES,
                               &dev_mhash[tid * SPX_FORS_MSG_BYTES]);
    }
    g.sync();

    /* Part 2, FORS */
    int fors_l_num = (1 << SPX_FORS_HEIGHT);

    // compute sk
    for (u32 i = tid; i < SPX_FORS_TREES * dp_num; i += tnum) {
        u32 ff = i % SPX_FORS_TREES; // which fors tree
        u32 tt = i / SPX_FORS_TREES; // which task
        const u8* for_sk_seed = sk + tt * SPX_SK_BYTES;
        u8* t_sig = sm + tt * SPX_SM_BYTES + SPX_N;
        u8* fors_sig = t_sig + ff * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
        u32 idx_offset = ff * (1 << SPX_FORS_HEIGHT);
        dev_set_tree_addr(fors_tree_addr, dev_tree[tt]);
        dev_set_keypair_addr(fors_tree_addr, dev_idx_leaf[tt]);

        dev_set_tree_height(fors_tree_addr, 0);
        dev_set_tree_index(fors_tree_addr, hp_indices[i] + idx_offset);

        dev_fors_gen_sk(fors_sig, for_sk_seed, fors_tree_addr);
    }
    // no sync is needed

    // compute leaf
    dev_set_tree_height(fors_tree_addr, 0);
    for (u32 i = tid; i < SPX_FORS_TREES * dp_num * fors_l_num; i += tnum) {
        u32 tt = i / (SPX_FORS_TREES * fors_l_num); // which task
        u32 ff = i / fors_l_num % SPX_FORS_TREES;   // which fors tree
        u32 ll = i % fors_l_num;                    // which leaf
        const u8* for_sk_seed = sk + tt * SPX_SK_BYTES;
        const u8* for_pub_seed = for_sk_seed + 2 * SPX_N;
        u32 idx_offset = ff * (1 << SPX_FORS_HEIGHT);

        dev_set_tree_addr(fors_tree_addr, dev_tree[tt]);
        dev_set_keypair_addr(fors_tree_addr, dev_idx_leaf[tt]);
        dev_set_tree_index(fors_tree_addr, hp_indices[i / fors_l_num] + idx_offset);

        dev_fors_gen_leaf(&dev_leaf[i * SPX_N], for_sk_seed, for_pub_seed, ll + idx_offset,
                          fors_tree_addr);
    }
    // g.sync();

    for (u32 i = tid; i < SPX_FORS_TREES * dp_num * fors_l_num; i += tnum) {
        u32 tt = i / (SPX_FORS_TREES * fors_l_num); // which task
        u32 ff = i / fors_l_num % SPX_FORS_TREES;   // which fors tree
        u32 ll = i % fors_l_num;                    // which leaf
        u8* fors_sig
            = sm + tt * SPX_SM_BYTES + SPX_N + ff * (SPX_N * SPX_FORS_HEIGHT + SPX_N) + SPX_N;

        // 叶节点的认证路径
        if ((hp_indices[i / fors_l_num] ^ 0x1) == ll) {
            memcpy(fors_sig, &dev_leaf[i * SPX_N], SPX_N);
        }
    }

    for (int i = 1, ii = 1; i <= SPX_FORS_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(fors_tree_addr, i);
        int li = (fors_l_num >> i);
        for (u32 j = tid; j < SPX_FORS_TREES * li * dp_num; j += tnum) {
            u32 tt = j / (SPX_FORS_TREES * li); // which task
            u32 ff = j / li % SPX_FORS_TREES;   // which fors tree
            u32 ll = j % li;                    // which leaf
            const u8* for_sk_seed = sk + tt * SPX_SK_BYTES;
            const u8* for_pub_seed = for_sk_seed + 2 * SPX_N;
            u8* t_sig = sm + tt * SPX_SM_BYTES + SPX_N;
            u8* fors_sig = t_sig + ff * (SPX_N * SPX_FORS_HEIGHT + SPX_N) + SPX_N;
            dev_set_tree_addr(fors_tree_addr, dev_tree[tt]);
            dev_set_keypair_addr(fors_tree_addr, dev_idx_leaf[tt]);

            int off = 2 * j * ii * SPX_N;
            dev_set_tree_index(fors_tree_addr, j % (SPX_FORS_TREES * li));
            u8 temp[SPX_N * 2];
            memcpy(temp, dev_leaf + off, SPX_N);
            memcpy(&temp[SPX_N], dev_leaf + off + ii * SPX_N, SPX_N);
            dev_thash(dev_leaf + off, temp, 2, for_pub_seed, fors_tree_addr);

            if (ll == ((hp_indices[j / li] >> i) ^ 0x1)) {
                memcpy(fors_sig + i * SPX_N, dev_leaf + off, SPX_N);
            }
        }
        ii *= 2;
    }

    for (int i = tid; i < SPX_FORS_TREES * dp_num; i += tnum) {
        memcpy(hp_roots + i * SPX_N, dev_leaf + i * fors_l_num * SPX_N, SPX_N);
    }
    g.sync();

    // 对多颗fors树的根节点拼接值进行哈希，仅数据并行
    if (tid < dp_num) {
        pub_seed = sk + tid * SPX_SK_BYTES + 2 * SPX_N;
        uint32_t fors_pk_addr[8] = {0};
        dev_set_tree_addr(wots_addr, dev_tree[tid]);
        dev_set_keypair_addr(wots_addr, dev_idx_leaf[tid]);

        dev_copy_keypair_addr(fors_pk_addr, wots_addr);
        dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

        dev_thash(&dev_root[tid * SPX_N], hp_roots + tid * SPX_FORS_TREES * SPX_N, SPX_FORS_TREES,
                  pub_seed, fors_pk_addr);
    }

    // g.sync(); // fors can be parallelized with ht
    /* 第三部分：XMSS_treehash */
    /* 仅第一级并行 */
    u32 leaf_num = (1 << SPX_TREE_HEIGHT);
    for (u32 i = tid; i < SPX_D * dp_num * leaf_num; i += tnum) {
        u32 tt = i / (SPX_D * leaf_num); // which task
        u32 xx = i / leaf_num % SPX_D;   // which xmss tree
        u32 ll = i % leaf_num;           // which leaf

        sk_seed = sk + tt * SPX_SK_BYTES;
        pub_seed = t_sk + 2 * SPX_N;

        tree = dev_tree[tt] >> (xx * SPX_TREE_HEIGHT);
        if (xx == 0) {
            idx_leaf = dev_idx_leaf[tt];
        } else {
            u64 last_tree = dev_tree[tt] >> ((xx - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }

        dev_set_layer_addr(tree_addr, xx);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_wots_gen_leaf(&dev_leaf[i * SPX_N], sk_seed, pub_seed, ll, tree_addr);
    }
    // g.sync();

    for (u32 i = tid; i < SPX_D * dp_num * leaf_num; i += tnum) {
        u32 tt = i / (SPX_D * leaf_num); // which task
        u32 xx = i / leaf_num % SPX_D;   // which xmss tree
        u32 ll = i % leaf_num;           // which leaf

        sig = sm + tt * SPX_SM_BYTES + SPX_N + SPX_FORS_BYTES;

        if (xx == 0) {
            idx_leaf = dev_idx_leaf[tt];
        } else {
            u64 last_tree = dev_tree[tt] >> ((xx - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }

        u8* t_sig = sig + xx * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;

        if ((idx_leaf ^ 0x1) == ll) {
            memcpy(t_sig, &dev_leaf[i * SPX_N], SPX_N);
        }
    }

    for (int i = 1, ii = 1; i <= SPX_TREE_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(tree_addr, i);
        int li = (leaf_num >> i);
        for (u32 j = tid; j < SPX_D * li * dp_num; j += tnum) {
            u32 tt = j / (SPX_D * li); // which task
            u32 xx = j / li % SPX_D;   // which xmss tree
            u32 ll = j % li;           // which leaf
            sk_seed = sk + tt * SPX_SK_BYTES;
            pub_seed = sk_seed + 2 * SPX_N;
            sig = sm + tt * SPX_SM_BYTES + SPX_N + SPX_FORS_BYTES;
            u8* t_sig = sig + xx * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;
            tree = dev_tree[tt] >> (xx * SPX_TREE_HEIGHT);
            if (xx == 0) {
                idx_leaf = dev_idx_leaf[tt];
            } else {
                u64 last_tree = dev_tree[tt] >> ((xx - 1) * SPX_TREE_HEIGHT);
                idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
            }
            dev_set_layer_addr(tree_addr, xx);
            dev_set_tree_addr(tree_addr, tree);

            int off = 2 * j * ii * SPX_N;
            dev_set_tree_index(tree_addr, ll);
            u8 temp[SPX_N * 2];
            memcpy(temp, dev_leaf + off, SPX_N);
            memcpy(&temp[SPX_N], dev_leaf + off + ii * SPX_N, SPX_N);
            dev_thash(dev_leaf + off, temp, 2, pub_seed, tree_addr);

            if (ll == ((idx_leaf >> i) ^ 0x1)) {
                memcpy(t_sig + i * SPX_N, dev_leaf + off, SPX_N);
            }
        }
        ii *= 2;
    }
    // g.sync();

    for (int i = tid; i < SPX_D * dp_num; i += tnum) {
        id = i % SPX_D;
        ttid = i / SPX_D;

        // wrong: dev_root + i * SPX_N + dp_num * SPX_N
        memcpy(dev_root + dp_num * id * SPX_N + ttid * SPX_N + dp_num * SPX_N,
               dev_leaf + i * leaf_num * SPX_N, SPX_N);
    }
    g.sync();

    /* 第四部分：wots_sign 1+3 */
    // 签名连续存放，每层偏移固定位置
    unsigned int lengths[SPX_WOTS_LEN];
    for (u32 iter = tid; iter < dp_num * SPX_WOTS_LEN * SPX_D; iter += tnum) {
        u32 tt = iter / (SPX_WOTS_LEN * SPX_D); // 哪个任务
        u32 ll = iter / SPX_WOTS_LEN % SPX_D;   // 哪一层
        u32 ww = iter % SPX_WOTS_LEN;           // 哪一个wots
        sk_seed = sk + tt * SPX_SK_BYTES;
        pk = sk_seed + 2 * SPX_N;
        u8* t_sig = sm + tt * SPX_SM_BYTES + SPX_FORS_BYTES + SPX_N
            + ll * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);

        dev_chain_lengths(lengths, &dev_root[dp_num * ll * SPX_N + tt * SPX_N]);
        tree = dev_tree[tt] >> (ll * SPX_TREE_HEIGHT);
        if (ll == 0) {
            idx_leaf = dev_idx_leaf[tt];
        } else {
            u64 last_tree = dev_tree[tt] >> ((ll - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }
        dev_set_layer_addr(tree_addr, ll);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_set_chain_addr(wots_addr, ww);
        dev_wots_gen_sk(t_sig + ww * SPX_N, sk_seed, wots_addr);
        dev_gen_chain(t_sig + ww * SPX_N, t_sig + ww * SPX_N, 0, lengths[ww], pk, wots_addr);
    }

    // g.sync();
    // 第5部分，将数据写回，数据并行
    t_sm = sm + tid * SPX_SM_BYTES;
    t_m = m + tid * SPX_MLEN;

    if (tid < dp_num) {
        memcpy(t_sm + SPX_BYTES, t_m, mlen);
        *smlen = SPX_BYTES + mlen;
    }
}

__global__ void global_mhp_crypto_sign_scheme2_compare(u8* sm, u64* smlen, const u8* m, u64 mlen,
                                                       const u8* sk, u32 dp_num) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;

    u8 optrand[SPX_N];
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};

    u32 id, ttid;
    u8 *t_sm, *sig;
    const u8 *t_m, *t_sk, *sk_seed, *sk_prf, *pk, *pub_seed;

#ifdef DEBUG_MODE
    for (int jj = 0; jj < SPX_N; jj++)
        optrand[jj] = 0xad;
#else
    dev_randombytes(optrand, SPX_N);
#endif // ifdef DEBUG_MODE
    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);

    /* 第一部分，此部分并行度始终与任务数相同 */
    sig = sm + tid * SPX_SM_BYTES;
    t_m = m + tid * SPX_MLEN;
    t_sk = sk + tid * SPX_SK_BYTES;
    pk = t_sk + 2 * SPX_N;

    sk_seed = t_sk;
    sk_prf = t_sk + SPX_N;
    pub_seed = t_sk + 2 * SPX_N;

    uint32_t fors_tree_addr[8] = {0};

    dev_copy_keypair_addr(fors_tree_addr, wots_addr);
    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);

    // if (tid < dp_num) {
    //     dev_initialize_hash_function(pub_seed, sk_seed);
    //     dev_gen_message_random(sig, sk_prf, optrand, t_m, mlen);
    //     dev_hash_message(&dev_mhash[tid * SPX_FORS_MSG_BYTES], &dev_tree[tid],
    //     &dev_idx_leaf[tid],
    //                      sig, pk, t_m, mlen);
    //     dev_message_to_indices(hp_indices + tid * SPX_FORS_TREES,
    //                            &dev_mhash[tid * SPX_FORS_MSG_BYTES]);
    // }
    // g.sync();

    /* Part 2, FORS */
    int fors_l_num = (1 << SPX_FORS_HEIGHT);

    // compute sk
    for (u32 i = tid; i < SPX_FORS_TREES * dp_num; i += tnum) {
        u32 ff = i % SPX_FORS_TREES; // which fors tree
        u32 tt = i / SPX_FORS_TREES; // which task
        const u8* for_sk_seed = sk + tt * SPX_SK_BYTES;
        u8* t_sig = sm + tt * SPX_SM_BYTES + SPX_N;
        u8* fors_sig = t_sig + ff * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
        u32 idx_offset = ff * (1 << SPX_FORS_HEIGHT);
        dev_set_tree_addr(fors_tree_addr, dev_tree[tt]);
        dev_set_keypair_addr(fors_tree_addr, dev_idx_leaf[tt]);

        dev_set_tree_height(fors_tree_addr, 0);
        dev_set_tree_index(fors_tree_addr, hp_indices[i] + idx_offset);

        dev_fors_gen_sk(fors_sig, for_sk_seed, fors_tree_addr);
    }
    // no sync is needed

    // compute leaf
    dev_set_tree_height(fors_tree_addr, 0);
    for (u32 i = tid; i < SPX_FORS_TREES * dp_num * fors_l_num; i += tnum) {
        u32 tt = i / (SPX_FORS_TREES * fors_l_num); // which task
        u32 ff = i / fors_l_num % SPX_FORS_TREES;   // which fors tree
        u32 ll = i % fors_l_num;                    // which leaf
        const u8* for_sk_seed = sk + tt * SPX_SK_BYTES;
        const u8* for_pub_seed = for_sk_seed + 2 * SPX_N;
        u32 idx_offset = ff * (1 << SPX_FORS_HEIGHT);

        dev_set_tree_addr(fors_tree_addr, dev_tree[tt]);
        dev_set_keypair_addr(fors_tree_addr, dev_idx_leaf[tt]);
        dev_set_tree_index(fors_tree_addr, hp_indices[i / fors_l_num] + idx_offset);

        // u8 lll[SPX_N];
        // dev_fors_gen_leaf(lll, for_sk_seed, for_pub_seed, ll + idx_offset,
        dev_fors_gen_leaf(&dev_leaf[i * SPX_N], for_sk_seed, for_pub_seed, ll + idx_offset,
                          fors_tree_addr);
    }
    // g.sync();

    for (u32 i = tid; i < SPX_FORS_TREES * dp_num * fors_l_num; i += tnum) {
        u32 tt = i / (SPX_FORS_TREES * fors_l_num); // which task
        u32 ff = i / fors_l_num % SPX_FORS_TREES;   // which fors tree
        u32 ll = i % fors_l_num;                    // which leaf
        u8* fors_sig
            = sm + tt * SPX_SM_BYTES + SPX_N + ff * (SPX_N * SPX_FORS_HEIGHT + SPX_N) + SPX_N;

        // 叶节点的认证路径
        if ((hp_indices[i / fors_l_num] ^ 0x1) == ll) {
            memcpy(fors_sig, &dev_leaf[i * SPX_N], SPX_N);
        }
    }

    for (int i = 1, ii = 1; i <= SPX_FORS_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(fors_tree_addr, i);
        int li = (fors_l_num >> i);
        for (u32 j = tid; j < SPX_FORS_TREES * li * dp_num; j += tnum) {
            u32 tt = j / (SPX_FORS_TREES * li); // which task
            u32 ff = j / li % SPX_FORS_TREES;   // which fors tree
            u32 ll = j % li;                    // which leaf
            const u8* for_sk_seed = sk + tt * SPX_SK_BYTES;
            const u8* for_pub_seed = for_sk_seed + 2 * SPX_N;
            u8* t_sig = sm + tt * SPX_SM_BYTES + SPX_N;
            u8* fors_sig = t_sig + ff * (SPX_N * SPX_FORS_HEIGHT + SPX_N) + SPX_N;
            dev_set_tree_addr(fors_tree_addr, dev_tree[tt]);
            dev_set_keypair_addr(fors_tree_addr, dev_idx_leaf[tt]);

            int off = 2 * j * ii * SPX_N;
            dev_set_tree_index(fors_tree_addr, j % (SPX_FORS_TREES * li));
            u8 temp[SPX_N * 2];
            memcpy(temp, dev_leaf + off, SPX_N);
            memcpy(&temp[SPX_N], dev_leaf + off + ii * SPX_N, SPX_N);
            dev_thash(dev_leaf + off, temp, 2, for_pub_seed, fors_tree_addr);

            if (ll == ((hp_indices[j / li] >> i) ^ 0x1)) {
                memcpy(fors_sig + i * SPX_N, dev_leaf + off, SPX_N);
            }
        }
        ii *= 2;
    }

    for (int i = tid; i < SPX_FORS_TREES * dp_num; i += tnum) {
        memcpy(hp_roots + i * SPX_N, dev_leaf + i * fors_l_num * SPX_N, SPX_N);
    }
    g.sync();

    // 对多颗fors树的根节点拼接值进行哈希，仅数据并行
    if (tid < dp_num) {
        pub_seed = sk + tid * SPX_SK_BYTES + 2 * SPX_N;
        uint32_t fors_pk_addr[8] = {0};
        dev_set_tree_addr(wots_addr, dev_tree[tid]);
        dev_set_keypair_addr(wots_addr, dev_idx_leaf[tid]);

        dev_copy_keypair_addr(fors_pk_addr, wots_addr);
        dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

        dev_thash(&dev_root[tid * SPX_N], hp_roots + tid * SPX_FORS_TREES * SPX_N, SPX_FORS_TREES,
                  pub_seed, fors_pk_addr);
    }

    // g.sync(); // fors can be parallelized with ht
    /* 第三部分：XMSS_treehash */
    /* 仅第一级并行 */
    u32 leaf_num = (1 << SPX_TREE_HEIGHT);
    for (u32 i = tid; i < SPX_D * dp_num * leaf_num; i += tnum) {
        u32 tt = i / (SPX_D * leaf_num); // which task
        u32 xx = i / leaf_num % SPX_D;   // which xmss tree
        u32 ll = i % leaf_num;           // which leaf

        sk_seed = sk + tt * SPX_SK_BYTES;
        pub_seed = t_sk + 2 * SPX_N;

        tree = dev_tree[tt] >> (xx * SPX_TREE_HEIGHT);
        if (xx == 0) {
            idx_leaf = dev_idx_leaf[tt];
        } else {
            u64 last_tree = dev_tree[tt] >> ((xx - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }

        dev_set_layer_addr(tree_addr, xx);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_wots_gen_leaf(&dev_leaf[i * SPX_N], sk_seed, pub_seed, ll, tree_addr);
    }
    // g.sync();

    for (u32 i = tid; i < SPX_D * dp_num * leaf_num; i += tnum) {
        u32 tt = i / (SPX_D * leaf_num); // which task
        u32 xx = i / leaf_num % SPX_D;   // which xmss tree
        u32 ll = i % leaf_num;           // which leaf

        sig = sm + tt * SPX_SM_BYTES + SPX_N + SPX_FORS_BYTES;

        if (xx == 0) {
            idx_leaf = dev_idx_leaf[tt];
        } else {
            u64 last_tree = dev_tree[tt] >> ((xx - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }

        u8* t_sig = sig + xx * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;

        if ((idx_leaf ^ 0x1) == ll) {
            memcpy(t_sig, &dev_leaf[i * SPX_N], SPX_N);
        }
    }

    for (int i = 1, ii = 1; i <= SPX_TREE_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(tree_addr, i);
        int li = (leaf_num >> i);
        for (u32 j = tid; j < SPX_D * li * dp_num; j += tnum) {
            u32 tt = j / (SPX_D * li); // which task
            u32 xx = j / li % SPX_D;   // which xmss tree
            u32 ll = j % li;           // which leaf
            sk_seed = sk + tt * SPX_SK_BYTES;
            pub_seed = sk_seed + 2 * SPX_N;
            sig = sm + tt * SPX_SM_BYTES + SPX_N + SPX_FORS_BYTES;
            u8* t_sig = sig + xx * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N) + SPX_WOTS_BYTES;
            tree = dev_tree[tt] >> (xx * SPX_TREE_HEIGHT);
            if (xx == 0) {
                idx_leaf = dev_idx_leaf[tt];
            } else {
                u64 last_tree = dev_tree[tt] >> ((xx - 1) * SPX_TREE_HEIGHT);
                idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
            }
            dev_set_layer_addr(tree_addr, xx);
            dev_set_tree_addr(tree_addr, tree);

            int off = 2 * j * ii * SPX_N;
            dev_set_tree_index(tree_addr, ll);
            u8 temp[SPX_N * 2];
            memcpy(temp, dev_leaf + off, SPX_N);
            memcpy(&temp[SPX_N], dev_leaf + off + ii * SPX_N, SPX_N);
            dev_thash(dev_leaf + off, temp, 2, pub_seed, tree_addr);

            if (ll == ((idx_leaf >> i) ^ 0x1)) {
                memcpy(t_sig + i * SPX_N, dev_leaf + off, SPX_N);
            }
        }
        ii *= 2;
    }
    // g.sync();

    for (int i = tid; i < SPX_D * dp_num; i += tnum) {
        id = i % SPX_D;
        ttid = i / SPX_D;

        // wrong: dev_root + i * SPX_N + dp_num * SPX_N
        memcpy(dev_root + dp_num * id * SPX_N + ttid * SPX_N + dp_num * SPX_N,
               dev_leaf + i * leaf_num * SPX_N, SPX_N);
    }
    g.sync();

    /* 第四部分：wots_sign 1+3 */
    // 签名连续存放，每层偏移固定位置
    unsigned int lengths[SPX_WOTS_LEN];
    for (u32 iter = tid; iter < dp_num * SPX_WOTS_LEN * SPX_D; iter += tnum) {
        u32 tt = iter / (SPX_WOTS_LEN * SPX_D); // 哪个任务
        u32 ll = iter / SPX_WOTS_LEN % SPX_D;   // 哪一层
        u32 ww = iter % SPX_WOTS_LEN;           // 哪一个wots
        sk_seed = sk + tt * SPX_SK_BYTES;
        pk = sk_seed + 2 * SPX_N;
        u8* t_sig = sm + tt * SPX_SM_BYTES + SPX_FORS_BYTES + SPX_N
            + ll * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);

        dev_chain_lengths(lengths, &dev_root[dp_num * ll * SPX_N + tt * SPX_N]);
        tree = dev_tree[tt] >> (ll * SPX_TREE_HEIGHT);
        if (ll == 0) {
            idx_leaf = dev_idx_leaf[tt];
        } else {
            u64 last_tree = dev_tree[tt] >> ((ll - 1) * SPX_TREE_HEIGHT);
            idx_leaf = (last_tree & ((1 << SPX_TREE_HEIGHT) - 1));
        }
        dev_set_layer_addr(tree_addr, ll);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_set_chain_addr(wots_addr, ww);
        dev_wots_gen_sk(t_sig + ww * SPX_N, sk_seed, wots_addr);
        dev_gen_chain(t_sig + ww * SPX_N, t_sig + ww * SPX_N, 0, lengths[ww], pk, wots_addr);
    }

    // g.sync();
    // 第5部分，将数据写回，数据并行
    t_sm = sm + tid * SPX_SM_BYTES;
    t_m = m + tid * SPX_MLEN;

    if (tid < dp_num) {
        memcpy(t_sm + SPX_BYTES, t_m, mlen);
        *smlen = SPX_BYTES + mlen;
    }
}

int face_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    void* kernelArgs[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk};
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, (SPX_BYTES + SPX_MLEN) * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_MLEN * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());

    // global_crypto_sign << < 1, 1 >> >
    // 	(dev_sm, dev_smlen, dev_m, mlen, dev_sk);
    cudaLaunchCooperativeKernel((void*) global_crypto_sign, 1, 1, kernelArgs);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(sm, dev_sm, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), D2H));
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
} // face_crypto_sign

int face_ht(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, int loop_num) {
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    void* kernelArgs[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &loop_num};
    u32 threads = 32;
    u32 t_spx = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN * SPX_D;
    u32 blocks = t_spx / threads + 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;

    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, (SPX_BYTES + SPX_MLEN) * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_MLEN * sizeof(u8), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ht, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(sm, dev_sm, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), D2H));

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
}

int face_ap_ht_1(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, int loop_num) {
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    void* kernelArgs[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &loop_num};
    u32 threads = 32;
    u32 t_spx = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN * SPX_D;
    u32 blocks = t_spx / threads + 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;

    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, (SPX_BYTES + SPX_MLEN) * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_MLEN * sizeof(u8), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_ht_1, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(sm, dev_sm, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), D2H));

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
}

int face_ap_ht_12(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, int loop_num) {
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    void* kernelArgs[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &loop_num};
    u32 threads = 32;
    u32 t_spx = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN * SPX_D;
    u32 blocks = t_spx / threads + 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;

    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, (SPX_BYTES + SPX_MLEN) * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_MLEN * sizeof(u8), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_ht_12, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(sm, dev_sm, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), D2H));

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
}

int face_ap_ht_123(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, int loop_num) {
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    void* kernelArgs[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &loop_num};
    u32 threads = 32;
    u32 t_spx = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN * SPX_D;
    u32 blocks = t_spx / threads + 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;

    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

#if defined(SPX_128S) || defined(SPX_192S) || defined(SPX_256S)
    blocks = maxallthreads * 2 / threads;
#endif

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, (SPX_BYTES + SPX_MLEN) * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_MLEN * sizeof(u8), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_ht_123, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(sm, dev_sm, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), D2H));

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
}

int face_ap_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    void* kernelArgs[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk};
    u32 threads = 32;
    u32 t_for = (1 << SPX_FORS_HEIGHT) * SPX_FORS_TREES;
    u32 t_spx = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN * SPX_D;
    u32 blocks = (t_for > t_spx ? t_for : t_spx) / threads + 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
#ifdef SIGN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef SIGN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_23,
                                                  threads, 0);
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef SIGN_SUITBLE_BLOCK

    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, (SPX_BYTES + SPX_MLEN) * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_MLEN * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_crypto_sign_123, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(sm, dev_sm, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
}

int face_ap_crypto_sign_1(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    void* kernelArgs[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk};
    u32 threads = 32;
    u32 t_for = (1 << SPX_FORS_HEIGHT) * SPX_FORS_TREES;
    u32 t_spx = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN * SPX_D;
    u32 blocks = (t_for > t_spx ? t_for : t_spx) / threads + 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
#ifdef SIGN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef SIGN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_1, threads,
                                                  0);
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef SIGN_SUITBLE_BLOCK

    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, (SPX_BYTES + SPX_MLEN) * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_MLEN * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_crypto_sign_1, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(sm, dev_sm, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
}

int face_ap_crypto_sign_12(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    void* kernelArgs[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk};
    u32 threads = 32;
    u32 t_for = (1 << SPX_FORS_HEIGHT) * SPX_FORS_TREES;
    u32 t_spx = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN * SPX_D;
    u32 blocks = (t_for > t_spx ? t_for : t_spx) / threads + 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
#ifdef SIGN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef SIGN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_12,
                                                  threads, 0);
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef SIGN_SUITBLE_BLOCK

    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;
#if defined(SPX_192S) || defined(SPX_256S)
    blocks = maxallthreads * 2 / threads;
#endif

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, (SPX_BYTES + SPX_MLEN) * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_MLEN * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_crypto_sign_12, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(sm, dev_sm, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
}

int face_ap_crypto_sign_123(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk) {
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int maxallthreads;
    cudaDeviceProp deviceProp;
    void* kernelArgs[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk};
    u32 threads = 32;
    u32 t_for = (1 << SPX_FORS_HEIGHT) * SPX_FORS_TREES;
    u32 t_spx = (1 << SPX_TREE_HEIGHT) * SPX_WOTS_LEN * SPX_D;
    u32 blocks = (t_for > t_spx ? t_for : t_spx) / threads + 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
#ifdef SIGN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef SIGN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_123,
                                                  threads, 0);
    int maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef SIGN_SUITBLE_BLOCK

    if (maxallthreads < threads * blocks) blocks = maxallthreads / threads;

#if defined(SPX_192S)
    blocks = maxallthreads * 2 / threads;
#endif

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, (SPX_BYTES + SPX_MLEN) * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_MLEN * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_crypto_sign_123, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(sm, dev_sm, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(smlen, dev_smlen, sizeof(u64), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
}

int face_mdp_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, u32 num) {
    struct timespec start, stop, b2, e2;
    double result;
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

#ifdef SIGN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    maxblocks = maxallthreads / threads;
    if (maxallthreads % threads != 0) printf("wrong in dp threads\n");
#else  // ifdef KEYGEN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_dp_crypto_sign_keypair,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef KEYGEN_SUITBLE_BLOCK

    if (num < maxallthreads)
        malloc_size = num / threads * threads + threads;
    else
        malloc_size = maxallthreads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;

    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = num / loop;
        int r = num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1

#ifdef PRINT_ALL
        printf("mdp sign: maxblocks, blocks, threads s %u %u %u %d %d\n", maxblocks, blocks,
               threads, s, malloc_size);
#endif

        CHECK(cudaMemcpy(dev_sk, sk, s * SPX_SK_BYTES * sizeof(u8), H2D));
        CHECK(cudaMemcpy(dev_m, m, s * SPX_MLEN * sizeof(u8), H2D));

        void* Args[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &s};

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &b2);
        CHECK(cudaDeviceSynchronize());
        cudaLaunchCooperativeKernel((void*) global_mdp_crypto_sign, blocks, threads, Args);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &e2);
        result = (e2.tv_sec - b2.tv_sec) * 1e6 + (e2.tv_nsec - b2.tv_nsec) / 1e3;
        g_inner_result += result;

        CHECK(cudaMemcpy(sm, dev_sm, s * SPX_SM_BYTES * sizeof(u8), D2H));

        sk += s * SPX_SK_BYTES;
        m += s * SPX_MLEN;
        sm += s * SPX_SM_BYTES;
        left -= s;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    *smlen = SPX_SM_BYTES;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
}

int face_mgpu_mdp_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, u32 num) {
    struct timespec start, stop, b2, e2;
    double result;
    int ngpu = 2;
    u8 *dev_sm[ngpu], *dev_m[ngpu], *dev_sk[ngpu];
    u64* dev_smlen[ngpu];
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxblocks, maxallthreads;
    num /= ngpu;

    // CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, 0);

#ifdef SIGN_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    maxblocks = maxallthreads / threads;
    if (maxallthreads % threads != 0) printf("wrong in dp threads\n");
#else  // ifdef KEYGEN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_dp_crypto_sign_keypair,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef KEYGEN_SUITBLE_BLOCK

    if (num < maxallthreads)
        malloc_size = num / threads * threads + threads;
    else
        malloc_size = maxallthreads;

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaMalloc((void**) &dev_sk[j], SPX_SK_BYTES * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_m[j], SPX_MLEN * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_sm[j], SPX_SM_BYTES * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_smlen[j], sizeof(u64)));
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;

    for (u32 iter = 0; iter < loop; iter++) {
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }

#ifdef PRINT_ALL
        printf("mdp sign: maxblocks, blocks, threads s %u %u %u %d %d\n", maxblocks, blocks,
               threads, s, malloc_size);
#endif
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpy(dev_sk[j], sk + j * s * SPX_SK_BYTES, s * SPX_SK_BYTES * sizeof(u8),
                             H2D));
            CHECK(cudaMemcpy(dev_m[j], m + j * s * SPX_MLEN, s * SPX_MLEN * sizeof(u8), H2D));
        }

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &b2);
        CHECK(cudaDeviceSynchronize());
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            void* Args[] = {&dev_sm[j], &dev_smlen[j], &dev_m[j], &mlen, &dev_sk[j], &s};
            cudaLaunchCooperativeKernel((void*) global_mdp_crypto_sign, blocks, threads, Args);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &e2);
        result = (e2.tv_sec - b2.tv_sec) * 1e6 + (e2.tv_nsec - b2.tv_nsec) / 1e3;
        g_inner_result += result;

        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpy(sm + j * s * SPX_SM_BYTES, dev_sm[j], s * SPX_SM_BYTES * sizeof(u8),
                             D2H));
        }

        sk += ngpu * s * SPX_SK_BYTES;
        m += ngpu * s * SPX_MLEN;
        sm += ngpu * s * SPX_SM_BYTES;
        left -= s;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    *smlen = SPX_SM_BYTES;

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaFree(dev_m[j]));
        CHECK(cudaFree(dev_sm[j]));
        CHECK(cudaFree(dev_sk[j]));
        CHECK(cudaFree(dev_smlen[j]));
    }

    return 0;
}

int face_ms_mdp_crypto_sign(u8* sm, u64* smlen, u8* m, u64 mlen, u8* sk, u32 num) {
    struct timespec start, stop;
    double result;
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;

    int device = DEVICE_USED;
    int threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

    malloc_size = num / threads * threads + (num % threads ? threads : 0);

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * malloc_size));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size));

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef USING_STREAM_1
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mdp_crypto_sign, threads,
                                                  0);
    maxallthreads = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
#endif // ifdef USING_STREAM_1

    int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;

#ifdef PRINT_ALL
    // printf("malloc_size = %d, loop = %d\n", malloc_size, loop);
#endif
    cudaStream_t stream[loop];

    u8 *p_sk[loop], *p_m[loop], *p_sm[loop], *p_dev_sk[loop], *p_dev_m[loop], *p_dev_sm[loop];
    for (int i = 0; i < loop; i++)
        CHECK(cudaStreamCreate(&stream[i]));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    u32 sum = 0;
    u32 ss[loop], ssum[loop], bblocks[loop];
    for (u32 i = 0; i < loop; i++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxallthreads > left) {
            s = left;
            bblocks[i] = s / threads + (s % threads ? 1 : 0);
        } else {
            bblocks[i] = maxallthreads / threads;
            s = maxallthreads;
        }
        ssum[i] = sum;
        ss[i] = s;
        sum += s;
        left -= s;
#else  // if LARGE_SCHEME == 1
        int q = num / loop;
        int r = num % loop;
        int s = q + ((i < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        printf("ms mdp sign: maxallthreads, blocks, threads, s: %u %u %u %u\n", maxallthreads,
               blocks, threads, s);
#endif // ifdef PRINT_ALL
        p_sk[i] = sk + ssum[i] * SPX_SK_BYTES;
        p_m[i] = m + ssum[i] * SPX_MLEN;
        p_sm[i] = sm + ssum[i] * SM_BYTES;
        p_dev_sk[i] = dev_sk + ssum[i] * SPX_SK_BYTES;
        p_dev_m[i] = dev_m + ssum[i] * SPX_MLEN;
        p_dev_sm[i] = dev_sm + ssum[i] * SM_BYTES;
    }

    for (int i = 0; i < loop; i++) {
        CHECK(cudaMemcpyAsync(p_dev_sk[i], p_sk[i], ss[i] * SPX_SK_BYTES, H2D, stream[i]));
        CHECK(cudaMemcpyAsync(p_dev_m[i], p_m[i], ss[i] * SPX_MLEN, H2D, stream[i]));
    }
    // CHECK(cudaStreamSynchronize(stream[1]));
    for (int i = 0; i < loop; i++) {
        void* Args[] = {&p_dev_sm[i], &p_dev_m[i], &p_dev_sk[i], &ss[i]};

        cudaLaunchCooperativeKernel((void*) t_global_mdp_crypto_sign, bblocks[i], threads, Args, 0,
                                    stream[i]);
    }

    for (int i = 0; i < loop; i++) {
        CHECK(cudaMemcpyAsync(p_sm[i], p_dev_sm[i], ss[i] * SM_BYTES, D2H, stream[i]));
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    *smlen = SM_BYTES;

    for (u32 i = 0; i < loop; i++) {
        cudaStreamDestroy(stream[i]);
    }

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_sk));
    // CHECK(cudaFree(dev_smlen));

    return 0;
}

int face_mgpu_ms_mdp_crypto_sign(u8* sm, u64* smlen, u8* m, u64 mlen, u8* sk, u32 num) {
    struct timespec start, stop;
    double result;
    int ngpu = 2;
    u8 *dev_sm[ngpu], *dev_m[ngpu], *dev_sk[ngpu];
    int threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxallthreads;
    num /= ngpu;

    cudaGetDeviceProperties(&deviceProp, 0);

    malloc_size = num / threads * threads + (num % threads ? threads : 0);

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaMalloc((void**) &dev_sk[j], SPX_SK_BYTES * malloc_size));
        CHECK(cudaMalloc((void**) &dev_m[j], SPX_MLEN * malloc_size));
        CHECK(cudaMalloc((void**) &dev_sm[j], SPX_SM_BYTES * malloc_size));
    }

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef USING_STREAM_1
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mdp_crypto_sign, threads,
                                                  0);
    maxallthreads = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
#endif // ifdef USING_STREAM_1

    int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;

#ifdef PRINT_ALL
    // printf("malloc_size = %d, loop = %d\n", malloc_size, loop);
#endif
    cudaStream_t stream[ngpu][loop];

    u8 *p_sk[ngpu][loop], *p_m[ngpu][loop], *p_sm[ngpu][loop];
    u8 *p_dev_sk[ngpu][loop], *p_dev_m[ngpu][loop], *p_dev_sm[ngpu][loop];
    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaStreamCreate(&stream[j][i]));
        }
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    u32 sum = 0;
    u32 ss[loop], ssum[loop], bblocks[loop];
    for (u32 i = 0; i < loop; i++) {
        u32 s;
        if (maxallthreads > left) {
            s = left;
            bblocks[i] = s / threads + (s % threads ? 1 : 0);
        } else {
            bblocks[i] = maxallthreads / threads;
            s = maxallthreads;
        }
        ssum[i] = sum;
        ss[i] = s;
        sum += s;
        left -= s;
#ifdef PRINT_ALL
        printf("ms mdp sign: maxallthreads, blocks, threads, s: %u %u %u %u\n", maxallthreads,
               blocks, threads, s);
#endif // ifdef PRINT_ALL
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            p_sk[j][i] = sk + j * s * SPX_SK_BYTES + ngpu * ssum[i] * SPX_SK_BYTES;
            p_m[j][i] = m + j * s * SPX_MLEN + ngpu * ssum[i] * SPX_MLEN;
            p_sm[j][i] = sm + j * s * SM_BYTES + ngpu * ssum[i] * SM_BYTES;
            p_dev_sk[j][i] = dev_sk[j] + ssum[i] * SPX_SK_BYTES;
            p_dev_m[j][i] = dev_m[j] + ssum[i] * SPX_MLEN;
            p_dev_sm[j][i] = dev_sm[j] + ssum[i] * SM_BYTES;
        }
    }

    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpyAsync(p_dev_sk[j][i], p_sk[j][i], ss[i] * SPX_SK_BYTES, H2D,
                                  stream[j][i]));
            CHECK(cudaMemcpyAsync(p_dev_m[j][i], p_m[j][i], ss[i] * SPX_MLEN, H2D, stream[j][i]));
        }
    }
    // CHECK(cudaStreamSynchronize(stream[1]));
    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            void* Args[] = {&p_dev_sm[j][i], &p_dev_m[j][i], &p_dev_sk[j][i], &ss[i]};

            cudaLaunchCooperativeKernel((void*) t_global_mdp_crypto_sign, bblocks[i], threads, Args,
                                        0, stream[j][i]);
        }
    }

    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpyAsync(p_sm[j][i], p_dev_sm[j][i], ss[i] * SM_BYTES, D2H, stream[j][i]));
        }
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    *smlen = SM_BYTES;

    for (u32 i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            cudaStreamDestroy(stream[j][i]);
        }
    }

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaFree(dev_m[j]));
        CHECK(cudaFree(dev_sm[j]));
        CHECK(cudaFree(dev_sk[j]));
    }
    // CHECK(cudaFree(dev_smlen));

    return 0;
}

int face_sdp_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, u32 dp_num) {
    struct timespec start, stop;
    double result;
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int malloc_size;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mdp_crypto_sign, threads,
                                                  0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
    if (dp_num < maxallthreads)
        malloc_size = dp_num / threads * threads + threads;
    else
        malloc_size = maxallthreads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = dp_num / maxallthreads + (dp_num % maxallthreads ? 1 : 0);
    u32 left = dp_num;

    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = dp_num / loop;
        int r = dp_num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        printf("mdp sign: maxblocks, blocks, threads %u %u %u\n", maxblocks, blocks, threads);
#endif // ifdef PRINT_ALL

        CHECK(cudaMemcpy(dev_m, m, s * SPX_MLEN * sizeof(u8), H2D));

        void* Args[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &s};

        CHECK(cudaDeviceSynchronize());
        cudaLaunchCooperativeKernel((void*) global_sdp_crypto_sign, blocks, threads, Args);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(sm, dev_sm, s * SM_BYTES * sizeof(u8), D2H));
        m += s * SPX_MLEN;
        sm += s * SM_BYTES;
        left -= s;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    *smlen = SM_BYTES;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
} // face_sdp_crypto_sign

int face_ms_sdp_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, u32 dp_num) {
    struct timespec start, stop;
    double result;
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int malloc_size;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mdp_crypto_sign, threads,
                                                  0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;

    malloc_size = dp_num / threads * threads + threads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef USING_STREAM_1
       // Remain the same
#endif // ifdef USING_STREAM_1

    int loop = dp_num / maxallthreads + (dp_num % maxallthreads ? 1 : 0);
    u32 left = dp_num;

    cudaStream_t stream[loop];
    // for free
    u8* free_m = dev_m;
    u8* free_sm = dev_sm;

    for (int i = 0; i < loop; i++)
        CHECK(cudaStreamCreate(&stream[i]));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaMemcpy(dev_sk, sk, SPX_SK_BYTES * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());
    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxallthreads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxallthreads / threads;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = dp_num / loop;
        int r = dp_num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        printf("ms sdp sign: maxblocks, blocks, threads %u %u %u\n", maxblocks, blocks, threads);
#endif // ifdef PRINT_ALL

        CHECK(cudaMemcpyAsync(dev_m, m, s * SPX_MLEN * sizeof(u8), H2D, stream[iter]));

        void* Args[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &s};

        cudaLaunchCooperativeKernel((void*) global_sdp_crypto_sign, blocks, threads, Args, 0,
                                    stream[iter]);

        CHECK(cudaMemcpyAsync(sm, dev_sm, s * SM_BYTES * sizeof(u8), D2H, stream[iter]));
        m += s * SPX_MLEN;
        sm += s * SM_BYTES;
        dev_m += s * SPX_MLEN;
        dev_sm += s * SM_BYTES;
        left -= s;
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    *smlen = SM_BYTES;

    cudaFree(free_m);
    cudaFree(free_sm);
    cudaFree(dev_sk);
    cudaFree(dev_smlen);

    return 0;
} // face_ms_sdp_crypto_sign

int face_mhp_crypto_sign(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, u32 dp_num) {
    struct timespec start, stop;
    double result;
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int malloc_size;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mhp_crypto_sign, threads,
                                                  0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
    if (dp_num < maxallthreads)
        malloc_size = dp_num / threads * threads + threads;
    else
        malloc_size = maxallthreads;
    // printf("malloc_size = %d\n", malloc_size);

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = dp_num / maxallthreads + (dp_num % maxallthreads ? 1 : 0);
    u32 left = dp_num;

    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = dp_num / loop;
        int r = dp_num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1

        CHECK(cudaMemcpy(dev_sk, sk, s * SPX_SK_BYTES * sizeof(u8), H2D));
        CHECK(cudaMemcpy(dev_m, m, s * SPX_MLEN * sizeof(u8), H2D));

        void* Args[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &s};
        blocks = dp_num * HP_PARALLELISM / threads;
        // blocks = dp_num * 2 * HP_PARALLELISM / threads;
        if (blocks == 0) blocks = 1;

#ifdef PRINT_ALL
        printf("mhp sign: maxblocks, blocks, threads s: %u %u %u %d\n", maxblocks, blocks, threads,
               s);
#endif // ifdef PRINT_ALL
        CHECK(cudaDeviceSynchronize());
        cudaLaunchCooperativeKernel((void*) global_mhp_crypto_sign, blocks, threads, Args);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(sm, dev_sm, s * SPX_SM_BYTES * sizeof(u8), D2H));

        sk += s * SPX_SK_BYTES;
        m += s * SPX_MLEN;
        sm += s * SPX_SM_BYTES;
        left -= s;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    *smlen = SPX_SM_BYTES;

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_sk));
    CHECK(cudaFree(dev_smlen));

    return 0;
} // face_mhp_crypto_sign

int face_mhp_crypto_sign_1(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk, u32 dp_num) {
    struct timespec start, stop;
    double result;
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int malloc_size;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mhp_crypto_sign_1,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
    if (dp_num < maxallthreads)
        malloc_size = dp_num / threads * threads + threads;
    else
        malloc_size = maxallthreads;
    printf("malloc_size = %d\n", malloc_size);

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = dp_num / maxallthreads + (dp_num % maxallthreads ? 1 : 0);
    u32 left = dp_num;

    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = dp_num / loop;
        int r = dp_num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1

        CHECK(cudaMemcpy(dev_sk, sk, s * SPX_SK_BYTES * sizeof(u8), H2D));
        CHECK(cudaMemcpy(dev_m, m, s * SPX_MLEN * sizeof(u8), H2D));

        void* Args[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &s};
        blocks = dp_num * HP_PARALLELISM / threads;
        // blocks = dp_num * 2 * HP_PARALLELISM / threads;
        if (blocks == 0) blocks = 1;

        // #ifdef PRINT_ALL
        printf("mhp sign: maxblocks, blocks, threads s: %u %u %u %d\n", maxblocks, blocks, threads,
               s);
        // #endif // ifdef PRINT_ALL
        CHECK(cudaDeviceSynchronize());
        cudaLaunchCooperativeKernel((void*) global_mhp_crypto_sign_1, blocks, threads, Args);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(sm, dev_sm, s * SPX_SM_BYTES * sizeof(u8), D2H));

        sk += s * SPX_SK_BYTES;
        m += s * SPX_MLEN;
        sm += s * SPX_SM_BYTES;
        left -= s;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    *smlen = SPX_SM_BYTES;

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_sk));
    CHECK(cudaFree(dev_smlen));

    return 0;
} // face_mhp_crypto_sign

int face_mhp_crypto_sign_scheme2(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk,
                                 u32 dp_num, u32 intra_para) {
    struct timespec start, stop;
    double result;
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mhp_crypto_sign_scheme2,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaMemcpy(dev_sk, sk, dp_num * SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, dp_num * SPX_MLEN * sizeof(u8), H2D));

    void* Args[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &dp_num};
    blocks = dp_num * intra_para / threads;
    // blocks = maxallthreads / threads;
    if (blocks == 0) blocks = 1;
    if (blocks * threads > maxallthreads) blocks = maxallthreads / threads;

#ifdef PRINT_ALL
    printf("mhp sign: maxblocks, blocks, threads s: %u %u %u %d\n", maxblocks, blocks, threads, s);
#endif // ifdef PRINT_ALL
    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_mhp_crypto_sign_scheme2, blocks, threads, Args);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(sm, dev_sm, dp_num * SPX_SM_BYTES * sizeof(u8), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    *smlen = SPX_SM_BYTES;

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_sk));
    CHECK(cudaFree(dev_smlen));

    return 0;
}

int face_mhp_crypto_sign_scheme2_compare(u8* sm, u64* smlen, const u8* m, u64 mlen, const u8* sk,
                                         u32 dp_num, u32 intra_para) {
    struct timespec start, stop;
    double result;
    u8 *dev_sm = NULL, *dev_m = NULL, *dev_sk = NULL;
    u64* dev_smlen = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mhp_crypto_sign_scheme2_compare,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;

    CHECK(cudaMalloc((void**) &dev_sk, SPX_SK_BYTES * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_smlen, sizeof(u64)));

    CHECK(cudaMemcpy(dev_sk, sk, dp_num * SPX_SK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, dp_num * SPX_MLEN * sizeof(u8), H2D));

    void* Args[] = {&dev_sm, &dev_smlen, &dev_m, &mlen, &dev_sk, &dp_num};
    blocks = dp_num * intra_para / threads;
    // blocks = maxallthreads / threads;
    if (blocks == 0) blocks = 1;
    if (blocks * threads > maxallthreads) blocks = maxallthreads / threads;

    float elapsed_time_ms = 0.0f;
    cudaEvent_t start1, stop1;
    cudaError_t err;

    CHECK(cudaDeviceSynchronize());
#ifdef PRINT_ALL
    printf("mhp sign: maxblocks, blocks, threads s: %u %u %u %d\n", maxblocks, blocks, threads, s);
#endif // ifdef PRINT_ALL
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);
    cudaLaunchCooperativeKernel((void*) global_mhp_crypto_sign_scheme2_compare, blocks, threads,
                                Args);
    CHECK(cudaMemcpy(sm, dev_sm, dp_num * SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaGetLastError());
    cudaEventRecord(stop1, 0);
    cudaDeviceSynchronize();
    cudaEventSynchronize(start1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&elapsed_time_ms, start1, stop1);
    elapsed_time_ms *= 1000;

    // clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    // result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += elapsed_time_ms;

    *smlen = SPX_SM_BYTES;

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_sk));
    CHECK(cudaFree(dev_smlen));

    return 0;
}

//*****************SIGN OPEN*****************//
//*****************SIGN OPEN*****************//
//*****************SIGN OPEN*****************//
//*****************SIGN OPEN*****************//
//*****************SIGN OPEN*****************//
//*****************SIGN OPEN*****************//

/**
 * Verifies a given signature-message pair under a given public key.
 */
int crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk) {
    /* The API caller does not necessarily know what size a signature should be
       but SPHINCS+ signatures are always exactly SPX_BYTES. */
    if (smlen < SPX_BYTES) {
        memset(m, 0, smlen);
        *mlen = 0;
        return -1;
    }

    *mlen = smlen - SPX_BYTES;

    if (crypto_sign_verify(sm, SPX_BYTES, sm + SPX_BYTES, (size_t) *mlen, pk)) {
        memset(m, 0, smlen);
        *mlen = 0;
        return -1;
    }

    /* If verification was successful, move the message to the right place. */
    memmove(m, sm + SPX_BYTES, *mlen);

    return 0;
} // crypto_sign_open

__device__ int dev_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk) {
    /* The API caller does not necessarily know what size a signature should be
       but SPHINCS+ signatures are always exactly SPX_BYTES. */
    if (smlen < SPX_BYTES) {
        memset(m, 0, smlen);
        *mlen = 0;
        return -1;
    }

    *mlen = smlen - SPX_BYTES;

    if (dev_crypto_sign_verify(sm, SPX_BYTES, sm + SPX_BYTES, (size_t) *mlen, pk)) {
        memset(m, 0, smlen);
        *mlen = 0;
        // printf("error\n");
        return -1;
    }

    /* If verification was successful, move the message to the right place. */
    memcpy(m, sm + SPX_BYTES, *mlen);

    return 0;
} // dev_crypto_sign_open

__global__ void global_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk,
                                        int* right) {
    *right = dev_crypto_sign_open(m, mlen, sm, smlen, pk);
} // global_crypto_sign_open

__device__ int dev_ap_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk) {
    // const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    /* The API caller does not necessarily know what size a signature should be
       but SPHINCS+ signatures are always exactly SPX_BYTES. */
    if (smlen < SPX_BYTES) {
        memset(m, 0, smlen);
        *mlen = 0;
        return -1;
    }

    *mlen = smlen - SPX_BYTES;
    const u8* t_sm = sm + blockIdx.x * SPX_SM_BYTES;

    if (dev_ap_crypto_sign_verify(t_sm, SPX_BYTES, t_sm + SPX_BYTES, (size_t) *mlen, pk)) {
        memset(m, 0, smlen);
        *mlen = 0;
        // printf("error\n");
        return -1;
    }

    /* If verification was successful, move the message to the right place. */
    if (threadIdx.x == 0) memcpy(m + blockIdx.x * (*mlen), t_sm + SPX_BYTES, *mlen);

    return 0;
}

__global__ void global_ap_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk,
                                           int* right) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int r = dev_ap_crypto_sign_open(m, mlen, sm, smlen, pk);

    if (tid == 0) *right = r;
}

__global__ void global_mdp_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk,
                                            u32 dp_num, int* right) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ret = 0;

    if (tid < dp_num) {
        ret = dev_crypto_sign_open(m + tid * SPX_MLEN, mlen, sm + tid * SPX_SM_BYTES, smlen,
                                   pk + tid * SPX_PK_BYTES);
    }
    if (tid == 0) *right = ret;
} // global_mdp_crypto_sign_open

__global__ void global_sdp_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk,
                                            u32 dp_num, int* right) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ret = 0;

    if (tid < dp_num) {
        ret = dev_crypto_sign_open(m + tid * SPX_MLEN, mlen, sm + tid * SPX_SM_BYTES, smlen, pk);
    }
    if (tid == 0) *right = ret;
} // global_sdp_crypto_sign_open

// something wrong which some number threads
__global__ void global_mhp_crypto_sign_open_whole(u8* m, u64* mlen, const u8* sm, u64 smlen,
                                                  const u8* pk, u32 dp_num, int* right) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    u32 para = tnum / dp_num;
    const u32 id = tid % para;
    const u32 ttid = tid / para;
    const u8* sig = sm + (tid / para) * SPX_SM_BYTES;
    const u8* t_pk = pk + (tid / para) * SPX_PK_BYTES;
    const u8* t_m = sm + (tid / para) * SPX_SM_BYTES + SPX_BYTES;
    const u8* pub_seed = t_pk;
    const u8* pub_root = t_pk + SPX_N;
    u8 mhash[SPX_FORS_MSG_BYTES];
    // u8 wots_pk[SPX_WOTS_BYTES];
    u8 root[SPX_N];
    u8 leaf[SPX_N];
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    *mlen = smlen - SPX_BYTES;
    if (smlen < SPX_BYTES) {
        memset(m + tid * SPX_MLEN, 0, smlen);
        *mlen = 0;
        if (tid == 0) *right = -1;
        return;
    }

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    dev_initialize_hash_function(pub_seed, NULL);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);

    /* Derive the message digest and leaf index from R || PK || M. */
    /* The additional SPX_N is a result of the hash domain separator. */
    dev_hash_message(mhash, &tree, &idx_leaf, sig, t_pk, t_m, *mlen);

    // if (tid == 0) printf("mhp: dp_num = %d, tnum = %d\n", dp_num, tnum);
    sig += SPX_N;

    /* Layer correctly defaults to 0, so no need to set_layer_addr */
    dev_set_tree_addr(wots_addr, tree);
    dev_set_keypair_addr(wots_addr, idx_leaf);

    // dev_fors_pk_from_sig(root, sig, mhash, pub_seed, wots_addr);
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;

    // const u8 *t_sig = sig;

    dev_copy_keypair_addr(fors_tree_addr, wots_addr);
    dev_copy_keypair_addr(fors_pk_addr, wots_addr);

    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

#ifdef HYBRID_VERIFY_FORS_LOAD_BALANCING
    if (id == 0) {
        memcpy(hp_fors_tree_addr + ttid * 8, fors_tree_addr, 8 * sizeof(int));
        dev_message_to_indices(hp_indices + ttid * SPX_FORS_TREES, mhash);
    }
    // __syncthreads();
    g.sync();

    // if (tid < dp_num * para) {
    uint32_t t_fors_tree_addr[8];
    dev_set_tree_height(t_fors_tree_addr, 0);
    for (u32 i = tid; i < SPX_FORS_TREES * dp_num; i += tnum) {
        u32 tree_idx = (i % SPX_FORS_TREES);
        u32 task_idx = (i / SPX_FORS_TREES);
        const u8* fors_pub_seed = pk + task_idx * SPX_PK_BYTES;
        const u8* fors_sig
            = sm + task_idx * SPX_SM_BYTES + SPX_N + tree_idx * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
        idx_offset = tree_idx * (1 << SPX_FORS_HEIGHT);
        memcpy(t_fors_tree_addr, hp_fors_tree_addr + task_idx * 8, 8 * sizeof(int));

        dev_set_tree_index(t_fors_tree_addr, hp_indices[i] + idx_offset);
        dev_fors_sk_to_leaf(leaf, fors_sig, fors_pub_seed, t_fors_tree_addr);
        fors_sig += SPX_N;

        dev_compute_root(hp_roots + i * SPX_N, leaf, hp_indices[i], idx_offset, fors_sig,
                         SPX_FORS_HEIGHT, fors_pub_seed, t_fors_tree_addr);
    }
    // }

#else  // ifdef HYBRID_VERIFY_FORS_LOAD_BALANCING
    uint32_t indices[SPX_FORS_TREES];
    dev_message_to_indices(indices, mhash);

    if (tid < dp_num * para) {
        for (u32 i = id; i < SPX_FORS_TREES; i += para) {
            const u8* t_sig = sig + i * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
            idx_offset = i * (1 << SPX_FORS_HEIGHT);

            dev_set_tree_height(fors_tree_addr, 0);
            dev_set_tree_index(fors_tree_addr, indices[i] + idx_offset);

            dev_fors_sk_to_leaf(leaf, t_sig, pub_seed, fors_tree_addr);
            t_sig += SPX_N;

            dev_compute_root(hp_roots + ttid * SPX_FORS_TREES * SPX_N + i * SPX_N, leaf, indices[i],
                             idx_offset, t_sig, SPX_FORS_HEIGHT, pub_seed, fors_tree_addr);
        }
    }
#endif // ifdef HYBRID_VERIFY_FORS_LOAD_BALANCING

    // g.sync();
    __syncthreads();
    if (tid < dp_num * para) {
        /* Hash horizontally across all tree roots to derive the public key. */
        dev_thash(root, hp_roots + ttid * SPX_FORS_TREES * SPX_N, SPX_FORS_TREES, pub_seed,
                  fors_pk_addr);
    }
    sig += SPX_FORS_BYTES;

    /* For each subtree.. */
    for (u32 i = 0; i < SPX_D; i++) {
        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_copy_keypair_addr(wots_pk_addr, wots_addr);

        unsigned int lengths[SPX_WOTS_LEN];

        dev_chain_lengths(lengths, root);

#if defined(WOTS_VERIFY_LOAD_BALANCING1)
        __syncthreads();
        if (ttid == 0) {
            u32 ll[SPX_WOTS_LEN] = {0};
            for (int i = 0; i < SPX_WOTS_LEN; i++) {
                int min = 999999;
                int mintid = -1;
                for (int j = 0; j < para; j++) {
                    if (ll[j] < min) {
                        min = ll[j];
                        mintid = j;
                    }
                }
                hp_worktid[i] = mintid;
                ll[mintid] += (SPX_WOTS_W - 1 - lengths[i]);
            }
        }
        __syncthreads();
        for (int i = 0; i < SPX_WOTS_LEN; i++) {
            if (hp_worktid[i] == id) {
                dev_set_chain_addr(wots_addr, i);
                dev_gen_chain(dev_wpk + ttid * SPX_WOTS_BYTES + i * SPX_N, sig + i * SPX_N,
                              lengths[i], SPX_WOTS_W - 1 - lengths[i], pub_seed, wots_addr);
            }
        }
#elif defined(WOTS_VERIFY_LOAD_BALANCING2)
        // required: 8 threads

        __shared__ u8 look[SPX_WOTS_LEN * 4];

        int share_num = 32 / para;
        int sttid = ttid % 4;
        // if (id == 0)
        memset(look + sttid * SPX_WOTS_LEN, 0, SPX_WOTS_LEN);
        // __syncthreads();

        if (tid < dp_num * para) {
            for (u32 j = 0; j < SPX_WOTS_LEN; j++) {
                if (look[j + sttid * SPX_WOTS_LEN] == 0) {
                    look[j + sttid * SPX_WOTS_LEN]++;
                    dev_set_chain_addr(wots_addr, j);
                    dev_gen_chain(dev_wpk + ttid * SPX_WOTS_BYTES + j * SPX_N, sig + j * SPX_N,
                                  lengths[j], SPX_WOTS_W - 1 - lengths[j], pub_seed, wots_addr);
                }
            }
        }
#elif defined(WOTS_VERIFY_LOAD_BALANCING3)
        if (id == 0) {
            memcpy(hp_fors_tree_addr + ttid * 8, wots_addr, 8 * sizeof(int));
            memcpy(hp_length + ttid * SPX_WOTS_LEN, lengths, SPX_WOTS_LEN * sizeof(int));
        }
        g.sync();

        // if (tid < dp_num * para) {
        u32 t_wots_addr[8];
        for (u32 j = tid; j < SPX_WOTS_LEN * dp_num; j += tnum) {
            u32 tree_idx = (j % SPX_WOTS_LEN);
            u32 task_idx = (j / SPX_WOTS_LEN);
            const u32 t_length = hp_length[j];
            const u8* t_pub_seed = pk + task_idx * SPX_PK_BYTES;
            const u8* t_sig = sm + task_idx * SPX_SM_BYTES + SPX_N + SPX_FORS_BYTES
                + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);
            memcpy(t_wots_addr, hp_fors_tree_addr + task_idx * 8, 8 * sizeof(int));
            dev_set_chain_addr(t_wots_addr, tree_idx);
            dev_gen_chain(dev_wpk + j * SPX_N, t_sig + tree_idx * SPX_N, t_length,
                          SPX_WOTS_W - 1 - t_length, t_pub_seed, t_wots_addr);
        }
        // }
        g.sync();

#else // if defined(WOTS_VERIFY_LOAD_BALANCING1)
        if (tid < dp_num * para) {
            for (u32 j = id; j < SPX_WOTS_LEN; j += para) {
                dev_set_chain_addr(wots_addr, j);
                dev_gen_chain(dev_wpk + ttid * SPX_WOTS_BYTES + j * SPX_N, sig + j * SPX_N,
                              lengths[j], SPX_WOTS_W - 1 - lengths[j], pub_seed, wots_addr);
            }
        }

#endif // if defined(WOTS_VERIFY_LOAD_BALANCING1)

        __syncthreads();
        sig += SPX_WOTS_BYTES;

        /* Compute the leaf node using the WOTS public key. */
        dev_thash(leaf, dev_wpk + ttid * SPX_WOTS_BYTES, SPX_WOTS_LEN, pub_seed, wots_pk_addr);

        /* Compute the root node of this subtree. */
        dev_compute_root(root, leaf, idx_leaf, 0, sig, SPX_TREE_HEIGHT, pub_seed, tree_addr);
        sig += SPX_TREE_HEIGHT * SPX_N;

        /* Update the indices for the next layer. */
        idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
        tree = tree >> SPX_TREE_HEIGHT;
    }

    // g.sync();
    if (id == 0) {
        /* only check thread 0 */
        /* Check if the root node equals the root node in the public key. */
        for (int i = 0; i < SPX_N; i++) {
            if (root[i] != pub_root[i]) {
                memset(m + ttid * SPX_MLEN, 0, smlen);
                *mlen = 0;
                if (tid == 0) *right = -1;
                // if (tid == 1) printf("wrong global_mhp_crypto_sign_open\n");
                // printf("wrong global_mhp_crypto_sign_open %d %d %d\n", tid, ttid, id);
                return;
            }
        }

        /* If verification was successful, move the message to the right place. */
        memcpy(m + ttid * SPX_MLEN, sm + ttid * SPX_SM_BYTES + SPX_BYTES, *mlen);

        if (tid == 0) *right = 0;
    }
}

// global sync version
__global__ void global_mhp_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk,
                                            u32 dp_num, int* right) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    u32 para = tnum / dp_num;
    const u32 id = tid % para;
    const u32 ttid = tid / para;
    const u8* sig = sm + (tid / para) * SPX_SM_BYTES;
    const u8* t_pk = pk + (tid / para) * SPX_PK_BYTES;
    const u8* t_m = sm + (tid / para) * SPX_SM_BYTES + SPX_BYTES;
    const u8* pub_seed = t_pk;
    const u8* pub_root = t_pk + SPX_N;
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    u8 leaf[SPX_N];
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    *mlen = smlen - SPX_BYTES;
    if (smlen < SPX_BYTES) {
        memset(m + tid * SPX_MLEN, 0, smlen);
        *mlen = 0;
        if (tid == 0) *right = -1;
        return;
    }

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    dev_initialize_hash_function(pub_seed, NULL);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);

    /* Derive the message digest and leaf index from R || PK || M. */
    /* The additional SPX_N is a result of the hash domain separator. */
    dev_hash_message(mhash, &tree, &idx_leaf, sig, t_pk, t_m, *mlen);

    // if (tid == 0) printf("mhp: dp_num = %d, tnum = %d\n", dp_num, tnum);
    sig += SPX_N;

    /* Layer correctly defaults to 0, so no need to set_layer_addr */
    dev_set_tree_addr(wots_addr, tree);
    dev_set_keypair_addr(wots_addr, idx_leaf);

    // dev_fors_pk_from_sig(root, sig, mhash, pub_seed, wots_addr);
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;

    // const u8 *t_sig = sig;

    dev_copy_keypair_addr(fors_tree_addr, wots_addr);
    dev_copy_keypair_addr(fors_pk_addr, wots_addr);

    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

    if (id == 0) {
        memcpy(hp_fors_tree_addr + ttid * 8, fors_tree_addr, 8 * sizeof(int));
        dev_message_to_indices(hp_indices + ttid * SPX_FORS_TREES, mhash);
    }
    // __syncthreads();
    g.sync();

    uint32_t t_fors_tree_addr[8];
    dev_set_tree_height(t_fors_tree_addr, 0);
    for (u32 i = tid; i < SPX_FORS_TREES * dp_num; i += tnum) {
        u32 tree_idx = (i % SPX_FORS_TREES);
        u32 task_idx = (i / SPX_FORS_TREES);
        const u8* fors_pub_seed = pk + task_idx * SPX_PK_BYTES;
        const u8* fors_sig
            = sm + task_idx * SPX_SM_BYTES + SPX_N + tree_idx * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
        idx_offset = tree_idx * (1 << SPX_FORS_HEIGHT);
        memcpy(t_fors_tree_addr, hp_fors_tree_addr + task_idx * 8, 8 * sizeof(int));

        dev_set_tree_index(t_fors_tree_addr, hp_indices[i] + idx_offset);
        dev_fors_sk_to_leaf(leaf, fors_sig, fors_pub_seed, t_fors_tree_addr);
        fors_sig += SPX_N;

        dev_compute_root(hp_roots + i * SPX_N, leaf, hp_indices[i], idx_offset, fors_sig,
                         SPX_FORS_HEIGHT, fors_pub_seed, t_fors_tree_addr);
    }

    // g.sync();
    __syncthreads();
    if (tid < dp_num * para) {
        /* Hash horizontally across all tree roots to derive the public key. */
        dev_thash(root, hp_roots + ttid * SPX_FORS_TREES * SPX_N, SPX_FORS_TREES, pub_seed,
                  fors_pk_addr);
    }
    sig += SPX_FORS_BYTES;

    /* For each subtree.. */
    for (u32 i = 0; i < SPX_D; i++) {
        dev_set_layer_addr(tree_addr, i);
        dev_set_tree_addr(tree_addr, tree);

        dev_copy_subtree_addr(wots_addr, tree_addr);
        dev_set_keypair_addr(wots_addr, idx_leaf);

        dev_copy_keypair_addr(wots_pk_addr, wots_addr);

        unsigned int lengths[SPX_WOTS_LEN];

        dev_chain_lengths(lengths, root);

        if (id == 0) {
            memcpy(hp_fors_tree_addr + ttid * 8, wots_addr, 8 * sizeof(int));
            memcpy(hp_length + ttid * SPX_WOTS_LEN, lengths, SPX_WOTS_LEN * sizeof(int));
        }
        g.sync();

        u32 t_wots_addr[8];
        for (u32 j = tid; j < SPX_WOTS_LEN * dp_num; j += tnum) {
            u32 tree_idx = (j % SPX_WOTS_LEN);
            u32 task_idx = (j / SPX_WOTS_LEN);
            const u32 t_length = hp_length[j];
            const u8* t_pub_seed = pk + task_idx * SPX_PK_BYTES;
            const u8* t_sig = sm + task_idx * SPX_SM_BYTES + SPX_N + SPX_FORS_BYTES
                + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);
            memcpy(t_wots_addr, hp_fors_tree_addr + task_idx * 8, 8 * sizeof(int));
            dev_set_chain_addr(t_wots_addr, tree_idx);
            dev_gen_chain(dev_wpk + j * SPX_N, t_sig + tree_idx * SPX_N, t_length,
                          SPX_WOTS_W - 1 - t_length, t_pub_seed, t_wots_addr);
        }
        g.sync();
        sig += SPX_WOTS_BYTES;

        /* Compute the leaf node using the WOTS public key. */
        dev_thash(leaf, dev_wpk + ttid * SPX_WOTS_BYTES, SPX_WOTS_LEN, pub_seed, wots_pk_addr);

        /* Compute the root node of this subtree. */
        dev_compute_root(root, leaf, idx_leaf, 0, sig, SPX_TREE_HEIGHT, pub_seed, tree_addr);
        sig += SPX_TREE_HEIGHT * SPX_N;

        /* Update the indices for the next layer. */
        idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
        tree = tree >> SPX_TREE_HEIGHT;
    }

    // g.sync();
    if (id == 0) {
        /* only check thread 0 */
        /* Check if the root node equals the root node in the public key. */
        for (int i = 0; i < SPX_N; i++) {
            if (root[i] != pub_root[i]) {
                memset(m + ttid * SPX_MLEN, 0, smlen);
                *mlen = 0;
                if (tid == 0) *right = -1;
                // if (tid == 1) printf("wrong global_mhp_crypto_sign_open\n");
                return;
            }
        }

        /* If verification was successful, move the message to the right place. */
        memcpy(m + ttid * SPX_MLEN, sm + ttid * SPX_SM_BYTES + SPX_BYTES, *mlen);

        if (tid == 0) *right = 0;
    }
}

__global__ void global_mhp_crypto_sign_open_compare(u8* m, u64* mlen, const u8* sm, u64 smlen,
                                                    const u8* pk, u32 dp_num, int* right) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    u32 para = tnum / dp_num;
    const u32 id = tid % para;
    const u32 ttid = tid / para;
    const u8* sig = sm + (tid / para) * SPX_SM_BYTES;
    const u8* t_pk = pk + (tid / para) * SPX_PK_BYTES;
    const u8* t_m = sm + (tid / para) * SPX_SM_BYTES + SPX_BYTES;
    const u8* pub_seed = t_pk;
    const u8* pub_root = t_pk + SPX_N;
    u8 mhash[SPX_FORS_MSG_BYTES];
    u8 root[SPX_N];
    u8 leaf[SPX_N];
    uint64_t tree;
    uint32_t idx_leaf;
    uint32_t wots_addr[8] = {0};
    uint32_t tree_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    *mlen = smlen - SPX_BYTES;
    if (smlen < SPX_BYTES) {
        memset(m + tid * SPX_MLEN, 0, smlen);
        *mlen = 0;
        if (tid == 0) *right = -1;
        return;
    }

    /* This hook allows the hash function instantiation to do whatever
       preparation or computation it needs, based on the public seed. */
    dev_initialize_hash_function(pub_seed, NULL);

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(tree_addr, SPX_ADDR_TYPE_HASHTREE);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);

    /* Derive the message digest and leaf index from R || PK || M. */
    /* The additional SPX_N is a result of the hash domain separator. */
    // dev_hash_message(mhash, &tree, &idx_leaf, sig, t_pk, t_m, *mlen);

    // if (tid == 0) printf("mhp: dp_num = %d, tnum = %d\n", dp_num, tnum);
    sig += SPX_N;

    /* Layer correctly defaults to 0, so no need to set_layer_addr */
    dev_set_tree_addr(wots_addr, tree);
    dev_set_keypair_addr(wots_addr, idx_leaf);

    // dev_fors_pk_from_sig(root, sig, mhash, pub_seed, wots_addr);
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;

    // const u8 *t_sig = sig;

    dev_copy_keypair_addr(fors_tree_addr, wots_addr);
    dev_copy_keypair_addr(fors_pk_addr, wots_addr);

    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

    if (id == 0) {
        memcpy(hp_fors_tree_addr + ttid * 8, fors_tree_addr, 8 * sizeof(int));
        dev_message_to_indices(hp_indices + ttid * SPX_FORS_TREES, mhash);
    }
    // __syncthreads();
    g.sync();

    uint32_t t_fors_tree_addr[8];
    dev_set_tree_height(t_fors_tree_addr, 0);
    for (u32 i = tid; i < SPX_FORS_TREES * dp_num; i += tnum) {
        u32 tree_idx = (i % SPX_FORS_TREES);
        u32 task_idx = (i / SPX_FORS_TREES);
        const u8* fors_pub_seed = pk + task_idx * SPX_PK_BYTES;
        const u8* fors_sig
            = sm + task_idx * SPX_SM_BYTES + SPX_N + tree_idx * (SPX_N * SPX_FORS_HEIGHT + SPX_N);
        idx_offset = tree_idx * (1 << SPX_FORS_HEIGHT);
        memcpy(t_fors_tree_addr, hp_fors_tree_addr + task_idx * 8, 8 * sizeof(int));

        dev_set_tree_index(t_fors_tree_addr, hp_indices[i] + idx_offset);
        dev_fors_sk_to_leaf(leaf, fors_sig, fors_pub_seed, t_fors_tree_addr);
        fors_sig += SPX_N;

        dev_compute_root(hp_roots + i * SPX_N, leaf, hp_indices[i], idx_offset, fors_sig,
                         SPX_FORS_HEIGHT, fors_pub_seed, t_fors_tree_addr);
    }

    // g.sync();
    __syncthreads();
    if (tid < dp_num * para) {
        /* Hash horizontally across all tree roots to derive the public key. */
        dev_thash(root, hp_roots + ttid * SPX_FORS_TREES * SPX_N, SPX_FORS_TREES, pub_seed,
                  fors_pk_addr);
    }
    sig += SPX_FORS_BYTES;

    /* For each subtree.. */
    // for (u32 i = 0; i < SPX_D; i++) {
    //     dev_set_layer_addr(tree_addr, i);
    //     dev_set_tree_addr(tree_addr, tree);

    //     dev_copy_subtree_addr(wots_addr, tree_addr);
    //     dev_set_keypair_addr(wots_addr, idx_leaf);

    //     dev_copy_keypair_addr(wots_pk_addr, wots_addr);

    //     unsigned int lengths[SPX_WOTS_LEN];

    //     dev_chain_lengths(lengths, root);

    //     if (id == 0) {
    //         memcpy(hp_fors_tree_addr + ttid * 8, wots_addr, 8 * sizeof(int));
    //         memcpy(hp_length + ttid * SPX_WOTS_LEN, lengths, SPX_WOTS_LEN * sizeof(int));
    //     }
    //     g.sync();

    //     u32 t_wots_addr[8];
    //     for (u32 j = tid; j < SPX_WOTS_LEN * dp_num; j += tnum) {
    //         u32 tree_idx = (j % SPX_WOTS_LEN);
    //         u32 task_idx = (j / SPX_WOTS_LEN);
    //         const u32 t_length = hp_length[j];
    //         const u8* t_pub_seed = pk + task_idx * SPX_PK_BYTES;
    //         const u8* t_sig = sm + task_idx * SPX_SM_BYTES + SPX_N + SPX_FORS_BYTES
    //             + i * (SPX_WOTS_BYTES + SPX_TREE_HEIGHT * SPX_N);
    //         memcpy(t_wots_addr, hp_fors_tree_addr + task_idx * 8, 8 * sizeof(int));
    //         dev_set_chain_addr(t_wots_addr, tree_idx);
    //         dev_gen_chain(dev_wpk + j * SPX_N, t_sig + tree_idx * SPX_N, t_length,
    //                       SPX_WOTS_W - 1 - t_length, t_pub_seed, t_wots_addr);
    //     }
    //     g.sync();
    //     sig += SPX_WOTS_BYTES;

    //     /* Compute the leaf node using the WOTS public key. */
    //     dev_thash(leaf, dev_wpk + ttid * SPX_WOTS_BYTES, SPX_WOTS_LEN, pub_seed, wots_pk_addr);

    //     /* Compute the root node of this subtree. */
    //     dev_compute_root(root, leaf, idx_leaf, 0, sig, SPX_TREE_HEIGHT, pub_seed, tree_addr);
    //     sig += SPX_TREE_HEIGHT * SPX_N;

    //     /* Update the indices for the next layer. */
    //     idx_leaf = (tree & ((1 << SPX_TREE_HEIGHT) - 1));
    //     tree = tree >> SPX_TREE_HEIGHT;
    // }

    // g.sync();
    // if (id == 0) {
    //     /* only check thread 0 */
    //     /* Check if the root node equals the root node in the public key. */
    //     for (int i = 0; i < SPX_N; i++) {
    //         if (root[i] != pub_root[i]) {
    //             memset(m + ttid * SPX_MLEN, 0, smlen);
    //             *mlen = 0;
    //             if (tid == 0) *right = -1;
    //             // if (tid == 1) printf("wrong global_mhp_crypto_sign_open\n");
    //             return;
    //         }
    //     }

    //     /* If verification was successful, move the message to the right place. */
    //     memcpy(m + ttid * SPX_MLEN, sm + ttid * SPX_SM_BYTES + SPX_BYTES, *mlen);

    //     if (tid == 0) *right = 0;
    // }
}

int face_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk) {
    u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
    u64* dev_mlen = NULL;
    int right, *dev_right = NULL;
    int device = DEVICE_USED;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_m, SPX_SM_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_mlen, 1 * sizeof(u64)));
    CHECK(cudaMalloc((void**) &dev_right, 1 * sizeof(int)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaMemcpy(dev_sm, sm, SPX_SM_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_pk, pk, SPX_PK_BYTES * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());

    global_crypto_sign_open<<<1, 1>>>(dev_m, dev_mlen, dev_sm, smlen, dev_pk, dev_right);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(m, dev_m, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), D2H));
    CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), D2H));
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_pk);
    cudaFree(dev_mlen);
    cudaFree(dev_right);

    return right;
} // face_crypto_sign_open

int face_ap_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk) {
    u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
    u64* dev_mlen = NULL;
    int right, *dev_right = NULL;
    int device = DEVICE_USED;
    int numBlocksPerSm, maxblocks, maxallthreads;
    cudaDeviceProp deviceProp;
    void* kernelArgs[] = {&dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &dev_right};
    // u32 threads = 32;
    // u32 t_for = SPX_FORS_TREES / 2 + 1;
    // u32 t_spx = SPX_WOTS_LEN;
    // // 2(35, 51) or 3 (67)
    // u32 blocks = (t_for > t_spx ? t_for : t_spx) / threads + 1;
    u32 threads = SPX_WOTS_LEN;
    u32 blocks = 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_open,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;

    CHECK(cudaMalloc((void**) &dev_m, SPX_SM_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_mlen, 1 * sizeof(u64)));
    CHECK(cudaMalloc((void**) &dev_right, 1 * sizeof(int)));

    if (g_count == 0)
        printf("blocks, threads: %d * %d\tall = %d\tmax = %d\t", blocks, threads, threads * blocks,
               maxallthreads);
    g_count++;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaMemcpy(dev_sm, sm, SPX_SM_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_pk, pk, SPX_PK_BYTES * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_crypto_sign_open, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(m, dev_m, SPX_SM_BYTES * sizeof(u8), D2H));
    CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), D2H));
    CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), D2H));
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_pk);
    cudaFree(dev_mlen);
    cudaFree(dev_right);

    return right;
}

int face_mdp_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk, u32 dp_num) {
    struct timespec start, stop, b2, e2;
    double result;
    u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
    u64* dev_mlen = NULL;
    int right, *dev_right = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

#ifdef VERIFY_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    maxblocks = maxallthreads / threads;
    if (maxallthreads % threads != 0) printf("wrong in dp threads\n");
#else  // ifdef KEYGEN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_dp_crypto_sign_keypair,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef KEYGEN_SUITBLE_BLOCK

    if (dp_num < maxallthreads)
        malloc_size = dp_num / threads * threads + threads;
    else
        malloc_size = maxallthreads;

    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_mlen, 1 * sizeof(u64)));
    CHECK(cudaMalloc((void**) &dev_right, 1 * sizeof(int)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = dp_num / maxallthreads + (dp_num % maxallthreads ? 1 : 0);
    u32 left = dp_num;

    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = dp_num / loop;
        int r = dp_num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        printf("mdp verify: maxblocks, blocks, threads, s: %u %u %u %u\n", maxblocks, blocks,
               threads, s);
#endif // ifdef PRINT_ALL

        void* Args[] = {&dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &s, &dev_right};

        CHECK(cudaMemcpy(dev_pk, pk, s * SPX_PK_BYTES * sizeof(u8), H2D));
        CHECK(cudaMemcpy(dev_sm, sm, s * SPX_SM_BYTES * sizeof(u8), H2D));

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &b2);
        CHECK(cudaDeviceSynchronize());
        cudaLaunchCooperativeKernel((void*) global_mdp_crypto_sign_open, blocks, threads, Args);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &e2);
        result = (e2.tv_sec - b2.tv_sec) * 1e6 + (e2.tv_nsec - b2.tv_nsec) / 1e3;
        g_inner_result += result;

        CHECK(cudaMemcpy(m, dev_m, s * SPX_MLEN * sizeof(u8), D2H));
        pk += s * SPX_PK_BYTES;
        sm += s * SPX_SM_BYTES;
        m += s * SPX_MLEN;
        left -= s;
    }
    CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), D2H));
    CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_mlen));
    CHECK(cudaFree(dev_right));

    return right;
}

int face_mgpu_mdp_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk,
                                   u32 dp_num) {
    struct timespec start, stop, b2, e2;
    double result;
    int ngpu = 2;
    u8 *dev_m[ngpu], *dev_sm[ngpu], *dev_pk[ngpu];
    u64* dev_mlen[ngpu];
    int right, *dev_right[ngpu];
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxblocks, maxallthreads;
    dp_num /= ngpu;

    cudaGetDeviceProperties(&deviceProp, 0); // device 0

#ifdef VERIFY_SUITBLE_BLOCK
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    maxblocks = maxallthreads / threads;
    if (maxallthreads % threads != 0) printf("wrong in dp threads\n");
#else  // ifdef KEYGEN_SUITBLE_BLOCK
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_dp_crypto_sign_keypair,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
#endif // ifdef KEYGEN_SUITBLE_BLOCK

    if (dp_num < maxallthreads)
        malloc_size = dp_num / threads * threads + threads;
    else
        malloc_size = maxallthreads;

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaMalloc((void**) &dev_m[j], SPX_MLEN * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_sm[j], SPX_SM_BYTES * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_pk[j], SPX_PK_BYTES * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_mlen[j], 1 * sizeof(u64)));
        CHECK(cudaMalloc((void**) &dev_right[j], 1 * sizeof(int)));
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = dp_num / maxallthreads + (dp_num % maxallthreads ? 1 : 0);
    u32 left = dp_num;

    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = dp_num / loop;
        int r = dp_num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        printf("mdp verify: maxblocks, blocks, threads, s: %u %u %u %u\n", maxblocks, blocks,
               threads, s);
#endif // ifdef PRINT_ALL

        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpy(dev_pk[j], pk + j * s * SPX_PK_BYTES, s * SPX_PK_BYTES * sizeof(u8),
                             H2D));
            CHECK(cudaMemcpy(dev_sm[j], sm + j * s * SPX_SM_BYTES, s * SPX_SM_BYTES * sizeof(u8),
                             H2D));
        }

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &b2);
        CHECK(cudaDeviceSynchronize());

        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            void* Args[]
                = {&dev_m[j], &dev_mlen[j], &dev_sm[j], &smlen, &dev_pk[j], &s, &dev_right[j]};
            cudaLaunchCooperativeKernel((void*) global_mdp_crypto_sign_open, blocks, threads, Args);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &e2);
        result = (e2.tv_sec - b2.tv_sec) * 1e6 + (e2.tv_nsec - b2.tv_nsec) / 1e3;
        g_inner_result += result;

        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpy(m + j * s * SPX_MLEN, dev_m[j], s * SPX_MLEN * sizeof(u8), D2H));
        }
        pk += ngpu * s * SPX_PK_BYTES;
        sm += ngpu * s * SPX_SM_BYTES;
        m += ngpu * s * SPX_MLEN;
        left -= s;
    }
    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaMemcpy(mlen, dev_mlen[j], 1 * sizeof(u64), D2H));
        CHECK(cudaMemcpy(&right, dev_right[j], 1 * sizeof(int), D2H));
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        cudaFree(dev_m[j]);
        cudaFree(dev_sm[j]);
        cudaFree(dev_pk[j]);
        cudaFree(dev_mlen[j]);
        cudaFree(dev_right[j]);
    }

    return right;
}

int face_ms_mdp_crypto_sign_open(u8* m, u64* mlen, u8* sm, u64 smlen, u8* pk, u32 num) {
    struct timespec start, stop;
    double result;
    u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
    u64* dev_mlen = NULL;
    int right, *dev_right = NULL;
    int device = DEVICE_USED;
    int threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

    malloc_size = num / threads * threads + (num % threads ? threads : 0);
    // printf("size = %d\n", malloc_size);

    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_mlen, 1 * sizeof(u64)));
    CHECK(cudaMalloc((void**) &dev_right, 1 * sizeof(int)));

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef USING_STREAM_1
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mdp_crypto_sign_open,
                                                  threads, 0);
    maxallthreads = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
#endif // ifdef USING_STREAM_1

    int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;

    cudaStream_t stream[loop];
    u8 *p_pk[loop], *p_m[loop], *p_sm[loop], *p_dev_pk[loop], *p_dev_m[loop], *p_dev_sm[loop];

    for (int i = 0; i < loop; ++i) {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    u32 sum = 0;
    u32 ss[loop], ssum[loop], bblocks[loop];
    for (u32 i = 0; i < loop; i++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxallthreads > left) {
            s = left;
            bblocks[i] = s / threads + (s % threads ? 1 : 0);
        } else {
            bblocks[i] = maxallthreads / threads;
            s = maxallthreads;
        }
        ssum[i] = sum;
        ss[i] = s;
        sum += s;
        left -= s;
#else  // if LARGE_SCHEME == 1
        int q = num / loop;
        int r = num % loop;
        int s = q + ((i < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        printf("i = %d, loop %d\n", i, loop);
        printf("ms mdp verify: blocks, threads: %u %u\n", bblocks[i], threads);
#endif // ifdef PRINT_ALL

        p_pk[i] = pk + ssum[i] * SPX_PK_BYTES;
        p_sm[i] = sm + ssum[i] * SPX_SM_BYTES;
        p_m[i] = m + ssum[i] * SPX_MLEN;
        p_dev_pk[i] = dev_pk + ssum[i] * SPX_PK_BYTES;
        p_dev_sm[i] = dev_sm + ssum[i] * SPX_SM_BYTES;
        p_dev_m[i] = dev_m + ssum[i] * SPX_MLEN;
    }

    for (int i = 0; i < loop; i++) {
        CHECK(cudaMemcpyAsync(p_dev_pk[i], p_pk[i], ss[i] * SPX_PK_BYTES * sizeof(u8), H2D,
                              stream[i]));
        CHECK(cudaMemcpyAsync(p_dev_sm[i], p_sm[i], ss[i] * SM_BYTES * sizeof(u8), H2D, stream[i]));
        // CHECK(cudaGetLastError());
        // CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < loop; i++) {
        void* Args[]
            = {&p_dev_m[i], &dev_mlen, &p_dev_sm[i], &smlen, &p_dev_pk[i], &ss[i], &dev_right};
        cudaLaunchCooperativeKernel((void*) global_mdp_crypto_sign_open, bblocks[i], threads, Args,
                                    0, stream[i]);
        // CHECK(cudaGetLastError());
        // CHECK(cudaDeviceSynchronize());
        // printf("i = %d, ss[i]= %d, bblock[i] = %d\n", i, ss[i], bblocks[i]);
    }

    for (int i = 0; i < loop; i++) {
        CHECK(cudaMemcpyAsync(p_m[i], p_dev_m[i], ss[i] * SPX_MLEN * sizeof(u8), D2H, stream[i]));
        // CHECK(cudaGetLastError());
        // CHECK(cudaDeviceSynchronize());
        // printf("i = %d,\n", i);
    }
    // printf("3\n");

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), D2H));
    CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    for (int i = 0; i < loop; i++) {
        CHECK(cudaStreamDestroy(stream[i]));
    }

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_mlen));
    CHECK(cudaFree(dev_right));

    return right;
}

int face_mgpu_ms_mdp_crypto_sign_open(u8* m, u64* mlen, u8* sm, u64 smlen, u8* pk, u32 num) {
    struct timespec start, stop;
    double result;
    int ngpu = 2;
    u8 *dev_m[ngpu], *dev_sm[ngpu], *dev_pk[ngpu];
    u64* dev_mlen[ngpu];
    int right, *dev_right[ngpu];
    int threads = 32;
    cudaDeviceProp deviceProp;
    int malloc_size;
    int maxallthreads;
    num /= ngpu;

    cudaGetDeviceProperties(&deviceProp, 0);

    malloc_size = num / threads * threads + (num % threads ? threads : 0);
    // printf("size = %d\n", malloc_size);

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaMalloc((void**) &dev_m[j], SPX_MLEN * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_sm[j], SPX_SM_BYTES * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_pk[j], SPX_PK_BYTES * malloc_size * sizeof(u8)));
        CHECK(cudaMalloc((void**) &dev_mlen[j], 1 * sizeof(u64)));
        CHECK(cudaMalloc((void**) &dev_right[j], 1 * sizeof(int)));
    }

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef USING_STREAM_1
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mdp_crypto_sign_open,
                                                  threads, 0);
    maxallthreads = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
#endif // ifdef USING_STREAM_1

    int loop = num / maxallthreads + (num % maxallthreads ? 1 : 0);
    u32 left = num;

    cudaStream_t stream[ngpu][loop];
    u8 *p_pk[ngpu][loop], *p_m[ngpu][loop], *p_sm[ngpu][loop];
    u8 *p_dev_pk[ngpu][loop], *p_dev_m[ngpu][loop], *p_dev_sm[ngpu][loop];

    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaStreamCreate(&stream[j][i]));
        }
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    u32 sum = 0;
    u32 ss[loop], ssum[loop], bblocks[loop];
    for (u32 i = 0; i < loop; i++) {
        u32 s;
        if (maxallthreads > left) {
            s = left;
            bblocks[i] = s / threads + (s % threads ? 1 : 0);
        } else {
            bblocks[i] = maxallthreads / threads;
            s = maxallthreads;
        }
        ssum[i] = sum;
        ss[i] = s;
        sum += s;
        left -= s;
#ifdef PRINT_ALL
        printf("i = %d, loop %d\n", i, loop);
        printf("ms mdp verify: blocks, threads: %u %u\n", bblocks[i], threads);
#endif // ifdef PRINT_ALL
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            p_pk[j][i] = pk + j * s * SPX_PK_BYTES + ngpu * ssum[i] * SPX_PK_BYTES;
            p_sm[j][i] = sm + j * s * SPX_SM_BYTES + ngpu * ssum[i] * SPX_SM_BYTES;
            p_m[j][i] = m + j * s * SPX_MLEN + ngpu * ssum[i] * SPX_MLEN;
            p_dev_pk[j][i] = dev_pk[j] + ssum[i] * SPX_PK_BYTES;
            p_dev_sm[j][i] = dev_sm[j] + ssum[i] * SPX_SM_BYTES;
            p_dev_m[j][i] = dev_m[j] + ssum[i] * SPX_MLEN;
        }
    }

    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpyAsync(p_dev_pk[j][i], p_pk[j][i], ss[i] * SPX_PK_BYTES * sizeof(u8),
                                  H2D, stream[j][i]));
            CHECK(cudaMemcpyAsync(p_dev_sm[j][i], p_sm[j][i], ss[i] * SM_BYTES * sizeof(u8), H2D,
                                  stream[j][i]));
        }
        // CHECK(cudaGetLastError());
        // CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            void* Args[] = {&p_dev_m[j][i],  &dev_mlen[j], &p_dev_sm[j][i], &smlen,
                            &p_dev_pk[j][i], &ss[i],       &dev_right[j]};
            cudaLaunchCooperativeKernel((void*) global_mdp_crypto_sign_open, bblocks[i], threads,
                                        Args, 0, stream[j][i]);
        }
        // CHECK(cudaGetLastError());
        // CHECK(cudaDeviceSynchronize());
        // printf("i = %d, ss[i]= %d, bblock[i] = %d\n", i, ss[i], bblocks[i]);
    }

    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaMemcpyAsync(p_m[j][i], p_dev_m[j][i], ss[i] * SPX_MLEN * sizeof(u8), D2H,
                                  stream[j][i]));
        }
        // CHECK(cudaGetLastError());
        // CHECK(cudaDeviceSynchronize());
        // printf("i = %d,\n", i);
    }
    // printf("3\n");

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaMemcpy(mlen, dev_mlen[j], 1 * sizeof(u64), D2H));
        CHECK(cudaMemcpy(&right, dev_right[j], 1 * sizeof(int), D2H));
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    for (int i = 0; i < loop; i++) {
        for (int j = 0; j < ngpu; j++) {
            CHECK(cudaSetDevice(j));
            CHECK(cudaStreamCreate(&stream[j][i]));
        }
    }

    for (int j = 0; j < ngpu; j++) {
        CHECK(cudaSetDevice(j));
        CHECK(cudaFree(dev_m[j]));
        CHECK(cudaFree(dev_sm[j]));
        CHECK(cudaFree(dev_pk[j]));
        CHECK(cudaFree(dev_mlen[j]));
        CHECK(cudaFree(dev_right[j]));
    }

    return right;
}

int face_sdp_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk, u32 dp_num) {
    struct timespec start, stop;
    double result;
    u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
    u64* dev_mlen = NULL;
    int right, *dev_right = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int malloc_size;
    int maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mdp_crypto_sign_open,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
    if (dp_num < maxallthreads)
        malloc_size = dp_num / threads * threads + threads;
    else
        malloc_size = maxallthreads;

    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_mlen, 1 * sizeof(u64)));
    CHECK(cudaMalloc((void**) &dev_right, 1 * sizeof(int)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaMemcpy(dev_pk, pk, SPX_PK_BYTES * sizeof(u8), H2D));

    int loop = dp_num / maxallthreads + (dp_num % maxallthreads ? 1 : 0);
    u32 left = dp_num;

    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxblocks * threads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxblocks;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = dp_num / loop;
        int r = dp_num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        printf("mdp verify: maxblocks, blocks, threads: %u %u %u\n", maxblocks, blocks, threads);
#endif // ifdef PRINT_ALL

        void* Args[] = {&dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &s, &dev_right};

        CHECK(cudaMemcpy(dev_sm, sm, s * SPX_SM_BYTES * sizeof(u8), H2D));

        CHECK(cudaDeviceSynchronize());
        cudaLaunchCooperativeKernel((void*) global_sdp_crypto_sign_open, blocks, threads, Args);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(m, dev_m, s * SPX_MLEN * sizeof(u8), D2H));
        sm += s * SPX_SM_BYTES;
        m += s * SPX_MLEN;
        left -= s;
    }
    CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), D2H));
    CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    cudaFree(dev_m);
    cudaFree(dev_sm);
    cudaFree(dev_pk);
    cudaFree(dev_mlen);
    cudaFree(dev_right);

    return right;
} // face_sdp_crypto_sign_open

int face_ms_sdp_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk,
                                 u32 dp_num) {
    struct timespec start, stop;
    double result;
    u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
    u64* dev_mlen = NULL;
    int right, *dev_right = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm;
    int malloc_size;
    int maxblocks, maxallthreads;
    // for free
    u8* free_m = dev_m;
    u8* free_sm = dev_sm;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_sdp_crypto_sign_open,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;
    malloc_size = dp_num / threads * threads + threads;

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * 32;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else  // ifdef USING_STREAM_1
       // Remain the same
#endif // ifdef USING_STREAM_1

    int loop = dp_num / maxallthreads + (dp_num % maxallthreads ? 1 : 0);
    u32 left = dp_num;

    cudaStream_t stream[loop];

    for (int i = 0; i < loop; ++i) {
        CHECK(cudaStreamCreate(&stream[i]));
    }

    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_mlen, 1 * sizeof(u64)));
    CHECK(cudaMalloc((void**) &dev_right, 1 * sizeof(int)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaMemcpy(dev_pk, pk, SPX_PK_BYTES * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());
    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        u32 s;
        if (maxallthreads > left) {
            s = left;
            blocks = s / threads + (s % threads ? 1 : 0);
        } else {
            blocks = maxallthreads / threads;
            s = maxallthreads;
        }
#else  // if LARGE_SCHEME == 1
        int q = dp_num / loop;
        int r = dp_num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        printf("ms mdp verify: maxblocks, blocks, threads: %u %u %u\n", maxblocks, blocks, threads);
#endif // ifdef PRINT_ALL

        void* Args[] = {&dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &s, &dev_right};

        CHECK(cudaMemcpyAsync(dev_sm, sm, s * SM_BYTES * sizeof(u8), H2D, stream[iter]));

        cudaLaunchCooperativeKernel((void*) global_sdp_crypto_sign_open, blocks, threads, Args, 0,
                                    stream[iter]);

        CHECK(cudaMemcpyAsync(m, dev_m, s * SPX_MLEN * sizeof(u8), D2H, stream[iter]));
        sm += s * SPX_SM_BYTES;
        m += s * SPX_MLEN;
        dev_sm += s * SPX_SM_BYTES;
        dev_m += s * SPX_MLEN;
        left -= s;
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), D2H));
    CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    cudaFree(dev_pk);
    cudaFree(free_sm);
    cudaFree(free_m);
    cudaFree(dev_mlen);
    cudaFree(dev_right);

    return right;
} // face_ms_sdp_crypto_sign_open

int face_mhp_crypto_sign_open(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk, u32 dp_num,
                              u32 intra_para) {
    struct timespec start, stop;
    double result;
    u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
    u64* dev_mlen = NULL;
    int right, *dev_right = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm, maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mhp_crypto_sign_open,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;

    blocks = dp_num * intra_para / threads;
    if (blocks == 0) blocks = 1;
    if (blocks * threads > maxallthreads) blocks = maxallthreads / threads;

    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_mlen, 1 * sizeof(u64)));
    CHECK(cudaMalloc((void**) &dev_right, 1 * sizeof(int)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

#ifdef PRINT_ALL
    printf("hp verify: maxblocks, blocks, threads: %u %u %u\n", maxblocks, blocks, threads);
#endif // ifdef PRINT_ALL

    void* Args[] = {&dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &dp_num, &dev_right};

    CHECK(cudaMemcpy(dev_pk, pk, dp_num * SPX_PK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_sm, sm, dp_num * SPX_SM_BYTES * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_mhp_crypto_sign_open, blocks, threads, Args);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(m, dev_m, dp_num * SPX_MLEN * sizeof(u8), D2H));

    CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), D2H));
    CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_mlen));
    CHECK(cudaFree(dev_right));

    return right;
}

int face_mhp_crypto_sign_open_compare(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk,
                                      u32 dp_num, u32 intra_para) {
    struct timespec start, stop;
    double result;
    u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
    u64* dev_mlen = NULL;
    int right, *dev_right = NULL;
    int device = DEVICE_USED;
    int blocks = 1, threads = 32;
    cudaDeviceProp deviceProp;
    int numBlocksPerSm, maxblocks, maxallthreads;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_mhp_crypto_sign_open,
                                                  threads, 0);
    maxblocks = numBlocksPerSm * deviceProp.multiProcessorCount;
    maxallthreads = maxblocks * threads;

    blocks = dp_num * intra_para / threads;
    if (blocks == 0) blocks = 1;
    if (blocks * threads > maxallthreads) blocks = maxallthreads / threads;

    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * dp_num * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_mlen, 1 * sizeof(u64)));
    CHECK(cudaMalloc((void**) &dev_right, 1 * sizeof(int)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

#ifdef PRINT_ALL
    printf("hp verify: maxblocks, blocks, threads: %u %u %u\n", maxblocks, blocks, threads);
#endif // ifdef PRINT_ALL

    void* Args[] = {&dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &dp_num, &dev_right};

    CHECK(cudaMemcpy(dev_pk, pk, dp_num * SPX_PK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_sm, sm, dp_num * SPX_SM_BYTES * sizeof(u8), H2D));

    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_mhp_crypto_sign_open_compare, blocks, threads, Args);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(m, dev_m, dp_num * SPX_MLEN * sizeof(u8), D2H));

    CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), D2H));
    CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_mlen));
    CHECK(cudaFree(dev_right));

    return right;
}

// 线程数可以小于35，都要试试
int face_mhp_sign_open_seperate(u8* m, u64* mlen, const u8* sm, u64 smlen, const u8* pk, u32 num) {
    struct timespec start, stop;
    double result;
    u8 *dev_m = NULL, *dev_sm = NULL, *dev_pk = NULL;
    u64* dev_mlen = NULL;
    int right, *dev_right = NULL;
    int device = DEVICE_USED;
    int blocks = num, threads = SPX_WOTS_LEN;
    // threads = SPX_WOTS_LEN / 32 * 32 + (SPX_WOTS_LEN % 32 ? 32 : 0);
    cudaDeviceProp deviceProp;
    int malloc_size, maxallthreads, maxblocks, numBlocksPerSm;

    CHECK(cudaSetDevice(device));
    cudaGetDeviceProperties(&deviceProp, device);

    malloc_size = num / threads * threads + (num % threads ? threads : 0);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_open,
                                                  threads, 0);

#if USING_STREAM == 1
    maxallthreads = deviceProp.multiProcessorCount * threads;
#elif USING_STREAM == 2
    maxallthreads
        = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
#else                                    // ifdef USING_STREAM_1
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, global_ap_crypto_sign_open,
                                                  threads, 0);
    maxallthreads = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
#endif                                   // ifdef USING_STREAM_1
    maxblocks = maxallthreads / threads; // 向下取整，避免超过上限

    CHECK(cudaMalloc((void**) &dev_m, SPX_MLEN * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sm, SPX_SM_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * malloc_size * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_mlen, 1 * sizeof(u64)));
    CHECK(cudaMalloc((void**) &dev_right, 1 * sizeof(int)));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    int loop = num / maxblocks + (num % maxblocks ? 1 : 0);
    u32 left = num;

    for (u32 iter = 0; iter < loop; iter++) {
#if LARGE_SCHEME == 1
        // u32 s;
        if (maxblocks > left) {
            blocks = left;
        } else {
            blocks = maxblocks;
        }
        left -= maxblocks;
#else  // if LARGE_SCHEME == 1
        int q = num / loop;
        int r = num % loop;
        int s = q + ((iter < r) ? 1 : 0);
        blocks = s / threads + (s % threads ? 1 : 0);
#endif // if LARGE_SCHEME == 1
#ifdef PRINT_ALL
        int max = numBlocksPerSm * deviceProp.multiProcessorCount * threads;
        int need_threads = threads * num;
        printf("seperate hp verify: max, need_threads, blocks, threads, this: %d %u %u %u %u\n",
               max, need_threads, blocks, threads, blocks * threads);
#endif // ifdef PRINT_ALL

        void* Args[] = {&dev_m, &dev_mlen, &dev_sm, &smlen, &dev_pk, &dev_right};
        // blocks = num * HP_PARALLELISM / threads;

        CHECK(cudaMemcpy(dev_pk, pk, blocks * SPX_PK_BYTES * sizeof(u8), H2D));
        CHECK(cudaMemcpy(dev_sm, sm, blocks * SPX_SM_BYTES * sizeof(u8), H2D));

        CHECK(cudaDeviceSynchronize());
        cudaLaunchCooperativeKernel((void*) global_ap_crypto_sign_open, blocks, threads, Args);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaMemcpy(m, dev_m, blocks * SPX_MLEN * sizeof(u8), D2H));
        pk += blocks * SPX_PK_BYTES;
        sm += blocks * SPX_SM_BYTES;
        m += blocks * SPX_MLEN;
    }
    CHECK(cudaMemcpy(mlen, dev_mlen, 1 * sizeof(u64), D2H));
    CHECK(cudaMemcpy(&right, dev_right, 1 * sizeof(int), D2H));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    CHECK(cudaFree(dev_m));
    CHECK(cudaFree(dev_sm));
    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_mlen));
    CHECK(cudaFree(dev_right));

    return right;
}

__global__ void global_thash(u8* out, u8* in, unsigned int l, int loop_num) {
    u8 seed[SPX_N];
    u32 addr[8];
    memcpy(seed, in, SPX_N);

    for (int i = 0; i < loop_num; i++)
        dev_thash(out, in, l, seed, addr);
}

__global__ void global_prf(u8* out, u8* seed, int loop_num) {
    u32 addr[8];

    for (int i = 0; i < loop_num; i++)
        dev_prf_addr(out, seed, addr);
}

__global__ void global_prf_msg(u8* R, u8* seed, u8* optrand, u8* m, int loop_num) {

    for (int i = 0; i < loop_num; i++)
        dev_gen_message_random(R, seed, optrand, m, SPX_MLEN);
}

__global__ void global_h_msg(u8* mhash, u8* R, u8* pk, u8* m, int loop_num) {
    uint64_t tree;
    uint32_t idx_leaf;
    for (int i = 0; i < loop_num; i++)
        dev_hash_message(mhash, &tree, &idx_leaf, R, pk, m, SPX_MLEN);
}

int face_tl(int l, int loop_num) {
    struct timespec start, stop;
    int device = DEVICE_USED;
    u8 in[l * SPX_N];
    u8 out[l * SPX_N];
    memset(in, 1, SPX_N * l);

    u8 *dev_out = NULL, *dev_in = NULL;

    CHECK(cudaSetDevice(device));
    CHECK(cudaMalloc((void**) &dev_in, SPX_N * l * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_out, SPX_N * l * sizeof(u8)));

    CHECK(cudaMemcpy(dev_in, in, SPX_N * l * sizeof(u8), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_thash<<<1, 1>>>(dev_out, dev_in, l, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    CHECK(cudaMemcpy(out, dev_out, SPX_N * l * sizeof(u8), D2H));

    CHECK(cudaFree(dev_out));
    CHECK(cudaFree(dev_in));

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    return 0;
}

int face_h(int loop_num) {
    face_tl(2, loop_num);

    return 0;
}

int face_f(int loop_num) {
    face_tl(1, loop_num);

    return 0;
}

int face_prf(int loop_num) {
    struct timespec start, stop;
    int device = DEVICE_USED;
    u8 seed[SPX_N];
    u8 out[128 * SPX_N];
    memset(seed, 1, SPX_N);

    u8 *dev_out = NULL, *dev_seed = NULL;

    CHECK(cudaSetDevice(device));
    CHECK(cudaMalloc((void**) &dev_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_out, SPX_N * 128 * sizeof(u8)));

    CHECK(cudaMemcpy(dev_seed, seed, SPX_N * sizeof(u8), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_prf<<<1, 1>>>(dev_out, dev_seed, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    CHECK(cudaMemcpy(out, dev_out, SPX_N * 128 * sizeof(u8), D2H));

    CHECK(cudaFree(dev_out));
    CHECK(cudaFree(dev_seed));

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    return 0;
}

int face_prf_msg(int loop_num) {
    struct timespec start, stop;
    int device = DEVICE_USED;
    u8 R[SPX_N];
    u8 seed[SPX_N];
    u8 optrand[SPX_N];
    u8 m[SPX_N * 16];
    memset(seed, 1, SPX_N);
    memset(optrand, 1, SPX_N);
    memset(m, 1, SPX_N * 16);

    u8 *dev_R = NULL, *dev_seed = NULL, *dev_optrand = NULL, *dev_m = NULL;

    CHECK(cudaSetDevice(device));
    CHECK(cudaMalloc((void**) &dev_R, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_optrand, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_N * 16 * sizeof(u8)));

    CHECK(cudaMemcpy(dev_seed, seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_optrand, optrand, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_N * 16 * sizeof(u8), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_prf_msg<<<1, 1>>>(dev_R, dev_seed, dev_optrand, dev_m, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    CHECK(cudaMemcpy(R, dev_R, SPX_N * sizeof(u8), D2H));

    CHECK(cudaFree(dev_R));
    CHECK(cudaFree(dev_seed));
    CHECK(cudaFree(dev_optrand));
    CHECK(cudaFree(dev_m));

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    return 0;
}

int face_h_msg(int loop_num) {
    struct timespec start, stop;
    int device = DEVICE_USED;
    u8 mhash[SPX_N];
    u8 R[SPX_N];
    u8 pk[SPX_PK_BYTES];
    u8 m[SPX_N];
    memset(R, 1, SPX_N);
    memset(pk, 1, SPX_PK_BYTES);
    memset(m, 1, SPX_N);

    u8 *dev_mhash = NULL, *dev_R = NULL, *dev_pk = NULL, *dev_m = NULL;

    CHECK(cudaSetDevice(device));
    CHECK(cudaMalloc((void**) &dev_mhash, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_R, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pk, SPX_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_m, SPX_N * sizeof(u8)));

    CHECK(cudaMemcpy(dev_R, R, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_pk, pk, SPX_PK_BYTES * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_m, m, SPX_N * sizeof(u8), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_h_msg<<<1, 1>>>(dev_mhash, dev_R, dev_pk, dev_m, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    CHECK(cudaMemcpy(mhash, dev_mhash, SPX_N * sizeof(u8), D2H));

    CHECK(cudaFree(dev_mhash));
    CHECK(cudaFree(dev_R));
    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_m));

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    return 0;
}