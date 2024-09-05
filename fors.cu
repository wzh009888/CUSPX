#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

#include "all_option.h"

#include "address.h"
#include "fors.h"
#include "hash.h"
#include "thash.h"
#include "utils.h"

#include <cooperative_groups.h>

#if SPX_FORS_TREES > 32
__device__ unsigned char fors_roots[SPX_FORS_TREES * SPX_N];
#endif // if SPX_FORS_TREES > 32

extern __device__ u8 dev_leaf[SPX_N * 1024 * 1024 * 44];        // 存放xmss fors的叶节点
__device__ u8 dev_fors_auth_path[SPX_FORS_HEIGHT * SPX_N * SPX_FORS_TREES];

static void fors_gen_sk(unsigned char* sk, const unsigned char* sk_seed,
                        uint32_t fors_leaf_addr[8]) {
    prf_addr(sk, sk_seed, fors_leaf_addr);
} // fors_gen_sk

__device__ void dev_fors_gen_sk(unsigned char* sk, const unsigned char* sk_seed,
                                uint32_t fors_leaf_addr[8]) {
    dev_prf_addr(sk, sk_seed, fors_leaf_addr);
} // fors_gen_sk

static void fors_sk_to_leaf(unsigned char* leaf, const unsigned char* sk,
                            const unsigned char* pub_seed, uint32_t fors_leaf_addr[8]) {
    thash(leaf, sk, 1, pub_seed, fors_leaf_addr);
} // fors_sk_to_leaf

__device__ void dev_fors_sk_to_leaf(unsigned char* leaf, const unsigned char* sk,
                                    const unsigned char* pub_seed, uint32_t fors_leaf_addr[8]) {
    dev_thash(leaf, sk, 1, pub_seed, fors_leaf_addr);
} // fors_sk_to_leaf

static void fors_gen_leaf(unsigned char* leaf, const unsigned char* sk_seed,
                          const unsigned char* pub_seed, uint32_t addr_idx,
                          const uint32_t fors_tree_addr[8]) {
    uint32_t fors_leaf_addr[8] = {0};

    /* Only copy the parts that must be kept in fors_leaf_addr. */
    copy_keypair_addr(fors_leaf_addr, fors_tree_addr);
    set_type(fors_leaf_addr, SPX_ADDR_TYPE_FORSTREE);
    set_tree_index(fors_leaf_addr, addr_idx);

    fors_gen_sk(leaf, sk_seed, fors_leaf_addr);
    fors_sk_to_leaf(leaf, leaf, pub_seed, fors_leaf_addr);
} // fors_gen_leaf

__device__ void dev_fors_gen_leaf(unsigned char* leaf, const unsigned char* sk_seed,
                                  const unsigned char* pub_seed, uint32_t addr_idx,
                                  const uint32_t fors_tree_addr[8]) {
    uint32_t fors_leaf_addr[8] = {0};

    /* Only copy the parts that must be kept in fors_leaf_addr. */
    dev_copy_keypair_addr(fors_leaf_addr, fors_tree_addr);
    dev_set_type(fors_leaf_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_tree_index(fors_leaf_addr, addr_idx);

    dev_fors_gen_sk(leaf, sk_seed, fors_leaf_addr);
    dev_fors_sk_to_leaf(leaf, leaf, pub_seed, fors_leaf_addr);
} // fors_gen_leaf

/**
 * Interprets m as SPX_FORS_HEIGHT-bit unsigned integers.
 * Assumes m contains at least SPX_FORS_HEIGHT * SPX_FORS_TREES bits.
 * Assumes indices has space for SPX_FORS_TREES integers.
 */
static void message_to_indices(uint32_t* indices, const unsigned char* m) {
    unsigned int i, j;
    unsigned int offset = 0;

    for (i = 0; i < SPX_FORS_TREES; i++) {
        indices[i] = 0;
        for (j = 0; j < SPX_FORS_HEIGHT; j++) {
            indices[i] ^= ((m[offset >> 3] >> (offset & 0x7)) & 0x1) << j;
            offset++;
        }
    }
} // message_to_indices

__device__ void dev_message_to_indices(uint32_t* indices, const u8* m) {
    unsigned int i, j;
    unsigned int offset = 0;

    for (i = 0; i < SPX_FORS_TREES; i++) {
        indices[i] = 0;
        for (j = 0; j < SPX_FORS_HEIGHT; j++) {
            indices[i] ^= ((m[offset >> 3] >> (offset & 0x7)) & 0x1) << j;
            offset++;
        }
    }
} // dev_message_to_indices

/**
 * Signs a message m, deriving the secret key from sk_seed and the FTS address.
 * Assumes m contains at least SPX_FORS_HEIGHT * SPX_FORS_TREES bits.
 */
void fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* m,
               const unsigned char* sk_seed, const unsigned char* pub_seed,
               const uint32_t fors_addr[8]) {
    uint32_t indices[SPX_FORS_TREES];
    unsigned char roots[SPX_FORS_TREES * SPX_N];
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;
    unsigned int i;

    copy_keypair_addr(fors_tree_addr, fors_addr);
    copy_keypair_addr(fors_pk_addr, fors_addr);

    set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

    message_to_indices(indices, m);

    for (i = 0; i < SPX_FORS_TREES; i++) {
        idx_offset = i * (1 << SPX_FORS_HEIGHT);

        set_tree_height(fors_tree_addr, 0);
        set_tree_index(fors_tree_addr, indices[i] + idx_offset);

        /* Include the secret key part that produces the selected leaf node. */
        fors_gen_sk(sig, sk_seed, fors_tree_addr);
        sig += SPX_N;

        /* Compute the authentication path for this leaf node. */
        treehash(roots + i * SPX_N, sig, sk_seed, pub_seed, indices[i], idx_offset, SPX_FORS_HEIGHT,
                 fors_gen_leaf, fors_tree_addr);
        sig += SPX_N * SPX_FORS_HEIGHT;
    }

    /* Hash horizontally across all tree roots to derive the public key. */
    thash(pk, roots, SPX_FORS_TREES, pub_seed, fors_pk_addr);
} // fors_sign

__device__ void dev_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                              const unsigned char* sk_seed, const unsigned char* pub_seed,
                              const uint32_t fors_addr[8]) {
    uint32_t indices[SPX_FORS_TREES];
    unsigned char roots[SPX_FORS_TREES * SPX_N];
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;
    unsigned int i;

    dev_copy_keypair_addr(fors_tree_addr, fors_addr);
    dev_copy_keypair_addr(fors_pk_addr, fors_addr);

    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

    dev_message_to_indices(indices, m);

    for (i = 0; i < SPX_FORS_TREES; i++) {
        idx_offset = i * (1 << SPX_FORS_HEIGHT);

        dev_set_tree_height(fors_tree_addr, 0);
        dev_set_tree_index(fors_tree_addr, indices[i] + idx_offset);

        /* Include the secret key part that produces the selected leaf node. */
        dev_fors_gen_sk(sig, sk_seed, fors_tree_addr);
        sig += SPX_N;

        /* Compute the authentication path for this leaf node. */
        dev_treehash(roots + i * SPX_N, sig, sk_seed, pub_seed, indices[i], idx_offset,
                     SPX_FORS_HEIGHT, dev_fors_gen_leaf, fors_tree_addr);
        sig += SPX_N * SPX_FORS_HEIGHT;
    }

    /* Hash horizontally across all tree roots to derive the public key. */
    dev_thash(pk, roots, SPX_FORS_TREES, pub_seed, fors_pk_addr);
} // dev_fors_sign

__device__ void dev_ap_fors_sign_1(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                                   const unsigned char* sk_seed, const unsigned char* pub_seed,
                                   const uint32_t fors_addr[8]) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t indices[SPX_FORS_TREES];
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;

#if SPX_FORS_TREES <= 32
    __shared__ unsigned char fors_roots[SPX_FORS_TREES * SPX_N];
#endif // if SPX_FORS_TREES <= 32

    memset(fors_roots, 0, SPX_FORS_TREES * SPX_N);

    dev_copy_keypair_addr(fors_tree_addr, fors_addr);
    dev_copy_keypair_addr(fors_pk_addr, fors_addr);

    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

    dev_message_to_indices(indices, m);

    // only level 1, undefine USING_PARALLEL_SIGN_FORS_LEAF
    if (tid < SPX_FORS_TREES) {
        unsigned char* temp_sig;

        idx_offset = tid * (1 << SPX_FORS_HEIGHT);

        dev_set_tree_height(fors_tree_addr, 0);
        dev_set_tree_index(fors_tree_addr, indices[tid] + idx_offset);

        temp_sig = sig + tid * SPX_N + tid * SPX_N * SPX_FORS_HEIGHT;

        /* Include the secret key part that produces the selected leaf node. */
        dev_fors_gen_sk(temp_sig, sk_seed, fors_tree_addr);
        temp_sig += SPX_N;
        /* Compute the authentication path for this leaf node. */
        dev_treehash(fors_roots + tid * SPX_N, temp_sig, sk_seed, pub_seed, indices[tid],
                     idx_offset, SPX_FORS_HEIGHT, dev_fors_gen_leaf, fors_tree_addr);
    }
    g.sync();

    /* Hash horizontally across all tree roots to derive the public key. */
    if (tid == 0) dev_thash(pk, fors_roots, SPX_FORS_TREES, pub_seed, fors_pk_addr);
}

__device__ void dev_ap_fors_sign_12(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                                    const unsigned char* sk_seed, const unsigned char* pub_seed,
                                    const uint32_t fors_addr[8]) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t indices[SPX_FORS_TREES];
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;

#if SPX_FORS_TREES <= 32
    __shared__ unsigned char fors_roots[SPX_FORS_TREES * SPX_N];
#endif // if SPX_FORS_TREES <= 32

    memset(fors_roots, 0, SPX_FORS_TREES * SPX_N);

    dev_copy_keypair_addr(fors_tree_addr, fors_addr);
    dev_copy_keypair_addr(fors_pk_addr, fors_addr);

    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

    dev_message_to_indices(indices, m);

    const u32 leaf_num = (uint32_t) (1 << SPX_FORS_HEIGHT);

    unsigned int tnum = gridDim.x * blockDim.x;
    u32 max_para = leaf_num * SPX_FORS_TREES;
    u32 max = (tnum < max_para ? tnum : max_para);
    if (tid < max) {
        for (u32 i = tid; i < leaf_num * SPX_FORS_TREES; i += max) {
            u32 id = i / leaf_num;
            idx_offset = id * leaf_num;
            dev_set_tree_height(fors_tree_addr, 0);
            dev_set_tree_index(fors_tree_addr, indices[id] + idx_offset);
            u8* temp_sig = sig + id * SPX_N + id * SPX_N * SPX_FORS_HEIGHT;
            dev_fors_gen_sk(temp_sig, sk_seed, fors_tree_addr);
            dev_fors_gen_leaf(dev_leaf + i * SPX_N, sk_seed, pub_seed, i, fors_tree_addr);
        }
    }
    g.sync();

    // inner parallelism
    u8* leaf_node = dev_leaf;
    int branch_para = 16;
    branch_para = tnum;
    if (tid < SPX_FORS_TREES)
        memcpy(dev_fors_auth_path + tid * SPX_FORS_HEIGHT * SPX_N,
               dev_leaf + tid * leaf_num * SPX_N + (indices[tid] ^ 0x1) * SPX_N, SPX_N);

    for (int i = 1, ii = 1; i <= SPX_FORS_HEIGHT; i++) {
        g.sync();
        dev_set_tree_height(fors_tree_addr, i);
        if (tid < branch_para) {
            for (int j = tid; j < SPX_FORS_TREES * (leaf_num >> i); j += branch_para) {
                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(fors_tree_addr, j);
                memcpy(leaf_node + off + SPX_N, leaf_node + off + ii * SPX_N, SPX_N);
                dev_thash(leaf_node + off, leaf_node + off, 2, pub_seed, fors_tree_addr);
                int tree = j / (leaf_num >> i);
                if (j % (leaf_num >> i) == ((indices[tree] >> i) ^ 0x1)) {
                    memcpy(dev_fors_auth_path + i * SPX_N + tree * SPX_FORS_HEIGHT * SPX_N,
                           leaf_node + off, SPX_N);
                }
            }
        }
        ii *= 2;
    }
    g.sync();

    if (tid < SPX_FORS_TREES) {
        memcpy(sig + SPX_N * (1 + tid + SPX_FORS_HEIGHT * tid),
               dev_fors_auth_path + tid * SPX_FORS_HEIGHT * SPX_N, SPX_N * SPX_FORS_HEIGHT);
        memcpy(fors_roots + tid * SPX_N, leaf_node + tid * leaf_num * SPX_N, SPX_N);
    }
    g.sync();

    /* Hash horizontally across all tree roots to derive the public key. */
    if (tid == 0) dev_thash(pk, fors_roots, SPX_FORS_TREES, pub_seed, fors_pk_addr);
}

__device__ void dev_ap_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                                 const unsigned char* sk_seed, const unsigned char* pub_seed,
                                 const uint32_t fors_addr[8]) {
#ifdef USING_PARALLEL_SIGN_FORS_LEAF
    // level 1 + level 2, define USING_PARALLEL_SIGN_FORS_LEAF
    // const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // if (tid == 0) printf("fors level 1 + 2\n");
    dev_ap_fors_sign_12(sig, pk, m, sk_seed, pub_seed, fors_addr);

#else // ifdef USING_PARALLEL_SIGN_FORS_LEAF
    // only level 1, undefine USING_PARALLEL_SIGN_FORS_LEAF
    // const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // if (tid == 0) printf("fors level 1\n");
    dev_ap_fors_sign_1(sig, pk, m, sk_seed, pub_seed, fors_addr);

#endif // ifdef USING_PARALLEL_SIGN_FORS_LEAF
}

__global__ void global_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                                 const unsigned char* sk_seed, const unsigned char* pub_seed,
                                 const uint32_t fors_addr[8], uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_fors_sign(sig, pk, m, sk_seed, pub_seed, fors_addr);
}

__global__ void global_ap_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                                    const unsigned char* sk_seed, const unsigned char* pub_seed,
                                    const uint32_t fors_addr[8], uint32_t loop_num) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    for (int i = 0; i < loop_num; i++) {
        dev_ap_fors_sign(sig, pk, m, sk_seed, pub_seed, fors_addr);
        g.sync();
    }
}

__global__ void global_ap_fors_sign_1(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                                      const unsigned char* sk_seed, const unsigned char* pub_seed,
                                      const uint32_t fors_addr[8], uint32_t loop_num) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    for (int i = 0; i < loop_num; i++) {
        dev_ap_fors_sign_1(sig, pk, m, sk_seed, pub_seed, fors_addr);
        g.sync();
    }
}

__global__ void global_ap_fors_sign_12(unsigned char* sig, unsigned char* pk,
                                       const unsigned char* m, const unsigned char* sk_seed,
                                       const unsigned char* pub_seed, const uint32_t fors_addr[8],
                                       uint32_t loop_num) {
    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    for (int i = 0; i < loop_num; i++) {
        dev_ap_fors_sign_12(sig, pk, m, sk_seed, pub_seed, fors_addr);
        g.sync();
    }
}

void face_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* msg,
                    const unsigned char* sk_seed, const unsigned char* pub_seed,
                    const uint32_t fors_addr[8], uint32_t loop_num) {
    struct timespec start, stop;
    double result;
    int device = DEVICE_USED;
    u8 *dev_sig = NULL, *dev_pk = NULL, *dev_msg = NULL;
    u8 *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_sig, SPX_FORS_BYTES * sizeof(u8)));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_FORS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pk, pk, SPX_FORS_PK_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_msg, SPX_FORS_MSG_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_msg, msg, SPX_FORS_MSG_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(dev_addr, fors_addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    global_fors_sign<<<1, 1>>>(dev_sig, dev_pk, dev_msg, dev_sk_seed, dev_pub_seed, dev_addr,
                               loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    g_result += result;

    CHECK(cudaMemcpy(sig, dev_sig, SPX_FORS_BYTES * sizeof(u8), DEVICE_2_HOST));
    CHECK(cudaMemcpy(pk, dev_pk, SPX_FORS_PK_BYTES * sizeof(u8), DEVICE_2_HOST));

    cudaFree(dev_sig);
    cudaFree(dev_pk);
    cudaFree(dev_msg);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);
} // face_fors_sign

void face_ap_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* msg,
                       const unsigned char* sk_seed, const unsigned char* pub_seed,
                       const uint32_t fors_addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_sig = NULL, *dev_pk = NULL, *dev_msg = NULL;
    u8 *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    u32 threads = 32;
    u32 blocks = (1 << SPX_FORS_HEIGHT) * SPX_FORS_TREES / threads + 1;
    if (blocks > 10496 / 32) blocks = 10496 / 32;
    // printf("blocks = %u\n", blocks);
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_sig, SPX_FORS_BYTES * sizeof(u8)));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_FORS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pk, pk, SPX_FORS_PK_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_msg, SPX_FORS_MSG_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_msg, msg, SPX_FORS_MSG_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(dev_addr, fors_addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    void* kernelArgs[]
        = {&dev_sig, &dev_pk, &dev_msg, &dev_sk_seed, &dev_pub_seed, &dev_addr, &loop_num};

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_fors_sign, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(sig, dev_sig, SPX_FORS_BYTES * sizeof(u8), DEVICE_2_HOST));
    CHECK(cudaMemcpy(pk, dev_pk, SPX_FORS_PK_BYTES * sizeof(u8), DEVICE_2_HOST));

    cudaFree(dev_sig);
    cudaFree(dev_pk);
    cudaFree(dev_msg);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);
}

void face_ap_fors_sign_1(unsigned char* sig, unsigned char* pk, const unsigned char* msg,
                         const unsigned char* sk_seed, const unsigned char* pub_seed,
                         const uint32_t fors_addr[8], uint32_t loop_num) {

    int device = DEVICE_USED;
    u8 *dev_sig = NULL, *dev_pk = NULL, *dev_msg = NULL;
    u8 *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    u32 threads = 32;
    u32 blocks = (1 << SPX_FORS_HEIGHT) * SPX_FORS_TREES / threads + 1;
    if (blocks > 10496 / 32) blocks = 10496 / 32;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_sig, SPX_FORS_BYTES * sizeof(u8)));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_FORS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pk, pk, SPX_FORS_PK_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_msg, SPX_FORS_MSG_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_msg, msg, SPX_FORS_MSG_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(dev_addr, fors_addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    void* kernelArgs[]
        = {&dev_sig, &dev_pk, &dev_msg, &dev_sk_seed, &dev_pub_seed, &dev_addr, &loop_num};

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_fors_sign_1, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(sig, dev_sig, SPX_FORS_BYTES * sizeof(u8), DEVICE_2_HOST));
    CHECK(cudaMemcpy(pk, dev_pk, SPX_FORS_PK_BYTES * sizeof(u8), DEVICE_2_HOST));

    cudaFree(dev_sig);
    cudaFree(dev_pk);
    cudaFree(dev_msg);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);
}

void face_ap_fors_sign_12(unsigned char* sig, unsigned char* pk, const unsigned char* msg,
                          const unsigned char* sk_seed, const unsigned char* pub_seed,
                          const uint32_t fors_addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_sig = NULL, *dev_pk = NULL, *dev_msg = NULL;
    u8 *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    u32 threads = 32;
    u32 blocks = (1 << SPX_FORS_HEIGHT) * SPX_FORS_TREES / threads + 1;
    if (blocks > 10496 / 32) blocks = 10496 / 32;

// #if defined(SPX_128S) || defined(SPX_192S) || defined(SPX_256S)
//     blocks = 10496 * 2 / threads;
// #endif

    // printf("blocks = %u\n", blocks);
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_sig, SPX_FORS_BYTES * sizeof(u8)));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_FORS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pk, pk, SPX_FORS_PK_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_msg, SPX_FORS_MSG_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_msg, msg, SPX_FORS_MSG_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(dev_addr, fors_addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    void* kernelArgs[]
        = {&dev_sig, &dev_pk, &dev_msg, &dev_sk_seed, &dev_pub_seed, &dev_addr, &loop_num};

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    cudaLaunchCooperativeKernel((void*) global_ap_fors_sign_12, blocks, threads, kernelArgs);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(sig, dev_sig, SPX_FORS_BYTES * sizeof(u8), DEVICE_2_HOST));
    CHECK(cudaMemcpy(pk, dev_pk, SPX_FORS_PK_BYTES * sizeof(u8), DEVICE_2_HOST));

    cudaFree(dev_sig);
    cudaFree(dev_pk);
    cudaFree(dev_msg);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);
}

/**
 * Derives the FORS public key from a signature.
 * This can be used for verification by comparing to a known public key, or to
 * subsequently verify a signature on the derived public key. The latter is the
 * typical use-case when used as an FTS below an OTS in a hypertree.
 * Assumes m contains at least SPX_FORS_HEIGHT * SPX_FORS_TREES bits.
 */
void fors_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* m,
                      const unsigned char* pub_seed, const uint32_t fors_addr[8]) {
    uint32_t indices[SPX_FORS_TREES];
    unsigned char roots[SPX_FORS_TREES * SPX_N];
    unsigned char leaf[SPX_N];
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;
    unsigned int i;

    copy_keypair_addr(fors_tree_addr, fors_addr);
    copy_keypair_addr(fors_pk_addr, fors_addr);

    set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

    message_to_indices(indices, m);

    for (i = 0; i < SPX_FORS_TREES; i++) {
        idx_offset = i * (1 << SPX_FORS_HEIGHT);

        set_tree_height(fors_tree_addr, 0);
        set_tree_index(fors_tree_addr, indices[i] + idx_offset);

        /* Derive the leaf from the included secret key part. */
        fors_sk_to_leaf(leaf, sig, pub_seed, fors_tree_addr);
        sig += SPX_N;

        /* Derive the corresponding root node of this tree. */
        compute_root(roots + i * SPX_N, leaf, indices[i], idx_offset, sig, SPX_FORS_HEIGHT,
                     pub_seed, fors_tree_addr);
        sig += SPX_N * SPX_FORS_HEIGHT;
    }

    /* Hash horizontally across all tree roots to derive the public key. */
    thash(pk, roots, SPX_FORS_TREES, pub_seed, fors_pk_addr);
} // fors_pk_from_sig

__device__ void dev_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                     const unsigned char* m, const unsigned char* pub_seed,
                                     const uint32_t fors_addr[8]) {
    uint32_t indices[SPX_FORS_TREES];
    unsigned char roots[SPX_FORS_TREES * SPX_N];
    unsigned char leaf[SPX_N];
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;
    unsigned int i;

    dev_copy_keypair_addr(fors_tree_addr, fors_addr);
    dev_copy_keypair_addr(fors_pk_addr, fors_addr);

    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

    dev_message_to_indices(indices, m);

    for (i = 0; i < SPX_FORS_TREES; i++) {
        idx_offset = i * (1 << SPX_FORS_HEIGHT);

        dev_set_tree_height(fors_tree_addr, 0);
        dev_set_tree_index(fors_tree_addr, indices[i] + idx_offset);

        /* Derive the leaf from the included secret key part. */
        dev_fors_sk_to_leaf(leaf, sig, pub_seed, fors_tree_addr);
        sig += SPX_N;

        /* Derive the corresponding root node of this tree. */
        dev_compute_root(roots + i * SPX_N, leaf, indices[i], idx_offset, sig, SPX_FORS_HEIGHT,
                         pub_seed, fors_tree_addr);
        sig += SPX_N * SPX_FORS_HEIGHT;
    }

    /* Hash horizontally across all tree roots to derive the public key. */
    dev_thash(pk, roots, SPX_FORS_TREES, pub_seed, fors_pk_addr);
} // dev_fors_pk_from_sig

unsigned char dev_buffer[2 * SPX_N * SPX_FORS_TREES * 2];

__device__ void dev_ap_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                        const unsigned char* m, const unsigned char* pub_seed,
                                        const uint32_t fors_addr[8]) {
    // const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    // cooperative_groups::grid_group g = cooperative_groups::this_grid();
    uint32_t indices[SPX_FORS_TREES];
    unsigned char leaf[SPX_N];
    uint32_t fors_tree_addr[8] = {0};
    uint32_t fors_pk_addr[8] = {0};
    uint32_t idx_offset;

    __shared__ u8 s_fors_roots[SPX_FORS_TREES * SPX_N];

    dev_copy_keypair_addr(fors_tree_addr, fors_addr);
    dev_copy_keypair_addr(fors_pk_addr, fors_addr);

    dev_set_type(fors_tree_addr, SPX_ADDR_TYPE_FORSTREE);
    dev_set_type(fors_pk_addr, SPX_ADDR_TYPE_FORSPK);

    dev_message_to_indices(indices, m);

#ifdef USING_PARALLEL_COMPUTE_ROOT

    if (tid < SPX_FORS_TREES) {
        idx_offset = tid * (1 << SPX_FORS_HEIGHT);

        dev_set_tree_height(fors_tree_addr, 0);
        dev_set_tree_index(fors_tree_addr, indices[tid] + idx_offset);

        sig += (tid * SPX_N + tid * SPX_N * SPX_FORS_HEIGHT);

        /* Derive the leaf from the included secret key part. */
        dev_fors_sk_to_leaf(leaf, sig, pub_seed, fors_tree_addr);
        sig += SPX_N;

        /* Derive the corresponding root node of this tree. */
        dev_compute_root(s_fors_roots + tid * SPX_N, leaf, indices[tid], idx_offset, sig,
                         SPX_FORS_HEIGHT, pub_seed, fors_tree_addr);
    }

    __syncthreads();

#else  // ifdef USING_PARALLEL_COMPUTE_ROOT

    for (unsigned int i = 0; i < SPX_FORS_TREES; i++) {
        idx_offset = i * (1 << SPX_FORS_HEIGHT);

        dev_set_tree_height(fors_tree_addr, 0);
        dev_set_tree_index(fors_tree_addr, indices[i] + idx_offset);

        /* Derive the leaf from the included secret key part. */
        if (tid == 0) dev_fors_sk_to_leaf(leaf, sig, pub_seed, fors_tree_addr);
        sig += SPX_N;

        /* Derive the corresponding root node of this tree. */
        if (tid == 0)
            dev_compute_root(fors_roots + i * SPX_N, leaf, indices[i], idx_offset, sig,
                             SPX_FORS_HEIGHT, pub_seed, fors_tree_addr);

        sig += SPX_N * SPX_FORS_HEIGHT;
    }
#endif // ifdef USING_PARALLEL_COMPUTE_ROOT

    /* Hash horizontally across all tree roots to derive the public key. */
    if (tid == 0) dev_thash(pk, s_fors_roots, SPX_FORS_TREES, pub_seed, fors_pk_addr);
} // dev_ap_fors_pk_from_sig

__global__ void global_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                        const unsigned char* m, const unsigned char* pub_seed,
                                        const uint32_t fors_addr[8], uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_fors_pk_from_sig(pk, sig, m, pub_seed, fors_addr);
} // global_fors_pk_from_sig

__global__ void global_ap_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                           const unsigned char* m, const unsigned char* pub_seed,
                                           const uint32_t fors_addr[8], uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ap_fors_pk_from_sig(pk, sig, m, pub_seed, fors_addr);
} // global_fors_pk_from_sig

void face_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                           const unsigned char* pub_seed, const uint32_t fors_addr[8],
                           uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sig = NULL, *dev_msg = NULL;
    u8* dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_FORS_PK_BYTES * sizeof(u8)));

    CHECK(cudaMalloc((void**) &dev_sig, SPX_FORS_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_sig, sig, SPX_FORS_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_msg, SPX_FORS_MSG_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_msg, msg, SPX_FORS_MSG_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(dev_addr, fors_addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    CHECK(cudaDeviceSynchronize());
    global_fors_pk_from_sig<<<1, 1>>>(dev_pk, dev_sig, dev_msg, dev_pub_seed, dev_addr, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(pk, dev_pk, SPX_FORS_PK_BYTES * sizeof(u8), DEVICE_2_HOST));

    cudaFree(dev_pk);
    cudaFree(dev_sig);
    cudaFree(dev_msg);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);
}

void face_ap_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                              const unsigned char* pub_seed, const uint32_t fors_addr[8],
                              uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sig = NULL, *dev_msg = NULL;
    u8* dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    u32 threads = 64;
    u32 blocks = 1;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_FORS_PK_BYTES * sizeof(u8)));

    CHECK(cudaMalloc((void**) &dev_sig, SPX_FORS_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_sig, sig, SPX_FORS_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_msg, SPX_FORS_MSG_BYTES * sizeof(u8)));
    CHECK(cudaMemcpy(dev_msg, msg, SPX_FORS_MSG_BYTES * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));
    CHECK(cudaMemcpy(dev_addr, fors_addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_fors_pk_from_sig<<<blocks, threads>>>(dev_pk, dev_sig, dev_msg, dev_pub_seed,
                                                    dev_addr, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(pk, dev_pk, SPX_FORS_PK_BYTES * sizeof(u8), DEVICE_2_HOST));

    cudaFree(dev_pk);
    cudaFree(dev_sig);
    cudaFree(dev_msg);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);
}
