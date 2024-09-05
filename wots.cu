#include <iostream>
#include <stdint.h>
#include <string.h>
using namespace std;

#include "all_option.h"

#include "address.h"
#include "hash.h"
#include "params.h"
#include "thash.h"
#include "utils.h"
#include "wots.h"

#include <cooperative_groups.h>

// TODO clarify address expectations, and make them more uniform.
// TODO i.e. do we expect types to be set already?
// TODO and do we expect modifications or copies?

extern __device__ u8 wots_pk[SPX_WOTS_BYTES * 512];

/**
 * Computes the starting value for a chain, i.e. the secret key.
 * Expects the address to be complete up to the chain address.
 */
static void wots_gen_sk(unsigned char* sk, const unsigned char* sk_seed, uint32_t wots_addr[8]) {
    /* Make sure that the hash address is actually zeroed. */
    set_hash_addr(wots_addr, 0);

    /* Generate sk element. */
    prf_addr(sk, sk_seed, wots_addr);
} /* wots_gen_sk */

__device__ void dev_wots_gen_sk(unsigned char* sk, const unsigned char* sk_seed,
                                uint32_t wots_addr[8]) {
    /* Make sure that the hash address is actually zeroed. */
    dev_set_hash_addr(wots_addr, 0);

    /* Generate sk element. */
    dev_prf_addr(sk, sk_seed, wots_addr);
} /* wots_gen_sk */

/**
 * Computes the chaining function.
 * out and in have to be n-byte arrays.
 *
 * Interprets in as start-th value of the chain.
 * addr has to contain the address of the chain.
 */
static void gen_chain(unsigned char* out, const unsigned char* in, unsigned int start,
                      unsigned int steps, const unsigned char* pub_seed, uint32_t addr[8]) {
    uint32_t i;

    /* Initialize out with the value at position 'start'. */
    memcpy(out, in, SPX_N);

    /* Iterate 'steps' calls to the hash function. */
    for (i = start; i < (start + steps) && i < SPX_WOTS_W; i++) {
        set_hash_addr(addr, i);
        thash(out, out, 1, pub_seed, addr);
    }
} /* gen_chain */

__device__ void dev_gen_chain(unsigned char* out, const unsigned char* in, unsigned int start,
                              unsigned int steps, const unsigned char* pub_seed, uint32_t addr[8]) {
    uint32_t i;

    /* Initialize out with the value at position 'start'. */
    memcpy(out, in, SPX_N);

    /* Iterate 'steps' calls to the hash function. */
    for (i = start; i < (start + steps) && i < SPX_WOTS_W; i++) {
        dev_set_hash_addr(addr, i);
        dev_thash(out, out, 1, pub_seed, addr);
    }
} /* gen_chain */

/**
 * base_w algorithm as described in draft.
 * Interprets an array of bytes as integers in base w.
 * This only works when log_w is a divisor of 8.
 */
static void base_w(unsigned int* output, const int out_len, const unsigned char* input) {
    int in = 0;
    int out = 0;
    unsigned char total;
    int bits = 0;
    int consumed;

    for (consumed = 0; consumed < out_len; consumed++) {
        if (bits == 0) {
            total = input[in];
            in++;
            bits += 8;
        }
        bits -= SPX_WOTS_LOGW;
        output[out] = (total >> bits) & (SPX_WOTS_W - 1);
        out++;
    }
} /* base_w */

__device__ void dev_base_w(unsigned int* output, const int out_len, const unsigned char* input) {
    int in = 0;
    int out = 0;
    unsigned char total;
    int bits = 0;
    int consumed;

    for (consumed = 0; consumed < out_len; consumed++) {
        if (bits == 0) {
            total = input[in];
            in++;
            bits += 8;
        }
        bits -= SPX_WOTS_LOGW;
        output[out] = (total >> bits) & (SPX_WOTS_W - 1);
        out++;
    }
} /* base_w */

/* Computes the WOTS+ checksum over a message (in base_w). */
static void wots_checksum(unsigned int* csum_base_w, const unsigned int* msg_base_w) {
    unsigned int csum = 0;
    unsigned char csum_bytes[(SPX_WOTS_LEN2 * SPX_WOTS_LOGW + 7) / 8];
    unsigned int i;

    /* Compute checksum. */
    for (i = 0; i < SPX_WOTS_LEN1; i++) {
        csum += SPX_WOTS_W - 1 - msg_base_w[i];
    }

    /* Convert checksum to base_w. */
    /* Make sure expected empty zero bits are the least significant bits. */
    csum = csum << ((8 - ((SPX_WOTS_LEN2 * SPX_WOTS_LOGW) % 8)) % 8);
    ull_to_bytes(csum_bytes, sizeof(csum_bytes), csum);
    base_w(csum_base_w, SPX_WOTS_LEN2, csum_bytes);
} /* wots_checksum */

__device__ void dev_wots_checksum(unsigned int* csum_base_w, const unsigned int* msg_base_w) {
    unsigned int csum = 0;
    unsigned char csum_bytes[(SPX_WOTS_LEN2 * SPX_WOTS_LOGW + 7) / 8];
    unsigned int i;

    /* Compute checksum. */
    for (i = 0; i < SPX_WOTS_LEN1; i++) {
        csum += SPX_WOTS_W - 1 - msg_base_w[i];
    }

    /* Convert checksum to base_w. */
    /* Make sure expected empty zero bits are the least significant bits. */
    csum = csum << ((8 - ((SPX_WOTS_LEN2 * SPX_WOTS_LOGW) % 8)) % 8);
    dev_ull_to_bytes(csum_bytes, sizeof(csum_bytes), csum);
    dev_base_w(csum_base_w, SPX_WOTS_LEN2, csum_bytes);
} /* wots_checksum */

/* Takes a message and derives the matching chain lengths. */
static void chain_lengths(unsigned int* lengths, const unsigned char* msg) {
    base_w(lengths, SPX_WOTS_LEN1, msg);
    wots_checksum(lengths + SPX_WOTS_LEN1, lengths);
} /* chain_lengths */

__device__ void dev_chain_lengths(unsigned int* lengths, const unsigned char* msg) {
    dev_base_w(lengths, SPX_WOTS_LEN1, msg);
    dev_wots_checksum(lengths + SPX_WOTS_LEN1, lengths);
} /* chain_lengths */

/**
 * WOTS key generation. Takes a 32 byte sk_seed, expands it to WOTS private key
 * elements and computes the corresponding public key.
 * It requires the seed pub_seed (used to generate bitmasks and hash keys)
 * and the address of this WOTS key pair.
 *
 * Writes the computed public key to 'pk'.
 */
void wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed, const unsigned char* pub_seed,
                 uint32_t addr[8]) {
    uint32_t i;

    for (i = 0; i < SPX_WOTS_LEN; i++) {
        set_chain_addr(addr, i);
        wots_gen_sk(pk + i * SPX_N, sk_seed, addr);
        gen_chain(pk + i * SPX_N, pk + i * SPX_N, 0, SPX_WOTS_W - 1, pub_seed, addr);
    }
} // wots_gen_pk

__device__ void dev_wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed,
                                const unsigned char* pub_seed, uint32_t addr[8]) {
    uint32_t i;

    for (i = 0; i < SPX_WOTS_LEN; i++) {
        dev_set_chain_addr(addr, i);
        dev_wots_gen_sk(pk + i * SPX_N, sk_seed, addr);
        dev_gen_chain(pk + i * SPX_N, pk + i * SPX_N, 0, SPX_WOTS_W - 1, pub_seed, addr);
    }

} // dev_wots_gen_pk

// assume thread > SPX_WOTS_LEN
__device__ void dev_ap_wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed,
                                   const unsigned char* pub_seed, uint32_t addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < SPX_WOTS_LEN) {
        dev_set_chain_addr(addr, tid);
        dev_wots_gen_sk(pk + tid * SPX_N, sk_seed, addr);
        dev_gen_chain(pk + tid * SPX_N, pk + tid * SPX_N, 0, SPX_WOTS_W - 1, pub_seed, addr);
    }
    __syncthreads();
}

__device__ void dev_ht_wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed,
                                   const unsigned char* pub_seed, uint32_t addr[8]) {
    uint32_t i, j;

    for (j = 0; j < SPX_D; j++) {
        for (i = 0; i < SPX_WOTS_LEN; i++) {
            dev_set_chain_addr(addr, i);
            dev_wots_gen_sk(pk + j * SPX_WOTS_BYTES + i * SPX_N, sk_seed, addr);
            dev_gen_chain(pk + j * SPX_WOTS_BYTES + i * SPX_N, pk + i * SPX_N, 0, SPX_WOTS_W - 1,
                          pub_seed, addr);
        }
    }
}

__device__ void dev_ap_ht_wots_gen_pk_12(unsigned char* pk, const unsigned char* sk_seed,
                                         const unsigned char* pub_seed, uint32_t addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    if (tid < SPX_WOTS_LEN * SPX_D) {
        dev_set_chain_addr(addr, tid);
        dev_wots_gen_sk(pk + tid * SPX_N, sk_seed, addr);
        dev_gen_chain(pk + tid * SPX_N, pk + tid * SPX_N, 0, SPX_WOTS_W - 1, pub_seed, addr);
    }
    g.sync();
}

__device__ void dev_ap_ht_wots_gen_pk_1(unsigned char* pk, const unsigned char* sk_seed,
                                        const unsigned char* pub_seed, uint32_t addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // cooperative_groups::grid_group g = cooperative_groups::this_grid();

    if (tid < SPX_D) {
        dev_wots_gen_pk(pk + tid * SPX_WOTS_BYTES, sk_seed, pub_seed, addr);
    }
    __syncthreads();
    // g.sync();
}

// assume thread > SPX_WOTS_LEN
__device__ void dev_ap_wots_gen_pk_dis(unsigned char* pk, const unsigned char* sk_seed,
                                       const unsigned char* pub_seed, uint32_t addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // 位于不同warp，也是位于不用块
    if (tid < 32) {
        dev_set_chain_addr(addr, tid / 32);
        dev_wots_gen_sk(pk + tid * 32 * SPX_N, sk_seed, addr);
        dev_gen_chain(pk + tid * 32 * SPX_N, pk + tid * 32 * SPX_N, 0, SPX_WOTS_W - 1, pub_seed,
                      addr);
    }
    __syncthreads();
}

__device__ void dev_ap_wots_gen_pk_cc(unsigned char* pk, const unsigned char* sk_seed,
                                      const unsigned char* pub_seed, uint32_t addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // int num = 32;

    // 一个wots内部连续存储，和线程一一对应
    dev_set_chain_addr(addr, tid % 35);
    dev_wots_gen_sk(pk + tid * SPX_N, sk_seed, addr);
    dev_gen_chain(pk + tid * SPX_N, wots_pk + tid * SPX_N, 0, SPX_WOTS_W - 1, pub_seed, addr);
    __syncthreads();
}

__device__ void dev_ap_wots_gen_pk_wocc(unsigned char* pk, const unsigned char* sk_seed,
                                        const unsigned char* pub_seed, uint32_t addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int num = 32;
    int id = tid % num;
    int ttid = tid / num;

    // 每个任务的第0部分连续存储，然后是每个任务的第1部分连续存储
    dev_set_chain_addr(addr, tid / 32);
    dev_wots_gen_sk(pk + id * SPX_N * num + ttid * SPX_N, sk_seed, addr);
    dev_gen_chain(pk + id * SPX_N * num + ttid * SPX_N, pk + id * SPX_N * num + ttid * SPX_N, 0,
                  SPX_WOTS_W - 1, pub_seed, addr);
    __syncthreads();
}

__global__ void global_wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed,
                                   const unsigned char* pub_seed, uint32_t addr[8],
                                   uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_wots_gen_pk(pk, sk_seed, pub_seed, addr);
}

__global__ void global_ap_wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed,
                                      const unsigned char* pub_seed, uint32_t addr[8],
                                      uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ap_wots_gen_pk(pk, sk_seed, pub_seed, addr);
} // global_ap_wots_gen_pk

__global__ void global_ap_wots_gen_pk_dis(unsigned char* pk, const unsigned char* sk_seed,
                                          const unsigned char* pub_seed, uint32_t addr[8],
                                          uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ap_wots_gen_pk_dis(pk, sk_seed, pub_seed, addr);
} // global_ap_wots_gen_pk

__global__ void global_ap_wots_gen_pk_cc(unsigned char* pk, const unsigned char* sk_seed,
                                         const unsigned char* pub_seed, uint32_t addr[8],
                                         uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ap_wots_gen_pk_cc(pk, sk_seed, pub_seed, addr);
} // global_ap_wots_gen_pk

__global__ void global_ap_wots_gen_pk_wocc(unsigned char* pk, const unsigned char* sk_seed,
                                           const unsigned char* pub_seed, uint32_t addr[8],
                                           uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ap_wots_gen_pk_wocc(pk, sk_seed, pub_seed, addr);
} // global_ap_wots_gen_pk

__global__ void global_ht_wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed,
                                      const unsigned char* pub_seed, uint32_t addr[8],
                                      uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ht_wots_gen_pk(pk, sk_seed, pub_seed, addr);
}

__global__ void global_ap_ht_wots_gen_pk_12(unsigned char* pk, const unsigned char* sk_seed,
                                            const unsigned char* pub_seed, uint32_t addr[8],
                                            uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ap_ht_wots_gen_pk_12(pk, sk_seed, pub_seed, addr);
}

__global__ void global_ap_ht_wots_gen_pk_1(unsigned char* pk, const unsigned char* sk_seed,
                                           const unsigned char* pub_seed, uint32_t addr[8],
                                           uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ap_ht_wots_gen_pk_1(pk, sk_seed, pub_seed, addr);
}

void face_wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed,
                      const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_addr, addr, 8 * sizeof(uint32_t), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_wots_gen_pk<<<1, 1>>>(dev_pk, dev_sk_seed, dev_pub_seed, dev_addr, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(pk, dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8), D2H));

    cudaFree(dev_pk);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);

} // face_wots_gen_pk

void face_ap_wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed,
                         const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;
    int blocks = 1;
    int threads = SPX_WOTS_LEN;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_addr, addr, 8 * sizeof(uint32_t), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_wots_gen_pk<<<blocks, threads>>>(dev_pk, dev_sk_seed, dev_pub_seed, dev_addr,
                                               loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(pk, dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8), D2H));

    cudaFree(dev_pk);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);

} // face_wots_gen_pk

// 用来测试wots使用合并访存是否有效，考虑32个任务计算wots_gen_pk
void face_ap_wots_gen_pk_cc(unsigned char* pk, const unsigned char* sk_seed,
                            const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;
    int num = 32;
    int threads = 32;
    int blocks = num * SPX_WOTS_LEN / threads;
    printf("using num = %d tasks\n", num);

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, num * SPX_WOTS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk_seed, num * SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, num * SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, num * 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, num * SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, num * SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_addr, addr, num * 8 * sizeof(uint32_t), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_wots_gen_pk<<<blocks, threads>>>(dev_pk, dev_sk_seed, dev_pub_seed, dev_addr,
                                               loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    printf("%11.3lf ms\n", g_result / loop_num / 1e3);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_wots_gen_pk_dis<<<blocks, threads>>>(dev_pk, dev_sk_seed, dev_pub_seed, dev_addr,
                                                   loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    printf("%11.3lf ms\n", g_result / loop_num / 1e3);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_wots_gen_pk_cc<<<blocks, threads>>>(dev_pk, dev_sk_seed, dev_pub_seed, dev_addr,
                                                  loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    printf("%11.3lf ms\n", g_result / loop_num / 1e3);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_wots_gen_pk_wocc<<<blocks, threads>>>(dev_pk, dev_sk_seed, dev_pub_seed, dev_addr,
                                                    loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;
    printf("%11.3lf ms\n", g_result / loop_num / 1e3);

    CHECK(cudaMemcpy(pk, dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8), D2H));

    cudaFree(dev_pk);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);

} // face_wots_gen_pk

void face_ht_wots_gen_pk(unsigned char* pk, const unsigned char* sk_seed,
                         const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_D * SPX_WOTS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_addr, addr, 8 * sizeof(uint32_t), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ht_wots_gen_pk<<<1, 1>>>(dev_pk, dev_sk_seed, dev_pub_seed, dev_addr, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(pk, dev_pk, SPX_D * SPX_WOTS_PK_BYTES * sizeof(u8), D2H));

    cudaFree(dev_pk);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);
}

void face_ap_ht_wots_gen_pk_1(unsigned char* pk, const unsigned char* sk_seed,
                              const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;
    int threads = 32;
    int blocks = SPX_D * SPX_WOTS_LEN / threads + 1;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_addr, addr, 8 * sizeof(uint32_t), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_ht_wots_gen_pk_1<<<blocks, threads>>>(dev_pk, dev_sk_seed, dev_pub_seed, dev_addr,
                                                    loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(pk, dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8), D2H));

    cudaFree(dev_pk);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);
}

void face_ap_ht_wots_gen_pk_12(unsigned char* pk, const unsigned char* sk_seed,
                               const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;
    int threads = 32;
    int blocks = SPX_D * SPX_WOTS_LEN / threads + 1;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), H2D));
    CHECK(cudaMemcpy(dev_addr, addr, 8 * sizeof(uint32_t), H2D));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_ht_wots_gen_pk_12<<<blocks, threads>>>(dev_pk, dev_sk_seed, dev_pub_seed, dev_addr,
                                                     loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(pk, dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8), D2H));

    cudaFree(dev_pk);
    cudaFree(dev_sk_seed);
    cudaFree(dev_pub_seed);
    cudaFree(dev_addr);
}

/**
 * Takes a n-byte message and the 32-byte sk_see to compute a signature 'sig'.
 */
void wots_sign(unsigned char* sig, const unsigned char* msg, const unsigned char* sk_seed,
               const unsigned char* pub_seed, uint32_t addr[8]) {
    unsigned int lengths[SPX_WOTS_LEN];
    uint32_t i;

    chain_lengths(lengths, msg);

    for (i = 0; i < SPX_WOTS_LEN; i++) {
        set_chain_addr(addr, i);
        wots_gen_sk(sig + i * SPX_N, sk_seed, addr);
        gen_chain(sig + i * SPX_N, sig + i * SPX_N, 0, lengths[i], pub_seed, addr);
    }
} /* wots_sign */

__device__ void dev_wots_sign(unsigned char* sig, const unsigned char* msg,
                              const unsigned char* sk_seed, const unsigned char* pub_seed,
                              uint32_t addr[8]) {
    unsigned int lengths[SPX_WOTS_LEN];
    uint32_t i;

    dev_chain_lengths(lengths, msg);

    for (i = 0; i < SPX_WOTS_LEN; i++) {
        dev_set_chain_addr(addr, i);
        dev_wots_gen_sk(sig + i * SPX_N, sk_seed, addr);
        dev_gen_chain(sig + i * SPX_N, sig + i * SPX_N, 0, lengths[i], pub_seed, addr);
    }
} // dev_wots_sign

__device__ void dev_ap_wots_sign(unsigned char* sig, const unsigned char* msg,
                                 const unsigned char* seed, const unsigned char* pub_seed,
                                 uint32_t addr[8], uint32_t offset) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    // cooperative_groups::grid_group g = cooperative_groups::this_grid();

    // g.sync();
    if (tid >= offset && tid < offset + SPX_WOTS_LEN) {
        u32 lengths[SPX_WOTS_LEN];
        // u8 buf[N + 32];
        u32 ttid = tid - offset;

        dev_chain_lengths(lengths, msg); // cannot be parallelized
        dev_set_chain_addr(addr, ttid);
        dev_wots_gen_sk(sig + ttid * SPX_N, seed, addr);
        dev_gen_chain(sig + ttid * SPX_N, sig + ttid * SPX_N, 0, lengths[ttid], pub_seed, addr);
        // dev_set_key_and_mask(addr, 0);
        // memcpy(buf, pub_seed, N);
        // dev_set_chain_addr(addr, ttid);
        // dev_set_hash_addr(addr, 0);
        // dev_prf_addr(sig + ttid * SPX_N, sk_seed, addr);
        // dev_addr_to_bytes(buf + N, addr);
        // #ifdef USING_LOCAL_MEMORY
        // u8 temp[N];
        // dev_prf_keygen(temp, buf, sk_seed);
        // dev_gen_chain(temp, temp, 0, lengths[ttid], pub_seed, addr);
        // memcpy(sig + ttid * N, temp, N);
        // #else // ifdef USING_LOCAL_MEMORY
        // dev_prf_keygen(sig + ttid * N, buf, sk_seed);
        // dev_gen_chain(sig + ttid * N, sig + ttid * N,
        // 	      0, lengths[ttid], pub_seed, addr);
        // #endif // ifdef USING_LOCAL_MEMORY
    }
    __syncthreads();
} // dev_wots_sign_parallel

__global__ void global_wots_sign(unsigned char* sig, const unsigned char* msg,
                                 const unsigned char* sk_seed, const unsigned char* pub_seed,
                                 uint32_t addr[8], uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_wots_sign(sig, msg, sk_seed, pub_seed, addr);
} // global_wots_sign

__global__ void global_ap_wots_sign(unsigned char* sig, const unsigned char* msg,
                                    const unsigned char* sk_seed, const unsigned char* pub_seed,
                                    uint32_t addr[8], uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ap_wots_sign(sig, msg, sk_seed, pub_seed, addr, 0);
} // global_wots_sign

void face_wots_sign(unsigned char* sig, const unsigned char* msg, const unsigned char* sk_seed,
                    const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_sig = NULL, *dev_msg = NULL;
    u8 *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_sig, SPX_WOTS_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_msg, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_msg, msg, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_addr, addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_wots_sign<<<1, 1>>>(dev_sig, dev_msg, dev_sk_seed, dev_pub_seed, dev_addr, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(sig, dev_sig, SPX_WOTS_BYTES * sizeof(u8), DEVICE_2_HOST));

    CHECK(cudaFree(dev_sig));
    CHECK(cudaFree(dev_msg));
    CHECK(cudaFree(dev_sk_seed));
    CHECK(cudaFree(dev_pub_seed));
    CHECK(cudaFree(dev_addr));
} // global_wots_sign

void face_ap_wots_sign(unsigned char* sig, const unsigned char* msg, const unsigned char* sk_seed,
                       const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_sig = NULL, *dev_msg = NULL;
    u8 *dev_sk_seed = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;
    u32 blocks = 1;
    u32 threads = SPX_WOTS_LEN;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_sig, SPX_WOTS_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_msg, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sk_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_msg, msg, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_addr, addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_wots_sign<<<blocks, threads>>>(dev_sig, dev_msg, dev_sk_seed, dev_pub_seed, dev_addr,
                                             loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(sig, dev_sig, SPX_WOTS_BYTES * sizeof(u8), DEVICE_2_HOST));

    CHECK(cudaFree(dev_sig));
    CHECK(cudaFree(dev_msg));
    CHECK(cudaFree(dev_sk_seed));
    CHECK(cudaFree(dev_pub_seed));
    CHECK(cudaFree(dev_addr));
} // global_wots_sign

/**
 * Takes a WOTS signature and an n-byte message, computes a WOTS public key.
 *
 * Writes the computed public key to 'pk'.
 */
void wots_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                      const unsigned char* pub_seed, uint32_t addr[8]) {
    unsigned int lengths[SPX_WOTS_LEN];
    uint32_t i;

    chain_lengths(lengths, msg);

    for (i = 0; i < SPX_WOTS_LEN; i++) {
        set_chain_addr(addr, i);
        gen_chain(pk + i * SPX_N, sig + i * SPX_N, lengths[i], SPX_WOTS_W - 1 - lengths[i],
                  pub_seed, addr);
    }
} // wots_pk_from_sig

__device__ void dev_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                     const unsigned char* msg, const unsigned char* pub_seed,
                                     uint32_t addr[8]) {
    unsigned int lengths[SPX_WOTS_LEN];
    uint32_t i;

    dev_chain_lengths(lengths, msg);

    for (i = 0; i < SPX_WOTS_LEN; i++) {
        dev_set_chain_addr(addr, i);
        dev_gen_chain(pk + i * SPX_N, sig + i * SPX_N, lengths[i], SPX_WOTS_W - 1 - lengths[i],
                      pub_seed, addr);
    }
} // dev_wots_pk_from_sig

__device__ int worktid[SPX_WOTS_LEN];

__device__ void dev_ap_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                        const unsigned char* msg, const unsigned char* pub_seed,
                                        uint32_t addr[8]) {
    // const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    register unsigned int lengths[SPX_WOTS_LEN];

    __syncthreads();
    if (tid < SPX_WOTS_LEN) {
        dev_chain_lengths(lengths, msg);
        dev_set_chain_addr(addr, tid);
        dev_gen_chain(pk + tid * SPX_N, sig + tid * SPX_N, lengths[tid],
                      SPX_WOTS_W - 1 - lengths[tid], pub_seed, addr);
    }
    __syncthreads();

    // int para = 8;

    // if (tid < SPX_WOTS_LEN) dev_chain_lengths(lengths, msg);
    //
    // __shared__ u8 look[SPX_WOTS_LEN];
    //
    // memset(look, 0, SPX_WOTS_LEN);
    //
    // __syncthreads();
    // if (tid < para) {
    // 	for (int i = 0; i < SPX_WOTS_LEN; i++) {
    // 		if (look[i] == 0) {
    // 			look[i] = 1;
    // 				dev_set_chain_addr(addr, i);
    // 				dev_gen_chain(pk + i * SPX_N, sig + i * SPX_N, lengths[i],
    // 					      SPX_WOTS_W - 1 - lengths[i], pub_seed, addr);
    // 		}
    // 	}
    // }

    // if (tid == 0) {
    // 	u32 steps[SPX_WOTS_LEN];        // 每个链长
    // 	u32 ll[SPX_WOTS_LEN];           // 每个线程当前的长度
    // 	for (int i = 0; i < para; i++)
    // 		ll[i] = 0;
    // 	for (int i = 0; i < SPX_WOTS_LEN; i++)
    // 		steps[i] = SPX_WOTS_W - 1 - lengths[i];
    // 	for (int i = 0; i < SPX_WOTS_LEN; i++) {
    // 		int min = 999999;
    // 		int mintid = -1;
    // 		for (int j = 0; j < para; j++) {
    // 			if (ll[j] < min) {
    // 				min = ll[j];
    // 				mintid = j;
    // 			}
    // 		}
    // 		worktid[i] = mintid;
    // 		ll[mintid] += steps[i];
    // 	}
    // }
    //
    // g.sync();
    // for (int i = 0; i < SPX_WOTS_LEN; i++) {
    // 	if (worktid[i] == tid && tid < para) {
    // 		dev_set_chain_addr(addr, i);
    // 		dev_gen_chain(pk + i * SPX_N, sig + i * SPX_N, lengths[i],
    // 			      SPX_WOTS_W - 1 - lengths[i], pub_seed, addr);
    // 	}
    // }

} // dev_wots_pk_from_sig

__global__ void global_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                        const unsigned char* msg, const unsigned char* pub_seed,
                                        uint32_t addr[8], uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_wots_pk_from_sig(pk, sig, msg, pub_seed, addr);
} // global_wots_pk_from_sig

__global__ void global_ap_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                           const unsigned char* msg, const unsigned char* pub_seed,
                                           uint32_t addr[8], uint32_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_ap_wots_pk_from_sig(pk, sig, msg, pub_seed, addr);
} // global_wots_pk_from_sig

void face_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                           const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sig = NULL, *dev_msg = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sig, SPX_WOTS_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_msg, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_sig, sig, SPX_WOTS_BYTES * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_msg, msg, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_addr, addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_wots_pk_from_sig<<<1, 1>>>(dev_pk, dev_sig, dev_msg, dev_pub_seed, dev_addr, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(pk, dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8), DEVICE_2_HOST));

    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_sig));
    CHECK(cudaFree(dev_msg));
    CHECK(cudaFree(dev_pub_seed));
    CHECK(cudaFree(dev_addr));
} // face_wots_pk_from_sig

void face_ap_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                              const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num) {
    int device = DEVICE_USED;
    u8 *dev_pk = NULL, *dev_sig = NULL, *dev_msg = NULL, *dev_pub_seed = NULL;
    uint32_t* dev_addr = NULL;
    struct timespec start, stop;
    u32 blocks = 1;
    u32 threads = SPX_WOTS_LEN;

    CHECK(cudaSetDevice(device));

    CHECK(cudaMalloc((void**) &dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_sig, SPX_WOTS_BYTES * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_msg, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_pub_seed, SPX_N * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_addr, 8 * sizeof(uint32_t)));

    CHECK(cudaMemcpy(dev_sig, sig, SPX_WOTS_BYTES * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_msg, msg, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));
    CHECK(cudaMemcpy(dev_addr, addr, 8 * sizeof(uint32_t), HOST_2_DEVICE));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    CHECK(cudaDeviceSynchronize());
    global_ap_wots_pk_from_sig<<<blocks, threads>>>(dev_pk, dev_sig, dev_msg, dev_pub_seed,
                                                    dev_addr, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result += (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(pk, dev_pk, SPX_WOTS_PK_BYTES * sizeof(u8), DEVICE_2_HOST));

    CHECK(cudaFree(dev_pk));
    CHECK(cudaFree(dev_sig));
    CHECK(cudaFree(dev_msg));
    CHECK(cudaFree(dev_pub_seed));
    CHECK(cudaFree(dev_addr));
} // face_ap_wots_pk_from_sig

