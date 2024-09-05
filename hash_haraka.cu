#include <stdint.h>
#include <string.h>

#include "address.h"
#include "utils.h"
#include "params.h"

#include "haraka.h"
#include "hash.h"
#include <iostream>
using namespace std;

#ifdef HARAKA

void initialize_hash_function(const unsigned char *pk_seed,
			      const unsigned char *sk_seed)
{
	tweak_constants(pk_seed, sk_seed, SPX_N);
} /* initialize_hash_function */

__device__ void dev_initialize_hash_function(const unsigned char *pk_seed,
				  const unsigned char *sk_seed)
{
	dev_tweak_constants(pk_seed, sk_seed, SPX_N);
} /* initialize_hash_function */

__global__ void global_initialize_hash_function(const unsigned char *pub_seed,
						const unsigned char *sk_seed)
{
	dev_initialize_hash_function(pub_seed, sk_seed);
} // dev_initialize_hash_function

void face_initialize_hash_function(const unsigned char *pub_seed,
				   const unsigned char *sk_seed)
{
	int device = DEVICE_USED;
	u8 *dev_pub_seed = NULL, *dev_sk_seed = NULL;

	CHECK(cudaSetDevice(device));

	CHECK(cudaMalloc((void **)&dev_pub_seed, SPX_N * sizeof(u8)));
	CHECK(cudaMemcpy(dev_pub_seed, pub_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaMalloc((void **)&dev_sk_seed, SPX_N * sizeof(u8)));
	CHECK(cudaMemcpy(dev_sk_seed, sk_seed, SPX_N * sizeof(u8), HOST_2_DEVICE));

	CHECK(cudaDeviceSynchronize());
	global_initialize_hash_function << < 1, 1 >> >
		(dev_pub_seed, dev_sk_seed);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	cudaFree(dev_pub_seed); cudaFree(dev_sk_seed);
} // face_initialize_hash_function

/*
 * Computes PRF(key, addr), given a secret key of SPX_N bytes and an address
 */
void prf_addr(unsigned char *out, const unsigned char *key,
	      const uint32_t addr[8])
{
	/* Since SPX_N may be smaller than 32, we need a temporary buffer. */
	unsigned char outbuf[32];

	(void)key; /* Suppress an 'unused parameter' warning. */

	haraka256_sk(outbuf, (const unsigned char *)addr);
	memcpy(out, outbuf, SPX_N);
} // prf_addr

__device__ void dev_prf_addr(unsigned char *out, const unsigned char *key,
			     const uint32_t addr[8])
{
	/* Since SPX_N may be smaller than 32, we need a temporary buffer. */
	unsigned char outbuf[32];

	(void)key; /* Suppress an 'unused parameter' warning. */

	dev_haraka256_sk(outbuf, (const unsigned char *)addr);
	memcpy(out, outbuf, SPX_N);
} // prf_addr

/**
 * Computes the message-dependent randomness R, using a secret seed and an
 * optional randomization value as well as the message.
 */
void gen_message_random(unsigned char *R, const unsigned char *sk_prf,
			const unsigned char *optrand,
			const unsigned char *m, unsigned long long mlen)
{
	uint8_t s_inc[65];

	haraka_S_inc_init(s_inc);
	haraka_S_inc_absorb(s_inc, sk_prf, SPX_N);
	haraka_S_inc_absorb(s_inc, optrand, SPX_N);
	haraka_S_inc_absorb(s_inc, m, mlen);
	haraka_S_inc_finalize(s_inc);
	haraka_S_inc_squeeze(R, SPX_N, s_inc);
} // gen_message_random

__device__ void dev_gen_message_random(unsigned char *R, const unsigned char *sk_prf,
				       const unsigned char *optrand,
				       const unsigned char *m, unsigned long long mlen)
{
	uint8_t s_inc[65];

	dev_haraka_S_inc_init(s_inc);
	dev_haraka_S_inc_absorb(s_inc, sk_prf, SPX_N);
	dev_haraka_S_inc_absorb(s_inc, optrand, SPX_N);
	dev_haraka_S_inc_absorb(s_inc, m, mlen);
	dev_haraka_S_inc_finalize(s_inc);
	dev_haraka_S_inc_squeeze(R, SPX_N, s_inc);
} // dev_gen_message_random

/**
 * Computes the message hash using R, the public key, and the message.
 * Outputs the message digest and the index of the leaf. The index is split in
 * the tree index and the leaf index, for convenient copying to an address.
 */
void hash_message(unsigned char *digest, uint64_t *tree, uint32_t *leaf_idx,
		  const unsigned char *R, const unsigned char *pk,
		  const unsigned char *m, unsigned long long mlen)
{
#define SPX_TREE_BITS (SPX_TREE_HEIGHT * (SPX_D - 1))
#define SPX_TREE_BYTES ((SPX_TREE_BITS + 7) / 8)
#define SPX_LEAF_BITS SPX_TREE_HEIGHT
#define SPX_LEAF_BYTES ((SPX_LEAF_BITS + 7) / 8)
#define SPX_DGST_BYTES (SPX_FORS_MSG_BYTES + SPX_TREE_BYTES + SPX_LEAF_BYTES)

	unsigned char buf[SPX_DGST_BYTES];
	unsigned char *bufp = buf;
	uint8_t s_inc[65];

	haraka_S_inc_init(s_inc);
	haraka_S_inc_absorb(s_inc, R, SPX_N);
	haraka_S_inc_absorb(s_inc, pk + SPX_N, SPX_N); // Only absorb root part of pk
	haraka_S_inc_absorb(s_inc, m, mlen);
	haraka_S_inc_finalize(s_inc);
	haraka_S_inc_squeeze(buf, SPX_DGST_BYTES, s_inc);

	memcpy(digest, bufp, SPX_FORS_MSG_BYTES);
	bufp += SPX_FORS_MSG_BYTES;

#if SPX_TREE_BITS > 64
    #error For given height and depth, 64 bits cannot represent all subtrees
#endif /* if SPX_TREE_BITS > 64 */

	*tree = bytes_to_ull(bufp, SPX_TREE_BYTES);
	*tree &= (~(uint64_t)0) >> (64 - SPX_TREE_BITS);
	bufp += SPX_TREE_BYTES;

	*leaf_idx = bytes_to_ull(bufp, SPX_LEAF_BYTES);
	*leaf_idx &= (~(uint32_t)0) >> (32 - SPX_LEAF_BITS);
} // hash_message

__device__ void dev_hash_message(unsigned char *digest, uint64_t *tree, uint32_t *leaf_idx,
				 const unsigned char *R, const unsigned char *pk,
				 const unsigned char *m, unsigned long long mlen)
{
#define SPX_TREE_BITS (SPX_TREE_HEIGHT * (SPX_D - 1))
#define SPX_TREE_BYTES ((SPX_TREE_BITS + 7) / 8)
#define SPX_LEAF_BITS SPX_TREE_HEIGHT
#define SPX_LEAF_BYTES ((SPX_LEAF_BITS + 7) / 8)
#define SPX_DGST_BYTES (SPX_FORS_MSG_BYTES + SPX_TREE_BYTES + SPX_LEAF_BYTES)

	unsigned char buf[SPX_DGST_BYTES];
	unsigned char *bufp = buf;
	uint8_t s_inc[65];

	dev_haraka_S_inc_init(s_inc);
	dev_haraka_S_inc_absorb(s_inc, R, SPX_N);
	dev_haraka_S_inc_absorb(s_inc, pk + SPX_N, SPX_N); // Only absorb root part of pk
	dev_haraka_S_inc_absorb(s_inc, m, mlen);
	dev_haraka_S_inc_finalize(s_inc);
	dev_haraka_S_inc_squeeze(buf, SPX_DGST_BYTES, s_inc);

	memcpy(digest, bufp, SPX_FORS_MSG_BYTES);
	bufp += SPX_FORS_MSG_BYTES;

#if SPX_TREE_BITS > 64
    #error For given height and depth, 64 bits cannot represent all subtrees
#endif /* if SPX_TREE_BITS > 64 */

	*tree = dev_bytes_to_ull(bufp, SPX_TREE_BYTES);
	*tree &= (~(uint64_t)0) >> (64 - SPX_TREE_BITS);
	bufp += SPX_TREE_BYTES;

	*leaf_idx = dev_bytes_to_ull(bufp, SPX_LEAF_BYTES);
	*leaf_idx &= (~(uint32_t)0) >> (32 - SPX_LEAF_BITS);
} // dev_hash_message

#endif /* ifdef HARAKA */
