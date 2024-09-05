#include <stdint.h>
#include <string.h>

#include "address.h"
#include "utils.h"
#include "params.h"
#include "hash.h"
#include "fips202.h"

#ifdef SHAKE256

/* For SHAKE256, there is no immediate reason to initialize at the start,
   so this function is an empty operation. */
void initialize_hash_function(const unsigned char *pub_seed,
                              const unsigned char *sk_seed)
{
    (void)pub_seed; /* Suppress an 'unused parameter' warning. */
    (void)sk_seed; /* Suppress an 'unused parameter' warning. */
}

__device__ void dev_initialize_hash_function(const unsigned char *pub_seed,
                              const unsigned char *sk_seed)
{
    (void)pub_seed; /* Suppress an 'unused parameter' warning. */
    (void)sk_seed; /* Suppress an 'unused parameter' warning. */
}

void face_initialize_hash_function(const unsigned char *pub_seed,
				   const unsigned char *sk_seed)
{
    (void)pub_seed; /* Suppress an 'unused parameter' warning. */
    (void)sk_seed; /* Suppress an 'unused parameter' warning. */
} // face_initialize_hash_function

/*
 * Computes PRF(key, addr), given a secret key of SPX_N bytes and an address
 */
void prf_addr(unsigned char *out, const unsigned char *key,
              const uint32_t addr[8])
{
    unsigned char buf[SPX_N + SPX_ADDR_BYTES];

    memcpy(buf, key, SPX_N);
    memcpy(buf + SPX_N, addr, SPX_ADDR_BYTES);

    shake256(out, SPX_N, buf, SPX_N + SPX_ADDR_BYTES);
}

__device__ void dev_prf_addr(unsigned char *out, const unsigned char *key,
              const uint32_t addr[8])
{
    unsigned char buf[SPX_N + SPX_ADDR_BYTES];

    memcpy(buf, key, SPX_N);
    memcpy(buf + SPX_N, addr, SPX_ADDR_BYTES);

    dev_shake256(out, SPX_N, buf, SPX_N + SPX_ADDR_BYTES);
}

/**
 * Computes the message-dependent randomness R, using a secret seed and an
 * optional randomization value as well as the message.
 */
void gen_message_random(unsigned char *R, const unsigned char *sk_prf,
                        const unsigned char *optrand,
                        const unsigned char *m, unsigned long long mlen)
{
    uint64_t s_inc[26];

    shake256_inc_init(s_inc);
    shake256_inc_absorb(s_inc, sk_prf, SPX_N);
    shake256_inc_absorb(s_inc, optrand, SPX_N);
    shake256_inc_absorb(s_inc, m, mlen);
    shake256_inc_finalize(s_inc);
    shake256_inc_squeeze(R, SPX_N, s_inc);

}

__device__ void dev_gen_message_random(unsigned char *R, const unsigned char *sk_prf,
                        const unsigned char *optrand,
                        const unsigned char *m, unsigned long long mlen)
{
    unsigned char temp[2 * SPX_N + 32];
    memcpy(temp, sk_prf, SPX_N);
    memcpy(temp + SPX_N, optrand, SPX_N);
    memcpy(temp + 2 * SPX_N, m, mlen);
    dev_shake256(R, SPX_N, temp, 2 * SPX_N + 32);
}


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
    uint64_t s_inc[26];

    shake256_inc_init(s_inc);
    shake256_inc_absorb(s_inc, R, SPX_N);
    shake256_inc_absorb(s_inc, pk, SPX_PK_BYTES);
    shake256_inc_absorb(s_inc, m, mlen);
    shake256_inc_finalize(s_inc);
    shake256_inc_squeeze(buf, SPX_DGST_BYTES, s_inc);

    memcpy(digest, bufp, SPX_FORS_MSG_BYTES);
    bufp += SPX_FORS_MSG_BYTES;

#if SPX_TREE_BITS > 64
    #error For given height and depth, 64 bits cannot represent all subtrees
#endif

    *tree = bytes_to_ull(bufp, SPX_TREE_BYTES);
    *tree &= (~(uint64_t)0) >> (64 - SPX_TREE_BITS);
    bufp += SPX_TREE_BYTES;

    *leaf_idx = bytes_to_ull(bufp, SPX_LEAF_BYTES);
    *leaf_idx &= (~(uint32_t)0) >> (32 - SPX_LEAF_BITS);
}

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

    unsigned char temp[SPX_N + SPX_PK_BYTES + 32];
    memcpy(temp, R, SPX_N);
    memcpy(temp + SPX_N, pk, SPX_PK_BYTES);
    memcpy(temp + SPX_N + SPX_PK_BYTES, m, mlen);
    dev_shake256(buf, SPX_DGST_BYTES, temp, SPX_N + SPX_PK_BYTES + 32);

    memcpy(digest, bufp, SPX_FORS_MSG_BYTES);
    bufp += SPX_FORS_MSG_BYTES;

#if SPX_TREE_BITS > 64
    #error For given height and depth, 64 bits cannot represent all subtrees
#endif

    *tree = dev_bytes_to_ull(bufp, SPX_TREE_BYTES);
    *tree &= (~(uint64_t)0) >> (64 - SPX_TREE_BITS);
    bufp += SPX_TREE_BYTES;

    *leaf_idx = dev_bytes_to_ull(bufp, SPX_LEAF_BYTES);
    *leaf_idx &= (~(uint32_t)0) >> (32 - SPX_LEAF_BITS);
}

#endif
