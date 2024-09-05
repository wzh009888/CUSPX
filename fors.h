#ifndef SPX_FORS_H
#define SPX_FORS_H

#include <stdint.h>

#include "params.h"

/**
 * intenal function
 */
__device__ void dev_fors_sk_to_leaf(unsigned char* leaf, const unsigned char* sk,
                                    const unsigned char* pub_seed, uint32_t fors_leaf_addr[8]);
__device__ void dev_fors_gen_sk(unsigned char* sk, const unsigned char* sk_seed,
                                uint32_t fors_leaf_addr[8]);
__device__ void dev_fors_gen_leaf(unsigned char* leaf, const unsigned char* sk_seed,
                                  const unsigned char* pub_seed, uint32_t addr_idx,
                                  const uint32_t fors_tree_addr[8]);
__device__ void dev_message_to_indices(uint32_t* indices, const unsigned char* m);

/**
 * Signs a message m, deriving the secret key from sk_seed and the FTS address.
 * Assumes m contains at least SPX_FORS_HEIGHT * SPX_FORS_TREES bits.
 */
void fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* m,
               const unsigned char* sk_seed, const unsigned char* pub_seed,
               const uint32_t fors_addr[8]);
__device__ void dev_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                              const unsigned char* sk_seed, const unsigned char* pub_seed,
                              const uint32_t fors_addr[8]);
void face_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* msg,
                    const unsigned char* sk_seed, const unsigned char* pub_seed,
                    const uint32_t fors_addr[8], uint32_t loop_num);

__device__ void dev_ap_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                                 const unsigned char* sk_seed, const unsigned char* pub_seed,
                                 const uint32_t fors_addr[8]);

__device__ void dev_ap_fors_sign_1(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                                   const unsigned char* sk_seed, const unsigned char* pub_seed,
                                   const uint32_t fors_addr[8]);

__device__ void dev_ap_fors_sign_12(unsigned char* sig, unsigned char* pk, const unsigned char* m,
                                    const unsigned char* sk_seed, const unsigned char* pub_seed,
                                    const uint32_t fors_addr[8]);

void face_ap_fors_sign(unsigned char* sig, unsigned char* pk, const unsigned char* msg,
                       const unsigned char* sk_seed, const unsigned char* pub_seed,
                       const uint32_t fors_addr[8], uint32_t loop_num);

void face_ap_fors_sign_1(unsigned char* sig, unsigned char* pk, const unsigned char* msg,
                         const unsigned char* sk_seed, const unsigned char* pub_seed,
                         const uint32_t fors_addr[8], uint32_t loop_num);

void face_ap_fors_sign_12(unsigned char* sig, unsigned char* pk, const unsigned char* msg,
                          const unsigned char* sk_seed, const unsigned char* pub_seed,
                          const uint32_t fors_addr[8], uint32_t loop_num);
/**
 * Derives the FORS public key from a signature.
 * This can be used for verification by comparing to a known public key, or to
 * subsequently verify a signature on the derived public key. The latter is the
 * typical use-case when used as an FTS below an OTS in a hypertree.
 * Assumes m contains at least SPX_FORS_HEIGHT * SPX_FORS_TREES bits.
 */
void fors_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                      const unsigned char* pub_seed, const uint32_t fors_addr[8]);
__device__ void dev_ap_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                        const unsigned char* m, const unsigned char* pub_seed,
                                        const uint32_t fors_addr[8]);
__device__ void dev_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                     const unsigned char* msg, const unsigned char* pub_seed,
                                     const uint32_t fors_addr[8]);
void face_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                           const unsigned char* pub_seed, const uint32_t fors_addr[8],
                           uint32_t loop_num);
void face_ap_fors_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                              const unsigned char* pub_seed, const uint32_t fors_addr[8],
                              uint32_t loop_num);

#endif /* ifndef SPX_FORS_H */
