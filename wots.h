#ifndef SPX_WOTS_H
#define SPX_WOTS_H

#include "params.h"
#include <stdint.h>

__device__ void dev_wots_gen_sk(unsigned char* sk, const unsigned char* sk_seed,
                                uint32_t wots_addr[8]);
__device__ void dev_gen_chain(unsigned char* out, const unsigned char* in, unsigned int start,
                              unsigned int steps, const unsigned char* pub_seed, uint32_t addr[8]);
__device__ void dev_chain_lengths(unsigned int* lengths, const unsigned char* msg);
/**
 * WOTS key generation. Takes a 32 byte seed for the private key, expands it to
 * a full WOTS private key and computes the corresponding public key.
 * It requires the seed pub_seed (used to generate bitmasks and hash keys)
 * and the address of this WOTS key pair.
 *
 * Writes the computed public key to 'pk'.
 */
void wots_gen_pk(unsigned char* pk, const unsigned char* seed, const unsigned char* pub_seed,
                 uint32_t addr[8]);

__device__ void dev_wots_gen_pk(unsigned char* pk, const unsigned char* seed,
                                const unsigned char* pub_seed, uint32_t addr[8]);

void face_wots_gen_pk(unsigned char* pk, const unsigned char* seed, const unsigned char* pub_seed,
                      uint32_t addr[8], uint32_t loop_num);

void face_ap_wots_gen_pk(unsigned char* pk, const unsigned char* seed,
                         const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num);

void face_ap_wots_gen_pk_cc(unsigned char* pk, const unsigned char* seed,
                            const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num);
void face_ht_wots_gen_pk(unsigned char* pk, const unsigned char* seed,
                         const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num);
void face_ap_ht_wots_gen_pk_1(unsigned char* pk, const unsigned char* seed,
                            const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num);

void face_ap_ht_wots_gen_pk_12(unsigned char* pk, const unsigned char* seed,
                            const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num);
/**
 * Takes a n-byte message and the 32-byte seed for the private key to compute a
 * signature that is placed at 'sig'.
 */
void wots_sign(unsigned char* sig, const unsigned char* msg, const unsigned char* seed,
               const unsigned char* pub_seed, uint32_t addr[8]);

__device__ void dev_ap_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                        const unsigned char* msg, const unsigned char* pub_seed,
                                        uint32_t addr[8]);

__device__ void dev_wots_sign(unsigned char* sig, const unsigned char* msg,
                              const unsigned char* seed, const unsigned char* pub_seed,
                              uint32_t addr[8]);

__device__ void dev_ap_wots_sign(unsigned char* sig, const unsigned char* msg,
                                 const unsigned char* seed, const unsigned char* pub_seed,
                                 uint32_t addr[8], uint32_t offset);

void face_wots_sign(unsigned char* sig, const unsigned char* msg, const unsigned char* seed,
                    const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num);

void face_ap_wots_sign(unsigned char* sig, const unsigned char* msg, const unsigned char* seed,
                       const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num);

/**
 * Takes a WOTS signature and an n-byte message, computes a WOTS public key.
 *
 * Writes the computed public key to 'pk'.
 */
void wots_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                      const unsigned char* pub_seed, uint32_t addr[8]);
__device__ void dev_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig,
                                     const unsigned char* msg, const unsigned char* pub_seed,
                                     uint32_t addr[8]);
void face_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                           const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num);

void face_ap_wots_pk_from_sig(unsigned char* pk, const unsigned char* sig, const unsigned char* msg,
                              const unsigned char* pub_seed, uint32_t addr[8], uint32_t loop_num);
#endif /* ifndef SPX_WOTS_H */
