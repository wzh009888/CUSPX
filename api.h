#ifndef SPX_API_H
#define SPX_API_H

#include <stddef.h>
#include <stdint.h>

#include "params.h"

#define CRYPTO_ALGNAME "SPHINCS+"

#define CRYPTO_SECRETKEYBYTES SPX_SK_BYTES
#define CRYPTO_PUBLICKEYBYTES SPX_PK_BYTES
#define CRYPTO_BYTES SPX_BYTES
#define CRYPTO_SEEDBYTES 3 * SPX_N

/*
 * Returns the length of a secret key, in bytes
 */
unsigned long long crypto_sign_secretkeybytes(void);

/*
 * Returns the length of a public key, in bytes
 */
unsigned long long crypto_sign_publickeybytes(void);

/*
 * Returns the length of a signature, in bytes
 */
unsigned long long crypto_sign_bytes(void);

/*
 * Returns the length of the seed required to generate a key pair, in bytes
 */
unsigned long long crypto_sign_seedbytes(void);

/*
 * Generates a SPHINCS+ key pair given a seed.
 * Format sk: [SK_SEED || SK_PRF || PUB_SEED || root]
 * Format pk: [root || PUB_SEED]
 */
int crypto_sign_seed_keypair(unsigned char* pk, unsigned char* sk, const unsigned char* seed);

/*
 * Generates a SPHINCS+ key pair.
 * Format sk: [SK_SEED || SK_PRF || PUB_SEED || root]
 * Format pk: [root || PUB_SEED]
 */
int crypto_sign_keypair(unsigned char* pk, unsigned char* sk);
int face_crypto_sign_keypair(unsigned char* pk, unsigned char* sk);
int face_ap_crypto_sign_keypair(unsigned char* pk, unsigned char* sk); // 23 version
int face_ap_crypto_sign_keypair_2(unsigned char* pk, unsigned char* sk);
int face_ap_crypto_sign_keypair_23(unsigned char* pk, unsigned char* sk);
int face_treehash_wots(uint32_t loop_num, uint32_t blocks, uint32_t threads,
                       uint32_t maxallthreads);
int face_ap_treehash_wots_2(uint32_t loop_num, uint32_t blocks, uint32_t threads,
                            uint32_t maxallthreads);
int face_ap_treehash_wots_23(uint32_t loop_num, uint32_t blocks, uint32_t threads,
                             uint32_t maxallthreads);
int face_mdp_crypto_sign_keypair(unsigned char* pk, unsigned char* sk, unsigned int num);
int face_mgpu_mdp_crypto_sign_keypair(unsigned char* pk, unsigned char* sk, unsigned int num);
int face_ms_mdp_crypto_sign_keypair(unsigned char* pk, unsigned char* sk, unsigned int num);
int face_mgpu_ms_mdp_crypto_sign_keypair(unsigned char* pk, unsigned char* sk, unsigned int num);
int face_mhp_crypto_sign_keypair(unsigned char* pk, unsigned char* sk, unsigned int num,
                                 unsigned int intra_para);
int face_mhp_crypto_sign_keypair_1(unsigned char* pk, unsigned char* sk, unsigned int num,
                                   unsigned int intra_para);
int face_mhp_crypto_sign_keypair_scheme2(unsigned char* pk, unsigned char* sk, unsigned int num,
                                         unsigned int intra_para);
int face_ms_mhp_crypto_sign_keypair(unsigned char* pk, unsigned char* sk, unsigned int num);
int face_mhp_sign_keypair_seperate(unsigned char* pk, unsigned char* sk, unsigned int num);
/**
 * Returns an array containing a detached signature.
 */
int crypto_sign_signature(uint8_t* sig, size_t* siglen, const uint8_t* m, size_t mlen,
                          const uint8_t* sk);

/**
 * Verifies a detached signature and message under a given public key.
 */
int crypto_sign_verify(const uint8_t* sig, size_t siglen, const uint8_t* m, size_t mlen,
                       const uint8_t* pk);

/**
 * Returns an array containing the signature followed by the message.
 */
int face_ht(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
            unsigned long long mlen, const unsigned char* sk, int loop_num);
int face_ap_ht_1(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                 unsigned long long mlen, const unsigned char* sk, int loop_num);
int face_ap_ht_12(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                  unsigned long long mlen, const unsigned char* sk, int loop_num);
int face_ap_ht_123(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                   unsigned long long mlen, const unsigned char* sk, int loop_num);

int crypto_sign(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                unsigned long long mlen, const unsigned char* sk);
int face_crypto_sign(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                     unsigned long long mlen, const unsigned char* sk);
int face_ap_crypto_sign(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                        unsigned long long mlen, const unsigned char* sk);
int face_ap_crypto_sign_1(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                          unsigned long long mlen, const unsigned char* sk);
int face_ap_crypto_sign_12(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                           unsigned long long mlen, const unsigned char* sk);
int face_ap_crypto_sign_123(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                            unsigned long long mlen, const unsigned char* sk);
int face_mdp_crypto_sign(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                         unsigned long long mlen, const unsigned char* sk, unsigned int dp_num);
int face_mgpu_mdp_crypto_sign(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                              unsigned long long mlen, const unsigned char* sk,
                              unsigned int dp_num);
int face_ms_mdp_crypto_sign(unsigned char* sm, unsigned long long* smlen, unsigned char* m,
                            unsigned long long mlen, unsigned char* sk, unsigned int dp_num);
int face_mgpu_ms_mdp_crypto_sign(unsigned char* sm, unsigned long long* smlen, unsigned char* m,
                                 unsigned long long mlen, unsigned char* sk, unsigned int dp_num);
int face_sdp_crypto_sign(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                         unsigned long long mlen, const unsigned char* sk, unsigned int dp_num);
int face_ms_sdp_crypto_sign(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                            unsigned long long mlen, const unsigned char* sk, unsigned int dp_num);
int face_mhp_crypto_sign(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                         unsigned long long mlen, const unsigned char* sk, unsigned int dp_num);
int face_mhp_crypto_sign_1(unsigned char* sm, unsigned long long* smlen, const unsigned char* m,
                           unsigned long long mlen, const unsigned char* sk, unsigned int dp_num);
int face_mhp_crypto_sign_scheme2(unsigned char* sm, unsigned long long* smlen,
                                 const unsigned char* m, unsigned long long mlen,
                                 const unsigned char* sk, unsigned int dp_num,
                                 unsigned int intra_para);

int face_mhp_crypto_sign_scheme2_compare(unsigned char* sm, unsigned long long* smlen,
                                         const unsigned char* m, unsigned long long mlen,
                                         const unsigned char* sk, unsigned int dp_num,
                                         unsigned int intra_para); // with kim
/**
 * Verifies a given signature-message pair under a given public key.
 */
int crypto_sign_open(unsigned char* m, unsigned long long* mlen, const unsigned char* sm,
                     unsigned long long smlen, const unsigned char* pk);
int face_crypto_sign_open(unsigned char* m, unsigned long long* mlen, const unsigned char* sm,
                          unsigned long long smlen, const unsigned char* pk);
int face_ap_crypto_sign_open(unsigned char* m, unsigned long long* mlen, const unsigned char* sm,
                             unsigned long long smlen, const unsigned char* pk);
int face_mdp_crypto_sign_open(unsigned char* m, unsigned long long* mlen, const unsigned char* sm,
                              unsigned long long smlen, const unsigned char* pk,
                              unsigned int dp_num);
int face_mgpu_mdp_crypto_sign_open(unsigned char* m, unsigned long long* mlen,
                                   const unsigned char* sm, unsigned long long smlen,
                                   const unsigned char* pk, unsigned int dp_num);
int face_mhp_sign_open_seperate(unsigned char* m, unsigned long long* mlen, const unsigned char* sm,
                                unsigned long long smlen, const unsigned char* pk,
                                unsigned int dp_num);
int face_ms_mdp_crypto_sign_open(unsigned char* m, unsigned long long* mlen, unsigned char* sm,
                                 unsigned long long smlen, unsigned char* pk, unsigned int dp_num);
int face_mgpu_ms_mdp_crypto_sign_open(unsigned char* m, unsigned long long* mlen, unsigned char* sm,
                                      unsigned long long smlen, unsigned char* pk,
                                      unsigned int dp_num);
int face_sdp_crypto_sign_open(unsigned char* m, unsigned long long* mlen, const unsigned char* sm,
                              unsigned long long smlen, const unsigned char* pk,
                              unsigned int dp_num);
int face_ms_sdp_crypto_sign_open(unsigned char* m, unsigned long long* mlen,
                                 const unsigned char* sm, unsigned long long smlen,
                                 const unsigned char* pk, unsigned int dp_num);
int face_mhp_crypto_sign_open(unsigned char* m, unsigned long long* mlen, const unsigned char* sm,
                              unsigned long long smlen, const unsigned char* pk,
                              unsigned int dp_num, unsigned int intra_para);
int face_mhp_crypto_sign_open_compare(unsigned char* m, unsigned long long* mlen, const unsigned char* sm,
unsigned long long smlen, const unsigned char* pk,
unsigned int dp_num, unsigned int intra_para);// with kim
int face_ms_mhp_crypto_sign_open(unsigned char* m, unsigned long long* mlen,
                                 const unsigned char* sm, unsigned long long smlen,
                                 const unsigned char* pk, unsigned int dp_num);
int face_shp_crypto_sign_open(unsigned char* m, unsigned long long* mlen, const unsigned char* sm,
                              unsigned long long smlen, const unsigned char* pk,
                              unsigned int dp_num);
int face_ms_shp_crypto_sign_open(unsigned char* m, unsigned long long* mlen,
                                 const unsigned char* sm, unsigned long long smlen,
                                 const unsigned char* pk, unsigned int dp_num);

int face_tl(int l, int loop_num);
int face_h(int loop_num);
int face_f(int loop_num);
int face_prf(int loop_num);
int face_prf_msg(int loop_num);
int face_h_msg(int loop_num);
#endif /* ifndef SPX_API_H */
