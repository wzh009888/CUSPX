#ifndef SPX_PARAMS_H
#define SPX_PARAMS_H

/* choosing parameter set starts */
#ifndef VARIANT
#define SHA256
// #define SHAKE256

// #define SPX_128S
// #define SPX_128F
// #define SPX_192S
#define SPX_192F
// #define SPX_256S
// #define SPX_256F

#endif

#if (!defined(SPX_128S) && !defined(SPX_128F) \
	&& !defined(SPX_192S) && !defined(SPX_192F) \
	&& !defined(SPX_256S) && !defined(SPX_256F))
    #error SPX_128S or SPX_128F or SPX_192S \
	or SPX_192F or SPX_256S or SPX_256F should be defined
#endif /* if (!defined(SPX_128S) && !defined(SPX_128F)
          && !defined(SPX_192S)&& !defined(SPX_192F)
          && !defined(SPX_256S) && !defined(SPX_256F)) */

#if (!defined(SHAKE256) && !defined(SHA256))
    #error SHAKE256 or SHA256 should be defined
#endif /* if (!defined(SHAKE256) && !defined(SHA256)) */

/* choosing parameter set ends */

#ifdef SPX_128S
#define SPX_N 16
#define SPX_FULL_HEIGHT 63
#define SPX_D 7
#define SPX_FORS_HEIGHT 12
#define SPX_FORS_TREES 14
#endif /* ifdef SPX_128S */

#ifdef SPX_128F
#define SPX_N 16
#define SPX_FULL_HEIGHT 66
#define SPX_D 22
#define SPX_FORS_HEIGHT 6
#define SPX_FORS_TREES 33
#endif /* ifdef SPX_128F */

#ifdef SPX_192S
#define SPX_N 24
#define SPX_FULL_HEIGHT 63
#define SPX_D 7
#define SPX_FORS_HEIGHT 14
#define SPX_FORS_TREES 17
#endif /* ifdef SPX_192S */

#ifdef SPX_192F
#define SPX_N 24
#define SPX_FULL_HEIGHT 66
#define SPX_D 22
#define SPX_FORS_HEIGHT 8
#define SPX_FORS_TREES 33
#endif /* ifdef SPX_192F */

#ifdef SPX_256S
#define SPX_N 32
#define SPX_FULL_HEIGHT 64
#define SPX_D 8
#define SPX_FORS_HEIGHT 14
#define SPX_FORS_TREES 22
#endif /* ifdef SPX_256S */

#ifdef SPX_256F
#define SPX_N 32
#define SPX_FULL_HEIGHT 68
#define SPX_D 17
#define SPX_FORS_HEIGHT 9
#define SPX_FORS_TREES 35
#endif /* ifdef SPX_256F */

#define SPX_WOTS_W 16

/* The hash function is defined by linking a different hash.c file, as opposed
   to setting a #define constant. */

/* For clarity */
#define SPX_ADDR_BYTES 32

/* WOTS parameters. */
#if SPX_WOTS_W == 256
    #define SPX_WOTS_LOGW 8
#elif SPX_WOTS_W == 16
    #define SPX_WOTS_LOGW 4
#else  /* if SPX_WOTS_W == 256 */
    #error SPX_WOTS_W assumed 16 or 256
#endif /* if SPX_WOTS_W == 256 */

#define SPX_WOTS_LEN1 (8 * SPX_N / SPX_WOTS_LOGW)

/* SPX_WOTS_LEN2 is floor(log(len_1 * (w - 1)) / log(w)) + 1; we precompute */
#if SPX_WOTS_W == 256
    #if SPX_N <= 1
	#define SPX_WOTS_LEN2 1
    #elif SPX_N <= 256
	#define SPX_WOTS_LEN2 2
    #else  /* if SPX_N <= 1 */
	#error Did not precompute SPX_WOTS_LEN2 for n outside {2, .., 256}
    #endif /* if SPX_N <= 1 */
#elif SPX_WOTS_W == 16
    #if SPX_N <= 8
	#define SPX_WOTS_LEN2 2
    #elif SPX_N <= 136
	#define SPX_WOTS_LEN2 3
    #elif SPX_N <= 256
	#define SPX_WOTS_LEN2 4
    #else  /* if SPX_N <= 8 */
	#error Did not precompute SPX_WOTS_LEN2 for n outside {2, .., 256}
    #endif /* if SPX_N <= 8 */
#endif /* if SPX_WOTS_W == 256 */

#define SPX_WOTS_LEN (SPX_WOTS_LEN1 + SPX_WOTS_LEN2)
#define SPX_WOTS_BYTES (SPX_WOTS_LEN * SPX_N)
#define SPX_WOTS_PK_BYTES SPX_WOTS_BYTES

/* Subtree size. */
#define SPX_TREE_HEIGHT (SPX_FULL_HEIGHT / SPX_D)

#if SPX_TREE_HEIGHT * SPX_D != SPX_FULL_HEIGHT
    #error SPX_D should always divide SPX_FULL_HEIGHT
#endif /* if SPX_TREE_HEIGHT * SPX_D != SPX_FULL_HEIGHT */

/* FORS parameters. */
#define SPX_FORS_MSG_BYTES ((SPX_FORS_HEIGHT * SPX_FORS_TREES + 7) / 8)
#define SPX_FORS_BYTES ((SPX_FORS_HEIGHT + 1) * SPX_FORS_TREES * SPX_N)
#define SPX_FORS_PK_BYTES SPX_N

/* Resulting SPX sizes. */
#define SPX_BYTES (SPX_N + SPX_FORS_BYTES + SPX_D * SPX_WOTS_BYTES + \
		   SPX_FULL_HEIGHT * SPX_N)
#define SPX_PK_BYTES (2 * SPX_N)
#define SPX_SK_BYTES (2 * SPX_N + SPX_PK_BYTES)

/* Optionally, signing can be made non-deterministic using optrand.
   This can help counter side-channel attacks that would benefit from
   getting a large number of traces when the signer uses the same nodes. */
#define SPX_OPTRAND_BYTES 32

#include "sha256_offsets.h"
#define SPX_MLEN 32
#define SPX_SM_BYTES (SPX_MLEN + SPX_BYTES)
#define SM_BYTES (SPX_MLEN + SPX_BYTES)

#endif /* ifndef SPX_PARAMS_H */
