#ifndef SPX_THASH_H
#define SPX_THASH_H

#include <stdint.h>
#include "all_option.h"

void thash(unsigned char *out, const unsigned char *in, unsigned int inblocks,
	   const unsigned char *pub_seed, uint32_t addr[8]);

__device__ void dev_thash(unsigned char *out, const unsigned char *in, unsigned int inblocks,
			  const unsigned char *pub_seed, uint32_t addr[8]);

__device__ void dev_ap_thash_two(unsigned char *out, const unsigned char *in,
				 unsigned int inblocks, const unsigned char *pub_seed,
				 uint32_t addr[8]);

__device__ void dev_ap_thash(unsigned char *out, const unsigned char *in, unsigned int inblocks,
			     const unsigned char *pub_seed, uint32_t addr[8]);


#endif /* ifndef SPX_THASH_H */
