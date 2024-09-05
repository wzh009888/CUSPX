
#include <stdint.h>
#include <string.h>

#include "thash.h"
#include "address.h"
#include "params.h"
#include "sha256.h"
#include "utils.h"

#ifdef SHA256

#include <cooperative_groups.h>
// __device__ u8 one_bm[SPX_WOTS_BYTES];

void thash(unsigned char *out, const unsigned char *in, unsigned int inblocks,
	   const unsigned char *pub_seed, uint32_t addr[8])
{
	unsigned char buf[SPX_SHA256_ADDR_BYTES + inblocks * SPX_N];
	unsigned char outbuf[SPX_SHA256_OUTPUT_BYTES];
	uint8_t sha2_state[40];

	(void)pub_seed; /* Suppress an 'unused parameter' warning. */

	/* Retrieve precomputed state containing pub_seed */
	memcpy(sha2_state, state_seeded, 40 * sizeof(uint8_t));

	memcpy(buf, addr, SPX_SHA256_ADDR_BYTES);
	memcpy(buf + SPX_SHA256_ADDR_BYTES, in, inblocks * SPX_N);

	sha256_inc_finalize(outbuf, sha2_state, buf,
			    SPX_SHA256_ADDR_BYTES + inblocks * SPX_N);
	memcpy(out, outbuf, SPX_N);
} // thash

extern __device__ uint8_t dev_state_seeded[40];

__device__ void dev_thash(unsigned char *out, const unsigned char *in,
			  unsigned int inblocks,
			  const unsigned char *pub_seed, uint32_t addr[8])
{
	unsigned char buf[SPX_SHA256_ADDR_BYTES + SPX_WOTS_LEN * SPX_N];
	unsigned char outbuf[SPX_SHA256_OUTPUT_BYTES];
	uint8_t sha2_state[40];

	(void)pub_seed; /* Suppress an 'unused parameter' warning. */

	/* Retrieve precomputed state containing pub_seed */
	memcpy(sha2_state, dev_state_seeded, 40 * sizeof(uint8_t));

	memcpy(buf, addr, SPX_SHA256_ADDR_BYTES);
	memcpy(buf + SPX_SHA256_ADDR_BYTES, in, inblocks * SPX_N);

	dev_sha256_inc_finalize(outbuf, sha2_state, buf,
				SPX_SHA256_ADDR_BYTES + inblocks * SPX_N);
	memcpy(out, outbuf, SPX_N);

} // dev_thash

__device__ void dev_ap_thash(unsigned char *out, const unsigned char *in,
			     unsigned int inblocks,
			     const unsigned char *pub_seed, uint32_t addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	cooperative_groups::grid_group g = cooperative_groups::this_grid();
	unsigned char buf[SPX_SHA256_ADDR_BYTES + SPX_WOTS_LEN * SPX_N];
	unsigned char outbuf[SPX_SHA256_OUTPUT_BYTES];
	uint8_t sha2_state[40];

	(void)pub_seed; /* Suppress an 'unused parameter' warning. */

	g.sync();
	if (tid == 0) {
		/* Retrieve precomputed state containing pub_seed */
		memcpy(sha2_state, dev_state_seeded, 40 * sizeof(uint8_t));

		memcpy(buf, addr, SPX_SHA256_ADDR_BYTES);
		memcpy(buf + SPX_SHA256_ADDR_BYTES, in, inblocks * SPX_N);

		dev_sha256_inc_finalize(outbuf, sha2_state, buf,
					SPX_SHA256_ADDR_BYTES + inblocks * SPX_N);
		memcpy(out, outbuf, SPX_N);
	}

} // dev_ap_thash

#endif // ifdef SHA256
