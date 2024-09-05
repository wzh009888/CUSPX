#include <stdint.h>
#include <string.h>

#include "thash.h"
#include "address.h"
#include "params.h"

#include "fips202.h"

#ifdef SHAKE256

#include <cooperative_groups.h>


void thash(unsigned char *out, const unsigned char *in, unsigned int inblocks,
	   const unsigned char *pub_seed, uint32_t addr[8])
{
	unsigned char buf[SPX_N + SPX_ADDR_BYTES + SPX_WOTS_LEN * SPX_N];

	memcpy(buf, pub_seed, SPX_N);
	memcpy(buf + SPX_N, addr, SPX_ADDR_BYTES);
	memcpy(buf + SPX_N + SPX_ADDR_BYTES, in, inblocks * SPX_N);

	shake256(out, SPX_N, buf, SPX_N + SPX_ADDR_BYTES + inblocks * SPX_N);
} // thash

__device__ void dev_thash(unsigned char *out, const unsigned char *in,
			  unsigned int inblocks,
			  const unsigned char *pub_seed, uint32_t addr[8])
{
	unsigned char buf[SPX_N + SPX_ADDR_BYTES + SPX_WOTS_LEN * SPX_N];

	memcpy(buf, pub_seed, SPX_N);
	memcpy(buf + SPX_N, addr, SPX_ADDR_BYTES);
	memcpy(buf + SPX_N + SPX_ADDR_BYTES, in, inblocks * SPX_N);

	dev_shake256(out, SPX_N, buf, SPX_N + SPX_ADDR_BYTES + inblocks * SPX_N);
} // dev_thash

__device__ void dev_ap_thash_two(unsigned char *out, const unsigned char *in,
				 unsigned int inblocks,
				 const unsigned char *pub_seed, uint32_t addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned char buf[SPX_N + SPX_ADDR_BYTES + 2 * SPX_N];

	__syncthreads();
	if (tid == 0) {
		memcpy(buf, pub_seed, SPX_N);
		memcpy(buf + SPX_N, addr, SPX_ADDR_BYTES);
		memcpy(buf + SPX_N + SPX_ADDR_BYTES, in, inblocks * SPX_N);

		dev_shake256(out, SPX_N, buf, SPX_N + SPX_ADDR_BYTES + inblocks * SPX_N);
	}
} // dev_ap_thash_two

__device__ void dev_ap_thash(unsigned char *out, const unsigned char *in,
			     unsigned int inblocks,
			     const unsigned char *pub_seed, uint32_t addr[8])
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	// cooperative_groups::grid_group g = cooperative_groups::this_grid();
	unsigned char buf[SPX_N + SPX_ADDR_BYTES + SPX_WOTS_LEN * SPX_N];

	// g.sync();
	if (tid == 0) {
		memcpy(buf, pub_seed, SPX_N);
		memcpy(buf + SPX_N, addr, SPX_ADDR_BYTES);
		memcpy(buf + SPX_N + SPX_ADDR_BYTES, in, inblocks * SPX_N);

		dev_shake256(out, SPX_N, buf, SPX_N + SPX_ADDR_BYTES + inblocks * SPX_N);
	}
} // dev_ap_thash

#endif // ifdef SHAKE256
