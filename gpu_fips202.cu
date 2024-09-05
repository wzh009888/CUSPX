/* Based on the public domain implementation in
 * crypto_hash/keccakc512/simple/ from http://bench.cr.yp.to/supercop.html
 * by Ronny Van Keer
 * and the public domain "TweetFips202" implementation
 * from https://twitter.com/tweetfips202
 * by Gilles Van Assche, Daniel J. Bernstein, and Peter Schwabe */

#include <stddef.h>
#include <stdint.h>

#include "fips202.h"

#ifdef SHAKE256

#define NROUNDS 24
static __device__ __forceinline__ uint2 operator^(uint2 a, uint2 b)
{
	return make_uint2(a.x ^ b.x, a.y ^ b.y);
} // ^
static __device__ __forceinline__ void operator^=(uint2 &a, uint2 b)
{
	a = a ^ b;
} // ^=
static __device__ __forceinline__ uint2 operator&(uint2 a, uint2 b)
{
	return make_uint2(a.x & b.x, a.y & b.y);
} // &
static __device__ __forceinline__ uint2 operator~(uint2 a)
{
	return make_uint2(~a.x, ~a.y);
} // ~

__device__ __forceinline__
uint2 u64_to_uint2(uint64_t v)
{
	uint2 result;

#ifdef USING_SHAKE_PTX
	asm ("mov.b64 {%0,%1},%2; \n\t"
	     : "=r" (result.x), "=r" (result.y) : "l" (v));
#else // ifdef USING_SHAKE_PTX
	result.x = (uint32_t)v;
	result.y = (uint32_t)(v >> 32);
#endif // ifdef USING_SHAKE_PTX
	return result;
} // u64_to_uint2

__device__ __forceinline__
uint64_t uint2_to_u64(uint2 a)
{
#ifdef USING_SHAKE_PTX
	return __double_as_longlong(__hiloint2double(a.y, a.x));
#else // ifdef USING_SHAKE_PTX
	return (((u64)(a.y) << 32) | a.x);
#endif // ifdef USING_SHAKE_PTX
} // u64_to_uint2

static __device__ __forceinline__
uint64_t ROL(const uint64_t x, const int offset)
{
	uint64_t result;

#ifdef USING_SHAKE_PTX
	asm ("{ // ROTL64 \n\t"
	     ".reg .b64 lhs;\n\t"
	     ".reg .u32 roff;\n\t"
	     "shl.b64 lhs, %1, %2;\n\t"
	     "sub.u32 roff, 64, %2;\n\t"
	     "shr.b64 %0, %1, roff;\n\t"
	     "add.u64 %0, lhs, %0;\n\t"
	     "}\n" : "=l" (result) : "l" (x), "r" (offset));
#else // ifdef USING_SHAKE_PTX
	result = ((x << offset) ^ (x >> (64 - offset)));
#endif // ifdef USING_SHAKE_PTX
	return result;
} // ROL

static __device__ __forceinline__
uint2 ROL(const uint2 a, const int offset)
{
	uint2 result;

// #ifdef USING_SHAKE_PTX
	if (offset >= 32) {
		asm ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r" (result.x) : "r" (a.x), "r" (a.y), "r" (offset));
		asm ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r" (result.y) : "r" (a.y), "r" (a.x), "r" (offset));
	}else {
		asm ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r" (result.x) : "r" (a.y), "r" (a.x), "r" (offset));
		asm ("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r" (result.y) : "r" (a.x), "r" (a.y), "r" (offset));
	}
	// below may leads to error when using link-opt, or maybe wrong
// #else // ifdef USING_SHAKE_PTX
// 	result.x = ((a.x << offset) | (a.x >> (32 - offset)));
// 	result.y = ((a.y << offset) | (a.y >> (32 - offset)));

	// new
// if(offset >= 32){
// result.x = ((a.x << offset)|(a.y >> (32-offset)));
// result.y = ((a.y << offset)|(a.x >> (32-offset)));
// } else {
// result.x = ((a.y << offset)|(a.x >> (32-offset)));
// result.y = ((a.x << offset)|(a.y >> (32-offset)));
// }
	// new end

// #endif // ifdef USING_SHAKE_PTX

	return result;
} // ROL
//
// result = a^b^c;
__device__ __forceinline__
uint2 xor3x(const uint2 a, const uint2 b, const uint2 c)
{
	uint2 result;

#ifdef USING_SHAKE_PTX
	asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r" (result.x) : "r" (a.x), "r" (b.x), "r" (c.x));
	asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r" (result.y) : "r" (a.y), "r" (b.y), "r" (c.y));
#else // ifdef USING_SHAKE_PTX
	result = a ^ b ^ c;
#endif // ifdef USING_SHAKE_PTX
	return result;
} // xor3x

// result = a ^ (~b) & c;
__device__ __forceinline__
uint2 chi(const uint2 a, const uint2 b, const uint2 c) // keccak chi
{
	uint2 result;

#ifdef USING_SHAKE_PTX
	asm ("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r" (result.x) : "r" (a.x), "r" (b.x), "r" (c.x));
	asm ("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r" (result.y) : "r" (a.y), "r" (b.y), "r" (c.y));
#else // ifdef USING_SHAKE_PTX
	result = a ^ (~b) & c;
#endif // ifdef USING_SHAKE_PTX

	return result;
} // chi

__constant__ uint2 cons2_KeccakF_RoundConstants[24] = {
	{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 }, { 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
	{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 }, { 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
	{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 }, { 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
	{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 }, { 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
	{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 }, { 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
	{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 }, { 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

__device__ __forceinline__
uint64_t xor5(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e)
{
	uint64_t result;

	asm ("xor.b64 %0, %1, %2;" : "=l" (result) : "l" (d), "l" (e));
	asm ("xor.b64 %0, %0, %1;" : "+l" (result) : "l" (c));
	asm ("xor.b64 %0, %0, %1;" : "+l" (result) : "l" (b));
	asm ("xor.b64 %0, %0, %1;" : "+l" (result) : "l" (a));
	return result;
} // xor5

/*************************************************
 * Name:        load64
 *
 * Description: Load 8 bytes into uint64_t in little-endian order
 *
 * Arguments:   - const uint8_t *x: pointer to input byte array
 *
 * Returns the loaded 64-bit unsigned integer
 **************************************************/
__device__ uint64_t dev_load64(const uint8_t *x)
{
	uint64_t r = 0;

	for (size_t i = 0; i < 8; ++i) {
		r |= (uint64_t)x[i] << 8 * i;
	}

	return r;
} // dev_load64

/*************************************************
 * Name:        store64
 *
 * Description: Store a 64-bit integer to a byte array in little-endian order
 *
 * Arguments:   - uint8_t *x: pointer to the output byte array
 *              - uint64_t u: input 64-bit unsigned integer
 **************************************************/
__device__ void dev_store64(uint8_t *x, uint64_t u)
{
	for (size_t i = 0; i < 8; ++i) {
		x[i] = (uint8_t)(u >> 8 * i);
	}
} // dev_store64

/* Keccak round constants */
__constant__ u64 cons_KeccakF_RoundConstants[NROUNDS] = {
	0x0000000000000001ULL, 0x0000000000008082ULL,
	0x800000000000808aULL, 0x8000000080008000ULL,
	0x000000000000808bULL, 0x0000000080000001ULL,
	0x8000000080008081ULL, 0x8000000000008009ULL,
	0x000000000000008aULL, 0x0000000000000088ULL,
	0x0000000080008009ULL, 0x000000008000000aULL,
	0x000000008000808bULL, 0x800000000000008bULL,
	0x8000000000008089ULL, 0x8000000000008003ULL,
	0x8000000000008002ULL, 0x8000000000000080ULL,
	0x000000000000800aULL, 0x800000008000000aULL,
	0x8000000080008081ULL, 0x8000000000008080ULL,
	0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ unsigned char cons_rhotates[5][5] = {
	{  0, 1,  62,  28,   27										    },
	{ 36, 44, 6,   55,   20										    },
	{  3, 10, 43,  25,   39										    },
	{ 41, 45, 15,  21,   8										    },
	{ 18, 2,  61,  56,   14										    }
};

/*************************************************
 * Name:        KeccakF1600_StatePermute
 *
 * Description: The Keccak F1600 Permutation
 *
 * Arguments:   - uint64_t *state: pointer to input/output Keccak state
 **************************************************/

#ifdef USING_SHAKE_HAND_UNROLL
// // 252
// // KECCAK_1X_ALT
__device__ void dev_KeccakF1600_StatePermute(uint64_t *state)
{
	uint2 A00, A01, A02, A03, A04;
	uint2 A10, A11, A12, A13, A14;
	uint2 A20, A21, A22, A23, A24;
	uint2 A30, A31, A32, A33, A34;
	uint2 A40, A41, A42, A43, A44;

	uint2 C0, C1, C2, C3, C4;
	uint2 D0, D1;

	A00 = u64_to_uint2(state[0]);
	A01 = u64_to_uint2(state[1]);
	A02 = u64_to_uint2(state[2]);
	A03 = u64_to_uint2(state[3]);
	A04 = u64_to_uint2(state[4]);
	A10 = u64_to_uint2(state[5]);
	A11 = u64_to_uint2(state[6]);
	A12 = u64_to_uint2(state[7]);
	A13 = u64_to_uint2(state[8]);
	A14 = u64_to_uint2(state[9]);
	A20 = u64_to_uint2(state[10]);
	A21 = u64_to_uint2(state[11]);
	A22 = u64_to_uint2(state[12]);
	A23 = u64_to_uint2(state[13]);
	A24 = u64_to_uint2(state[14]);
	A30 = u64_to_uint2(state[15]);
	A31 = u64_to_uint2(state[16]);
	A32 = u64_to_uint2(state[17]);
	A33 = u64_to_uint2(state[18]);
	A34 = u64_to_uint2(state[19]);
	A40 = u64_to_uint2(state[20]);
	A41 = u64_to_uint2(state[21]);
	A42 = u64_to_uint2(state[22]);
	A43 = u64_to_uint2(state[23]);
	A44 = u64_to_uint2(state[24]);

#ifdef USING_SHAKE_UNROLL
	#pragma unroll 25
#endif // ifdef USING_SHAKE_UNROLL
	for (u32 i = 0; i < 24; i++) {
		C0 = A00 ^ A10 ^ A20 ^ A30 ^ A40;
		C1 = A01 ^ A11 ^ A21 ^ A31 ^ A41;
		C2 = A02 ^ A12 ^ A22 ^ A32 ^ A42;
		C3 = A03 ^ A13 ^ A23 ^ A33 ^ A43;
		C4 = A04 ^ A14 ^ A24 ^ A34 ^ A44;

		D0 = C0 ^ ROL(C2, 1);
		D1 = C1 ^ ROL(C3, 1);
		C2 ^= ROL(C4, 1);
		C3 ^= ROL(C0, 1);
		C4 ^= ROL(C1, 1);

		A01 ^= D0;
		A11 ^= D0;
		A21 ^= D0;
		A31 ^= D0;
		A41 ^= D0;

		A02 ^= D1;
		A12 ^= D1;
		A22 ^= D1;
		A32 ^= D1;
		A42 ^= D1;

		A03 ^= C2;
		A13 ^= C2;
		A23 ^= C2;
		A33 ^= C2;
		A43 ^= C2;

		A04 ^= C3;
		A14 ^= C3;
		A24 ^= C3;
		A34 ^= C3;
		A44 ^= C3;

		A00 ^= C4;
		A10 ^= C4;
		A20 ^= C4;
		A30 ^= C4;
		A40 ^= C4;

		C1 = A01;
		C2 = A02;
		C3 = A03;
		C4 = A04;

		A01 = ROL(A11, 44);
		A02 = ROL(A22, 43);
		A03 = ROL(A33, 21);
		A04 = ROL(A44, 14);

		A11 = ROL(A14, 20);
		A22 = ROL(A23, 25);
		A33 = ROL(A32, 15);
		A44 = ROL(A41, 2);

		A14 = ROL(A42, 61);
		A23 = ROL(A34, 8);
		A32 = ROL(A21, 10);
		A41 = ROL(A13, 55);

		A42 = ROL(A24, 39);
		A34 = ROL(A43, 56);
		A21 = ROL(A12, 6);
		A13 = ROL(A31, 45);

		A24 = ROL(A40, 18);
		A43 = ROL(A30, 41);
		A12 = ROL(A20, 3);
		A31 = ROL(A10, 36);

		A10 = ROL(C3,  28);
		A20 = ROL(C1,  1);
		A30 = ROL(C4,  27);
		A40 = ROL(C2,  62);

		C0 = A00;
		C1 = A10;
		D0 = A01;
		D1 = A11;

		A00 = chi(A00, A01, A02);
		A10 = chi(A10, A11, A12);
		A01 = chi(A01, A02, A03);
		A11 = chi(A11, A12, A13);
		A02 = chi(A02, A03, A04);
		A12 = chi(A12, A13, A14);
		A03 = chi(A03, A04, C0);
		A13 = chi(A13, A14, C1);
		A04 = chi(A04, C0, D0);
		A14 = chi(A14, C1, D1);

		C2 = A20;
		C3 = A30;
		D0 = A21;
		D1 = A31;

		A20 = chi(A20, A21, A22);
		A30 = chi(A30, A31, A32);
		A21 = chi(A21, A22, A23);
		A31 = chi(A31, A32, A33);
		A22 = chi(A22, A23, A24);
		A32 = chi(A32, A33, A34);
		A23 = chi(A23, A24, C2);
		A33 = chi(A33, A34, C3);
		A24 = chi(A24, C2, D0);
		A34 = chi(A34, C3, D1);

		C4 = A40;
		D1 = A41;

		A40 = chi(A40, A41, A42);
		A41 = chi(A41, A42, A43);
		A42 = chi(A42, A43, A44);
		A43 = chi(A43, A44, C4);
		A44 = chi(A44, C4, D1);
		A00 ^= cons2_KeccakF_RoundConstants[i];
	}

	state[0] = uint2_to_u64(A00);
	state[1] = uint2_to_u64(A01);
	state[2] = uint2_to_u64(A02);
	state[3] = uint2_to_u64(A03);
	state[4] = uint2_to_u64(A04);
	state[5] = uint2_to_u64(A10);
	state[6] = uint2_to_u64(A11);
	state[7] = uint2_to_u64(A12);
	state[8] = uint2_to_u64(A13);
	state[9] = uint2_to_u64(A14);
	state[10] = uint2_to_u64(A20);
	state[11] = uint2_to_u64(A21);
	state[12] = uint2_to_u64(A22);
	state[13] = uint2_to_u64(A23);
	state[14] = uint2_to_u64(A24);
	state[15] = uint2_to_u64(A30);
	state[16] = uint2_to_u64(A31);
	state[17] = uint2_to_u64(A32);
	state[18] = uint2_to_u64(A33);
	state[19] = uint2_to_u64(A34);
	state[20] = uint2_to_u64(A40);
	state[21] = uint2_to_u64(A41);
	state[22] = uint2_to_u64(A42);
	state[23] = uint2_to_u64(A43);
	state[24] = uint2_to_u64(A44);

}       // dev_KeccakF1600_StatePermute
#else   // ifdef USING_SHAKE_HAND_UNROLL
// openssl ref verison
__device__ void dev_KeccakF1600_StatePermute(uint64_t *state)
{
#ifdef USING_SHAKE_UINT2
	uint2 A[5][5];
	uint2 *A1 = &A[0][0];
	uint2 C[5], D[5];
	uint2 T[5][5];

	for (int i = 0; i < 25; i++)
		A1[i] = u64_to_uint2(state[i]);
#else // ifdef USING_SHAKE_UINT2
	uint64_t A[5][5];
	uint64_t C[5], D[5];
	uint64_t T[5][5];

	memcpy(A, state, sizeof(uint64_t) * 25);
#endif // ifdef USING_SHAKE_UINT2

	// #pragma unroll 24
	for (size_t i = 0; i < 24; i++) {
		// 1

		C[0] = A[0][0];
		C[1] = A[0][1];
		C[2] = A[0][2];
		C[3] = A[0][3];
		C[4] = A[0][4];

		for (size_t y = 1; y < 5; y++) {
			C[0] ^= A[y][0];
			C[1] ^= A[y][1];
			C[2] ^= A[y][2];
			C[3] ^= A[y][3];
			C[4] ^= A[y][4];
		}

		D[0] = ROL(C[1], 1) ^ C[4];
		D[1] = ROL(C[2], 1) ^ C[0];
		D[2] = ROL(C[3], 1) ^ C[1];
		D[3] = ROL(C[4], 1) ^ C[2];
		D[4] = ROL(C[0], 1) ^ C[3];

		for (size_t y = 0; y < 5; y++) {
			A[y][0] ^= D[0];
			A[y][1] ^= D[1];
			A[y][2] ^= D[2];
			A[y][3] ^= D[3];
			A[y][4] ^= D[4];
		}

		// 2
		for (size_t y = 0; y < 5; y++) {
			A[y][0] = ROL(A[y][0], cons_rhotates[y][0]);
			A[y][1] = ROL(A[y][1], cons_rhotates[y][1]);
			A[y][2] = ROL(A[y][2], cons_rhotates[y][2]);
			A[y][3] = ROL(A[y][3], cons_rhotates[y][3]);
			A[y][4] = ROL(A[y][4], cons_rhotates[y][4]);
		}

		// 3
		memcpy(T, A, sizeof(T));

		A[0][0] = T[0][0];
		A[0][1] = T[1][1];
		A[0][2] = T[2][2];
		A[0][3] = T[3][3];
		A[0][4] = T[4][4];

		A[1][0] = T[0][3];
		A[1][1] = T[1][4];
		A[1][2] = T[2][0];
		A[1][3] = T[3][1];
		A[1][4] = T[4][2];

		A[2][0] = T[0][1];
		A[2][1] = T[1][2];
		A[2][2] = T[2][3];
		A[2][3] = T[3][4];
		A[2][4] = T[4][0];

		A[3][0] = T[0][4];
		A[3][1] = T[1][0];
		A[3][2] = T[2][1];
		A[3][3] = T[3][2];
		A[3][4] = T[4][3];

		A[4][0] = T[0][2];
		A[4][1] = T[1][3];
		A[4][2] = T[2][4];
		A[4][3] = T[3][0];
		A[4][4] = T[4][1];

		//4
		for (size_t y = 0; y < 5; y++) {
			C[0] = A[y][0] ^ (~A[y][1] & A[y][2]);
			C[1] = A[y][1] ^ (~A[y][2] & A[y][3]);
			C[2] = A[y][2] ^ (~A[y][3] & A[y][4]);
			C[3] = A[y][3] ^ (~A[y][4] & A[y][0]);
			C[4] = A[y][4] ^ (~A[y][0] & A[y][1]);

			A[y][0] = C[0];
			A[y][1] = C[1];
			A[y][2] = C[2];
			A[y][3] = C[3];
			A[y][4] = C[4];
		}

		//5
	#ifdef USING_SHAKE_UINT2
		A[0][0] ^= cons2_KeccakF_RoundConstants[i];
	#else // ifdef USING_SHAKE_UINT2
		A[0][0] ^= cons_KeccakF_RoundConstants[i];
	#endif // ifdef USING_SHAKE_UINT2
	}
	memcpy(state, A, sizeof(uint64_t) * 25);
} // dev_KeccakF1600_StatePermute
#endif // ifdef USING_SHAKE_HAND_UNROLL

/*************************************************
 * Name:        keccak_absorb
 *
 * Description: Absorb step of Keccak;
 *              non-incremental, starts by zeroeing the state.
 *
 * Arguments:   - uint64_t *s: pointer to (uninitialized) output Keccak state
 *              - uint32_t r: rate in bytes (e.g., 168 for SHAKE128)
 *              - const uint8_t *m: pointer to input to be absorbed into s
 *              - size_t mlen: length of input in bytes
 *              - uint8_t p: domain-separation byte for different
 *                                 Keccak-derived functions
 **************************************************/
__device__ void dev_keccak_absorb(uint64_t *s, uint32_t r, const uint8_t *m,
				  size_t mlen, uint8_t p)
{
	size_t i;
	uint8_t t[200];

	/* Zero state */
	for (i = 0; i < 25; ++i) {
		s[i] = 0;
	}

	while (mlen >= r) {
		for (i = 0; i < r / 8; ++i) {
		#ifdef USING_SHAKE_INTEGER
			uint64_t Ai;
			memcpy(&Ai, &m[8 * i], 8);
			s[i] ^= Ai;
		#else // ifdef USING_SHAKE_INTEGER
			s[i] ^= dev_load64(m + 8 * i);
		#endif // ifdef USING_SHAKE_INTEGER
		}

		dev_KeccakF1600_StatePermute(s);
		mlen -= r;
		m += r;
	}

	for (i = 0; i < r; ++i) {
		t[i] = 0;
	}
	for (i = 0; i < mlen; ++i) {
		t[i] = m[i];
	}
	t[i] = p;
	t[r - 1] |= 128;
	for (i = 0; i < r / 8; ++i) {
		#ifdef USING_SHAKE_INTEGER
		uint64_t Ai;
		memcpy(&Ai, &t[8 * i], 8);
		s[i] ^= Ai;
		#else // ifdef USING_SHAKE_INTEGER
		s[i] ^= dev_load64(t + 8 * i);
		#endif // ifdef USING_SHAKE_INTEGER
	}
} // dev_keccak_absorb

/*************************************************
 * Name:        keccak_squeezeblocks
 *
 * Description: Squeeze step of Keccak. Squeezes full blocks of r bytes each.
 *              Modifies the state. Can be called multiple times to keep
 *              squeezing, i.e., is incremental.
 *
 * Arguments:   - uint8_t *h: pointer to output blocks
 *              - size_t nblocks: number of blocks to be
 *                                                squeezed (written to h)
 *              - uint64_t *s: pointer to input/output Keccak state
 *              - uint32_t r: rate in bytes (e.g., 168 for SHAKE128)
 **************************************************/
__device__ void dev_keccak_squeezeblocks(uint8_t *h, size_t nblocks,
					 uint64_t *s, uint32_t r)
{
	while (nblocks > 0) {
		dev_KeccakF1600_StatePermute(s);
		for (size_t i = 0; i < (r >> 3); i++) {
			#ifdef USING_SHAKE_INTEGER
			memcpy(h + 8 * i, &s[i], 8);
			#else // ifdef USING_SHAKE_INTEGER
			dev_store64(h + 8 * i, s[i]);
			#endif // ifdef USING_SHAKE_INTEGER
		}
		h += r;
		nblocks--;
	}
} // dev_keccak_squeezeblocks

/*************************************************
 * Name:        keccak_inc_init
 *
 * Description: Initializes the incremental Keccak state to zero.
 *
 * Arguments:   - uint64_t *s_inc: pointer to input/output incremental state
 *                First 25 values represent Keccak state.
 *                26th value represents either the number of absorbed bytes
 *                that have not been permuted, or not-yet-squeezed bytes.
 **************************************************/
__device__ void dev_keccak_inc_init(uint64_t *s_inc)
{
	size_t i;

	for (i = 0; i < 25; ++i) {
		s_inc[i] = 0;
	}
	s_inc[25] = 0;
} // dev_keccak_inc_init

/*************************************************
 * Name:        keccak_inc_absorb
 *
 * Description: Incremental keccak absorb
 *              Preceded by keccak_inc_init, succeeded by keccak_inc_finalize
 *
 * Arguments:   - uint64_t *s_inc: pointer to input/output incremental state
 *                First 25 values represent Keccak state.
 *                26th value represents either the number of absorbed bytes
 *                that have not been permuted, or not-yet-squeezed bytes.
 *              - uint32_t r: rate in bytes (e.g., 168 for SHAKE128)
 *              - const uint8_t *m: pointer to input to be absorbed into s
 *              - size_t mlen: length of input in bytes
 **************************************************/
__device__ void dev_keccak_inc_absorb(uint64_t *s_inc, uint32_t r, const uint8_t *m,
				      size_t mlen)
{
	size_t i;

	/* Recall that s_inc[25] is the non-absorbed bytes xored into the state */
	while (mlen + s_inc[25] >= r) {
		for (i = 0; i < r - s_inc[25]; i++) {
			/* Take the i'th byte from message
			   xor with the s_inc[25] + i'th byte of the state; little-endian */
			s_inc[(s_inc[25] + i) >> 3] ^= (uint64_t)m[i] << (8 * ((s_inc[25] + i) & 0x07));
		}
		mlen -= (size_t)(r - s_inc[25]);
		m += r - s_inc[25];
		s_inc[25] = 0;

		dev_KeccakF1600_StatePermute(s_inc);
	}

	for (i = 0; i < mlen; i++) {
		s_inc[(s_inc[25] + i) >> 3] ^= (uint64_t)m[i] << (8 * ((s_inc[25] + i) & 0x07));
	}
	s_inc[25] += mlen;
} // dev_keccak_inc_absorb

/*************************************************
 * Name:        keccak_inc_finalize
 *
 * Description: Finalizes Keccak absorb phase, prepares for squeezing
 *
 * Arguments:   - uint64_t *s_inc: pointer to input/output incremental state
 *                First 25 values represent Keccak state.
 *                26th value represents either the number of absorbed bytes
 *                that have not been permuted, or not-yet-squeezed bytes.
 *              - uint32_t r: rate in bytes (e.g., 168 for SHAKE128)
 *              - uint8_t p: domain-separation byte for different
 *                                 Keccak-derived functions
 **************************************************/
__device__ void dev_keccak_inc_finalize(uint64_t *s_inc, uint32_t r, uint8_t p)
{
	/* After keccak_inc_absorb, we are guaranteed that s_inc[25] < r,
	   so we can always use one more byte for p in the current state. */
	s_inc[s_inc[25] >> 3] ^= (uint64_t)p << (8 * (s_inc[25] & 0x07));
	s_inc[(r - 1) >> 3] ^= (uint64_t)128 << (8 * ((r - 1) & 0x07));
	s_inc[25] = 0;
} // dev_keccak_inc_finalize

/*************************************************
 * Name:        keccak_inc_squeeze
 *
 * Description: Incremental Keccak squeeze; can be called on byte-level
 *
 * Arguments:   - uint8_t *h: pointer to output bytes
 *              - size_t outlen: number of bytes to be squeezed
 *              - uint64_t *s_inc: pointer to input/output incremental state
 *                First 25 values represent Keccak state.
 *                26th value represents either the number of absorbed bytes
 *                that have not been permuted, or not-yet-squeezed bytes.
 *              - uint32_t r: rate in bytes (e.g., 168 for SHAKE128)
 **************************************************/
__device__ void dev_keccak_inc_squeeze(uint8_t *h, size_t outlen,
				       uint64_t *s_inc, uint32_t r)
{
	size_t i;

	/* First consume any bytes we still have sitting around */
	for (i = 0; i < outlen && i < s_inc[25]; i++) {
		/* There are s_inc[25] bytes left, so r - s_inc[25] is the first
		   available byte. We consume from there, i.e., up to r. */
		h[i] = (uint8_t)(s_inc[(r - s_inc[25] + i) >> 3] >> (8 * ((r - s_inc[25] + i) & 0x07)));
	}
	h += i;
	outlen -= i;
	s_inc[25] -= i;

	/* Then squeeze the remaining necessary blocks */
	while (outlen > 0) {
		dev_KeccakF1600_StatePermute(s_inc);

		for (i = 0; i < outlen && i < r; i++) {
			h[i] = (uint8_t)(s_inc[i >> 3] >> (8 * (i & 0x07)));
		}
		h += i;
		outlen -= i;
		s_inc[25] = r - i;
	}
} // dev_keccak_inc_squeeze

__device__ void dev_shake256_inc_init(uint64_t *s_inc)
{
	dev_keccak_inc_init(s_inc);
} // dev_shake256_inc_init

__device__ void dev_shake256_inc_absorb(uint64_t *s_inc, const uint8_t *input, size_t inlen)
{
	dev_keccak_inc_absorb(s_inc, SHAKE256_RATE, input, inlen);
} // dev_shake256_inc_absorb

__device__ void dev_shake256_inc_finalize(uint64_t *s_inc)
{
	dev_keccak_inc_finalize(s_inc, SHAKE256_RATE, 0x1F);
} // dev_shake256_inc_finalize

__device__ void dev_shake256_inc_squeeze(uint8_t *output, size_t outlen, uint64_t *s_inc)
{
	dev_keccak_inc_squeeze(output, outlen, s_inc, SHAKE256_RATE);
} // dev_shake256_inc_squeeze


/*************************************************
 * Name:        shake256_absorb
 *
 * Description: Absorb step of the SHAKE256 XOF.
 *              non-incremental, starts by zeroeing the state.
 *
 * Arguments:   - uint64_t *s: pointer to (uninitialized) output Keccak state
 *              - const uint8_t *input: pointer to input to be absorbed
 *                                            into s
 *              - size_t inlen: length of input in bytes
 **************************************************/
__device__ void dev_shake256_absorb(uint64_t *s, const uint8_t *input, size_t inlen)
{
	dev_keccak_absorb(s, SHAKE256_RATE, input, inlen, 0x1F);
} // dev_shake256_absorb

/*************************************************
 * Name:        shake256_squeezeblocks
 *
 * Description: Squeeze step of SHAKE256 XOF. Squeezes full blocks of
 *              SHAKE256_RATE bytes each. Modifies the state. Can be called
 *              multiple times to keep squeezing, i.e., is incremental.
 *
 * Arguments:   - uint8_t *output: pointer to output blocks
 *              - size_t nblocks: number of blocks to be squeezed
 *                                (written to output)
 *              - uint64_t *s: pointer to input/output Keccak state
 **************************************************/
__device__ void dev_shake256_squeezeblocks(uint8_t *output, size_t nblocks, uint64_t *s)
{
	dev_keccak_squeezeblocks(output, nblocks, s, SHAKE256_RATE);
} // dev_shake256_squeezeblocks

/*************************************************
 * Name:        shake256
 *
 * Description: SHAKE256 XOF with non-incremental API
 *
 * Arguments:   - uint8_t *output: pointer to output
 *              - size_t outlen: requested output length in bytes
 *              - const uint8_t *input: pointer to input
 *              - size_t inlen: length of input in bytes
 **************************************************/
__device__ void dev_shake256(uint8_t *output, size_t outlen,
			     const uint8_t *input, size_t inlen)
{
	size_t nblocks = outlen / SHAKE256_RATE;
	uint8_t t[SHAKE256_RATE];
	uint64_t s[25];

	dev_shake256_absorb(s, input, inlen);
	dev_shake256_squeezeblocks(output, nblocks, s);

	output += nblocks * SHAKE256_RATE;
	outlen -= nblocks * SHAKE256_RATE;

	if (outlen) {
		dev_shake256_squeezeblocks(t, 1, s);
		for (size_t i = 0; i < outlen; ++i) {
			output[i] = t[i];
		}
	}
} // dev_shake256

#endif // ifdef SHAKE256
