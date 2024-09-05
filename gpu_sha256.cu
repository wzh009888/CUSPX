
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "sha256.h"
#include "utils.h"
#include <iostream>
using namespace std;

#ifdef SHA256

const __constant__ u32 __align__(8) cons_K256[64]
    = {0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL, 0x3956c25bUL, 0x59f111f1UL,
       0x923f82a4UL, 0xab1c5ed5UL, 0xd807aa98UL, 0x12835b01UL, 0x243185beUL, 0x550c7dc3UL,
       0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL, 0xc19bf174UL, 0xe49b69c1UL, 0xefbe4786UL,
       0x0fc19dc6UL, 0x240ca1ccUL, 0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL,
       0x983e5152UL, 0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL, 0xc6e00bf3UL, 0xd5a79147UL,
       0x06ca6351UL, 0x14292967UL, 0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL, 0x53380d13UL,
       0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL, 0xa2bfe8a1UL, 0xa81a664bUL,
       0xc24b8b70UL, 0xc76c51a3UL, 0xd192e819UL, 0xd6990624UL, 0xf40e3585UL, 0x106aa070UL,
       0x19a4c116UL, 0x1e376c08UL, 0x2748774cUL, 0x34b0bcb5UL, 0x391c0cb3UL, 0x4ed8aa4aUL,
       0x5b9cca4fUL, 0x682e6ff3UL, 0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL,
       0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL}; // __align__

__device__ uint32_t dev_load_bigendian_32(const uint8_t* x) {
    return (uint32_t) (x[3]) | (((uint32_t) (x[2])) << 8) | (((uint32_t) (x[1])) << 16)
        | (((uint32_t) (x[0])) << 24);
} // dev_load_bigendian_32

__device__ uint64_t dev_load_bigendian_64(const uint8_t* x) {
    return (uint64_t) (x[7]) | (((uint64_t) (x[6])) << 8) | (((uint64_t) (x[5])) << 16)
        | (((uint64_t) (x[4])) << 24) | (((uint64_t) (x[3])) << 32) | (((uint64_t) (x[2])) << 40)
        | (((uint64_t) (x[1])) << 48) | (((uint64_t) (x[0])) << 56);
} // dev_load_bigendian_64

__device__ void dev_store_bigendian_32(uint8_t* x, uint64_t u) {
    x[3] = (uint8_t) u;
    u >>= 8;
    x[2] = (uint8_t) u;
    u >>= 8;
    x[1] = (uint8_t) u;
    u >>= 8;
    x[0] = (uint8_t) u;
} // dev_store_bigendian_32

__device__ void dev_store_bigendian_64(uint8_t* x, uint64_t u) {
    x[7] = (uint8_t) u;
    u >>= 8;
    x[6] = (uint8_t) u;
    u >>= 8;
    x[5] = (uint8_t) u;
    u >>= 8;
    x[4] = (uint8_t) u;
    u >>= 8;
    x[3] = (uint8_t) u;
    u >>= 8;
    x[2] = (uint8_t) u;
    u >>= 8;
    x[1] = (uint8_t) u;
    u >>= 8;
    x[0] = (uint8_t) u;
} // dev_store_bigendian_64

#ifdef USING_SHA256_PTX

#if USING_SHA256_PTX_MODE == 0 // sign & default

#define Ch(a, b, c)                                                                                \
    ({                                                                                             \
        u32 result;                                                                                \
        asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(result) : "r"(a), "r"(b), "r"(c));             \
        result;                                                                                    \
    })

#define Maj(a, b, c)                                                                               \
    ({                                                                                             \
        u32 result;                                                                                \
        asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(result) : "r"(a), "r"(b), "r"(c));             \
        result;                                                                                    \
    })

#define ROL(v, n)                                                                                  \
    ({                                                                                             \
        u32 result;                                                                                \
        asm("shf.l.clamp.b32 %0, %1, %1, %2;\n\t" : "=r"(result) : "r"(v), "r"(n));                \
        result;                                                                                    \
    })

#define Sigma0_32(x)                                                                               \
    ({                                                                                             \
        u32 t1 = 0, t2 = 0;                                                                        \
        asm("shf.l.clamp.b32 %0, %2, %2, 30;\n\t"                                                  \
            "shf.l.clamp.b32 %1, %2, %2, 19;\n\t"                                                  \
            "xor.b32 %0, %0, %1;\n\t"                                                              \
            "shf.l.clamp.b32 %1, %2, %2, 10;\n\t"                                                  \
            "xor.b32 %0, %0, %1;\n\t"                                                              \
            : "+r"(t1), "+r"(t2)                                                                   \
            : "r"(x));                                                                             \
        t1;                                                                                        \
    })

#define Sigma1_32(x)                                                                               \
    ({                                                                                             \
        u32 t1 = 0, t2 = 0;                                                                        \
        asm("shf.l.clamp.b32 %0, %2, %2, 26;\n\t"                                                  \
            "shf.l.clamp.b32 %1, %2, %2, 21;\n\t"                                                  \
            "xor.b32 %0, %0, %1;\n\t"                                                              \
            "shf.l.clamp.b32 %1, %2, %2, 7;\n\t"                                                   \
            "xor.b32 %0, %0, %1;\n\t"                                                              \
            : "+r"(t1), "+r"(t2)                                                                   \
            : "r"(x));                                                                             \
        t1;                                                                                        \
    })

#define sigma0_32(x)                                                                               \
    ({                                                                                             \
        u32 t1 = 0, t2 = 0;                                                                        \
        asm("shf.l.clamp.b32 %0, %2, %2, 25;\n\t"                                                  \
            "shf.l.clamp.b32 %1, %2, %2, 14;\n\t"                                                  \
            "xor.b32 %0, %0, %1;\n\t"                                                              \
            "shr.b32 %1, %2, 3;\n\t"                                                               \
            "xor.b32 %0, %0, %1;\n\t"                                                              \
            : "+r"(t1), "+r"(t2)                                                                   \
            : "r"(x));                                                                             \
        t1;                                                                                        \
    })

#define sigma1_32(x)                                                                               \
    ({                                                                                             \
        u32 t1 = 0, t2 = 0;                                                                        \
        asm("shf.l.clamp.b32 %0, %2, %2, 15;\n\t"                                                  \
            "shf.l.clamp.b32 %1, %2, %2, 13;\n\t"                                                  \
            "xor.b32 %0, %0, %1;\n\t"                                                              \
            "shr.b32 %1, %2, 10;\n\t"                                                              \
            "xor.b32 %0, %0, %1;\n\t"                                                              \
            : "+r"(t1), "+r"(t2)                                                                   \
            : "r"(x));                                                                             \
        t1;                                                                                        \
    })

// __device__ __forceinline__ u32 Sigma1_32_Ch(u32 e, u32 f, u32 g) {
//     u32 t1 = 0, t2 = 0;
//     asm("shf.l.clamp.b32 %1, %0, %0, 26;\n\t"
//         "shf.l.clamp.b32 %2, %0, %0, 21;\n\t"
//         "xor.b32 %1, %1, %2;\n\t"
//         "shf.l.clamp.b32 %2, %0, %0, 7;\n\t"
//         "xor.b32 %1, %1, %2;\n\t"
//         "lop3.b32 %0, %0, %3, %4, 0xCA;\n\t"
//         "add.u32 %0, %0, %1;\n\t"
//         : "+r"(e), "+r"(t1), "+r"(t2)
//         : "r"(f), "r"(g));
//     return e;
// }

// __device__ __forceinline__ u32 Sigma0_32_Maj(u32 a, u32 b, u32 c) {
//     u32 t1 = 0, t2 = 0;
//     asm("shf.l.clamp.b32 %1, %0, %0, 26;\n\t"
//         "shf.l.clamp.b32 %2, %0, %0, 21;\n\t"
//         "xor.b32 %1, %1, %2;\n\t"
//         "shf.l.clamp.b32 %2, %0, %0, 7;\n\t"
//         "xor.b32 %1, %1, %2;\n\t"
//         "lop3.b32 %0, %0, %3, %4, 0xE8;\n\t"
//         "add.u32 %0, %0, %1;\n\t"
//         : "+r"(a), "+r"(t1), "+r"(t2)
//         : "r"(b), "r"(c));
//     return a;
// }

#elif USING_SHA256_PTX_MODE == 1 // kg

#define Ch(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define Maj(x, y, z) (((y) & ((x) | (z))) | ((x) & (z)))
#define ROL(v, n) (((v) << (n)) | ((v) >> (32 - (n))))

#define Sigma0_32(x) (ROL((x), 30) ^ ROL((x), 19) ^ ROL((x), 10))
#define Sigma1_32(x) (ROL((x), 26) ^ ROL((x), 21) ^ ROL((x), 7))
#define sigma0_32(x) (ROL((x), 25) ^ ROL((x), 14) ^ ((x) >> 3))
#define sigma1_32(x) (ROL((x), 15) ^ ROL((x), 13) ^ ((x) >> 10))

#elif USING_SHA256_PTX_MODE == 2 // verify
#define Ch(a, b, c)                                                                                \
    ({                                                                                             \
        u32 result;                                                                                \
        asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(result) : "r"(a), "r"(b), "r"(c));             \
        result;                                                                                    \
    })

#define Maj(a, b, c)                                                                               \
    ({                                                                                             \
        u32 result;                                                                                \
        asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(result) : "r"(a), "r"(b), "r"(c));             \
        result;                                                                                    \
    })

#define ROL(v, n) (((v) << (n)) | ((v) >> (32 - (n))))

#define Sigma0_32(x) (ROL((x), 30) ^ ROL((x), 19) ^ ROL((x), 10))
#define Sigma1_32(x) (ROL((x), 26) ^ ROL((x), 21) ^ ROL((x), 7))
#define sigma0_32(x) (ROL((x), 25) ^ ROL((x), 14) ^ ((x) >> 3))
#define sigma1_32(x) (ROL((x), 15) ^ ROL((x), 13) ^ ((x) >> 10))

#endif

#else // ifdef USING_SHA256_PTX

#define Ch(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define Maj(x, y, z) (((y) & ((x) | (z))) | ((x) & (z)))
#define ROL(v, n) (((v) << (n)) | ((v) >> (32 - (n))))
#define Sigma0_32(x) (ROL((x), 30) ^ ROL((x), 19) ^ ROL((x), 10))
#define Sigma1_32(x) (ROL((x), 26) ^ ROL((x), 21) ^ ROL((x), 7))
#define sigma0_32(x) (ROL((x), 25) ^ ROL((x), 14) ^ ((x) >> 3))
#define sigma1_32(x) (ROL((x), 15) ^ ROL((x), 13) ^ ((x) >> 10))

#endif // ifdef USING_SHA256_PTX

#ifdef USING_SHA256_INTEGER
#define HOST_c2l(c, l) (l = __byte_perm(*(c++), 0, 0x0123))
#else // ifdef USING_SHA256_INTEGER
#define HOST_c2l(c, l)                                                                             \
    (l = (((unsigned long) (*((c)++))) << 24), l |= (((unsigned long) (*((c)++))) << 16),          \
     l |= (((unsigned long) (*((c)++))) << 8), l |= (((unsigned long) (*((c)++)))))
#endif // ifdef USING_SHA256_INTEGER

#ifdef FASTER

#define ROUND_00_15(i, a, b, c, d, e, f, g, h)                                                     \
    T1 += h + Sigma1_32(e) + Ch(e, f, g) + cons_K256[i];                                           \
    h = Sigma0_32(a) + Maj(a, b, c);                                                               \
    d += T1;                                                                                       \
    h += T1;

#ifdef USING_SHA256_X_UNROLL
// x unroll version
__device__ void dev_crypto_hashblocks_sha256(uint8_t* __restrict__ statebytes,
                                             const void* __restrict__ in, size_t inlen) {

    u32 state[8];
    u32 a, b, c, d, e, f, g, h, s0, s1, T1;
    u32 X0, X1, X2, X3;
    u32 X4, X5, X6, X7;
    u32 X8, X9, X10, X11;
    u32 X12, X13, X14, X15;
    u32 num = inlen / 64;

    for (int i = 0; i < 8; i++)
        state[i] = dev_load_bigendian_32(statebytes + 4 * i);

#ifdef USING_SHA256_INTEGER
    const u32* data = (const u32*) in;
#else  // ifdef USING_SHA256_INTEGER
    const u8* data = (const u8*) in;
#endif // ifdef USING_SHA256_INTEGER

    while (num--) {
        a = state[0];
        b = state[1];
        c = state[2];
        d = state[3];
        e = state[4];
        f = state[5];
        g = state[6];
        h = state[7];

        u32 l;

        (void) HOST_c2l(data, l);
        T1 = X0 = l;
        ROUND_00_15(0, a, b, c, d, e, f, g, h);
        (void) HOST_c2l(data, l);
        T1 = X1 = l;
        ROUND_00_15(1, h, a, b, c, d, e, f, g);
        (void) HOST_c2l(data, l);
        T1 = X2 = l;
        ROUND_00_15(2, g, h, a, b, c, d, e, f);
        (void) HOST_c2l(data, l);
        T1 = X3 = l;
        ROUND_00_15(3, f, g, h, a, b, c, d, e);
        (void) HOST_c2l(data, l);
        T1 = X4 = l;
        ROUND_00_15(4, e, f, g, h, a, b, c, d);
        (void) HOST_c2l(data, l);
        T1 = X5 = l;
        ROUND_00_15(5, d, e, f, g, h, a, b, c);
        (void) HOST_c2l(data, l);
        T1 = X6 = l;
        ROUND_00_15(6, c, d, e, f, g, h, a, b);
        (void) HOST_c2l(data, l);
        T1 = X7 = l;
        ROUND_00_15(7, b, c, d, e, f, g, h, a);
        (void) HOST_c2l(data, l);
        T1 = X8 = l;
        ROUND_00_15(8, a, b, c, d, e, f, g, h);
        (void) HOST_c2l(data, l);
        T1 = X9 = l;
        ROUND_00_15(9, h, a, b, c, d, e, f, g);
        (void) HOST_c2l(data, l);
        T1 = X10 = l;
        ROUND_00_15(10, g, h, a, b, c, d, e, f);
        (void) HOST_c2l(data, l);
        T1 = X11 = l;
        ROUND_00_15(11, f, g, h, a, b, c, d, e);
        (void) HOST_c2l(data, l);
        T1 = X12 = l;
        ROUND_00_15(12, e, f, g, h, a, b, c, d);
        (void) HOST_c2l(data, l);
        T1 = X13 = l;
        ROUND_00_15(13, d, e, f, g, h, a, b, c);
        (void) HOST_c2l(data, l);
        T1 = X14 = l;
        ROUND_00_15(14, c, d, e, f, g, h, a, b);
        (void) HOST_c2l(data, l);
        T1 = X15 = l;
        ROUND_00_15(15, b, c, d, e, f, g, h, a);

        // #pragma unroll
        for (int i = 16; i < 64; i += 16) {
            s0 = sigma0_32(X1);
            s1 = sigma1_32(X14);
            T1 = X0 += s0 + s1 + X9;
            ROUND_00_15(i + 0, a, b, c, d, e, f, g, h);

            s0 = sigma0_32(X2);
            s1 = sigma1_32(X15);
            T1 = X1 += s0 + s1 + X10;
            ROUND_00_15(i + 1, h, a, b, c, d, e, f, g);

            s0 = sigma0_32(X3);
            s1 = sigma1_32(X0);
            T1 = X2 += s0 + s1 + X11;
            ROUND_00_15(i + 2, g, h, a, b, c, d, e, f);

            s0 = sigma0_32(X4);
            s1 = sigma1_32(X1);
            T1 = X3 += s0 + s1 + X12;
            ROUND_00_15(i + 3, f, g, h, a, b, c, d, e);

            s0 = sigma0_32(X5);
            s1 = sigma1_32(X2);
            T1 = X4 += s0 + s1 + X13;
            ROUND_00_15(i + 4, e, f, g, h, a, b, c, d);

            s0 = sigma0_32(X6);
            s1 = sigma1_32(X3);
            T1 = X5 += s0 + s1 + X14;
            ROUND_00_15(i + 5, d, e, f, g, h, a, b, c);

            s0 = sigma0_32(X7);
            s1 = sigma1_32(X4);
            T1 = X6 += s0 + s1 + X15;
            ROUND_00_15(i + 6, c, d, e, f, g, h, a, b);

            s0 = sigma0_32(X8);
            s1 = sigma1_32(X5);
            T1 = X7 += s0 + s1 + X0;
            ROUND_00_15(i + 7, b, c, d, e, f, g, h, a);

            // 8 - 16
            s0 = sigma0_32(X9);
            s1 = sigma1_32(X6);
            T1 = X8 += s0 + s1 + X1;
            ROUND_00_15(i + 8, a, b, c, d, e, f, g, h);

            s0 = sigma0_32(X10);
            s1 = sigma1_32(X7);
            T1 = X9 += s0 + s1 + X2;
            ROUND_00_15(i + 9, h, a, b, c, d, e, f, g);

            s0 = sigma0_32(X11);
            s1 = sigma1_32(X8);
            T1 = X10 += s0 + s1 + X3;
            ROUND_00_15(i + 10, g, h, a, b, c, d, e, f);

            s0 = sigma0_32(X12);
            s1 = sigma1_32(X9);
            T1 = X11 += s0 + s1 + X4;
            ROUND_00_15(i + 11, f, g, h, a, b, c, d, e);

            s0 = sigma0_32(X13);
            s1 = sigma1_32(X10);
            T1 = X12 += s0 + s1 + X5;
            ROUND_00_15(i + 12, e, f, g, h, a, b, c, d);

            s0 = sigma0_32(X14);
            s1 = sigma1_32(X11);
            T1 = X13 += s0 + s1 + X6;
            ROUND_00_15(i + 13, d, e, f, g, h, a, b, c);

            s0 = sigma0_32(X15);
            s1 = sigma1_32(X12);
            T1 = X14 += s0 + s1 + X7;
            ROUND_00_15(i + 14, c, d, e, f, g, h, a, b);

            s0 = sigma0_32(X0);
            s1 = sigma1_32(X13);
            T1 = X15 += s0 + s1 + X8;
            ROUND_00_15(i + 15, b, c, d, e, f, g, h, a);
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    for (int i = 0; i < 8; i++)
        dev_store_bigendian_32(statebytes + 4 * i, state[i]);
}
#else // ifdef USING_SHA256_X_UNROLL

#define ROUND_16_63(i, a, b, c, d, e, f, g, h, X)                                                  \
    do {                                                                                           \
        s0 = X[(i + 1) & 0x0f];                                                                    \
        s0 = sigma0_32(s0);                                                                        \
        s1 = X[(i + 14) & 0x0f];                                                                   \
        s1 = sigma1_32(s1);                                                                        \
        T1 = X[(i) & 0x0f] += s0 + s1 + X[(i + 9) & 0x0f];                                         \
        ROUND_00_15(i, a, b, c, d, e, f, g, h);                                                    \
    } while (0)

__device__ void dev_crypto_hashblocks_sha256(uint8_t* statebytes, const void* in, size_t inlen) {
    u32 state[8];
    u32 a, b, c, d, e, f, g, h, s0, s1, T1;
    u32 X[16];
    u32 i;
    u32 num = inlen / 64;

    for (i = 0; i < 8; i++)
        state[i] = dev_load_bigendian_32(statebytes + 4 * i);

#ifdef USING_SHA256_INTEGER
    const u32* data = (const u32*) in;
#else  // ifdef USING_SHA256_INTEGER
    const u8* data = (const u8*) in;
#endif // ifdef USING_SHA256_INTEGER

    while (num--) {
        a = state[0];
        b = state[1];
        c = state[2];
        d = state[3];
        e = state[4];
        f = state[5];
        g = state[6];
        h = state[7];

        u32 l;

        (void) HOST_c2l(data, l);
        T1 = X[0] = l;
        ROUND_00_15(0, a, b, c, d, e, f, g, h);
        (void) HOST_c2l(data, l);
        T1 = X[1] = l;
        ROUND_00_15(1, h, a, b, c, d, e, f, g);
        (void) HOST_c2l(data, l);
        T1 = X[2] = l;
        ROUND_00_15(2, g, h, a, b, c, d, e, f);
        (void) HOST_c2l(data, l);
        T1 = X[3] = l;
        ROUND_00_15(3, f, g, h, a, b, c, d, e);
        (void) HOST_c2l(data, l);
        T1 = X[4] = l;
        ROUND_00_15(4, e, f, g, h, a, b, c, d);
        (void) HOST_c2l(data, l);
        T1 = X[5] = l;
        ROUND_00_15(5, d, e, f, g, h, a, b, c);
        (void) HOST_c2l(data, l);
        T1 = X[6] = l;
        ROUND_00_15(6, c, d, e, f, g, h, a, b);
        (void) HOST_c2l(data, l);
        T1 = X[7] = l;
        ROUND_00_15(7, b, c, d, e, f, g, h, a);
        (void) HOST_c2l(data, l);
        T1 = X[8] = l;
        ROUND_00_15(8, a, b, c, d, e, f, g, h);
        (void) HOST_c2l(data, l);
        T1 = X[9] = l;
        ROUND_00_15(9, h, a, b, c, d, e, f, g);
        (void) HOST_c2l(data, l);
        T1 = X[10] = l;
        ROUND_00_15(10, g, h, a, b, c, d, e, f);
        (void) HOST_c2l(data, l);
        T1 = X[11] = l;
        ROUND_00_15(11, f, g, h, a, b, c, d, e);
        (void) HOST_c2l(data, l);
        T1 = X[12] = l;
        ROUND_00_15(12, e, f, g, h, a, b, c, d);
        (void) HOST_c2l(data, l);
        T1 = X[13] = l;
        ROUND_00_15(13, d, e, f, g, h, a, b, c);
        (void) HOST_c2l(data, l);
        T1 = X[14] = l;
        ROUND_00_15(14, c, d, e, f, g, h, a, b);
        (void) HOST_c2l(data, l);
        T1 = X[15] = l;
        ROUND_00_15(15, b, c, d, e, f, g, h, a);

#pragma unroll 6
        for (i = 16; i < 64; i += 8) {
            ROUND_16_63(i + 0, a, b, c, d, e, f, g, h, X);
            ROUND_16_63(i + 1, h, a, b, c, d, e, f, g, X);
            ROUND_16_63(i + 2, g, h, a, b, c, d, e, f, X);
            ROUND_16_63(i + 3, f, g, h, a, b, c, d, e, X);
            ROUND_16_63(i + 4, e, f, g, h, a, b, c, d, X);
            ROUND_16_63(i + 5, d, e, f, g, h, a, b, c, X);
            ROUND_16_63(i + 6, c, d, e, f, g, h, a, b, X);
            ROUND_16_63(i + 7, b, c, d, e, f, g, h, a, X);
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    for (i = 0; i < 8; i++)
        dev_store_bigendian_32(statebytes + 4 * i, state[i]);

} // sha256_block_data_order
#endif // ifdef USING_SHA256_X_UNROLL

#else // ifdef FASTER

__device__ void dev_crypto_hashblocks_sha256(uint8_t* statebytes, const void* in, size_t inlen) {
    u32 state[8];
    u32 a, b, c, d, e, f, g, h, s0, s1, T1, T2;
    u32 X[16], l;
    u32 i;
    u32 num = inlen / 64;

    for (i = 0; i < 8; i++)
        state[i] = dev_load_bigendian_32(statebytes + 4 * i);

#ifdef USING_SHA256_INTEGER
    const u32* data = (const u32*) in;
#else  // ifdef USING_SHA256_INTEGER
    const u8* data = (const u8*) in;
#endif // ifdef USING_SHA256_INTEGER

    while (num--) {
        a = state[0];
        b = state[1];
        c = state[2];
        d = state[3];
        e = state[4];
        f = state[5];
        g = state[6];
        h = state[7];

#ifdef USING_SHA256_UNROLL
#pragma unroll
#endif // ifdef USING_SHA256_UNROLL
        for (i = 0; i < 16; i++) {
            (void) HOST_c2l(data, l);
            T1 = X[i] = l;
            T1 += h + Sigma1_32(e) + Ch(e, f, g) + cons_K256[i];
            T2 = Sigma0_32(a) + Maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
        }

#ifdef USING_SHA256_UNROLL
#pragma unroll
#endif // ifdef USING_SHA256_UNROLL
        for (i = 16; i < 64; i++) {
            s0 = X[(i + 1) & 0x0f];
            s0 = sigma0_32(s0);
            s1 = X[(i + 14) & 0x0f];
            s1 = sigma1_32(s1);

            T1 = X[i & 0xf] += s0 + s1 + X[(i + 9) & 0xf];
            T1 += h + Sigma1_32(e) + Ch(e, f, g) + cons_K256[i];
            T2 = Sigma0_32(a) + Maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + T1;
            d = c;
            c = b;
            b = a;
            a = T1 + T2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    for (i = 0; i < 8; i++)
        dev_store_bigendian_32(statebytes + 4 * i, state[i]);
} // dev_crypto_hashblocks_sha256

#endif // ifdef FASTER

__device__ void dev_sha256_inc_init(uint8_t* state) {
    u8 iv[40] = {0x6a, 0x09, 0xe6, 0x67, 0xbb, 0x67, 0xae, 0x85, 0x3c, 0x6e, 0xf3, 0x72, 0xa5, 0x4f,
                 0xf5, 0x3a, 0x51, 0x0e, 0x52, 0x7f, 0x9b, 0x05, 0x68, 0x8c, 0x1f, 0x83, 0xd9, 0xab,
                 0x5b, 0xe0, 0xcd, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    memcpy(state, iv, 40);
} // dev_sha256_inc_init

__device__ void dev_sha256_inc_blocks(uint8_t* state, const void* in, size_t inblocks) {
    if (inblocks == 0) return;

    uint64_t bytes = dev_load_bigendian_64(state + 32);

    dev_crypto_hashblocks_sha256(state, in, 64 * inblocks);
    bytes += 64 * inblocks;

    dev_store_bigendian_64(state + 32, bytes);
} // dev_sha256_inc_blocks

__device__ void dev_sha256_inc_finalize(uint8_t* out, uint8_t* state, void* in_, size_t inlen) {
    // u8 padded[128];
    uint32_t padded_[32];
    u8* padded = (u8*) padded_;

    memset(padded, 0, 128);
    uint64_t bytes = dev_load_bigendian_64(state + 32) + inlen;

    u8* in = (u8*) in_;

    dev_crypto_hashblocks_sha256(state, in, inlen);

    in += inlen;
    inlen &= 63;
    in -= inlen;

    if (inlen != 0) memcpy(padded, in, inlen);

    padded[inlen] = 0x80;

    u32 bytes_arr[8] = {53, 45, 37, 29, 21, 13, 5, 3};

    if (inlen < 56) {
        memset(padded + inlen + 1, 0, 56 - inlen - 1);
        // padded[63] = (uint8_t)(bytes << 3);
        // padded[56] = (uint8_t)(bytes >> 53);
        // padded[57] = (uint8_t)(bytes >> 45);
        // padded[58] = (uint8_t)(bytes >> 37);
        // padded[59] = (uint8_t)(bytes >> 29);
        // padded[60] = (uint8_t)(bytes >> 21);
        // padded[61] = (uint8_t)(bytes >> 13);
        // padded[62] = (uint8_t)(bytes >> 5);
        // padded[63] = (uint8_t)(bytes << 3);
        for (size_t i = 0; i < 7; i++)
            padded[i + 56] = (uint8_t) (bytes >> bytes_arr[i]);
        padded[63] = (uint8_t) (bytes << 3);

        dev_crypto_hashblocks_sha256(state, (void*) padded, 64);
    } else {
        memset(in + inlen + 1, 0, 120 - inlen - 1);
        padded[120] = (uint8_t) (bytes >> 53);
        padded[121] = (uint8_t) (bytes >> 45);
        padded[122] = (uint8_t) (bytes >> 37);
        padded[123] = (uint8_t) (bytes >> 29);
        padded[124] = (uint8_t) (bytes >> 21);
        padded[125] = (uint8_t) (bytes >> 13);
        padded[126] = (uint8_t) (bytes >> 5);
        padded[127] = (uint8_t) (bytes << 3);
        // for (size_t i = 0; i < 8; i++)
        // 	padded[i + 120] = (uint8_t)(bytes >> bytes_arr[i]);
        dev_crypto_hashblocks_sha256(state, (void*) padded, 128);
    }
    memcpy(out, state, 32);

} 

__device__ void dev_sha256(uint8_t* out, uint8_t* in, size_t inlen) {
    uint8_t state[40];
    static u8 m[32];

    // if (out == NULL) out = m;

    dev_sha256_inc_init(state);
    dev_sha256_inc_finalize(out, state, in, inlen);
}


// Kim version
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

// #define USE_GPU_SPHINCS_SECURITY_LEVEL1
// #define USE_GPU_SPHINCS_SHA256
// //!Data type Define
// // typedef unsigned char uint8_t;
// // typedef unsigned int uint32_t;
// // typedef unsigned long long uint64_t;

// //!SPHINCS+_iternal_index Define
// #define OFFSET_LAYER	0
// #define OFFSET_TREE		1
// #define OFFSET_KP_ADDR1	13
// #define OFFSET_KP_ADDR2	12
// #define OFFSET_TYPE		9
// #define OFFSET_TREE_HGT 17
// #define OFFSET_TREE_INDEX 18
// #define OFFSET_CHAIN_ADDR	17
// #define OFFSET_HASH_ADDR 21
// #define ADDR_TYPE_FORSTREE	3
// #define ADDR_TYPE_FORS_PK	4
// #define ADDR_TYPE_HASHTREE	2
// #define ADDR_TYPE_WOTS		0

// #ifdef USE_GPU_SPHINCS_SECURITY_LEVEL1
// //!Parameter Define
// #define HASH_DIGEST		16
// #define HASH_BLOCK		64
// #define HASH_OUTBYTE	32
// #define HASH_ADDR_BYTES	22

// //!SPHINCS+ Parameter Define
// #define optrand_size	32
// #define FULL_HEIGHT		66
// #define SUBTREE_LAYER	22
// #define TREE_HEIGHT		(FULL_HEIGHT / SUBTREE_LAYER)
// #define PK_BYTE			(2 * HASH_DIGEST)
// #define SK_BYTE			(2 * HASH_DIGEST + PK_BYTE)

// //!FORS Parameter Define
// #define FORS_HEIGHT		6
// #define FORS_TREE		33
// #define FORS_MSG_BYTE	((FORS_HEIGHT * FORS_TREE + 7) / 8)
// #define	FORS_BYTES		((FORS_HEIGHT + 1) * FORS_TREE * HASH_DIGEST)

// //!WOTS Parameter Define
// #define WOTS_W			16
// #define WOTS_LOGW		4
// #define WOTS_LEN1		(8 * HASH_DIGEST / WOTS_LOGW)
// #define WOTS_LEN2		3
// #define WOTS_LEN		(WOTS_LEN1 + WOTS_LEN2)
// #define WOTS_BYTES		(WOTS_LEN * HASH_DIGEST)

// //!SPHINCS+ Sig size Define
// #define SIG_BYTE	(HASH_DIGEST + FORS_BYTES + SUBTREE_LAYER * WOTS_BYTES + FULL_HEIGHT * HASH_DIGEST)
// #endif

// //!CPU_Phase Functions
// // void CPU_hash_initialize_hash_function(uint8_t* pub_seed, uint8_t* sk_seed, uint8_t* state_seed);
// // void CPU_randombytes(uint8_t* in, size_t len);
// // void CPU_gen_message_random(uint8_t* sig, uint8_t* sk_prf, uint8_t* optrand, uint8_t* m, size_t mlen);
// // void CPU_hash_message(uint8_t* digest, uint64_t* tree, uint32_t* leaf_idx, uint8_t* R, uint8_t* pk, uint8_t* m, uint64_t mlen);


// __device__ uint64_t GPU_bytes_to_ull(uint8_t* in, uint32_t inlen)
// {
//     uint64_t retval = 0;
//     uint32_t i;

//     for (i = 0; i < inlen; i++) {
//         retval |= ((uint64_t)in[i]) << (8 * (inlen - 1 - i));
//     }
//     return retval;
// }
// __device__ uint32_t GPU_load_bigendian_32(uint8_t* x) {
//     return (uint32_t)(x[3]) | (((uint32_t)(x[2])) << 8) |
//         (((uint32_t)(x[1])) << 16) | (((uint32_t)(x[0])) << 24);
// }
// __device__ uint64_t GPU_load_bigendian_64(uint8_t* x) {
//     return (uint64_t)(x[7]) | (((uint64_t)(x[6])) << 8) |
//         (((uint64_t)(x[5])) << 16) | (((uint64_t)(x[4])) << 24) |
//         (((uint64_t)(x[3])) << 32) | (((uint64_t)(x[2])) << 40) |
//         (((uint64_t)(x[1])) << 48) | (((uint64_t)(x[0])) << 56);
// }
// __device__ void GPU_store_bigendian_32(uint8_t* x, uint64_t u) {
//     x[3] = (uint8_t)u;
//     u >>= 8;
//     x[2] = (uint8_t)u;
//     u >>= 8;
//     x[1] = (uint8_t)u;
//     u >>= 8;
//     x[0] = (uint8_t)u;
// }
// __device__ void GPU_store_bigendian_64(uint8_t* x, uint64_t u) {
//     x[7] = (uint8_t)u;
//     u >>= 8;
//     x[6] = (uint8_t)u;
//     u >>= 8;
//     x[5] = (uint8_t)u;
//     u >>= 8;
//     x[4] = (uint8_t)u;
//     u >>= 8;
//     x[3] = (uint8_t)u;
//     u >>= 8;
//     x[2] = (uint8_t)u;
//     u >>= 8;
//     x[1] = (uint8_t)u;
//     u >>= 8;
//     x[0] = (uint8_t)u;
// }
// __device__ void GPU_u32_to_bytes(uint8_t* out, uint32_t in) {
//     out[0] = (uint8_t)(in >> 24);
//     out[1] = (uint8_t)(in >> 16);
//     out[2] = (uint8_t)(in >> 8);
//     out[3] = (uint8_t)in;
// }
// __device__ void GPU_set_type(uint32_t* addr, uint32_t type) {
//     ((uint8_t*)addr)[OFFSET_TYPE] = type;
// }
// __device__ void GPU_set_tree_height(uint32_t* addr, uint32_t tree_height) {
//     ((uint8_t*)addr)[OFFSET_TREE_HGT] = tree_height;
// }
// __device__ void GPU_set_tree_index(uint32_t* addr, uint32_t tree_index) {
//     GPU_u32_to_bytes(&((uint8_t*)addr)[OFFSET_TREE_INDEX], tree_index);
// }
// __device__ void GPU_set_layer_addr(uint32_t* addr, uint32_t layer) {
//     ((uint8_t*)addr)[OFFSET_LAYER] = layer;
// }
// __device__ void GPU_set_keypair_addr(uint32_t* addr, uint32_t keypair) {
//     ((uint8_t*)addr)[OFFSET_KP_ADDR1] = keypair;
// }
// __device__ void GPU_set_chain_addr(uint32_t* addr, uint32_t chain) {
//     ((uint8_t*)addr)[OFFSET_CHAIN_ADDR] = chain;
// }
// __device__ void GPU_set_hash_addr(uint32_t* addr, uint32_t hash) {
//     ((uint8_t*)addr)[OFFSET_HASH_ADDR] = hash;
// }
// __device__ void GPU_copy_keypair_addr(uint32_t* out, uint32_t* in) {
//     for (int i = 0; i < OFFSET_TREE + 8; i++)
//         ((uint8_t*)out)[i] = ((uint8_t*)in)[i];
//     ((uint8_t*)out)[OFFSET_KP_ADDR1] = ((uint8_t*)in)[OFFSET_KP_ADDR1];
// }
// __device__ void GPU_copy_subtree_addr(uint32_t* out, uint32_t* in) {
//     for (int i = 0; i < (OFFSET_TREE + 8); i++)
//         ((uint8_t*)out)[i] = ((uint8_t*)in)[i];
// }
// __device__ void GPU_ull_to_bytes(unsigned char* out, unsigned int outlen, unsigned long long in)
// {
//     int i;

//     /* Iterate over out in decreasing order, for big-endianness. */
//     for (i = outlen - 1; i >= 0; i--) {
//         out[i] = in & 0xff;
//         in = in >> 8;
//     }
// }
// __device__ void GPU_set_tree_addr(uint32_t* addr, uint64_t tree) {
//     GPU_ull_to_bytes(&((unsigned char*)addr)[OFFSET_TREE], 8, tree);
// }

// //! GPU Hash Function Define
// // #ifdef USE_GPU_SPHINCS_SHA256
// //!SHA256 MACRO
// #define hc_add3(a, b, c)	(a + b + c)
// #define hc_rotl32(x, n)		(((x) << (n)) | ((x) >> (32 - (n))))
// #define SHIFT_RIGHT_32(x,n) ((x) >> (n))

// #define SHA256_F0(x,y,z)	(((x) & (y)) | ((z) & ((x) ^ (y))))
// #define SHA256_F1(x,y,z)	((z) ^ ((x) & ((y) ^ (z))))
// #define SHA256_F0o(x,y,z) (SHA256_F0 ((x), (y), (z)))
// #define SHA256_F1o(x,y,z) (SHA256_F1 ((x), (y), (z)))

// #define SHA256_S0(x) (hc_rotl32 ((x), 25u) ^ hc_rotl32 ((x), 14u) ^ SHIFT_RIGHT_32 ((x),  3u))
// #define SHA256_S1(x) (hc_rotl32 ((x), 15u) ^ hc_rotl32 ((x), 13u) ^ SHIFT_RIGHT_32 ((x), 10u))
// #define SHA256_S2(x) (hc_rotl32 ((x), 30u) ^ hc_rotl32 ((x), 19u) ^ hc_rotl32 ((x), 10u))
// #define SHA256_S3(x) (hc_rotl32 ((x), 26u) ^ hc_rotl32 ((x), 21u) ^ hc_rotl32 ((x),  7u))

// #define SHA256_STEP(F0,F1,a,b,c,d,e,f,g,h,x,K)    \
// {                                                 \
//   h = hc_add3 (h, K, x);                          \
//   h = hc_add3 (h, SHA256_S3 (e), F1 (e,f,g));     \
//   d += h;                                         \
//   h = hc_add3 (h, SHA256_S2 (a), F0 (a,b,c));     \
// }

// #define SHA256_EXPAND(x,y,z,w) (SHA256_S1 (x) + y + SHA256_S0 (z) + w)
// #define ROTL32(x, n)			(((x) << (n)) | ((x) >> (32 - (n))))
// #define ROTR32(x, n)			(((x) >> (n)) | ((x) << (32 - (n))))

// __constant__ uint8_t GPU_iv_256[32] = {
//     0x6a, 0x09, 0xe6, 0x67, 0xbb, 0x67, 0xae, 0x85,
//     0x3c, 0x6e, 0xf3, 0x72, 0xa5, 0x4f, 0xf5, 0x3a,
//     0x51, 0x0e, 0x52, 0x7f, 0x9b, 0x05, 0x68, 0x8c,
//     0x1f, 0x83, 0xd9, 0xab, 0x5b, 0xe0, 0xcd, 0x19
// };
// __device__ size_t GPU_crypto_hashblock_sha256(uint8_t* statebytes, uint8_t* in, size_t inlen) {
//     uint32_t state[8];
//     uint32_t a = GPU_load_bigendian_32(statebytes + 0); state[0] = a;
//     uint32_t b = GPU_load_bigendian_32(statebytes + 4);	state[1] = b;
//     uint32_t c = GPU_load_bigendian_32(statebytes + 8);	state[2] = c;
//     uint32_t d = GPU_load_bigendian_32(statebytes + 12); state[3] = d;
//     uint32_t e = GPU_load_bigendian_32(statebytes + 16); state[4] = e;
//     uint32_t f = GPU_load_bigendian_32(statebytes + 20); state[5] = f;
//     uint32_t g = GPU_load_bigendian_32(statebytes + 24); state[6] = g;
//     uint32_t h = GPU_load_bigendian_32(statebytes + 28); state[7] = h;

//     while (inlen >= 64) {
//         uint32_t w0_t = GPU_load_bigendian_32(in + 0);
//         uint32_t w1_t = GPU_load_bigendian_32(in + 4);
//         uint32_t w2_t = GPU_load_bigendian_32(in + 8);
//         uint32_t w3_t = GPU_load_bigendian_32(in + 12);
//         uint32_t w4_t = GPU_load_bigendian_32(in + 16);
//         uint32_t w5_t = GPU_load_bigendian_32(in + 20);
//         uint32_t w6_t = GPU_load_bigendian_32(in + 24);
//         uint32_t w7_t = GPU_load_bigendian_32(in + 28);
//         uint32_t w8_t = GPU_load_bigendian_32(in + 32);
//         uint32_t w9_t = GPU_load_bigendian_32(in + 36);
//         uint32_t wa_t = GPU_load_bigendian_32(in + 40);
//         uint32_t wb_t = GPU_load_bigendian_32(in + 44);
//         uint32_t wc_t = GPU_load_bigendian_32(in + 48);
//         uint32_t wd_t = GPU_load_bigendian_32(in + 52);
//         uint32_t we_t = GPU_load_bigendian_32(in + 56);
//         uint32_t wf_t = GPU_load_bigendian_32(in + 60);

//         SHA256_STEP(SHA256_F0o, SHA256_F1o, a, b, c, d, e, f, g, h, w0_t, 0x428a2f98);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, h, a, b, c, d, e, f, g, w1_t, 0x71374491);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, g, h, a, b, c, d, e, f, w2_t, 0xb5c0fbcf);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, f, g, h, a, b, c, d, e, w3_t, 0xe9b5dba5);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, e, f, g, h, a, b, c, d, w4_t, 0x3956c25b);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, d, e, f, g, h, a, b, c, w5_t, 0x59f111f1);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, c, d, e, f, g, h, a, b, w6_t, 0x923f82a4);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, b, c, d, e, f, g, h, a, w7_t, 0xab1c5ed5);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, a, b, c, d, e, f, g, h, w8_t, 0xd807aa98);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, h, a, b, c, d, e, f, g, w9_t, 0x12835b01);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, g, h, a, b, c, d, e, f, wa_t, 0x243185be);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, f, g, h, a, b, c, d, e, wb_t, 0x550c7dc3);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, e, f, g, h, a, b, c, d, wc_t, 0x72be5d74);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, d, e, f, g, h, a, b, c, wd_t, 0x80deb1fe);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, c, d, e, f, g, h, a, b, we_t, 0x9bdc06a7);
//         SHA256_STEP(SHA256_F0o, SHA256_F1o, b, c, d, e, f, g, h, a, wf_t, 0xc19bf174);

//         w0_t = SHA256_EXPAND(we_t, w9_t, w1_t, w0_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, a, b, c, d, e, f, g, h, w0_t, 0xe49b69c1);
//         w1_t = SHA256_EXPAND(wf_t, wa_t, w2_t, w1_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, h, a, b, c, d, e, f, g, w1_t, 0xefbe4786);
//         w2_t = SHA256_EXPAND(w0_t, wb_t, w3_t, w2_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, g, h, a, b, c, d, e, f, w2_t, 0x0fc19dc6);
//         w3_t = SHA256_EXPAND(w1_t, wc_t, w4_t, w3_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, f, g, h, a, b, c, d, e, w3_t, 0x240ca1cc);
//         w4_t = SHA256_EXPAND(w2_t, wd_t, w5_t, w4_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, e, f, g, h, a, b, c, d, w4_t, 0x2de92c6f);
//         w5_t = SHA256_EXPAND(w3_t, we_t, w6_t, w5_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, d, e, f, g, h, a, b, c, w5_t, 0x4a7484aa);
//         w6_t = SHA256_EXPAND(w4_t, wf_t, w7_t, w6_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, c, d, e, f, g, h, a, b, w6_t, 0x5cb0a9dc);
//         w7_t = SHA256_EXPAND(w5_t, w0_t, w8_t, w7_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, b, c, d, e, f, g, h, a, w7_t, 0x76f988da);
//         w8_t = SHA256_EXPAND(w6_t, w1_t, w9_t, w8_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, a, b, c, d, e, f, g, h, w8_t, 0x983e5152);
//         w9_t = SHA256_EXPAND(w7_t, w2_t, wa_t, w9_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, h, a, b, c, d, e, f, g, w9_t, 0xa831c66d);
//         wa_t = SHA256_EXPAND(w8_t, w3_t, wb_t, wa_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, g, h, a, b, c, d, e, f, wa_t, 0xb00327c8);
//         wb_t = SHA256_EXPAND(w9_t, w4_t, wc_t, wb_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, f, g, h, a, b, c, d, e, wb_t, 0xbf597fc7);
//         wc_t = SHA256_EXPAND(wa_t, w5_t, wd_t, wc_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, e, f, g, h, a, b, c, d, wc_t, 0xc6e00bf3);
//         wd_t = SHA256_EXPAND(wb_t, w6_t, we_t, wd_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, d, e, f, g, h, a, b, c, wd_t, 0xd5a79147);
//         we_t = SHA256_EXPAND(wc_t, w7_t, wf_t, we_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, c, d, e, f, g, h, a, b, we_t, 0x06ca6351);
//         wf_t = SHA256_EXPAND(wd_t, w8_t, w0_t, wf_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, b, c, d, e, f, g, h, a, wf_t, 0x14292967);

//         w0_t = SHA256_EXPAND(we_t, w9_t, w1_t, w0_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, a, b, c, d, e, f, g, h, w0_t, 0x27b70a85);
//         w1_t = SHA256_EXPAND(wf_t, wa_t, w2_t, w1_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, h, a, b, c, d, e, f, g, w1_t, 0x2e1b2138);
//         w2_t = SHA256_EXPAND(w0_t, wb_t, w3_t, w2_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, g, h, a, b, c, d, e, f, w2_t, 0x4d2c6dfc);
//         w3_t = SHA256_EXPAND(w1_t, wc_t, w4_t, w3_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, f, g, h, a, b, c, d, e, w3_t, 0x53380d13);
//         w4_t = SHA256_EXPAND(w2_t, wd_t, w5_t, w4_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, e, f, g, h, a, b, c, d, w4_t, 0x650a7354);
//         w5_t = SHA256_EXPAND(w3_t, we_t, w6_t, w5_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, d, e, f, g, h, a, b, c, w5_t, 0x766a0abb);
//         w6_t = SHA256_EXPAND(w4_t, wf_t, w7_t, w6_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, c, d, e, f, g, h, a, b, w6_t, 0x81c2c92e);
//         w7_t = SHA256_EXPAND(w5_t, w0_t, w8_t, w7_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, b, c, d, e, f, g, h, a, w7_t, 0x92722c85);
//         w8_t = SHA256_EXPAND(w6_t, w1_t, w9_t, w8_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, a, b, c, d, e, f, g, h, w8_t, 0xa2bfe8a1);
//         w9_t = SHA256_EXPAND(w7_t, w2_t, wa_t, w9_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, h, a, b, c, d, e, f, g, w9_t, 0xa81a664b);
//         wa_t = SHA256_EXPAND(w8_t, w3_t, wb_t, wa_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, g, h, a, b, c, d, e, f, wa_t, 0xc24b8b70);
//         wb_t = SHA256_EXPAND(w9_t, w4_t, wc_t, wb_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, f, g, h, a, b, c, d, e, wb_t, 0xc76c51a3);
//         wc_t = SHA256_EXPAND(wa_t, w5_t, wd_t, wc_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, e, f, g, h, a, b, c, d, wc_t, 0xd192e819);
//         wd_t = SHA256_EXPAND(wb_t, w6_t, we_t, wd_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, d, e, f, g, h, a, b, c, wd_t, 0xd6990624);
//         we_t = SHA256_EXPAND(wc_t, w7_t, wf_t, we_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, c, d, e, f, g, h, a, b, we_t, 0xf40e3585);
//         wf_t = SHA256_EXPAND(wd_t, w8_t, w0_t, wf_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, b, c, d, e, f, g, h, a, wf_t, 0x106aa070);

//         w0_t = SHA256_EXPAND(we_t, w9_t, w1_t, w0_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, a, b, c, d, e, f, g, h, w0_t, 0x19a4c116);
//         w1_t = SHA256_EXPAND(wf_t, wa_t, w2_t, w1_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, h, a, b, c, d, e, f, g, w1_t, 0x1e376c08);
//         w2_t = SHA256_EXPAND(w0_t, wb_t, w3_t, w2_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, g, h, a, b, c, d, e, f, w2_t, 0x2748774c);
//         w3_t = SHA256_EXPAND(w1_t, wc_t, w4_t, w3_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, f, g, h, a, b, c, d, e, w3_t, 0x34b0bcb5);
//         w4_t = SHA256_EXPAND(w2_t, wd_t, w5_t, w4_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, e, f, g, h, a, b, c, d, w4_t, 0x391c0cb3);
//         w5_t = SHA256_EXPAND(w3_t, we_t, w6_t, w5_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, d, e, f, g, h, a, b, c, w5_t, 0x4ed8aa4a);
//         w6_t = SHA256_EXPAND(w4_t, wf_t, w7_t, w6_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, c, d, e, f, g, h, a, b, w6_t, 0x5b9cca4f);
//         w7_t = SHA256_EXPAND(w5_t, w0_t, w8_t, w7_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, b, c, d, e, f, g, h, a, w7_t, 0x682e6ff3);
//         w8_t = SHA256_EXPAND(w6_t, w1_t, w9_t, w8_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, a, b, c, d, e, f, g, h, w8_t, 0x748f82ee);
//         w9_t = SHA256_EXPAND(w7_t, w2_t, wa_t, w9_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, h, a, b, c, d, e, f, g, w9_t, 0x78a5636f);
//         wa_t = SHA256_EXPAND(w8_t, w3_t, wb_t, wa_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, g, h, a, b, c, d, e, f, wa_t, 0x84c87814);
//         wb_t = SHA256_EXPAND(w9_t, w4_t, wc_t, wb_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, f, g, h, a, b, c, d, e, wb_t, 0x8cc70208);
//         wc_t = SHA256_EXPAND(wa_t, w5_t, wd_t, wc_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, e, f, g, h, a, b, c, d, wc_t, 0x90befffa);
//         wd_t = SHA256_EXPAND(wb_t, w6_t, we_t, wd_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, d, e, f, g, h, a, b, c, wd_t, 0xa4506ceb);
//         we_t = SHA256_EXPAND(wc_t, w7_t, wf_t, we_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, c, d, e, f, g, h, a, b, we_t, 0xbef9a3f7);
//         wf_t = SHA256_EXPAND(wd_t, w8_t, w0_t, wf_t); SHA256_STEP(SHA256_F0o, SHA256_F1o, b, c, d, e, f, g, h, a, wf_t, 0xc67178f2);

//         a += state[0];
//         b += state[1];
//         c += state[2];
//         d += state[3];
//         e += state[4];
//         f += state[5];
//         g += state[6];
//         h += state[7];

//         state[0] = a;
//         state[1] = b;
//         state[2] = c;
//         state[3] = d;
//         state[4] = e;
//         state[5] = f;
//         state[6] = g;
//         state[7] = h;

//         in += 64;
//         inlen -= 64;
//     }
//     GPU_store_bigendian_32(statebytes + 0, state[0]);
//     GPU_store_bigendian_32(statebytes + 4, state[1]);
//     GPU_store_bigendian_32(statebytes + 8, state[2]);
//     GPU_store_bigendian_32(statebytes + 12, state[3]);
//     GPU_store_bigendian_32(statebytes + 16, state[4]);
//     GPU_store_bigendian_32(statebytes + 20, state[5]);
//     GPU_store_bigendian_32(statebytes + 24, state[6]);
//     GPU_store_bigendian_32(statebytes + 28, state[7]);
//     return inlen;
// }

// __device__ void GPU_sha256_inc_init(uint8_t* state) {
//     for (size_t i = 0; i < 32; i++)
//         state[i] = GPU_iv_256[i];
//     for (size_t i = 32; i < 40; i++)
//         state[i] = 0;
// }
// __device__ void GPU_sha256_inc_block(uint8_t* state, uint8_t* in, size_t inblocks) {
//     uint64_t bytes = GPU_load_bigendian_64(state + 32);
//     GPU_crypto_hashblock_sha256(state, in, 64 * inblocks);
//     bytes += 64 * inblocks;
//     GPU_store_bigendian_64(state + 32, bytes);
// }
// __device__ void GPU_sha256_inc_finalize(uint8_t* out, uint8_t* state, uint8_t* in, size_t inlen) {
//     uint8_t padded[128];
//     uint64_t bytes = GPU_load_bigendian_64(state + 32) + inlen;
//     GPU_crypto_hashblock_sha256(state, in, inlen);
//     in += inlen;
//     inlen &= 63;
//     in -= inlen;
//     for (size_t i = 0; i < inlen; i++)
//         padded[i] = in[i];
//     padded[inlen] = 0x80;
//     if (inlen < 56) {
//         for (size_t i = inlen + 1; i < 56; i++)
//             padded[i] = 0;
//         padded[56] = (uint8_t)(bytes >> 53);
//         padded[57] = (uint8_t)(bytes >> 45);
//         padded[58] = (uint8_t)(bytes >> 37);
//         padded[59] = (uint8_t)(bytes >> 29);
//         padded[60] = (uint8_t)(bytes >> 21);
//         padded[61] = (uint8_t)(bytes >> 13);
//         padded[62] = (uint8_t)(bytes >> 5);
//         padded[63] = (uint8_t)(bytes << 3);
//         GPU_crypto_hashblock_sha256(state, padded, 64);
//     }

//     else {
//         for (size_t i = inlen + 1; i < 120; i++)
//             padded[i] = 0;
//         padded[120] = (uint8_t)(bytes >> 53);
//         padded[121] = (uint8_t)(bytes >> 45);
//         padded[122] = (uint8_t)(bytes >> 37);
//         padded[123] = (uint8_t)(bytes >> 29);
//         padded[124] = (uint8_t)(bytes >> 21);
//         padded[125] = (uint8_t)(bytes >> 13);
//         padded[126] = (uint8_t)(bytes >> 5);
//         padded[127] = (uint8_t)(bytes << 3);
//         GPU_crypto_hashblock_sha256(state, padded, 128);
//     }

//     for (size_t i = 0; i < HASH_OUTBYTE; i++)
//         out[i] = state[i];
// }

// __device__ void dev_sha256(uint8_t* out, uint8_t* in, size_t inlen) {
//     uint8_t state[40];
//     GPU_sha256_inc_init(state);
//     GPU_sha256_inc_finalize(out, state, in, inlen);
// }

// kim version finished

__global__ void global_sha256(uint8_t* out, uint8_t* in, size_t inlen, size_t loop_num) {
    for (int i = 0; i < loop_num; i++)
        dev_sha256(out, in, inlen);
} // global_sha256

void face_sha256(uint8_t* md, uint8_t* d, size_t n, size_t loop_num) {
    struct timespec start, stop;
    CHECK(cudaSetDevice(DEVICE_USED));
    u8 *dev_d = NULL, *dev_md = NULL;

    CHECK(cudaMalloc((void**) &dev_d, n * sizeof(u8)));
    CHECK(cudaMalloc((void**) &dev_md, 32 * sizeof(u8)));
    CHECK(cudaMemcpy(dev_d, d, n * sizeof(u8), HOST_2_DEVICE));

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);

    CHECK(cudaDeviceSynchronize());
    global_sha256<<<1, 1>>>(dev_md, dev_d, n, loop_num);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);

    g_result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;

    CHECK(cudaMemcpy(md, dev_md, 32 * sizeof(u8), DEVICE_2_HOST));

    cudaFree(dev_d);
    cudaFree(dev_md);
}

/**
 * Note that inlen should be sufficiently small that it still allows for
 * an array to be allocated on the stack. Typically 'in' is merely a seed.
 * Outputs outlen number of bytes
 */
__device__ void dev_mgf1(unsigned char* out, unsigned long outlen, const unsigned char* in,
                         unsigned long inlen) {
    unsigned char inbuf[SPX_N + SPX_SHA256_ADDR_BYTES + 4];
    // unsigned char outbuf[SPX_SHA256_OUTPUT_BYTES]; // 715 wrong
    unsigned char outbuf[SPX_SHA256_OUTPUT_BYTES * 2];
    u32 i;

    memcpy(inbuf, in, inlen);

    /* While we can fit in at least another full block of SHA256 output.. */
    for (i = 0; (i + 1) * SPX_SHA256_OUTPUT_BYTES <= outlen; i++) {
        dev_u32_to_bytes(inbuf + inlen, i);
        dev_sha256(out, inbuf, inlen + 4);
        out += SPX_SHA256_OUTPUT_BYTES;
    }
    /* Until we cannot anymore, and we fill the remainder. */
    if (outlen > i * SPX_SHA256_OUTPUT_BYTES) {
        dev_u32_to_bytes(inbuf + inlen, i);
        dev_sha256(outbuf, inbuf, inlen + 4);
        memcpy(out, outbuf, outlen - i * SPX_SHA256_OUTPUT_BYTES);
    }

    // const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    //
    // out -= SPX_SHA256_OUTPUT_BYTES;
    // if (tid == 0) {
    // 	printf("gpu\n");
    // 	printf("out = %02x\n", out[0]);
    // 	printf("out = %02x\n", out[1]);
    // 	printf("out = %02x\n", out[2]);
    // }
    // int a = 1;
    //
    // while (a) {
    // }

} // dev_mgf1

__device__ void dev_mgf1_hg(unsigned char* out, unsigned long outlen, const unsigned char* in,
                            unsigned long inlen) {
    unsigned char inbuf[SPX_SHA256_OUTPUT_BYTES + 4]; // inlen + 4
    unsigned char outbuf[SPX_SHA256_OUTPUT_BYTES];
    unsigned long i;

    memcpy(inbuf, in, inlen);

    /* While we can fit in at least another full block of SHA256 output.. */
    for (i = 0; (i + 1) * SPX_SHA256_OUTPUT_BYTES <= outlen; i++) {
        dev_u32_to_bytes(inbuf + inlen, i);
        dev_sha256(out, inbuf, inlen + 4);
        out += SPX_SHA256_OUTPUT_BYTES;
    }
    /* Until we cannot anymore, and we fill the remainder. */
    if (outlen > i * SPX_SHA256_OUTPUT_BYTES) {
        dev_u32_to_bytes(inbuf + inlen, i);
        dev_sha256(outbuf, inbuf, inlen + 4);
        memcpy(out, outbuf, outlen - i * SPX_SHA256_OUTPUT_BYTES);
    }
    // if (outlen / SPX_SHA256_OUTPUT_BYTES > 1) no output
    // 	printf("outlen = %d\n", outlen / SPX_SHA256_OUTPUT_BYTES);

} // dev_mgf1_hg

__device__ uint8_t dev_state_seeded[40];

/**
 * Absorb the constant pub_seed using one round of the compression function
 * This initializes state_seeded, which can then be reused in thash
 **/
__device__ void dev_seed_state(const unsigned char* pub_seed) {
    uint8_t block[SPX_SHA256_BLOCK_BYTES];
    size_t i;

    for (i = 0; i < SPX_N; ++i) {
        block[i] = pub_seed[i];
    }
    for (i = SPX_N; i < SPX_SHA256_BLOCK_BYTES; ++i) {
        block[i] = 0;
    }

    dev_sha256_inc_init(dev_state_seeded);
    dev_sha256_inc_blocks(dev_state_seeded, block, 1);
} // seed_state

#endif // ifdef SHA256