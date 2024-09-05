#include <iostream>
using namespace std;

// The option for optimization

/* Algorithmic parallelism: start */

// SHA256 optimization
#define USING_SHA256_UNROLL
#define FASTER
#define USING_SHA256_X_UNROLL // used with FASTER
#define USING_SHA256_PTX  // kg/verify without this
#define USING_SHA256_INTEGER

#define USING_SHAKE_UINT2
#define USING_SHAKE_HAND_UNROLL // hand + array
#define USING_SHAKE_PTX
#define USING_SHAKE_INTEGER
#define USING_SHAKE_UNROLL // automatic, useless

#define USING_COALESCED_ACCESS // not used
#define USING_LOCAL_MEMORY

// default MAXIMUM_BLOCK
#define KEYGEN_SUITBLE_BLOCK
#define SIGN_SUITBLE_BLOCK
#define VERIFY_SUITBLE_BLOCK

#define USING_PARALLEL_PKGEN_TREEHASH_XMSS   // kg, below 3 belongs to this, level 2
#define USING_PARALLEL_SIGN_TREEHASH_XMSS    // sign, xmss leaf, level 2
#define USING_PARALLEL_XMSS_BRANCH           // branch node, level 2
#define USING_PARALLEL_WOTS_PKGEN            // wots_pkgen, level 3

#define USING_PARALLEL_FORS_SIGN             // sign, level 1
#define USING_PARALLEL_SIGN_FORS_LEAF        // sign, fors leaf, level 2
#define USING_PARALLEL_SIGN_FORS_BRNACH	     // sign, fors branch node, level 2
#define USING_PARALLEL_WOTS_SIGN             // sign, level 3
#define USING_PARALLEL_FORS_SIGN_THASH       // sign, level 4

#define USING_PARALLEL_FORS_PK_FROM_SIG      // fors
#define USING_PARALLEL_WOTS_PK_FROM_SIG      // verify
#define USING_PARALLEL_COMPUTE_ROOT          // verify
#define USING_PARALLEL_FORS_VERIFY_THASH     // verify, level 4
#define USING_PARALLEL_WOTS_THASH            // verify, level 4
#define USING_VERIFY_FOUR_BRANCH_NODE      // verfiy, level 4

/* test for algorithmic parallelism: finish */

#define USING_SHA256_PTX_MODE 1 

/* test for multi-keypair data parallelism: start */
// 1. maximum first
// 2. uniformly distributed
#define LARGE_SCHEME 1 // 1 or 2
// 1: 82 * 32
// 2: cuda core
// 3: maximum parallelism
#define USING_STREAM 2
/* test for multi-keypair data parallelism: finish */

/* Hybrid parallelism: start */
#define HP_PARALLELISM (8)

// #define HYBRID_SIGN_FORS_LOAD_BALANCING
// #define WOTS_SIGN_LOAD_BALANCING

#define HYBRID_VERIFY_FORS_LOAD_BALANCING // useful when HP_PARALLELISM is 2, 32
// #define WOTS_VERIFY_LOAD_BALANCING1
// #define WOTS_VERIFY_LOAD_BALANCING2
#define WOTS_VERIFY_LOAD_BALANCING3

/* Hybrid parallelism: finish */

#define DEBUG_MODE
#define SAME_CHECK

// #define USING_FIXED_SEEDS

#ifndef DEVICE_USED
#define DEVICE_USED 0
#endif /* ifndef DEVICE_USED */

// #define PRINT_ALL

// macros that do not need to be changed
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
#define HOST_2_DEVICE cudaMemcpyHostToDevice
#define H2D cudaMemcpyHostToDevice
#define DEVICE_2_HOST cudaMemcpyDeviceToHost
#define D2H cudaMemcpyDeviceToHost

#define CHECK(call) \
	if ((call) != cudaSuccess) { \
		cudaError_t err = cudaGetLastError(); \
		cerr << "CUDA error calling \""#call "\", code is " << err << endl; }

extern double g_result;
extern double g_inner_result;
extern double g_count;
