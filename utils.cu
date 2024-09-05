#include <string.h>

#include "address.h"
#include "hash.h"
#include "params.h"
#include "thash.h"
#include "utils.h"

#include "all_option.h"
#include "fors.h"
#include "sha256.h"
#include "wots.h"
#include <cooperative_groups.h>
#include <iostream>
using namespace std;

#define LEAM_NUM (1 << SPX_TREE_HEIGHT)
#define NN (LEAM_NUM * SPX_WOTS_LEN)
#define MM (1 << SPX_FORS_HEIGHT)

__device__ u8 wots_pk[SPX_WOTS_BYTES * 512]; // provided that thread size is 512
__device__ u8 leaf_node[SPX_N * 1024 * 22]; // 假设2048个叶节点，最大22层

extern __device__ uint8_t dev_state_seeded[40];
__device__ u8 dev_auth_path[20 * SPX_N]; // max < 20

extern __device__ u8 dev_wpk[512 * 512 * SPX_WOTS_BYTES];
extern __device__ u8 dev_leaf[SPX_N * 1024 * 1024 * 44];

/**
 * Converts the value of 'in' to 'outlen' bytes in big-endian byte order.
 */
void ull_to_bytes(unsigned char* out, unsigned int outlen, unsigned long long in) {
    int i;

    /* Iterate over out in decreasing order, for big-endianness. */
    for (i = outlen - 1; i >= 0; i--) {
        out[i] = in & 0xff;
        in = in >> 8;
    }
} // ull_to_bytes

__device__ void dev_ull_to_bytes(unsigned char* out, unsigned int outlen, unsigned long long in) {
    int i;

    /* Iterate over out in decreasing order, for big-endianness. */
    for (i = outlen - 1; i >= 0; i--) {
        out[i] = in & 0xff;
        in = in >> 8;
    }
} // dev_ull_to_bytes

void u32_to_bytes(unsigned char* out, uint32_t in) {
    out[0] = (unsigned char) (in >> 24);
    out[1] = (unsigned char) (in >> 16);
    out[2] = (unsigned char) (in >> 8);
    out[3] = (unsigned char) in;
} // u32_to_bytes

__device__ void dev_u32_to_bytes(unsigned char* out, uint32_t in) {
    out[0] = (unsigned char) (in >> 24);
    out[1] = (unsigned char) (in >> 16);
    out[2] = (unsigned char) (in >> 8);
    out[3] = (unsigned char) in;
} // dev_u32_to_bytes

/**
 * Converts the inlen bytes in 'in' from big-endian byte order to an integer.
 */
unsigned long long bytes_to_ull(const unsigned char* in, unsigned int inlen) {
    unsigned long long retval = 0;
    unsigned int i;

    for (i = 0; i < inlen; i++) {
        retval |= ((unsigned long long) in[i]) << (8 * (inlen - 1 - i));
    }
    return retval;
} // bytes_to_ull

__device__ unsigned long long dev_bytes_to_ull(const unsigned char* in, unsigned int inlen) {
    unsigned long long retval = 0;
    unsigned int i;

    for (i = 0; i < inlen; i++) {
        retval |= ((unsigned long long) in[i]) << (8 * (inlen - 1 - i));
    }
    return retval;
} // dev_bytes_to_ull

/**
 * Computes a root node given a leaf and an auth path.
 * Expects address to be complete other than the tree_height and tree_index.
 */
void compute_root(unsigned char* root, const unsigned char* leaf, uint32_t leaf_idx,
                  uint32_t idx_offset, const unsigned char* auth_path, uint32_t tree_height,
                  const unsigned char* pub_seed, uint32_t addr[8]) {
    uint32_t i;
    unsigned char buffer[2 * SPX_N];

    /* If leaf_idx is odd (last bit = 1), current path element is a right child
       and auth_path has to go left. Otherwise it is the other way around. */
    if (leaf_idx & 1) {
        memcpy(buffer + SPX_N, leaf, SPX_N);
        memcpy(buffer, auth_path, SPX_N);
    } else {
        memcpy(buffer, leaf, SPX_N);
        memcpy(buffer + SPX_N, auth_path, SPX_N);
    }
    auth_path += SPX_N;

    for (i = 0; i < tree_height - 1; i++) {
        leaf_idx >>= 1;
        idx_offset >>= 1;
        /* Set the address of the node we're creating. */
        set_tree_height(addr, i + 1);
        set_tree_index(addr, leaf_idx + idx_offset);

        /* Pick the right or left neighbor, depending on parity of the node. */
        if (leaf_idx & 1) {
            thash(buffer + SPX_N, buffer, 2, pub_seed, addr);
            memcpy(buffer, auth_path, SPX_N);
        } else {
            thash(buffer, buffer, 2, pub_seed, addr);
            memcpy(buffer + SPX_N, auth_path, SPX_N);
        }
        auth_path += SPX_N;
    }

    /* The last iteration is exceptional; we do not copy an auth_path node. */
    leaf_idx >>= 1;
    idx_offset >>= 1;
    set_tree_height(addr, tree_height);
    set_tree_index(addr, leaf_idx + idx_offset);
    thash(root, buffer, 2, pub_seed, addr);
} // compute_root

__device__ void dev_compute_root(unsigned char* root, const unsigned char* leaf, uint32_t leaf_idx,
                                 uint32_t idx_offset, const unsigned char* auth_path,
                                 uint32_t tree_height, const unsigned char* pub_seed,
                                 uint32_t addr[8]) {
    uint32_t i;
    unsigned char buffer[2 * SPX_N];

    /* If leaf_idx is odd (last bit = 1), current path element is a right child
       and auth_path has to go left. Otherwise it is the other way around. */
    if (leaf_idx & 1) {
        memcpy(buffer + SPX_N, leaf, SPX_N);
        memcpy(buffer, auth_path, SPX_N);
    } else {
        memcpy(buffer, leaf, SPX_N);
        memcpy(buffer + SPX_N, auth_path, SPX_N);
    }
    auth_path += SPX_N;

    for (i = 0; i < tree_height - 1; i++) {
        leaf_idx >>= 1;
        idx_offset >>= 1;
        /* Set the address of the node we're creating. */
        dev_set_tree_height(addr, i + 1);
        dev_set_tree_index(addr, leaf_idx + idx_offset);

        /* Pick the right or left neighbor, depending on parity of the node. */
        if (leaf_idx & 1) {
            dev_thash(buffer + SPX_N, buffer, 2, pub_seed, addr);
            memcpy(buffer, auth_path, SPX_N);
        } else {
            dev_thash(buffer, buffer, 2, pub_seed, addr);
            memcpy(buffer + SPX_N, auth_path, SPX_N);
        }
        auth_path += SPX_N;
    }

    /* The last iteration is exceptional; we do not copy an auth_path node. */
    leaf_idx >>= 1;
    idx_offset >>= 1;
    dev_set_tree_height(addr, tree_height);
    dev_set_tree_index(addr, leaf_idx + idx_offset);
    dev_thash(root, buffer, 2, pub_seed, addr);
} // dev_compute_root

/**
 * For a given leaf index, computes the authentication path and the resulting
 * root node using Merkle's TreeHash algorithm.
 * Expects the layer and tree parts of the tree_addr to be set, as well as the
 * tree type (i.e. SPX_ADDR_TYPE_HASHTREE or SPX_ADDR_TYPE_FORSTREE).
 * Applies the offset idx_offset to indices before building addresses, so that
 * it is possible to continue counting indices across trees.
 */
void treehash(unsigned char* root, unsigned char* auth_path, const unsigned char* sk_seed,
              const unsigned char* pub_seed, uint32_t leaf_idx, uint32_t idx_offset,
              uint32_t tree_height,
              void (*gen_leaf)(unsigned char* /* leaf */, const unsigned char* /* sk_seed */,
                               const unsigned char* /* pub_seed */, uint32_t /* addr_idx */,
                               const uint32_t[8] /* tree_addr */),
              uint32_t tree_addr[8]) {
    unsigned char stack[(tree_height + 1) * SPX_N];
    unsigned int heights[tree_height + 1];
    unsigned int offset = 0;
    uint32_t idx;
    uint32_t tree_idx;

    for (idx = 0; idx < (uint32_t) (1 << tree_height); idx++) {
        /* Add the next leaf node to the stack. */
        gen_leaf(stack + offset * SPX_N, sk_seed, pub_seed, idx + idx_offset, tree_addr);
        offset++;
        heights[offset - 1] = 0;

        /* If this is a node we need for the auth path.. */
        if ((leaf_idx ^ 0x1) == idx) {
            memcpy(auth_path, stack + (offset - 1) * SPX_N, SPX_N);
        }

        /* While the top-most nodes are of equal height.. */
        while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
            /* Compute index of the new node, in the next layer. */
            tree_idx = (idx >> (heights[offset - 1] + 1));

            /* Set the address of the node we're creating. */
            set_tree_height(tree_addr, heights[offset - 1] + 1);
            set_tree_index(tree_addr, tree_idx + (idx_offset >> (heights[offset - 1] + 1)));
            /* Hash the top-most nodes from the stack together. */
            thash(stack + (offset - 2) * SPX_N, stack + (offset - 2) * SPX_N, 2, pub_seed,
                  tree_addr);
            // for (size_t i = 0; i < SPX_N; i++) {
            // 	if (i % 8 == 0) printf("\n");
            // 	printf("%02x", stack[(offset - 2) * SPX_N + i]);
            // }
            // printf("\n");
            offset--;
            /* Note that the top-most node is now one layer higher. */
            heights[offset - 1]++;

            /* If this is a node we need for the auth path.. */
            if (((leaf_idx >> heights[offset - 1]) ^ 0x1) == tree_idx) {
                memcpy(auth_path + heights[offset - 1] * SPX_N, stack + (offset - 1) * SPX_N,
                       SPX_N);
            }
        }
    }
    memcpy(root, stack, SPX_N);
} // treehash

__device__ void
dev_treehash(unsigned char* root, unsigned char* auth_path, const unsigned char* sk_seed,
             const unsigned char* pub_seed, uint32_t leaf_idx, uint32_t idx_offset,
             uint32_t tree_height,
             void (*dev_gen_leaf)(unsigned char* /* leaf */, const unsigned char* /* sk_seed */,
                                  const unsigned char* /* pub_seed */, uint32_t /* addr_idx */,
                                  const uint32_t[8] /* tree_addr */),
             uint32_t tree_addr[8]) {
#define HIGHER_HEIGHT (SPX_TREE_HEIGHT > SPX_FORS_HEIGHT ? SPX_TREE_HEIGHT : SPX_FORS_HEIGHT)
    unsigned char stack[(HIGHER_HEIGHT + 1) * SPX_N];
    unsigned int heights[HIGHER_HEIGHT + 1];
    unsigned int offset = 0;
    uint32_t idx;
    uint32_t tree_idx;

    for (idx = 0; idx < (uint32_t) (1 << tree_height); idx++) {
        /* Add the next leaf node to the stack. */
        dev_gen_leaf(stack + offset * SPX_N, sk_seed, pub_seed, idx + idx_offset, tree_addr);
        offset++;
        heights[offset - 1] = 0;

        /* If this is a node we need for the auth path.. */
        if ((leaf_idx ^ 0x1) == idx) {
            memcpy(auth_path, stack + (offset - 1) * SPX_N, SPX_N);
        }

        /* While the top-most nodes are of equal height.. */
        while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
            /* Compute index of the new node, in the next layer. */
            tree_idx = (idx >> (heights[offset - 1] + 1));

            /* Set the address of the node we're creating. */
            dev_set_tree_height(tree_addr, heights[offset - 1] + 1);
            dev_set_tree_index(tree_addr, tree_idx + (idx_offset >> (heights[offset - 1] + 1)));
            /* Hash the top-most nodes from the stack together. */
            dev_thash(stack + (offset - 2) * SPX_N, stack + (offset - 2) * SPX_N, 2, pub_seed,
                      tree_addr);

            offset--;
            /* Note that the top-most node is now one layer higher. */
            heights[offset - 1]++;

            /* If this is a node we need for the auth path.. */
            if (((leaf_idx >> heights[offset - 1]) ^ 0x1) == tree_idx) {
                memcpy(auth_path + heights[offset - 1] * SPX_N, stack + (offset - 1) * SPX_N,
                       SPX_N);
            }
        }
    }
    memcpy(root, stack, SPX_N);
    // for (int jj = 0; jj < SPX_N; jj++) {
    // 	printf("%02x", root[jj]);
    // 	if (jj == SPX_N - 1) printf("\n");
    // }
}

__device__ void dev_ap_treehash_fors(
    unsigned char* root, unsigned char* auth_path, const unsigned char* sk_seed,
    const unsigned char* pub_seed, uint32_t leaf_idx, uint32_t idx_offset, uint32_t tree_height,
    void (*dev_gen_leaf)(unsigned char* /* leaf */, const unsigned char* /* sk_seed */,
                         const unsigned char* /* pub_seed */, uint32_t /* addr_idx */,
                         const uint32_t[8] /* tree_addr */),
    uint32_t tree_addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    const u32 leaf_num = (uint32_t) (1 << tree_height);

    // 内部无法并行
    if (tid < leaf_num) {
        for (int i = tid; i < leaf_num; i += tnum) {
            dev_fors_gen_leaf(leaf_node + i * SPX_N, sk_seed, pub_seed, i + idx_offset, tree_addr);
        }
    }

#ifdef USING_PARALLEL_XMSS_BRANCH
    // g.sync();
    if (tid == ((leaf_idx >> 0) ^ 0x1)) memcpy(dev_auth_path, leaf_node + tid * SPX_N, SPX_N);

    for (int i = 1, ii = 1; i <= tree_height; i++) {
        g.sync();
        int off = 2 * tid * ii * SPX_N;
        if (tid < (leaf_num >> i)) {
            dev_set_tree_height(tree_addr, i);
            dev_set_tree_index(tree_addr, tid + (idx_offset >> i));
            memcpy(leaf_node + off + SPX_N, leaf_node + off + ii * SPX_N, SPX_N);
            dev_thash(leaf_node + off, leaf_node + off, 2, pub_seed, tree_addr);
            if (tid == ((leaf_idx >> i) ^ 0x1))
                memcpy(dev_auth_path + i * SPX_N, leaf_node + off, SPX_N);
        }
        ii *= 2;
    }
    // g.sync();
    // 最后一个时，仅0号线程拼接，因此无需同步

    if (tid == 0) {
        for (int i = 0; i < tree_height; i++) {
            memcpy(auth_path + i * SPX_N, dev_auth_path + i * SPX_N, SPX_N);
            // for (int jj = 0; jj < SPX_N; jj++) {
            // 	printf("%02x", auth_path[i * SPX_N + jj]);
            // 	if (jj == SPX_N - 1) printf("\n");
            // }
        }
        // printf("\n");
        memcpy(root, leaf_node, SPX_N);
        // for (int jj = 0; jj < SPX_N; jj++) {
        // 	printf("%02x", root[jj]);
        // 	if (jj == SPX_N - 1) printf("\n");
        // }
    }
#else // ifdef USING_PARALLEL_XMSS_BRANCH
#define HIGHER_HEIGHT (SPX_TREE_HEIGHT > SPX_FORS_HEIGHT ? SPX_TREE_HEIGHT : SPX_FORS_HEIGHT)
    unsigned char stack[(HIGHER_HEIGHT + 1) * SPX_N];
    unsigned int heights[HIGHER_HEIGHT + 1];
    unsigned int offset = 0;
    uint32_t idx;
    uint32_t tree_idx;

    g.sync();

    if (tid == 0) {
        for (idx = 0; idx < (uint32_t) (1 << tree_height); idx++) {
            /* Add the next leaf node to the stack. */
            // dev_gen_leaf(stack + offset * SPX_N,
            // 		 sk_seed, pub_seed, idx + idx_offset, tree_addr);
            memcpy(stack + offset * SPX_N, leaf_node + idx * SPX_N, SPX_N);
            offset++;
            heights[offset - 1] = 0;

            /* If this is a node we need for the auth path.. */
            if ((leaf_idx ^ 0x1) == idx) {
                memcpy(auth_path, stack + (offset - 1) * SPX_N, SPX_N);
            }

            /* While the top-most nodes are of equal height.. */
            while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
                /* Compute index of the new node, in the next layer. */
                tree_idx = (idx >> (heights[offset - 1] + 1));

                /* Set the address of the node we're creating. */
                dev_set_tree_height(tree_addr, heights[offset - 1] + 1);
                dev_set_tree_index(tree_addr, tree_idx + (idx_offset >> (heights[offset - 1] + 1)));
                /* Hash the top-most nodes from the stack together. */
                dev_thash(stack + (offset - 2) * SPX_N, stack + (offset - 2) * SPX_N, 2, pub_seed,
                          tree_addr);
                offset--;
                /* Note that the top-most node is now one layer higher. */
                heights[offset - 1]++;

                /* If this is a node we need for the auth path.. */
                if (((leaf_idx >> heights[offset - 1]) ^ 0x1) == tree_idx) {
                    memcpy(auth_path + heights[offset - 1] * SPX_N, stack + (offset - 1) * SPX_N,
                           SPX_N);
                    // for (int jj = 0; jj < SPX_N; jj++) {
                    // 	printf("%02x", auth_path[heights[offset - 1] * SPX_N + jj]);
                    // 	if (jj == SPX_N - 1) printf("\n");
                    // }
                }
            }
        }
        // printf("\n");
        memcpy(root, stack, SPX_N);
        // for (int jj = 0; jj < SPX_N; jj++) {
        // 	printf("%02x", root[jj]);
        // }
        // printf("\n");
    }
#endif // ifdef USING_PARALLEL_XMSS_BRANCH
} // dev_ap_treehash_fors

__device__ void dev_ap_treehash_wots_2(
    unsigned char* root, unsigned char* auth_path, const unsigned char* sk_seed,
    const unsigned char* pub_seed, uint32_t leaf_idx, uint32_t idx_offset, uint32_t tree_height,
    void (*dev_gen_leaf)(unsigned char* /* leaf */, const unsigned char* /* sk_seed */,
                         const unsigned char* /* pub_seed */, uint32_t /* addr_idx */,
                         const uint32_t[8] /* tree_addr */),
    uint32_t tree_addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    u32 leaf_num = (1 << tree_height);
    // u32 max_threads = leaf_num * SPX_WOTS_LEN;
    // uint32_t wots_addr[8] = {0};
    // uint32_t wots_pk_addr[8] = {0};

    if (tid < leaf_num)
        dev_gen_leaf(leaf_node + tid * SPX_N, sk_seed, pub_seed, tid + idx_offset, tree_addr);

    if (tid == ((leaf_idx >> 0) ^ 0x1)) memcpy(dev_auth_path, leaf_node + tid * SPX_N, SPX_N);

    int branch_para = 4;
    branch_para = tnum;
    for (int i = 1, ii = 1; i <= tree_height; i++) {
        g.sync();
        dev_set_tree_height(tree_addr, i);
        if (tid < branch_para) {
            for (int j = tid; j < (leaf_num >> i); j += branch_para) {
                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(tree_addr, j);
                u8 temp[SPX_N * 2];
                memcpy(temp, leaf_node + off, SPX_N);
                memcpy(&temp[SPX_N], leaf_node + off + ii * SPX_N, SPX_N);
                dev_thash(leaf_node + off, temp, 2, pub_seed, tree_addr);

                if (j == ((leaf_idx >> i) ^ 0x1)) {
                    memcpy(dev_auth_path + i * SPX_N, leaf_node + off, SPX_N);
                }
            }
        }
        ii *= 2;
    }

    if (tid == 0) {
        memcpy(auth_path, dev_auth_path, SPX_N * tree_height);
        memcpy(root, leaf_node, SPX_N);
    }
}

__device__ void dev_ap_treehash_wots_23(
    unsigned char* root, unsigned char* auth_path, const unsigned char* sk_seed,
    const unsigned char* pub_seed, uint32_t leaf_idx, uint32_t idx_offset, uint32_t tree_height,
    void (*dev_gen_leaf)(unsigned char* /* leaf */, const unsigned char* /* sk_seed */,
                         const unsigned char* /* pub_seed */, uint32_t /* addr_idx */,
                         const uint32_t[8] /* tree_addr */),
    uint32_t tree_addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    u32 leaf_num = (1 << tree_height);
    u32 max_threads = leaf_num * SPX_WOTS_LEN;
    uint32_t wots_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};

    if (max_threads > tnum) max_threads = tnum - SPX_WOTS_LEN;

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);
    dev_copy_subtree_addr(wots_addr, tree_addr);

    if (tid < max_threads) {
        for (int i = tid; i < SPX_WOTS_LEN * leaf_num; i += max_threads) {
            dev_set_keypair_addr(wots_addr, i / SPX_WOTS_LEN);
            dev_set_chain_addr(wots_addr, i % SPX_WOTS_LEN);
            dev_set_hash_addr(wots_addr, 0);

            u8 temp[SPX_N];
            dev_prf_addr(temp, sk_seed, wots_addr);
            dev_gen_chain(temp, temp, 0, SPX_WOTS_W - 1, pub_seed, wots_addr);
            memcpy(wots_pk + i * SPX_N, temp, SPX_N);
        }
    }
    g.sync();

    if (tid < leaf_num) {
        dev_copy_keypair_addr(wots_pk_addr, wots_addr);
        dev_set_keypair_addr(wots_pk_addr, tid + idx_offset);
        u8 temp[SPX_WOTS_BYTES];
        memcpy(temp, wots_pk + tid * SPX_WOTS_BYTES, SPX_WOTS_BYTES);
        dev_thash(leaf_node + tid * SPX_N, temp, SPX_WOTS_LEN, pub_seed, wots_pk_addr);
    }

    if (tid == ((leaf_idx >> 0) ^ 0x1)) memcpy(dev_auth_path, leaf_node + tid * SPX_N, SPX_N);

    int branch_para = 4;
    branch_para = tnum;
    for (int i = 1, ii = 1; i <= tree_height; i++) {
        g.sync();
        dev_set_tree_height(tree_addr, i);
        if (tid < branch_para) {
            for (int j = tid; j < (leaf_num >> i); j += branch_para) {
                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(tree_addr, j);
                u8 temp[SPX_N * 2];
                memcpy(temp, leaf_node + off, SPX_N);
                memcpy(&temp[SPX_N], leaf_node + off + ii * SPX_N, SPX_N);
                dev_thash(leaf_node + off, temp, 2, pub_seed, tree_addr);

                if (j == ((leaf_idx >> i) ^ 0x1)) {
                    memcpy(dev_auth_path + i * SPX_N, leaf_node + off, SPX_N);
                }
            }
        }
        ii *= 2;
    }

    if (tid == 0) {
        memcpy(auth_path, dev_auth_path, SPX_N * tree_height);
        memcpy(root, leaf_node, SPX_N);
    }
}

__device__ void dev_ap_treehash_wots(
    unsigned char* root, unsigned char* auth_path, const unsigned char* sk_seed,
    const unsigned char* pub_seed, uint32_t leaf_idx, uint32_t idx_offset, uint32_t tree_height,
    void (*dev_gen_leaf)(unsigned char* /* leaf */, const unsigned char* /* sk_seed */,
                         const unsigned char* /* pub_seed */, uint32_t /* addr_idx */,
                         const uint32_t[8] /* tree_addr */),
    uint32_t tree_addr[8]) {
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int tnum = gridDim.x * blockDim.x;
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    u32 leaf_num = (1 << tree_height);
    u32 max_threads = leaf_num * SPX_WOTS_LEN;
    uint32_t wots_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};
    // __shared__ u8 s_wots_pk[64 * SPX_N];

    // if (max_threads > tnum) max_threads = tnum - SPX_WOTS_LEN;

#if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_THASH))
    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);
    dev_copy_subtree_addr(wots_addr, tree_addr);
#endif // if (defined(USING_PARALLEL_WOTS_PKGEN)) || (defined(USING_PARALLEL_THASH))

    // if (tid == 0) printf("max_threads = %d %d\n", leaf_num * SPX_WOTS_LEN, max_threads);
#ifdef USING_PARALLEL_WOTS_PKGEN
    if (tid < max_threads) {
        for (int i = tid; i < SPX_WOTS_LEN * leaf_num; i += max_threads) {
            dev_set_keypair_addr(wots_addr, i / SPX_WOTS_LEN);
            dev_set_chain_addr(wots_addr, i % SPX_WOTS_LEN);
            dev_set_hash_addr(wots_addr, 0);
#ifdef USING_LOCAL_MEMORY
            u8 temp[SPX_N];
            dev_prf_addr(temp, sk_seed, wots_addr);
            dev_gen_chain(temp, temp, 0, SPX_WOTS_W - 1, pub_seed, wots_addr);
            memcpy(wots_pk + i * SPX_N, temp, SPX_N);
#else  // ifdef USING_LOCAL_MEMORY
            dev_prf_addr(wots_pk + i * SPX_N, sk_seed, wots_addr);
            dev_gen_chain(wots_pk + i * SPX_N, wots_pk + i * SPX_N, 0, SPX_WOTS_W - 1, pub_seed,
                          wots_addr);
#endif // ifdef USING_LOCAL_MEMORY
        }
    }
#else  // ifdef USING_PARALLEL_WOTS_PKGEN
    if (tid < leaf_num) {
        u8* temp_pk = wots_pk + tid * SPX_WOTS_BYTES;
        dev_set_keypair_addr(wots_addr, tid);

        for (u32 i = 0; i < SPX_WOTS_LEN; i++) {
            dev_set_chain_addr(wots_addr, i);
            dev_set_hash_addr(wots_addr, 0);
            dev_prf_addr(temp_pk + i * SPX_N, sk_seed, wots_addr);
            dev_gen_chain(temp_pk + i * SPX_N, temp_pk + i * SPX_N, 0, SPX_WOTS_W - 1, pub_seed,
                          wots_addr);
        }
    }
#endif // ifdef USING_PARALLEL_WOTS_PKGEN
    g.sync();

    if (tid < leaf_num) {
        dev_copy_keypair_addr(wots_pk_addr, wots_addr);
        dev_set_keypair_addr(wots_pk_addr, tid + idx_offset);
#ifdef USING_LOCAL_MEMORY
        u8 temp[SPX_WOTS_BYTES];
        memcpy(temp, wots_pk + tid * SPX_WOTS_BYTES, SPX_WOTS_BYTES);
        dev_thash(leaf_node + tid * SPX_N, temp, SPX_WOTS_LEN, pub_seed, wots_pk_addr);
#else
        dev_thash(leaf_node + tid * SPX_N, wots_pk + tid * SPX_WOTS_BYTES, SPX_WOTS_LEN, pub_seed,
                  wots_pk_addr);
#endif
    }

#if (!defined(USING_PARALLEL_WOTS_PKGEN)) && (!defined(USING_PARALLEL_THASH))
    if (tid < leaf_num)
        dev_gen_leaf(leaf_node + tid * SPX_N, sk_seed, pub_seed, tid + idx_offset, tree_addr);
#endif // if (!defined(USING_PARALLEL_WOTS_PKGEN)) && (!defined(USING_PARALLEL_THASH))

#ifdef USING_PARALLEL_XMSS_BRANCH
    if (tid == ((leaf_idx >> 0) ^ 0x1)) memcpy(dev_auth_path, leaf_node + tid * SPX_N, SPX_N);

#define BRANCH_SCHEME2

#ifdef BRANCH_SCHEME2
    int branch_para = 4;
    branch_para = tnum;
    for (int i = 1, ii = 1; i <= tree_height; i++) {
        g.sync();
        // __syncthreads();
        dev_set_tree_height(tree_addr, i);
        if (tid < branch_para) {
            for (int j = tid; j < (leaf_num >> i); j += branch_para) {
                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(tree_addr, j);
#ifdef USING_LOCAL_MEMORY
                u8 temp[SPX_N * 2];
                memcpy(temp, leaf_node + off, SPX_N);
                memcpy(&temp[SPX_N], leaf_node + off + ii * SPX_N, SPX_N);
                dev_thash(leaf_node + off, temp, 2, pub_seed, tree_addr);
#else
                memcpy(leaf_node + off + SPX_N, leaf_node + off + ii * SPX_N, SPX_N);
                dev_thash(leaf_node + off, leaf_node + off, 2, pub_seed, tree_addr);
#endif
                // if(i == tree_height - 1 && tid == 1) {
                // 	for (int k = 0; k < SPX_N; k++)
                // 		printf("%02x", leaf_node[off + k]);
                // 	printf("\n");
                // }
                if (j == ((leaf_idx >> i) ^ 0x1)) {
                    memcpy(dev_auth_path + i * SPX_N, leaf_node + off, SPX_N);
                    // printf("i = %d\n", i);
                }
            }
        }
        ii *= 2;
    }

    if (tid == 0) {
        memcpy(auth_path, dev_auth_path, SPX_N * tree_height);
        memcpy(root, leaf_node, SPX_N);
    }

#else // ifdef BRANCH_SCHEME2
    // SCHEME 1
    unsigned char stack[(SPX_TREE_HEIGHT + 1) * SPX_N];
    unsigned int heights[SPX_FORS_HEIGHT + 1];
    unsigned int offset = 0;
    // uint32_t idx;
    uint32_t tree_idx;

    // if (tid == 0)
    // 	printf("leaf_num = %d\n", leaf_num); //512
    int para = 4;
    int stheight = tree_height - 2;
    int stNum = para;

    // int para = 2;
    // int stheight = tree_height - 1;
    // int stNum = 2;

    int stLeafNum = leaf_num / stNum;
    if (tid < para) {
        for (int i = tid * stLeafNum; i < (tid + 1) * stLeafNum; i++) {
            memcpy(stack + offset * SPX_N, leaf_node + i * SPX_N, SPX_N);
            offset++;
            heights[offset - 1] = 0;

            /* If this is a node we need for the auth path.. */
            if ((leaf_idx ^ 0x1) == i) {
                memcpy(dev_auth_path, stack + (offset - 1) * SPX_N, SPX_N);
            }

            /* While the top-most nodes are of equal height.. */
            while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
                /* Compute index of the new node, in the next layer. */
                tree_idx = (i >> (heights[offset - 1] + 1));

                /* Set the address of the node we're creating. */
                dev_set_tree_height(tree_addr, heights[offset - 1] + 1);
                dev_set_tree_index(tree_addr, tree_idx);
                /* Hash the top-most nodes from the stack together. */
                dev_thash(stack + (offset - 2) * SPX_N, stack + (offset - 2) * SPX_N, 2, pub_seed,
                          tree_addr);
                offset--;
                /* Note that the top-most node is now one layer higher. */
                heights[offset - 1]++;

                /* If this is a node we need for the auth path.. */
                if (((leaf_idx >> heights[offset - 1]) ^ 0x1) == tree_idx) {
                    memcpy(dev_auth_path + heights[offset - 1] * SPX_N,
                           stack + (offset - 1) * SPX_N, SPX_N);
                    // printf("i = %d %d\n", i, leaf_idx);
                }
            }
        }
        memcpy(leaf_node + tid * SPX_N, stack, SPX_N);
        // printf("%02x %d %d\n", leaf_node[tid * SPX_N], tid, leaf_idx);
    }
    for (int i = 1, ii = 1; i <= tree_height - stheight; i++) {
        g.sync();
        dev_set_tree_height(tree_addr, i + stheight);
        if (tid < para) {
            for (int j = tid; j < (stNum >> i); j += para) {
                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(tree_addr, j);
                memcpy(leaf_node + off + SPX_N, leaf_node + off + ii * SPX_N, SPX_N);
                dev_thash(leaf_node + off, leaf_node + off, 2, pub_seed, tree_addr);
                // printf("11 %02x %d %d\n", leaf_node[off], j, leaf_idx);
                if (j == ((leaf_idx >> (stheight + i)) ^ 0x1)) {
                    memcpy(dev_auth_path + (stheight + i) * SPX_N, leaf_node + off, SPX_N);
                    // printf("leaf_node + off = %02x %d %d\n", leaf_node[off], stheight, i);
                }
            }
        }
        ii *= 2;
    }

    if (tid == 0) {
        for (int i = 0; i < tree_height; i++) {
            memcpy(auth_path + i * SPX_N, dev_auth_path + i * SPX_N, SPX_N);
            // printf("%02x\n", auth_path[i * SPX_N]);
        }
        memcpy(root, leaf_node, SPX_N);
        // for (int i = 0; i < SPX_N; i++)
        // 	printf("%02x", root[i]);
        // printf("\n");
    }
    int aaa = 0;
    while (1) {
        aaa++;
    }

#endif // ifdef BRANCH_SCHEME2

#else  // ifdef USING_PARALLEL_XMSS_BRANCH
    unsigned char stack[(SPX_TREE_HEIGHT + 1) * SPX_N];
    unsigned int heights[SPX_FORS_HEIGHT + 1];
    unsigned int offset = 0;
    uint32_t idx;
    uint32_t tree_idx;
    g.sync();
    if (tid == 0) {
        for (idx = 0; idx < leaf_num; idx++) {
            memcpy(stack + offset * SPX_N, leaf_node + idx * SPX_N, SPX_N);
            offset++;
            heights[offset - 1] = 0;

            /* If this is a node we need for the auth path.. */
            if ((leaf_idx ^ 0x1) == idx) {
                memcpy(auth_path, stack + (offset - 1) * SPX_N, SPX_N);
            }

            /* While the top-most nodes are of equal height.. */
            while (offset >= 2 && heights[offset - 1] == heights[offset - 2]) {
                /* Compute index of the new node, in the next layer. */
                tree_idx = (idx >> (heights[offset - 1] + 1));

                /* Set the address of the node we're creating. */
                dev_set_tree_height(tree_addr, heights[offset - 1] + 1);
                dev_set_tree_index(tree_addr, tree_idx + (idx_offset >> (heights[offset - 1] + 1)));
                /* Hash the top-most nodes from the stack together. */
                dev_thash(stack + (offset - 2) * SPX_N, stack + (offset - 2) * SPX_N, 2, pub_seed,
                          tree_addr);
                offset--;
                /* Note that the top-most node is now one layer higher. */
                heights[offset - 1]++;

                /* If this is a node we need for the auth path.. */
                if (((leaf_idx >> heights[offset - 1]) ^ 0x1) == tree_idx) {
                    memcpy(auth_path + heights[offset - 1] * SPX_N, stack + (offset - 1) * SPX_N,
                           SPX_N);
                }
            }
        }
        memcpy(root, stack, SPX_N);
        // for (int jj = 0; jj < SPX_N; jj++) {
        // 	printf("%02x", root[jj]);
        // 	if (jj == SPX_N - 1) printf("\n");
        // }
    }
#endif // ifdef USING_PARALLEL_XMSS_BRANCH
}

__device__ void dev_ap_treehash_wots_shared(
    unsigned char* root, unsigned char* auth_path, const unsigned char* sk_seed,
    const unsigned char* pub_seed, uint32_t leaf_idx, uint32_t idx_offset, uint32_t tree_height,
    void (*dev_gen_leaf)(unsigned char* /* leaf */, const unsigned char* /* sk_seed */,
                         const unsigned char* /* pub_seed */, uint32_t /* addr_idx */,
                         const uint32_t[8] /* tree_addr */),
    uint32_t tree_addr[8]) {
    const unsigned int tid = threadIdx.x;
    // u8* wots_pk_off = wots_pk + (blockIdx.x * blockDim.x) * SPX_N;
    u8* leaf_node_off = dev_leaf + (blockIdx.x * blockDim.x) * SPX_N;
    u32 leaf_num = (1 << tree_height);
    uint32_t wots_addr[8] = {0};
    uint32_t wots_pk_addr[8] = {0};
    __shared__ u8 s_auth_path[SPX_TREE_HEIGHT * SPX_N];
    int para = blockDim.x;

    dev_set_type(wots_addr, SPX_ADDR_TYPE_WOTS);
    dev_set_type(wots_pk_addr, SPX_ADDR_TYPE_WOTSPK);
    dev_copy_subtree_addr(wots_addr, tree_addr);

    for (int i = tid; i < SPX_WOTS_LEN * leaf_num; i += para) {
        dev_set_keypair_addr(wots_addr, i / SPX_WOTS_LEN);
        dev_set_chain_addr(wots_addr, i % SPX_WOTS_LEN);
        dev_set_hash_addr(wots_addr, 0);
        u8 temp[SPX_N];
        dev_prf_addr(temp, sk_seed, wots_addr);
        dev_gen_chain(temp, temp, 0, SPX_WOTS_W - 1, pub_seed, wots_addr);
        memcpy(dev_wpk + (blockIdx.x * blockDim.x + i) * SPX_N, temp, SPX_N);
    }
    __syncthreads();

    if (tid < leaf_num) {
        dev_copy_keypair_addr(wots_pk_addr, wots_addr);
        dev_set_keypair_addr(wots_pk_addr, tid + idx_offset);
        u8 temp[SPX_WOTS_BYTES];
        memcpy(temp, dev_wpk + blockIdx.x * blockDim.x * SPX_N + tid * SPX_WOTS_BYTES,
               SPX_WOTS_BYTES);
        dev_thash(leaf_node_off + tid * SPX_N, temp, SPX_WOTS_LEN, pub_seed, wots_pk_addr);
    }
    __syncthreads();

    if (tid == ((leaf_idx >> 0) ^ 0x1)) memcpy(s_auth_path, leaf_node_off + tid * SPX_N, SPX_N);

    int branch_para = 4;
    branch_para = para;
    for (int i = 1, ii = 1; i <= tree_height; i++) {
        // g.sync();
        __syncthreads();
        dev_set_tree_height(tree_addr, i);
        if (tid < branch_para) {
            for (int j = tid; j < (leaf_num >> i); j += branch_para) {
                int off = 2 * j * ii * SPX_N;
                dev_set_tree_index(tree_addr, j);
                u8 temp[SPX_N * 2];
                memcpy(temp, leaf_node_off + off, SPX_N);
                memcpy(&temp[SPX_N], leaf_node_off + off + ii * SPX_N, SPX_N);
                dev_thash(leaf_node_off + off, temp, 2, pub_seed, tree_addr);
                if (j == ((leaf_idx >> i) ^ 0x1)) {
                    memcpy(s_auth_path + i * SPX_N, leaf_node_off + off, SPX_N);
                }
            }
        }
        ii *= 2;
    }

    if (tid == 0) {
        memcpy(auth_path, s_auth_path, SPX_N * tree_height);
        memcpy(root, leaf_node_off, SPX_N);
    }
}
