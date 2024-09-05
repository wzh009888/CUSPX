#include <stdio.h>
#include <string.h>

#include "../hash.h"
#include "../fors.h"
#include "../rng.h"
#include "../params.h"

#include "../common.h"

int main()
{
	/* Make stdout buffer more responsive. */
	setbuf(stdout, NULL);

	unsigned char sk_seed[SPX_N];
	unsigned char pub_seed[SPX_N];
	unsigned char pk1[SPX_FORS_PK_BYTES];
	unsigned char pk2[SPX_FORS_PK_BYTES];
	unsigned char sig[SPX_FORS_BYTES];
	unsigned char m[SPX_FORS_MSG_BYTES];
	uint32_t addr[8] = { 0 };

	randombytes(sk_seed, SPX_N);
	randombytes(pub_seed, SPX_N);
	randombytes(m, SPX_FORS_MSG_BYTES);
	randombytes((unsigned char *)addr, 8 * sizeof(uint32_t));

	printf("Testing FORS signature and PK derivation..\n");

	initialize_hash_function(pub_seed, sk_seed);

    face_initialize_hash_function(pub_seed, sk_seed);

	fors_sign(sig, pk1, m, sk_seed, pub_seed, addr);
	fors_pk_from_sig(pk2, sig, m, pub_seed, addr);

	if (memcmp(pk1, pk2, SPX_FORS_PK_BYTES)) {
		printf("CPU test failed!\n");
		return -1;
	}
	printf("CPU test succeeded.\n");

	face_fors_sign(sig, pk1, m, sk_seed, pub_seed, addr, 1);
	fors_pk_from_sig(pk2, sig, m, pub_seed, addr);

	if (memcmp(pk1, pk2, SPX_FORS_PK_BYTES)) {
		printf("fors_sign failed!\n");
		return -1;
	}
	printf("fors_sign succeeded.\n");

    fors_sign(sig, pk1, m, sk_seed, pub_seed, addr);
    face_fors_pk_from_sig(pk2, sig, m, pub_seed, addr, 1);

    if (memcmp(pk1, pk2, SPX_FORS_PK_BYTES)) {
        printf("fors_pk_from_sig failed!\n");
        return -1;
    }
    printf("fors_pk_from_sig succeeded.\n");

	face_fors_sign(sig, pk1, m, sk_seed, pub_seed, addr, 1);
	face_fors_pk_from_sig(pk2, sig, m, pub_seed, addr, 1);

	if (memcmp(pk1, pk2, SPX_FORS_PK_BYTES)) {
		printf("GPU fors test failed!\n");
		printf("%02x %02x \n", pk1[0], pk2[0]);
		return -1;
	}
	printf("GPU fors test succeeded.\n");
	return 0;
} // main
