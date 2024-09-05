#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../api.h"
#include "../params.h"
#include "../rng.h"

#define SPX_SIGNATURES 1

#include "all_option.h"
#include <iostream>
using namespace std;

int main()
{
	int ret = 0;
	int i;

	/* Make stdout buffer more responsive. */
	setbuf(stdout, NULL);
	u8 *pk, *sk, *test_pk, *test_sk;
	u8 *m, *sm, *mout;

	CHECK(cudaMallocHost(&pk, SPX_PK_BYTES * 10));
	CHECK(cudaMallocHost(&sk, SPX_SK_BYTES * 10));
	CHECK(cudaMallocHost(&test_pk, SPX_PK_BYTES * 10));
	CHECK(cudaMallocHost(&test_sk, SPX_SK_BYTES * 10));
	CHECK(cudaMallocHost(&m, SPX_MLEN * 10));
	CHECK(cudaMallocHost(&sm, (SPX_BYTES + SPX_MLEN) * 10));
	CHECK(cudaMallocHost(&mout, (SPX_BYTES + SPX_MLEN) * 10));
	unsigned long long smlen;
	unsigned long long mlen;

	randombytes(m, SPX_MLEN);

	printf("Testing SPX.. \n");
	printf("test gpu key generation\n");
	//使用相同种子，在函数内部进行了分别进行了替换

	if (crypto_sign_keypair(pk, sk)) {
		printf("failed!\n");
		return -1;
	}
	face_crypto_sign_keypair(test_pk, test_sk);
	int wrong = 0;

	for (size_t i = 0; i < SPX_PK_BYTES; i++) {
		if (pk[i] != test_pk[i]) {
			printf("wrong pk\n");
			wrong = 1;
			break;
		}
	}
	for (size_t i = 0; i < SPX_SK_BYTES; i++) {
		if (sk[i] != test_sk[i]) {
			printf("wrong sk\n");
			wrong = 1;
			break;
		}
	}
	if (wrong == 0)
		printf("pk & sk of spx are same, successful.\n");
	else {
		printf("cpu_pk:");
		for (size_t i = 0; i < SPX_PK_BYTES; i++) {
			if (i % 8 == 0) printf("\n");
			printf("%02x", pk[i]);
		}
		printf("\n");
		printf("cpu_sk:");
		for (size_t i = 0; i < SPX_SK_BYTES; i++) {
			if (i % 8 == 0) printf("\n");
			printf("%02x", sk[i]);
		}
		printf("\n");
		printf("\n");

		printf("gpu_pk:");
		for (size_t i = 0; i < SPX_PK_BYTES; i++) {
			if (i % 8 == 0) printf("\n");
			printf("%02x", test_pk[i]);
		}
		printf("\n");
		printf("gpu_sk:");
		for (size_t i = 0; i < SPX_SK_BYTES; i++) {
			if (i % 8 == 0) printf("\n");
			printf("%02x", test_sk[i]);
		}
		printf("\n");
		printf("please check the random number or the code\n");
	}

	printf("SPX_PK_BYTES = %d, SPX_SK_BYTES = %d\n",
	       SPX_PK_BYTES, SPX_SK_BYTES);

	printf("Testing %d signatures.. \n", SPX_SIGNATURES);

	printf("test gpu signature\n");
	for (i = 0; i < SPX_SIGNATURES; i++) {
		printf("  - iteration #%d:\n", i);

		face_crypto_sign(sm, &smlen, m, SPX_MLEN, sk);
#ifdef DEBUG_MODE
		u8 *sm_tmp;
		unsigned long long smlen_tmp;
		CHECK(cudaMallocHost(&sm_tmp, SPX_BYTES + SPX_MLEN));
		crypto_sign(sm_tmp, &smlen_tmp, m, SPX_MLEN, sk);
		int r = memcmp(sm_tmp, sm, SPX_BYTES + SPX_MLEN);
		printf("set the optrand is 0, test sig if same... ");
		if(r == 0) printf("Same signature, successful.\n");
		else printf("Different signature, failed.\n");
#endif

#ifdef VARIANT
		// we use different macro, so we do not check this
		if (smlen != SPX_BYTES + SPX_MLEN) {
			printf("  X smlen incorrect [%llu != %u]!\n",
			       smlen, SPX_BYTES + SPX_MLEN);
			ret = -1;
		}else {
			printf("    smlen as expected [%llu].\n", smlen);
		}
#endif

		/* Test if signature is valid. */
		if (crypto_sign_open(mout, &mlen, sm, smlen, pk)) {
			printf("  X verification failed!\n");
			ret = -1;
		} else {
			printf("    verification succeeded.\n");
		}

		/* Test if the correct message was recovered. */
		if (mlen != SPX_MLEN) {
			printf("  X mlen incorrect [%llu != %u]!\n", mlen, SPX_MLEN);
			ret = -1;
		} else {
			printf("    mlen as expected [%llu].\n", mlen);
		}
		if (memcmp(m, mout, SPX_MLEN)) {
			printf("  X output message incorrect!\n");
			ret = -1;
		} else {
			printf("    output message as expected.\n");
		}

		/* Test if signature is valid when validating in-place. */
		if (crypto_sign_open(sm, &mlen, sm, smlen, pk)) {
			printf("  X in-place verification failed!\n");
			ret = -1;
		} else {
			printf("    in-place verification succeeded.\n");
		}

		/* Test if flipping bits invalidates the signature (it should). */

		/* Flip the first bit of the message. Should invalidate. */
		sm[smlen - 1] ^= 1;
		if (!crypto_sign_open(mout, &mlen, sm, smlen, pk)) {
			printf("  X flipping a bit of m DID NOT invalidate signature!\n");
			ret = -1;
		} else {
			printf("    flipping a bit of m invalidates signature.\n");
		}
		sm[smlen - 1] ^= 1;

#ifdef SPX_TEST_INVALIDSIG
		int j;
		/* Flip one bit per hash; the signature is entirely hashes. */
		for (j = 0; j < (int)(smlen - SPX_MLEN); j += SPX_N) {
			sm[j] ^= 1;
			if (!crypto_sign_open(mout, &mlen, sm, smlen, pk)) {
				printf("  X flipping bit %d DID NOT invalidate sig + m!\n", j);
				sm[j] ^= 1;
				ret = -1;
				break;
			}
			sm[j] ^= 1;
		}
		if (j >= (int)(smlen - SPX_MLEN)) {
			printf("    changing any signature hash invalidates signature.\n");
		}
#endif // ifdef SPX_TEST_INVALIDSIG
	}

	printf("test gpu verification\n");
	for (i = 0; i < SPX_SIGNATURES; i++) {
		printf("  - iteration #%d:\n", i);

		crypto_sign(sm, &smlen, m, SPX_MLEN, sk);
	#ifdef VARIANT
		if (smlen != SPX_BYTES + SPX_MLEN) {
			printf("  X smlen incorrect [%llu != %u]!\n",
			       smlen, SPX_BYTES);
			ret = -1;
		}else {
			printf("    smlen as expected [%llu].\n", smlen);
		}
	#endif

		/* Test if signature is valid. */
		if (face_crypto_sign_open(mout, &mlen, sm, smlen, pk)) {
			printf("  X verification failed!\n");
			ret = -1;
		}else {
			printf("    verification succeeded.\n");
		}

		/* Test if the correct message was recovered. */
		if (mlen != SPX_MLEN) {
			printf("  X mlen incorrect [%llu != %u]!\n", mlen, SPX_MLEN);
			ret = -1;
		}else {
			printf("    mlen as expected [%llu].\n", mlen);
		}
		if (memcmp(m, mout, SPX_MLEN)) {
			printf("  X output message incorrect!\n");
			ret = -1;
		}else {
			printf("    output message as expected.\n");
		}

		/* Test if signature is valid when validating in-place. */
		if (face_crypto_sign_open(sm, &mlen, sm, smlen, pk)) {
			printf("  X in-place verification failed!\n");
			ret = -1;
		}else {
			printf("    in-place verification succeeded.\n");
		}

		/* Test if flipping bits invalidates the signature (it should). */

		/* Flip the first bit of the message. Should invalidate. */
		sm[smlen - 1] ^= 1;
		if (!face_crypto_sign_open(mout, &mlen, sm, smlen, pk)) {
			printf("  X flipping a bit of m DID NOT invalidate signature!\n");
			ret = -1;
		}else {
			printf("    flipping a bit of m invalidates signature.\n");
		}
		sm[smlen - 1] ^= 1;

#ifdef SPX_TEST_INVALIDSIG
		int j;
		/* Flip one bit per hash; the signature is entirely hashes. */
		for (j = 0; j < (int)(smlen - SPX_MLEN); j += SPX_N) {
			sm[j] ^= 1;
			if (!face_crypto_sign_open(mout, &mlen, sm, smlen, pk)) {
				printf("  X flipping bit %d DID NOT invalidate sig + m!\n", j);
				sm[j] ^= 1;
				ret = -1;
				break;
			}
			sm[j] ^= 1;
		}
		if (j >= (int)(smlen - SPX_MLEN)) {
			printf("    changing any signature hash invalidates signature.\n");
		}
#endif // ifdef SPX_TEST_INVALIDSIG
	}

	return ret;
} // main
