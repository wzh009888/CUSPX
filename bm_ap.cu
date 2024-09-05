// #define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "api.h"
#include "fors.h"
#include "wots.h"
#include "params.h"
#include "rng.h"

#if (defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F))
#define NTESTS 10
#else // if (defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F))
#define NTESTS 2
#endif // if (defined(SPX_128F) || defined(SPX_192F) || defined(SPX_256F))
#define NTESTS_GPU 300

#include "all_option.h"

static int cmp_llu(const void *a, const void *b)
{
	if (*(unsigned long long *)a < *(unsigned long long *)b) return -1;
	if (*(unsigned long long *)a > *(unsigned long long *)b) return 1;
	return 0;
} // cmp_llu

static unsigned long long median(unsigned long long *l, size_t llen)
{
	qsort(l, llen, sizeof(unsigned long long), cmp_llu);

	if (llen % 2) return l[llen / 2];
	else return (l[llen / 2 - 1] + l[llen / 2]) / 2;
} // median

static void delta(unsigned long long *l, size_t llen)
{
	unsigned int i;

	for (i = 0; i < llen - 1; i++) {
		l[i] = l[i + 1] - l[i];
	}
} // delta

static unsigned long long cpucycles(void)
{
	unsigned long long result;
	__asm volatile (".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
			: "=a" (result) ::  "%rdx");

	return result;
} // cpucycles

static void printfcomma(unsigned long long n)
{
	if (n < 1000) {
		printf("%llu", n);
		return;
	}
	printfcomma(n / 1000);
	printf(",%03llu", n % 1000);
} // printfcomma

static void printfalignedcomma(unsigned long long n, int len)
{
	unsigned long long ncopy = n;
	int i = 0;

	while (ncopy > 9) {
		len -= 1;
		ncopy /= 10;
		i += 1; // to account for commas
	}
	i = i / 3 - 1;  // to account for commas
	for (; i < len; i++) {
		printf(" ");
	}
	printfcomma(n);
} // printfalignedcomma

static void display_result(double result, unsigned long long *l, size_t llen, unsigned long long mul)
{
	unsigned long long med;

	result /= NTESTS;
	delta(l, NTESTS + 1);
	med = median(l, llen);
	printf("avg. %11.2lf us (%2.2lf sec); median ", result, result / 1e6);
	printfalignedcomma(med, 12);
	printf(" cycles,  %5llux: ", mul);
	printfalignedcomma(mul * med, 12);
	printf(" cycles\n");
} // display_result

#define MEASURE(TEXT, MUL, FNCALL) \
	printf(TEXT); \
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); \
	for (i = 0; i < NTESTS; i++) { \
		t[i] = cpucycles(); \
		FNCALL; \
	} \
	t[NTESTS] = cpucycles(); \
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop); \
	result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3; \
	g_result = result / NTESTS; \
	display_result(result, t, NTESTS, MUL);

#define MEASURE_GPU(TEXT, MUL, FNCALL) \
	g_result = 0; \
	g_count = 100000; \
	for (i = 0; i < MUL; i++) \
	FNCALL; \
	g_result /= MUL; \
	// printf("%11.2lf us (%2.2lf sec)\n", g_result, g_result / 1e6);

int main()
{
	/* Make stdout buffer more responsive. */
	setbuf(stdout, NULL);

	unsigned char *pk, *sk, *m, *sm, *mout;

	//
	CHECK(cudaMallocHost(&pk, SPX_PK_BYTES));
	CHECK(cudaMallocHost(&sk, SPX_SK_BYTES));
	CHECK(cudaMallocHost(&m, SPX_MLEN));
	CHECK(cudaMallocHost(&sm, (SPX_BYTES + SPX_MLEN)));
	CHECK(cudaMallocHost(&mout, (SPX_BYTES + SPX_MLEN)));

	unsigned char addr[SPX_ADDR_BYTES];

	unsigned long long smlen;
	unsigned long long mlen;
	int i;
	double result_kg, result_sign, result_verify;

	randombytes(m, SPX_MLEN);
	randombytes(addr, SPX_ADDR_BYTES);

	// warming up
	for (i = 0; i < NTESTS; i++) {
		face_crypto_sign_keypair(pk, sk);
		face_crypto_sign(sm, &smlen, m, SPX_MLEN, sk);
		face_crypto_sign_open(mout, &mlen, sm, smlen, pk);
	}

	MEASURE_GPU("Generating keypair.. ", NTESTS_GPU, face_ap_crypto_sign_keypair(pk, sk));
	// MEASURE_GPU("Generating keypair.. ", 1, face_crypto_sign_keypair(pk, sk));
	result_kg = g_result;
	MEASURE_GPU("Signing..            ", NTESTS_GPU, face_ap_crypto_sign(sm, &smlen, m, SPX_MLEN, sk));
	// MEASURE_GPU("Signing..            ", 1, face_crypto_sign(sm, &smlen, m, SPX_MLEN, sk));
	result_sign = g_result;
	MEASURE_GPU("Verifying..          ", NTESTS_GPU, face_ap_crypto_sign_open(mout, &mlen, sm, smlen, pk));
	// MEASURE_GPU("Verifying..          ", 1, face_crypto_sign_open(mout, &mlen, sm, smlen, pk));
	result_verify = g_result;
	// ms
	printf("%.3lf %.3lf %.3lf\n",
	       result_kg / 1000, result_sign / 1000, result_verify / 1000);

	CHECK(cudaFreeHost(m));
	CHECK(cudaFreeHost(sm));
	CHECK(cudaFreeHost(mout));

	return 0;
} // main
