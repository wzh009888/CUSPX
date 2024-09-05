#include "common.h"
#include "params.h"

double g_result = 0;
double g_inner_result = 0;
double g_count = -100000;

void show_para() {
    string res = "SLH-DSA-";

#ifdef SHA256
    res += "SHA-256";
#endif
#ifdef SHAKE256
    res += "SHAKE256";
#endif
#ifdef HARAKA
    res += "Haraka";
#endif

#ifdef SPX_128F
    res += "-128f";
#endif
#ifdef SPX_128S
    res += "-128s";
#endif
#ifdef SPX_192F
    res += "-192f";
#endif
#ifdef SPX_192S
    res += "-192s";
#endif
#ifdef SPX_256F
    res += "-256f";
#endif
#ifdef SPX_256S
    res += "-256s";
#endif

    printf("n = %d, h = %d, d = %d, b = %d, k = %d, w = %d, len = %d\n", SPX_N, SPX_FULL_HEIGHT,
           SPX_D, SPX_FORS_HEIGHT, SPX_FORS_TREES, SPX_WOTS_W, SPX_WOTS_LEN);

    cout << res << endl;
}