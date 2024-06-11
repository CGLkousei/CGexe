
#include <stdio.h>

#include "random.h"

/* Period parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0df   /* constant vector a */
#define UPPER_MASK 0x80000000 /* most significant w-r bits */
#define LOWER_MASK 0x7fffffff /* least significant r bits */

/* Tempering parameters */
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static uint32_t mt[N]; /* the array for the state vector  */
static int32_t mti = N + 1; /* mti==N+1 means mt[N] is not initialized */

/* initializing the array with a NONZERO seed */
void
#if 0
sgenrand(uint32_t seed)
#endif
seedMT(uint32_t seed) {
    /* setting initial seeds to mt[N] using         */
    /* the generator Line 25 of Table 1 in          */
    /* [KNUTH 1981, The Art of Computer Programming */
    /*    Vol. 2 (2nd Ed.), pp102]                  */
    mt[0] = seed & 0xffffffff;
    for (mti = 1; mti < N; mti++)
        mt[mti] = (69069 * mt[mti - 1]) & 0xffffffff;
}

float /* generating reals */
/* unsigned long */ /* for integer generation */
randomMT() {
    uint32_t y;
    static uint32_t mag01[2] = {0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int32_t kk;

        if (mti == N + 1)   /* if sgenrand() has not been called, */
            //sgenrand(4357); /* a default initial seed is used   */
            seedMT(4357); /* a default initial seed is used   */

        for (kk = 0; kk < N - M; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (; kk < N - 1; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

        mti = 0;
    }

    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);

    return ((float) y * (float) 2.3283064365386963e-10); /* reals: [0,1)-interval */
    /* return y; */ /* for integer generation */
}


uint32_t //*/ /* for integer generation */
randomMT_i() {
    uint32_t y;
    static uint32_t mag01[2] = {0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N + 1)   /* if sgenrand() has not been called, */
            //sgenrand(4357); /* a default initial seed is used   */
            seedMT(4357); /* a default initial seed is used   */

        for (kk = 0; kk < N - M; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        for (; kk < N - 1; kk++) {
            y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1];

        mti = 0;
    }

    y = mt[mti++];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);

    //   return ( (float)y * (float)2.3283064365386963e-10 ); /* reals: [0,1)-interval */
    return y; //*/ /* for integer generation */
}

#if 0
/* this main() outputs first 1000 generated numbers  */
main()
{ 
    int32_t j;

    //sgenrand(4357); /* any nonzero integer can be used as a seed */
    seedMT(4357); /* any nonzero integer can be used as a seed */
    for (j=0; j<1000; j++) {
        printf("%10.8f ", genrand());
        if (j%8==7) printf("\n");
    }
    printf("\n");
}
#endif
