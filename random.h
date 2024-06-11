
#ifndef RANDOM_H
#define RANDOM_H

//#include "vector.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern void seedMT(uint32_t seed);

extern float randomMT(void);

extern uint32_t randomMT_i(void);

//extern void      random_uniform_vector(ri_vector_t *dst);

#ifdef __cplusplus
}	/* extern "C" */
#endif

#endif
