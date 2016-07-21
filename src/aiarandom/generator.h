#ifndef AIA_RANDOM_GENERATOR_H
#define AIA_RANDOM_GENERATOR_H

#include <aiautil/util.h>

#define MERSENNE_STATE_N 624 /* degree of recurrence */
#define MERSENNE_STATE_M 397 /* middle word, an offset used in the recurrence relation defining the series x, 1 ≤ m < n */
#define RAND_MAX_VAL 0xFFFFFFFFUL
#define RAND_STATE_LEN 624

/* State container for a single random number stream */
typedef struct AIARandGen {
	unsigned long initSeed;
	int pos;
	int isSeeded;
	unsigned long state[RAND_STATE_LEN]; /* state vector */

	/* fields for normal distribution */
	double normalX;
	double normalY;
	double normalRho;
	int isNormalValid;
} AIARandGen;

/* create new seeded generator */
AIA_API AIARandGen* aiarandgen_new(void);

/* create new manually seeded generator */
AIA_API AIARandGen* aiarandgen_newWithSeed(unsigned long seed);

/* create copy of a generator from source */
AIA_API AIARandGen* aiarandgen_copy(AIARandGen *this, AIARandGen *src);

/* free memory for a generator */
AIA_API void aiarandgen_free(AIARandGen *this);

/* check if generator is valid */
AIA_API int aiarandgen_valid(AIARandGen *this);

/* return initial seed for the generator */
AIA_API unsigned long aiarandgen_initSeed(AIARandGen *this);

/* reseed the generator with a new seed */
AIA_API void aiarandgen_reset(AIARandGen *this, unsigned long seed);

/* return a random unsigned long between 0 and RAND_MAX */
AIA_API unsigned long aiarandgen_random(AIARandGen *gen);

/* return a random long between 0 and LONG_MAX */
AIA_API long aiarandgen_long(AIARandGen *gen);

/* return a random unsigned long between 0 and ULONG_MAX */
AIA_API unsigned long aiarandgen_ulong(AIARandGen *gen);

/* return a random double between 0.0 and 1.0  */
AIA_API double aiarandgen_double(AIARandGen *gen);

/* return a random double between 0.0 and 1.0  */
AIA_API float aiarandgen_float(AIARandGen *gen);

#endif
