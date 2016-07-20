#include "generator.h"

#ifndef WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

/* Macros for the Mersenne Twister random generator */
/* Period parameters */
/* Prime is 2^19937 - 1 which is a mersenne prime*/
#define MATRIX_A 0x9908b0dfUL /* constant vector a */
#define UMASK 0x80000000UL /* most significant w-r bits */
#define LMASK 0x7fffffffUL /* least significant r bits */

/* seed a generator with manually provided seed */
static void aiarandgen_seed_(AIARandGen *this, unsigned long seed) {
  int i;

  this->initSeed = seed;
  seed = this->initSeed & 0xffffffffUL;

  for (i = 0; i < RAND_STATE_LEN; i++) {
    this->state[i] = seed;
    /* knuth prng */
    seed = (1812433253UL * (seed ^ (seed >> 30)) + i + 1) & 0xffffffffUL;
  }
  this->pos = RAND_STATE_LEN;
  this->isSeeded = 1;
}

static unsigned long aiarandgen_random_(AIARandGen *gen) {
  unsigned long random;

  if (gen->pos == RAND_STATE_LEN) {
    int i;

    for (i = 0; i < MERSENNE_STATE_N - MERSENNE_STATE_M; i++) {
      random = (gen->state[i] & UMASK) | (gen->state[i + 1] & LMASK);
      gen->state[i] = gen->state[i + MERSENNE_STATE_M] ^ (random >> 1) ^ (-(random & 1) & MATRIX_A);
    }

    for (; i < MERSENNE_STATE_N - 1; i++) {
      random = (gen->state[i] & UMASK) | (gen->state[i + 1] & LMASK);
      gen->state[i] = gen->state[i + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (random >> 1) ^ (-(random & 1) & MATRIX_A);
    }

    random = (gen->state[MERSENNE_STATE_N - 1] & UMASK) | (gen->state[0] & LMASK);
    gen->state[MERSENNE_STATE_N - 1] = gen->state[MERSENNE_STATE_M - 1] ^ (random >> 1) ^ (-(random & 1) & MATRIX_A);

    gen->pos = 0;
  }

  random = gen->state[gen->pos++];

  /* Tempering */
  random ^= (random >> 11);
  random ^= (random << 7) & 0x9d2c5680UL;
  random ^= (random << 15) & 0xefc60000UL;
  random ^= (random >> 18);

  return random;
}

/* creates an unseeded generator */
static AIARandGen* aiarandgen_empty() {
  AIARandGen *this = aia_alloc(sizeof(AIARandGen));
  memset(this, 0, sizeof(AIARandGen));
  this->pos = 0;
  this->initSeed = 0;
  this->isSeeded = 0;
  this->isNormalValid = 0;
  return this;
}

/* read from /dev/urandom in unix */
#ifndef WIN32
static unsigned long aiarandgen_readURandom() {
  int randDev;
  unsigned long randValue;
  ssize_t readBytes;

  randDev = open("/dev/urandom", O_RDONLY);
  if (randDev < 0) {
    aia_error("Unable to open /dev/urandom");
  }

  readBytes = read(randDev, &randValue, sizeof(randValue));
  if (readBytes < sizeof(randValue)) {
    aia_error("Unable to read from /dev/urandom");
  }
  close(randDev);
  return randValue;
}
#endif


AIARandGen* aiarandgen_new() {
  unsigned long seed;

  AIARandGen *this = aiarandgen_empty();
  /* get seed from urandom or system time */
  #ifdef WIN32
    seed = (unsigned long)time(0);
  #else
    seed = aiarandgen_readURandom();
  #endif

  aiarandgen_seed_(this, seed);
  return this;
}

AIARandGen* aiarandgen_newWithSeed(unsigned long seed) {
  AIARandGen *this = aiarandgen_empty();
  aiarandgen_seed_(this, seed);
  return this;
}

void aiarandgen_reset(AIARandGen *this, unsigned long seed) {
  /* This ensures reseeding resets all of the state of generator*/
  AIARandGen *empty = aiarandgen_empty();
  aiarandgen_copy(this, empty);
  aiarandgen_free(empty);
  aiarandgen_seed_(this, seed);
}

AIARandGen* aiarandgen_copy(AIARandGen *this, AIARandGen *src) {
  memcpy(this, src, sizeof(AIARandGen));
  return this;
}

void aiarandgen_free(AIARandGen *this) {
  aia_free(this);
}

int aiarandgen_valid(AIARandGen *this) {
  if ((this->isSeeded == 1) && (this->pos > 0 &&
    this->pos <= RAND_STATE_LEN)) {
    return 1;
  }
  return 0;
}

unsigned long aiarandgen_initSeed(AIARandGen *this) {
  return this->initSeed;
}

unsigned long aiarandgen_random(AIARandGen *gen) {
  return aiarandgen_random_(gen);
}

unsigned long aiarandgen_ulong(AIARandGen *gen) {
  #if ULONG_MAX <= 0xffffffffUL
    return aiarandgen_random(gen);
  #else
    return (aiarandgen_random(gen) << 32) | (aiarandgen_random(gen));
  #endif
}

long aiarandgen_long(AIARandGen *gen) {
  return aiarandgen_ulong(gen) >> 1;
}

double aiarandgen_double(AIARandGen *gen) {
  /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
  long a = aiarandgen_random(gen) >> 5;
  long b = aiarandgen_random(gen) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

float aiarandgen_float(AIARandGen *gen) {
  return ((float)aiarandgen_random(gen) * 2.3283064365387e-10);
}


