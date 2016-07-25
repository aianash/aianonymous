#ifndef AIA_UTIL_MATH_H
#define AIA_UTIL_MATH_H

/* epsilon equality comaparision for double */
#define epsieq(a, b, epsi) (epsi > 0 ? (fabs(a - b) < epsi) : (fabs(a - b) < -epsi))

/* epsilon equality comaparision for float */
#define epsieqf(a, b, epsi) (epsi > 0 ? (fabsf(a - b) < epsi) : (fabsf(a - b) < -epsi))

#endif