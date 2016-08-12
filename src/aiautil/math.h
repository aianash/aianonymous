#ifndef AIA_UTIL_MATH_H
#define AIA_UTIL_MATH_H

/* epsilon equality comaparision for double */
#define epsieq(a, b, epsi) (epsi > 0 ? (fabs(a - b) < epsi) : (fabs(a - b) < -epsi))

/* epsilon equality comaparision for float */
#define epsieqf(a, b, epsi) (epsi > 0 ? (fabsf(a - b) < epsi) : (fabsf(a - b) < -epsi))

#define max2(a, b) ((a > b) ? a : b)
#define max3(a, b, c) max2(a, max2(b, c))
#define min2(a, b) ((a < b) ? a : b)

#ifndef PI
# define PI 3.14159265358979323846
#endif

#endif