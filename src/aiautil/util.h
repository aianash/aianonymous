#ifndef AIA_CORE_UTIL_H
#define AIA_CORE_UTIL_H

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <aiautil/bits.h>

#define AIA_EXTERNC extern
#define AIA_API AIA_EXTERNC

#define USE_BLAS
#define USE_LAPACK

typedef int bool;
#define TRUE 1
#define FALSE 0

#define fsigndiff(x, y) (*(x) * (*(y) / fabs(*(y))) < 0.)

#define SWAP(T, x, y) \
do {                  \
  T t = x;            \
  x = y;              \
  y = t;              \
} while(0);

#define aia_cleanup(...) __VA_ARGS__

#define AIA_CONCAT_STRING_2(x,y) AIA_CONCAT_STRING_2_EXPAND(x,y)
#define AIA_CONCAT_STRING_2_EXPAND(x,y) #x #y

#define AIA_CONCAT_STRING_3(x,y,z) AIA_CONCAT_STRING_3_EXPAND(x,y,z)
#define AIA_CONCAT_STRING_3_EXPAND(x,y,z) #x #y #z

#define AIA_CONCAT_STRING_4(x,y,z,w) AIA_CONCAT_STRING_4_EXPAND(x,y,z,w)
#define AIA_CONCAT_STRING_4_EXPAND(x,y,z,w) #x #y #z #w

#define AIA_CONCAT_2(x,y) AIA_CONCAT_2_EXPAND(x,y)
#define AIA_CONCAT_2_EXPAND(x,y) x ## y

#define AIA_CONCAT_3(x,y,z) AIA_CONCAT_3_EXPAND(x,y,z)
#define AIA_CONCAT_3_EXPAND(x,y,z) x ## y ## z

#define AIA_CONCAT_4_EXPAND(x,y,z,w) x ## y ## z ## w
#define AIA_CONCAT_4(x,y,z,w) AIA_CONCAT_4_EXPAND(x,y,z,w)

// Use this macro to define and use type variable names
// for default instances
#define default(type, name) AIA_CONCAT_3(default_, name, type)
#define default_(name) default(T_, name)

// For type generic instantiation
#define asT(val) ((T)(val))
#define asTp(ptr) ((T *)(val))

#include <aiautil/memory.h>
#include <aiautil/check.h>
#include <aiautil/array.h>
#include <aiautil/math.h>

#endif