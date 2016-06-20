#ifndef AIA_UTIL_CHECK_H
#define AIA_UTIL_CHECK_H

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

extern void _aia_error(const char *file, const int line, const char *fmt, ...);
extern void _aia_argcheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...);
extern void _aia_assertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...);

#define WITHIN_RANGE(x, from, to) (x >= from) && (x < to)

#define aia_error(...) _aia_error(__FILE__, __LINE__, __VA_ARGS__)

#define aia_argcheck(...)                             \
do {                                                  \
  _aia_argcheck(__FILE__, __LINE__, __VA_ARGS__);     \
} while(0)

#define aia_assert(exp)                                   \
do {                                                      \
  if(!(exp)) {                                            \
    _aia_assertionFailed(__FILE__, __LINE__, #exp, "");   \
  }                                                       \
} while(0)

#define aia_assertMsg(exp, ...)                                       \
do {                                                                  \
  if(!(exp)) {                                                        \
    _aia_assertionFailed(__FILE__, __LINE__, #exp, __VA_ARGS__);      \
  }                                                                   \
} while(0)

#endif