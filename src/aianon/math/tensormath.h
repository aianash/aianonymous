#include <aianon/tensor.h>

#ifndef AIA_TENSORMATH_H
#ifdef ERASED_TYPE_AVAILABLE


#define aiatensor__(name) aiatensor_(T_, name)
#define AIATensor_ AIATensor(T_)


extern void aiatensor__(cadd)(AIATensor_ *res, AIATensor_ *tensor1, T value, AIATensor_ *tensor2);
extern void aiatensor__(csub)(AIATensor_ *res, AIATensor_ *tensor1, T value, AIATensor_ *tensor2);
extern void aiatensor__(cmul)(AIATensor_ *res, AIATensor_ *tensor1, AIATensor_ *tensor2);
extern void aiatensor__(cpow)(AIATensor_ *res, AIATensor_ *base, AIATensor_ *exp);
extern void aiatensor__(cdiv)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom);
extern void aiatensor__(cfmod)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom);
extern void aiatensor__(cremainder)(AIATensor_ *res, AIATensor_ *numer, AIATensor_ *denom);

extern void aiatensor__(addbmm)(AIATensor_ *res, T alpha, AIATensor_ *tensor, T beta, AIATensor_ *batch1, AIATensor_ *batch2);
extern void aiatensor__(baddbmm)(AIATensor_ *res, T alpha, AIATensor_ *batch3, T beta, AIATensor_ *batch1, AIATensor_ *batch2);

#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aianon/math/tensormath.h"
#include <aianon/util/erasure.h>


#define AIA_TENSORMATH_H
#endif