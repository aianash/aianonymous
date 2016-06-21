#ifndef AIA_TENSOR_LINALG_H

#include <aianon/core/util.h>
#include <aianon/core/math/lapack.h>
#include <aianon/tensor/tensor.h>

#ifndef NON_ERASED_BLOCK
#define NON_ERASED_BLOCK

#define aia_lapackCheckWithCleanup(fmt, cleanup, func, info, ...)     \
if (info < 0) {                                                       \
  cleanup                                                             \
  aia_error("Lapack Error in %s : Illegal Argument %d", func, -info); \
} else if(info > 0) {                                                 \
  cleanup                                                             \
  aia_error(fmt, func, info, ##__VA_ARGS__);                          \
}

#endif

#ifdef EARASED_TYPE_PRESENT

AIA_API void aiatensor__(potrf)(AIATensor_ *res, AIATensor_ *mat, const char *uplo);
AIA_API void aiatensor__(gesvd)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *mat, const char *jobu);
AIA_API void aiatensor__(gesvd2)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *rmat, AIATensor_ *mat, const char *jobu);
AIA_API void aiatensor__(syev)(AIATensor_ *rese, AIATensor_ *resv, AIATensor_ *mat, const char *jobz, const char *uplo);

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/linalg.h"
#include <aianon/core/erasure.h>

#define AIA_TENSOR_LINALG_H
#endif