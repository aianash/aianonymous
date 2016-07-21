#ifndef AIA_TENSOR_LINALG_H

#include <aiautil/util.h>
#include <aianon/core/math/lapack.h>
#include <aiatensor/tensor.h>

#ifdef ERASED_TYPE_PRESENT

AIA_API void aiatensor__(potrf)(AIATensor_ *res, AIATensor_ *mat, const char *uplo);
AIA_API void aiatensor__(potrs)(AIATensor_ *res, AIATensor_ *b, AIATensor_ *a, const char *uplo);
AIA_API void aiatensor__(trtrs)(AIATensor_ *resa, AIATensor_ *resb, AIATensor_ *b, AIATensor_ *amat, const char *uplo, const char *trans, const char *diag);
AIA_API void aiatensor__(gesvd)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *mat, const char *jobu);
AIA_API void aiatensor__(gesvd2)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *rmat, AIATensor_ *mat, const char *jobu);
AIA_API void aiatensor__(syev)(AIATensor_ *rese, AIATensor_ *resv, AIATensor_ *mat, const char *jobz, const char *uplo);

#endif

#define ERASE_FLOAT
#define ERASE_DOUBLE
#define ERASURE_FILE "aiatensor/linalg.h"
#include <aiautil/erasure.h>

#define AIA_TENSOR_LINALG_H
#endif