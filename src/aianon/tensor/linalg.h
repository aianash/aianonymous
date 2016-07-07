#ifndef AIA_TENSOR_LINALG_H

#include <aianon/core/util.h>
#include <aianon/core/math/lapack.h>
#include <aianon/tensor/tensor.h>

#ifdef ERASED_TYPE_PRESENT

AIA_API void aiatensor__(potrf)(AIATensor_ *res, AIATensor_ *mat, const char *uplo);
AIA_API void aiatensor__(potrs)(AIATensor_ *res, AIATensor_ *b, AIATensor_ *a, const char *uplo);
AIA_API void aiatensor__(trtrs)(AIATensor_ *res, AIATensor_ *b, AIATensor_ *a, const char *uplo, const char *trans, const char *diag);
AIA_API void aiatensor__(gesvd)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *mat, const char *jobu);
AIA_API void aiatensor__(gesvd2)(AIATensor_ *resu, AIATensor_ *ress, AIATensor_ *resv, AIATensor_ *rmat, AIATensor_ *mat, const char *jobu);
AIA_API void aiatensor__(syev)(AIATensor_ *rese, AIATensor_ *resv, AIATensor_ *mat, const char *jobz, const char *uplo);

#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/linalg.h"
#include <aianon/core/erasure.h>

#define AIA_TENSOR_LINALG_H
#endif