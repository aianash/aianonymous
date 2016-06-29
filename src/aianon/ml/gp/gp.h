#ifndef AIA_ML_GP_H
#define AIA_ML_GP_H

#include <aianon/core/util.h>
#include <aianon/tensor/tensor.h>

#ifdef ERASED_TYPE_PRESENT

AIA_API void aiagp__(calcbeta)(AIATensor_ *beta, AIATensor_ *K, AIATensor_ *y, const char* uplo);

/** Certain input */
AIA_API void aiagp__(vpredc)(AIATensor_ *fmean, AIATensor_ *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *Kx, AIATensor_ *Kxx, AIATensor_ *beta);
AIA_API void aiagp__(soredc)(T *fmean, T *fcov, AIATensor_ *K, const char *uplo, AIATensor_ *Kx, T Kxx, AIATensor_ *beta);

/** Uncertain input */
AIA_API void aiagp__(preduc)();
AIA_API void aiagp__(graduc)();

#endif

#ifndef aiagp_
#define aiagp_(type, name) AIA_FN_ERASE_(gp, type, name)
#define aiagp__(name) aiagp_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/ml/gp/gp.h"
#include <aianon/core/erasure.h>

#endif