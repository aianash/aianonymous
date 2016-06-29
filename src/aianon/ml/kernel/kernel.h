#ifndef AIA_ML_KERNEL_H
#define AIA_ML_KERNEL_H

#include <aianon/core/util.h>
#include <aianon/core/util/memory.h>
#include <aianon/tensor/tensor.h>
#include <aianon/tensor/dimcrossapply.h>

#ifdef ERASED_TYPE_PRESENT

AIA_API void aiakernel_rbf__(create)(AIATensor_ *X, AIATensor_ *Y, AIATensor_ *K, long sigma);

#endif

#ifndef aiakernel_rbf_
#define aiakernel_rbf_(type, name) AIA_FN_ERASE_(kernel_rbf, type, name)
#define aiakernel_rbf__(name) aiakernel_rbf_(T_, name)
#endif

#define ERASE_FLOAT
#define ERASURE_FILE "aianon/ml/kernel/kernel.h"
#include <aianon/core/erasure.h>

#endif