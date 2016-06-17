#ifndef AIA_TENSOR_H
#ifdef ERASED_TYPE_AVAILABLE


T aiatensor_(T_, get)(T p);

#endif

#ifndef aiatensor_
#define aiatensor_(type, name) aiaerase_(tensor, type, name)
#endif

#define ERASE_ALL
#define ERASURE_FILE "aianon/tensor.h"
#include <aianon/util/erasure.h>

#define AIA_TENSOR_H
#endif