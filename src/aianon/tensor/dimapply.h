#ifndef AIA_TENSOR_DIM_APPLY_INC
#define AIA_TENSOR_DIM_APPLY_INC

#define AIA_TENSOR_DIM_APPLY3(T1, tensor1, T2, tensor2, T3, tensor3, dimension, code) { \
  T1 *tensor1##_data = NULL; \
  long tensor1##_stride = 0, tensor1##_size = 0; \
  T2 *tensor2##_data = NULL; \
  long tensor2##_stride = 0, tensor2##_size = 0; \
  T3 *tensor3##_data = NULL; \
  long tensor3##_stride = 0, tensor3##_size = 0; \
  long *aia_tensor_dim_apply_counter = NULL; \
  int aia_tensor_dim_apply_finished = 0; \
  int aia_tensor_dim_apply_i; \
\
  if((dimension < 0) || (dimension >= tensor1->nDimension)) \
    aia_error("invalid dimension"); \
  if(tensor1->dimension != tensor2->dimension) \
    aia_error("inconsistent tensor size"); \
  if(tensor1->dimension != tensor3->nDimension) \
    aia_error("inconsistent tensor size"); \
  for(aia_tensor_dim_apply_i = 0; aia_tensor_dim_apply_i < tensor1->nDimension; aia_tensor_dim_apply_i++) { \
    if(aia_tensor_dim_apply_i == dimension) continue; \
    if(tensor1->size[aia_tensor_dim_apply_i] != tensor2->size[aia_tensor_dim_apply_i]) aia_error("inconsistent tensor size"); \
    if(tensor1->size[aia_tensor_dim_apply_i] != tensor3->size[aia_tensor_dim_apply_i]) aia_error("inconsistent tensor size"); \
  } \
\
  aia_tensor_dim_apply_counter = (long*) aia_alloc(sizeof(long) * tensor1->nDimension); \
  for(aia_tensor_dim_apply_i = 0; aia_tensor_dim_apply_i < tensor1->nDimension; aia_tensor_dim_apply_i++) aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] = 0; \
\
  tensor1##_data = tensor1->storage->data + tensor1->storageOffset; \
  tensor1##_stride = tensor1->stride[dimension]; \
  tensor1##_size = tensor1->size[dimension]; \
\
  tensor2##_data = tensor2->storage->data + tensor2->storageOffset; \
  tensor2##_stride = tensor2->stride[dimension]; \
  tensor2##_size = tensor2->size[dimension]; \
\
  tensor3##_data = tensor3->storage->data + tensor3->storageOffset; \
  tensor3##_stride = tensor3->stride[dimension]; \
  tensor3##_size = tensor3->size[dimension]; \
\
  while(!aia_tensor_dim_apply_finished) { \
    code \
\
    if(tensor1->nDimension == 1) break; \
\
    for(aia_tensor_dim_apply_i = 0; aia_tensor_dim_apply_i < tensor1->nDimension; aia_tensor_dim_apply_i++) { \
      if(aia_tensor_dim_apply_i == dimension) { \
        if(aia_tensor_dim_apply_i == tensor1->nDimension - 1) { \
          aia_tensor_dim_apply_finished = 1; \
          break; \
        } \
        continue; \
      } \
\
      aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i]++; \
      tensor1##_data += tensor1->stride[aia_tensor_dim_apply_i]; \
      tensor2##_data += tensor2->stride[aia_tensor_dim_apply_i]; \
      tensor3##_data += tensor3->stride[aia_tensor_dim_apply_i]; \
\
      if(aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] == tensor1->size[aia_tensor_dim_apply_i]) { \
        if(aia_tensor_dim_apply_i == tensor1->nDimension - 1) { \
          aia_tensor_dim_apply_finished = 1; \
          break; \
        } else { \
          tensor1##_data -= aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] * tensor1->stride[aia_tensor_dim_apply_i]; \
          tensor2##_data -= aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] * tensor2->stride[aia_tensor_dim_apply_i]; \
          tensor3##_data -= aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] * tensor3->stride[aia_tensor_dim_apply_i]; \
          aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] = 0; \
        } \
      } else { \
        break; \
      } \
    } \
  } \
  aia_free(aia_tensor_dim_apply_counter); \
}


#define AIA_TENSOR_DIM_APPLY2(T1, tensor1, T2, tensor2, dimension, code) { \
  T1 *tensor1##_data = NULL; \
  long tensor1##_stride = 0, tensor1##_size = 0; \
  T1 *tensor2##_data = NULL; \
  long tensor2##_stride = 0, tensor2##_size = 0; \
  long *aia_tensor_dim_apply_counter = NULL; \
  int aia_tensor_dim_apply_finished = 0; \
  int aia_tensor_dim_apply_i; \
\
  if((dimension < 0) || (dimension > tensor1->nDimension)) \
    aia_error("invalid tensor size"); \
  if(tensor1->nDimension != tensor2->nDimension) \
    aia_error("inconsistent tensor sizes"); \
  for(aia_tensor_dim_apply_i = 0; aia_tensor_dim_apply_i < tensor1->nDimension; aia_tensor_dim_apply_i++) { \
    if(aia_tensor_dim_apply_i == dimension) continue; \
    if(tensor1->size[aia_tensor_dim_apply_i] != tensor2->size[aia_tensor_dim_apply_i]) \
      aia_error("inconsistent tensor sizes"); \
  } \
\
  aia_tensor_dim_apply_counter = (long*) aia_alloc(sizeof(long) * tensor1->nDimension); \
  for(aia_tensor_dim_apply_i = 0; aia_tensor_dim_apply_i < tensor1->nDimension; aia_tensor_dim_apply_i++) \
    aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] = 0; \
\
  tensor1##_data = tensor1->storage->data + tensor1->storageOffset; \
  tensor1##_stride = tensor1->stride[dimension]; \
  tensor1##_size = tensor1->size[dimension]; \
\
  tensor2##_data = tensor2->storage->data + tensor2->storageOffset; \
  tensor2##_stride = tensor2->stride[dimension]; \
  tensor2##_size = tensor2->size[dimension]; \
\
  while(!aia_tensor_dim_apply_finished) { \
    code \
\
    if(tensor1->nDimension == 1) break; \
\
    for(aia_tensor_dim_apply_i = 0; aia_tensor_dim_apply_i < tensor1->nDimension; aia_tensor_dim_apply_i++) { \
      if(aia_tensor_dim_apply_i == dimension) { \
        if(aia_tensor_dim_apply_i == tensor1->nDimension - 1) { \
          aia_tensor_dim_apply_finished = 1; \
          break; \
        } \
        continue; \
      } \
\
      aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i]++; \
      tensor1##_data += tensor1->stride[aia_tensor_dim_apply_i]; \
      tensor2##_data += tensor2->stride[aia_tensor_dim_apply_i]; \
\
      if(aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] == tensor1->size[aia_tensor_dim_apply_i]) { \
        if(aia_tensor_dim_apply_i == tensor1->nDimension - 1) { \
          aia_tensor_dim_apply_finished = 1; \
          break; \
        } else { \
          tensor1##_data -= aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] * tensor1->stride[aia_tensor_dim_apply_i]; \
          tensor2##_data -= aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] * tensor2->stride[aia_tensor_dim_apply_i]; \
          aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] = 0; \
        } \
      } else { \
        break; \
      } \
    } \
  } \
  aia_free(aia_tensor_dim_apply_counter); \
}


#define AIA_TENSOR_DIM_APPLY(T, tensor, dimension, code) { \
  T *tensor##_data = NULL; \
  long tensor##_stride = 0, tensor##_size = 0; \
  long *aia_tensor_dim_apply_counter = NULL; \
  int aia_tensor_dim_apply_finished = 0; \
  int aia_tensor_dim_apply_i; \
\
  if((dimension < 0) || (dimension > tensor->nDimension)) \
    aia_error("invalid tensor size"); \
\
  aia_tensor_dim_apply_counter = (long*) aia_alloc(sizeof(long) * tensor->nDimension); \
  for(aia_tensor_dim_apply_i = 0; aia_tensor_dim_apply_i < tensor->nDimension; aia_tensor_dim_apply_i++) \
    aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] = 0; \
\
  tensor##_data = tensor->storage->data + tensor->storageOffset; \
  tensor##_stride = tensor->stride[dimension]; \
  tensor##_size = tensor->size[dimension]; \
\
  while(!aia_tensor_dim_apply_finished) { \
    code \
\
    if(tensor->nDimension == 1) break; \
\
    for(aia_tensor_dim_apply_i = 0; aia_tensor_dim_apply_i < tensor->nDimension; aia_tensor_dim_apply_i++) { \
      if(aia_tensor_dim_apply_i == dimension) { \
        if(aia_tensor_dim_apply_i == tensor->nDimension - 1) { \
          aia_tensor_dim_apply_finished = 1; \
          break; \
        } \
        continue; \
      } \
\
      aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i]++; \
      tensor##_data += tensor->stride[aia_tensor_dim_apply_i]; \
\
      if(aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] == tensor->size[aia_tensor_dim_apply_i]) { \
        if(aia_tensor_dim_apply_i == tensor->nDimension - 1) { \
          aia_tensor_dim_apply_finished = 1; \
          break; \
        } else { \
          tensor##_data -= aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] * tensor->stride[aia_tensor_dim_apply_i]; \
          aia_tensor_dim_apply_counter[aia_tensor_dim_apply_i] = 0; \
        } \
      } else { \
        break; \
      } \
    } \
  } \
  aia_free(aia_tensor_dim_apply_counter); \
}

#endif