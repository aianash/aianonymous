#ifndef AIA_TENSOR_CROSS_DIM_APPLY_H
#define AIA_TENSOR_CROSS_DIM_APPLY_H

#define AIA_TENSOR_CROSS_DIM_APPLY3(T1, tensor1, T2, tensor2, T3, tensor3, dimension, code) { \
  int th_cross_dim_apply_finished = 0; \
  int tensor1##_finished = 0; \
  int tensor2##_finished = 0; \
\
  long *tensor1##_counter = (long*) aia_alloc(sizeof(long) * tensor1->nDimension); \
  long *tensor2##_counter = (long*) aia_alloc(sizeof(long) * tensor2->nDimension); \
\
  T1 *tensor1##_data = tensor1->storage->data + tensor1->storageOffset; \
  T2 *tensor2##_data = tensor2->storage->data + tensor2->storageOffset; \
  T3 *tensor3##_data = tensor3->storage->data + tensor3->storageOffset; \
\
  long tensor1##_stride = tensor1->stride[dimension], tensor1##_size = tensor1->size[dimension]; \
  long tensor2##_stride = tensor2->stride[dimension], tensor2##_size = tensor2->size[dimension]; \
  long tensor3##_stride = tensor3->stride[0], tensor3##_size = tensor3->size[0]; \
  long tensor1##_dim_i = 0, tensor2##_dim_i = 0; \
\
  for(tensor1##_dim_i = 0; tensor1##_dim_i < tensor1->nDimension; tensor1##_dim_i++) \
    tensor1##_counter[tensor1##_dim_i] = 0; \
\
  for(tensor2##_dim_i = 0; tensor2##_dim_i < tensor2->nDimension; tensor2##_dim_i++) \
    tensor2##_counter[tensor2##_dim_i] = 0; \
\
  while(!th_cross_dim_apply_finished) { \
    code \
\
    tensor3##_data += tensor3->stride[0]; \
\
    if(tensor1->nDimension == 1 && tensor2->nDimension == 1) { \
      break; \
    } \
\
    for(tensor2##_dim_i = 0; tensor2##_dim_i < tensor2->nDimension; tensor2##_dim_i++) { \
      if(tensor2##_dim_i == dimension) { \
        if(tensor2##_dim_i == tensor2->nDimension - 1) tensor2##_finished = 1; \
        else continue; \
      } \
\
      if(tensor2##_finished) { \
        for(tensor1##_dim_i = 0; tensor1##_dim_i < tensor1->nDimension; tensor1##_dim_i++) { \
          if(tensor1##_dim_i == dimension) { \
            if(tensor1##_dim_i == tensor1->nDimension - 1) { \
              tensor1##_finished = 1; \
              break; \
            } else { \
              continue; \
            } \
          } \
\
          tensor1##_counter[tensor1##_dim_i]++; \
          tensor1##_data += tensor1->stride[tensor1##_dim_i]; \
\
          if(tensor1##_counter[tensor1##_dim_i] == tensor1->size[tensor1##_dim_i]) { \
            if(tensor1##_dim_i == tensor1->nDimension - 1) { \
              tensor1##_finished = 1; \
              break; \
            } else { \
              tensor1##_data -= tensor1##_counter[tensor1##_dim_i] * tensor1->stride[tensor1##_dim_i]; \
              tensor1##_counter[tensor1##_dim_i] = 0; \
            } \
          } else { \
            break; \
          } \
        } \
        if(!tensor1##_finished) { \
          tensor2##_data = tensor2->storage->data + tensor2->storageOffset; \
          tensor2##_finished = 0; \
        } \
      } else { \
        tensor2##_counter[tensor2##_dim_i]++; \
        tensor2##_data += tensor2->stride[tensor2##_dim_i]; \
\
        if(tensor2##_counter[tensor2##_dim_i] == tensor2->size[tensor2##_dim_i]) { \
          if(tensor2##_dim_i == tensor2->nDimension - 1) { \
              tensor2##_finished = 1; \
          } else { \
            tensor2##_data -= tensor2##_counter[tensor2##_dim_i] * tensor2->stride[tensor2##_dim_i]; \
            tensor2##_counter[tensor2##_dim_i] = 0; \
          } \
        } else { \
          break; \
        } \
      } \
\
      if(tensor1##_finished && tensor2##_finished) { \
        th_cross_dim_apply_finished = 1; \
        break; \
      } \
    } \
  } \
  aia_free(tensor1##_counter); \
  aia_free(tensor2##_counter); \
}

#endif