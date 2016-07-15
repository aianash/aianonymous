#ifndef AIA_TENSOR_APPLY_INC
#define AIA_TENSOR_APPLY_INC

#define AIA_TENSOR_APPLY(T, tensor, code) { \
	T *tensor##_data = NULL; \
	long *tensor##_counter = NULL; \
	long tensor##_stride = 0; \
	long tensor##_size = 0; \
	long tensor##_dim = 0; \
	long tensor##_index = 0; \
	int tensor_apply_finished = 0; \
\
	if (tensor->nDimension == 0) \
		tensor_apply_finished = 1; \
	else { \
		tensor##_data = tensor->storage->data + tensor->storageOffset; \
\
		/* get first stride after ignoring dimensions with size value 1 */ \
		for (tensor##_dim = tensor->nDimension - 1; tensor##_dim >= 0; tensor##_dim--) { \
			if (tensor->size[tensor##_dim] != 1) \
				break; \
		} \
		tensor##_stride = (tensor##_dim == -1 ? 0 : tensor->stride[tensor##_dim]); \
\
		/* get largest contiguous section over multiple dimensions */ \
		tensor##_size = 1; \
		for (tensor##_dim = tensor->nDimension - 1; tensor##_dim >= 0; tensor##_dim--) { \
			if (tensor->size[tensor##_dim] != 1) { \
				if (tensor->stride[tensor##_dim] == tensor##_size) \
					tensor##_size *= tensor->size[tensor##_dim]; \
				else \
					break; \
			} \
		} \
\
		/* initialize counter for not contiguous dimensions */ \
		tensor##_counter = (long*) aia_alloc(sizeof(long) * (tensor##_dim + 1)); \
		for (tensor##_index = 0; tensor##_index <= tensor##_dim; tensor##_index++) \
			tensor##_counter[tensor##_index] = 0; \
	} \
\
	/* apply code to the elements */ \
	while(!tensor_apply_finished) { \
		/* apply code to the contiguous block */ \
		for (tensor##_index = 0; tensor##_index < tensor##_size; tensor##_index++, \
			tensor##_data += tensor##_stride) { \
				code \
		} \
\
		/* if whole tensor is contiguous then this condition is true */ \
		if (tensor##_dim == -1) \
			 break; \
\
		/* go to beginning of block */ \
		tensor##_data -= tensor##_index * tensor##_stride; \
\
		/* iterate through all contiguous blocks one by one */ \
		for (tensor##_index = tensor##_dim; tensor##_index >= 0; tensor##_index--) { \
			tensor##_counter[tensor##_index]++; \
			tensor##_data += tensor->stride[tensor##_index]; \
\
			/* check if dimension is finished */ \
			if (tensor##_counter[tensor##_index]  == tensor->size[tensor##_index]) { \
				if (tensor##_index == 0) { \
					/* done with all dimensions */ \
					tensor_apply_finished = 1; \
					break; \
				} else { \
					/* done wth current dimension */ \
					tensor##_data -= tensor##_counter[tensor##_index] * tensor->stride[tensor##_index]; \
					tensor##_counter[tensor##_index] = 0; \
				} \
			} \
			else \
				break; \
		} \
	} \
	aia_free(tensor##_counter); \
}


#define AIA_TENSOR_APPLY2(T1, tensor1, T2, tensor2, code) { \
	T1 *tensor1##_data = NULL; \
	long *tensor1##_counter = NULL; \
	long tensor1##_stride = 0; \
	long tensor1##_size = 0; \
	long tensor1##_dim = 0; \
	long tensor1##_index; \
	long tensor1##_nElems; \
\
	T2 *tensor2##_data = NULL; \
	long *tensor2##_counter = NULL; \
	long tensor2##_stride = 0; \
	long tensor2##_size = 0; \
	long tensor2##_dim = 0; \
	long tensor2##_index; \
	long tensor2##_nElems; \
 \
	int tensor_apply_finished = 0; \
\
	tensor1##_nElems = (tensor1->nDimension ? 1 : 0); \
	for (tensor1##_index = 0; tensor1##_index < tensor1->nDimension; tensor1##_index++) \
		tensor1##_nElems *= tensor1->size[tensor1##_index]; \
\
	tensor2##_nElems = (tensor2->nDimension ? 1 : 0); \
	for (tensor2##_index = 0; tensor2##_index < tensor2->nDimension; tensor2##_index++) \
		tensor2##_nElems *= tensor2->size[tensor2##_index]; \
\
	if (tensor1##_nElems != tensor2##_nElems) \
		aia_error("Inconsistent tensor sizes"); \
\
	if (tensor1->nDimension == 0) \
		tensor_apply_finished = 1; \
	else { \
		tensor1##_data = tensor1->storage->data + tensor1->storageOffset; \
		for (tensor1##_dim = tensor1->nDimension-1; tensor1##_dim >= 0; tensor1##_dim--) { \
			if (tensor1->size[tensor1##_dim] != 1) \
				break; \
		} \
		tensor1##_stride = (tensor1##_dim == -1 ? 0 : tensor1->stride[tensor1##_dim]); \
		tensor1##_size = 1; \
		for (tensor1##_dim = tensor1->nDimension - 1; tensor1##_dim >= 0; tensor1##_dim--) { \
			if (tensor1->size[tensor1##_dim] != 1) { \
				if (tensor1->stride[tensor1##_dim] == tensor1##_size) \
					tensor1##_size *= tensor1->size[tensor1##_dim]; \
				else \
					break; \
			} \
		} \
		tensor1##_counter = (long*) aia_alloc(sizeof(long) * (tensor1##_dim + 1)); \
		for (tensor1##_index = 0; tensor1##_index <= tensor1##_dim; tensor1##_index++) \
			tensor1##_counter[tensor1##_index] = 0; \
\
		tensor2##_data = tensor2->storage->data + tensor2->storageOffset; \
		for (tensor2##_dim = tensor2->nDimension-1; tensor2##_dim >= 0; tensor2##_dim--) { \
			if (tensor2->size[tensor2##_dim] != 1) \
				break; \
		} \
		tensor2##_stride = (tensor2##_dim == -1 ? 0 : tensor2->stride[tensor2##_dim]); \
		tensor2##_size = 1; \
		for (tensor2##_dim = tensor2->nDimension - 1; tensor2##_dim >= 0; tensor2##_dim--) { \
			if (tensor2->size[tensor2##_dim] != 1) { \
				if (tensor2->stride[tensor2##_dim] == tensor2##_size) \
					tensor2##_size *= tensor2->size[tensor2##_dim]; \
				else \
					break; \
			} \
		} \
		tensor2##_counter = (long*) aia_alloc(sizeof(long) * (tensor2##_dim + 1)); \
		for (tensor2##_index = 0; tensor2##_index <= tensor2##_dim; tensor2##_index++) \
			tensor2##_counter[tensor2##_index] = 0; \
	} \
\
	tensor1##_index = 0; \
	tensor2##_index = 0; \
	while (!tensor_apply_finished) { \
		for (; tensor1##_index < tensor1##_size && tensor2##_index < tensor2##_size; \
			tensor1##_index++, tensor2##_index++, \
			tensor1##_data += tensor1##_stride, tensor2##_data += tensor2##_stride) { \
			code \
		} \
\
		/* contiguous block can have differnet sizes */ \
		if (tensor1##_index == tensor1##_size) { \
			if(tensor1##_dim == -1) \
				 break; \
\
			tensor1##_data -= tensor1##_size * tensor1##_stride; \
			for (tensor1##_index = tensor1##_dim; tensor1##_index >= 0; tensor1##_index--) { \
				tensor1##_counter[tensor1##_index]++; \
				tensor1##_data += tensor1->stride[tensor1##_index]; \
\
				if (tensor1##_counter[tensor1##_index]  == tensor1->size[tensor1##_index]) { \
					if (tensor1##_index == 0) { \
						tensor_apply_finished = 1; \
						break; \
					} else { \
						tensor1##_data -= tensor1##_counter[tensor1##_index] * tensor1->stride[tensor1##_index]; \
						tensor1##_counter[tensor1##_index] = 0; \
					} \
				} \
				else \
					break; \
			} \
			tensor1##_index = 0; \
		} \
\
		if (tensor2##_index == tensor2##_size) { \
			if(tensor2##_dim == -1) \
				 break; \
\
			tensor2##_data -= tensor2##_size * tensor2##_stride; \
			for (tensor2##_index = tensor2##_dim; tensor2##_index >= 0; tensor2##_index--) { \
				tensor2##_counter[tensor2##_index]++; \
				tensor2##_data += tensor2->stride[tensor2##_index]; \
\
				if (tensor2##_counter[tensor2##_index]  == tensor2->size[tensor2##_index]) { \
					if (tensor2##_index == 0) { \
						tensor_apply_finished = 1; \
						break; \
					} else { \
						tensor2##_data -= tensor2##_counter[tensor2##_index] * tensor2->stride[tensor2##_index]; \
						tensor2##_counter[tensor2##_index] = 0; \
					} \
				} \
				else \
					break; \
			} \
			tensor2##_index = 0; \
		} \
	} \
	aia_free(tensor1##_counter); \
	aia_free(tensor2##_counter); \
}


#define AIA_TENSOR_APPLY3(T1, tensor1, T2, tensor2, T3, tensor3, code) { \
	T1 *tensor1##_data = NULL; \
	long *tensor1##_counter = NULL; \
	long tensor1##_stride = 0; \
	long tensor1##_size = 0; \
	long tensor1##_dim = 0; \
	long tensor1##_index; \
	long tensor1##_nElems; \
\
	T2 *tensor2##_data = NULL; \
	long *tensor2##_counter = NULL; \
	long tensor2##_stride = 0; \
	long tensor2##_size = 0; \
	long tensor2##_dim = 0; \
	long tensor2##_index; \
	long tensor2##_nElems; \
\
	T3 *tensor3##_data = NULL; \
	long *tensor3##_counter = NULL; \
	long tensor3##_stride = 0; \
	long tensor3##_size = 0; \
	long tensor3##_dim = 0; \
	long tensor3##_index; \
	long tensor3##_nElems; \
\
	int tensor_apply_finished = 0; \
\
	tensor1##_nElems = (tensor1->nDimension ? 1 : 0); \
	for (tensor1##_index = 0; tensor1##_index < tensor1->nDimension; tensor1##_index++) \
		tensor1##_nElems *= tensor1->size[tensor1##_index]; \
\
	tensor2##_nElems = (tensor2->nDimension ? 1 : 0); \
	for (tensor2##_index = 0; tensor2##_index < tensor2->nDimension; tensor2##_index++) \
		tensor2##_nElems *= tensor2->size[tensor2##_index]; \
\
	tensor3##_nElems = (tensor3->nDimension ? 1 : 0); \
	for (tensor3##_index = 0; tensor3##_index < tensor3->nDimension; tensor3##_index++) \
		tensor3##_nElems *= tensor3->size[tensor3##_index]; \
\
	if (tensor1##_nElems != tensor2##_nElems || tensor1##_nElems != tensor3##_nElems) \
		aia_error("Inconsistent tensor size"); \
\
	if (tensor1->nDimension == 0) \
		tensor_apply_finished = 1; \
	else { \
		tensor1##_data = tensor1->storage->data + tensor1->storageOffset; \
		for (tensor1##_dim = tensor1->nDimension - 1; tensor1##_dim >= 0; tensor1##_dim--) { \
			if (tensor1->size[tensor1##_dim] != 1) \
				break; \
		} \
		tensor1##_stride = (tensor1##_dim == -1 ? 0 : tensor1->stride[tensor1##_dim]); \
		tensor1##_size = 1; \
		for (tensor1##_dim = tensor1->nDimension - 1; tensor1##_dim >= 0; tensor1##_dim--) { \
			if (tensor1->size[tensor1##_dim] != 1) { \
				if (tensor1->stride[tensor1##_dim] == tensor1##_size) \
					tensor1##_size *= tensor1->size[tensor1##_dim]; \
				else \
					break; \
			} \
		} \
		tensor1##_counter = (long*) aia_alloc(sizeof(long)*(tensor1##_dim + 1)); \
		for (tensor1##_index = 0; tensor1##_index <= tensor1##_dim; tensor1##_index++) \
			tensor1##_counter[tensor1##_index] = 0; \
\
		tensor2##_data = tensor2->storage->data + tensor2->storageOffset; \
		for (tensor2##_dim = tensor2->nDimension - 1; tensor2##_dim >= 0; tensor2##_dim--) { \
			if (tensor2->size[tensor2##_dim] != 1) \
				break; \
		} \
		tensor2##_stride = (tensor2##_dim == -1 ? 0 : tensor2->stride[tensor2##_dim]); \
		tensor2##_size = 1; \
		for (tensor2##_dim = tensor2->nDimension - 1; tensor2##_dim >= 0; tensor2##_dim--) { \
			if (tensor2->size[tensor2##_dim] != 1) { \
				if (tensor2->stride[tensor2##_dim] == tensor2##_size) \
					tensor2##_size *= tensor2->size[tensor2##_dim]; \
				else \
					break; \
			} \
		} \
		tensor2##_counter = (long*) aia_alloc(sizeof(long)*(tensor2##_dim + 1)); \
		for (tensor2##_index = 0; tensor2##_index <= tensor2##_dim; tensor2##_index++) \
			tensor2##_counter[tensor2##_index] = 0; \
\
		tensor3##_data = tensor3->storage->data + tensor3->storageOffset; \
		for (tensor3##_dim = tensor3->nDimension - 1; tensor3##_dim >= 0; tensor3##_dim--) { \
			if (tensor3->size[tensor3##_dim] != 1) \
				break; \
		} \
		tensor3##_stride = (tensor3##_dim == -1 ? 0 : tensor3->stride[tensor3##_dim]); \
		tensor3##_size = 1; \
		for (tensor3##_dim = tensor3->nDimension - 1; tensor3##_dim >= 0; tensor3##_dim--) { \
			if (tensor3->size[tensor3##_dim] != 1) { \
				if (tensor3->stride[tensor3##_dim] == tensor3##_size) \
					tensor3##_size *= tensor3->size[tensor3##_dim]; \
				else \
					break; \
			} \
		} \
		tensor3##_counter = (long*) aia_alloc(sizeof(long)*(tensor3##_dim + 1)); \
		for (tensor3##_index = 0; tensor3##_index <= tensor3##_dim; tensor3##_index++) \
			tensor3##_counter[tensor3##_index] = 0; \
	} \
\
	tensor1##_index = 0; \
	tensor2##_index = 0; \
	tensor3##_index = 0; \
	while (!tensor_apply_finished) { \
		for (; tensor1##_index < tensor1##_size && tensor2##_index < tensor2##_size && \
			tensor3##_index < tensor3##_size; tensor1##_index++, tensor2##_index++, tensor3##_index++, \
			tensor1##_data += tensor1##_stride, tensor2##_data += tensor2##_stride, \
			tensor3##_data += tensor3##_stride) { \
				code \
		} \
\
		if (tensor1##_index == tensor1##_size) { \
			if (tensor1##_dim == -1) \
				 break; \
\
			tensor1##_data -= tensor1##_size * tensor1##_stride; \
			for (tensor1##_index = tensor1##_dim; tensor1##_index >= 0; tensor1##_index--) { \
				tensor1##_counter[tensor1##_index]++; \
				tensor1##_data += tensor1->stride[tensor1##_index]; \
\
				if (tensor1##_counter[tensor1##_index]  == tensor1->size[tensor1##_index]) { \
					if (tensor1##_index == 0) { \
						tensor_apply_finished = 1; \
						break; \
					} else { \
						tensor1##_data -= tensor1##_counter[tensor1##_index] * tensor1->stride[tensor1##_index]; \
						tensor1##_counter[tensor1##_index] = 0; \
					} \
				} \
				else \
					break; \
			} \
			tensor1##_index = 0; \
		} \
\
		if (tensor2##_index == tensor2##_size) { \
			if (tensor2##_dim == -1) \
				 break; \
\
			tensor2##_data -= tensor2##_size * tensor2##_stride; \
			for (tensor2##_index = tensor2##_dim; tensor2##_index >= 0; tensor2##_index--) { \
				tensor2##_counter[tensor2##_index]++; \
				tensor2##_data += tensor2->stride[tensor2##_index]; \
\
				if (tensor2##_counter[tensor2##_index]  == tensor2->size[tensor2##_index]) { \
					if (tensor2##_index == 0) { \
						tensor_apply_finished = 1; \
						break; \
					} else { \
						tensor2##_data -= tensor2##_counter[tensor2##_index] * tensor2->stride[tensor2##_index]; \
						tensor2##_counter[tensor2##_index] = 0; \
					} \
				} \
				else \
					break; \
			} \
			tensor2##_index = 0; \
		} \
\
		if (tensor3##_index == tensor3##_size) { \
			if (tensor3##_dim == -1) \
				 break; \
\
			tensor3##_data -= tensor3##_size * tensor3##_stride; \
			for (tensor3##_index = tensor3##_dim; tensor3##_index >= 0; tensor3##_index--) { \
				tensor3##_counter[tensor3##_index]++; \
				tensor3##_data += tensor3->stride[tensor3##_index]; \
\
				if (tensor3##_counter[tensor3##_index]  == tensor3->size[tensor3##_index]) { \
					if(tensor3##_index == 0) { \
						tensor_apply_finished = 1; \
						break; \
					} else { \
						tensor3##_data -= tensor3##_counter[tensor3##_index] * tensor3->stride[tensor3##_index]; \
						tensor3##_counter[tensor3##_index] = 0; \
					} \
				} \
				else \
					break; \
			} \
			tensor3##_index = 0; \
		} \
	} \
	aia_free(tensor1##_counter); \
	aia_free(tensor2##_counter); \
	aia_free(tensor3##_counter); \
}

#endif