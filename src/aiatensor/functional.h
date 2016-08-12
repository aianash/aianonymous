#ifndef AIA_TENSOR_FUNCTIONAL
#define AIA_TENSOR_FUNCTIONAL


/**************** FOREACH *****************/

/**
 * Description
 * -----------
 * Foreach matrix iteration for side effect
 *
 * Usage
 * -----
 * mforeach(ele, matrix) {
 *  // code
 * }
 * endmforeach(optional code)
 *
 * Input
 * -----
 * ele    : Pointer to each entry
 * matrix : Matrix to iterate
 */
#define mforeach(ele, matrix) { \
  long ele##_index = 0; \
  long ele##_stride = 0; \
  long matrix##_contsz = 0; \
  long matrix##_rowcntr = 0; \
  int fbreak = 0; \
\
  int matrix##_rmajr = (matrix->stride[1] == 1 && matrix->stride[0] == matrix->size[1]) ? 1 : 0; \
  ele##_stride = matrix->stride[1]; \
\
  if(matrix##_rmajr) { \
    matrix##_contsz = matrix->size[0] * matrix->size[1]; \
  } else { \
    matrix##_contsz = matrix->size[1]; \
  } \
\
  ele = matrix->storage->data + matrix->storageOffset - matrix->stride[0]; \
  while(!fbreak) { \
    if(matrix##_rmajr) fbreak = 1; \
    matrix##_rowcntr += 1; \
    if(matrix##_rowcntr == matrix->size[0]) break; \
    ele -= ele##_index * ele##_stride; \
    ele += matrix->stride[0]; \
\
    for(ele##_index = 0; ele##_index < matrix##_contsz; \
      ele##_index++, ele += ele##_stride) {

#define endmforeach(code) \
    } \
    code \
  } \
}

/**
 * Description
 * -----------
 * Foreach vector iteration for side effect
 *
 * Usage
 * -----
 * vforeach(ele, matrix) {
 *  // code
 * }
 * endvforeach(optional code)
 *
 * Input
 * -----
 * ele    : Pointer to each entry
 * vector : Vector to iterate
 */
#define vforeach(ele, vector) { \
  long ele##_index = 0; \
  long ele##_stride = vector->stride[0]; \
  long vector##_size = vector->size[0]; \
\
  ele = vector->storage->data + vector->storageOffset; \
  for(; ele##_index < vector##_size; ele##_index++, ele += ele##_stride) {

#define endvforeach(code) \
  } \
  code \
}


/**************** ZIP *****************/

/**
 * Description
 * -----------
 * Zip two matrix and iterate with side effect
 *
 * Usage
 * -----
 * mzip2(ele1, matrix1, ele2, matrix2) {
 *  // code
 * }
 * endmzip2
 *
 * Input
 * -----
 * ele1    : Pointer to each entry of first matrix
 * matrix1 : Matrix to iterate
 * ele2    : Pointer to each entry of second matrix
 * matrix2 : Matrix to iterate
 */
#define mzip2(ele1, matrix1, ele2, matrix2) { \
  long ele1##_index = 0; \
  long ele1##_stride = 0; \
  long matrix1##_contsz = 0; \
  long matrix1##_rowcntr = 0; \
\
  long ele2##_index = 0; \
  long ele2##_stride = 0; \
  long matrix2##_contsz = 0; \
  long matrix2##_rowcntr = 0; \
\
  int fbreak = 0; \
\
  int matrix1##_rmajr = (matrix1->stride[1] == 1 && matrix1->stride[0] == matrix1->size[1]) ? 1 : 0; \
  ele1##_stride = matrix1->stride[1]; \
\
  int matrix2##_rmajr = (matrix2->stride[1] == 1 && matrix2->stride[0] == matrix2->size[1]) ? 1 : 0; \
  ele2##_stride = matrix2->stride[1]; \
\
  if(matrix1##_rmajr) { \
    matrix1##_contsz = matrix1->size[0] * matrix1->size[1]; \
  } else { \
    matrix1##_contsz = matrix1->size[1]; \
  } \
\
  if(matrix2##_rmajr) { \
    matrix2##_contsz = matrix2->size[0] * matrix2->size[1]; \
  } else { \
    matrix2##_contsz = matrix2->size[1]; \
  } \
\
  ele1 = matrix1->storage->data + matrix1->storageOffset; \
  ele2 = matrix2->storage->data + matrix2->storageOffset; \
\
  while(!fbreak) { \
    if(ele1##_index == matrix1##_contsz) { \
      matrix1##_rowcntr += 1; \
      if(matrix1##_rowcntr == matrix1->size[0]) break; \
      ele1 -= ele1##_index * ele1##_stride; \
      ele1 += matrix1->stride[0]; \
      ele1##_index = 0; \
    } \
\
    if(ele2##_index == matrix2##_contsz) { \
      matrix2##_rowcntr += 1; \
      if(matrix2##_rowcntr == matrix2->size[0]) break; \
      ele2 -= ele2##_index * ele2##_stride; \
      ele2 += matrix2->stride[0]; \
      ele2##_index = 0; \
    } \
\
    if(matrix1##_rmajr && matrix2##_rmajr) fbreak = 1; \
\
    for(; ele1##_index < matrix1##_contsz && ele2##_index < matrix2##_contsz; \
      ele1##_index++, ele2##_index++, \
      ele1 += ele1##_stride, ele2 += ele2##_stride) {

#define endmzip2 }}}


/**
 * Description
 * -----------
 * Zip three matrix and iterate with side effect
 *
 * Usage
 * -----
 * mzip3(ele1, matrix1, ele2, matrix2) {
 *  // code
 * }
 * endmzip3
 *
 * Input
 * -----
 * ele1    : Pointer to each entry of first matrix
 * matrix1 : Matrix to iterate
 * ele2    : Pointer to each entry of second matrix
 * matrix2 : Matrix to iterate
 * ele3    : Pointer to each entry of third matrix
 * matrix3 : Matrix to iterate
 */
#define mzip3(ele1, matrix1, ele2, matrix2, ele3, matrix3) { \
  long ele1##_index = 0; \
  long ele1##_stride = 0; \
  long matrix1##_contsz = 0; \
  long matrix1##_rowcntr = 0; \
\
  long ele2##_index = 0; \
  long ele2##_stride = 0; \
  long matrix2##_contsz = 0; \
  long matrix2##_rowcntr = 0; \
\
  long ele3##_index = 0; \
  long ele3##_stride = 0; \
  long matrix3##_contsz = 0; \
  long matrix3##_rowcntr = 0; \
\
  int fbreak = 0; \
\
  int matrix1##_rmajr = (matrix1->stride[1] == 1 && matrix1->stride[0] == matrix1->size[1]) ? 1 : 0; \
  ele1##_stride = matrix1->stride[1]; \
\
  int matrix2##_rmajr = (matrix2->stride[1] == 1 && matrix2->stride[0] == matrix2->size[1]) ? 1 : 0; \
  ele2##_stride = matrix2->stride[1]; \
\
  int matrix3##_rmajr = (matrix3->stride[1] == 1 && matrix3->stride[0] == matrix3->size[1]) ? 1 : 0; \
  ele3##_stride = matrix3->stride[1]; \
\
  if(matrix1##_rmajr) { \
    matrix1##_contsz = matrix1->size[0] * matrix1->size[1]; \
  } else { \
    matrix1##_contsz = matrix1->size[1]; \
  } \
\
  if(matrix2##_rmajr) { \
    matrix2##_contsz = matrix2->size[0] * matrix2->size[1]; \
  } else { \
    matrix2##_contsz = matrix2->size[1]; \
  } \
\
  if(matrix3##_rmajr) { \
    matrix3##_contsz = matrix3->size[0] * matrix3->size[1]; \
  } else { \
    matrix3##_contsz = matrix3->size[1]; \
  } \
\
  ele1 = matrix1->storage->data + matrix1->storageOffset; \
  ele2 = matrix2->storage->data + matrix2->storageOffset; \
  ele3 = matrix3->storage->data + matrix3->storageOffset; \
\
  while(!fbreak) { \
    if(ele1##_index == matrix1##_contsz) { \
      matrix1##_rowcntr += 1; \
      if(matrix1##_rowcntr == matrix1->size[0]) break; \
      ele1 -= ele1##_index * ele1##_stride; \
      ele1 += matrix1->stride[0]; \
      ele1##_index = 0; \
    } \
\
    if(ele2##_index == matrix2##_contsz) { \
      matrix2##_rowcntr += 1; \
      if(matrix2##_rowcntr == matrix2->size[0]) break; \
      ele2 -= ele2##_index * ele2##_stride; \
      ele2 += matrix2->stride[0]; \
      ele2##_index = 0; \
    } \
\
    if(ele3##_index == matrix3##_contsz) { \
      matrix3##_rowcntr += 1; \
      if(matrix3##_rowcntr == matrix3->size[0]) break; \
      ele3 -= ele3##_index * ele3##_stride; \
      ele3 += matrix3->stride[0]; \
      ele3##_index = 0; \
    } \
\
    if(matrix1##_rmajr && matrix2##_rmajr && matrix3##_rmajr) fbreak = 1; \
\
    for(; ele1##_index < matrix1##_contsz && ele2##_index < matrix2##_contsz && ele3##_index < matrix3##_contsz; \
      ele1##_index++, ele2##_index++, ele3##_index++, \
      ele1 += ele1##_stride, ele2 += ele2##_stride, ele3 += ele3##_stride) {

#define endmzip3 }}}

/**
 * Description
 * -----------
 * Zip two vector and iterate with side effect
 *
 * Usage
 * -----
 * vzip2(ele1, vector1, ele2, vector2) {
 *  // code
 * }
 * endvzip2
 *
 * Input
 * -----
 * ele1    : Pointer to each entry of first vector
 * vector1 : Vector to iterate
 * ele2    : Pointer to each entry of second vector
 * vector2 : Vector to iterate
 */
#define vzip2(ele1, vector1, ele2, vector2) { \
  long ele1##_index = 0; \
  long ele1##_stride = vector1->stride[0]; \
  long vector1##_size = vector1->size[0]; \
\
  long ele2##_index = 0; \
  long ele2##_stride = vector2->stride[0]; \
  long vector2##_size = vector2->size[0]; \
\
  ele1 = vector1->storage->data + vector1->storageOffset; \
  ele2 = vector2->storage->data + vector2->storageOffset; \
  for(; ele1##_index < vector1##_size && ele2##_index < vector2##_size; \
      ele1##_index++, ele2##_index++, \
      ele1 += ele1##_stride, ele2 += ele2##_stride) {

#define endvzip2 }}

/**
 * Description
 * -----------
 * Zip three vector and iterate with side effect
 *
 * Usage
 * -----
 * vzip3(ele1, vector1, ele2, vector2, ele3, vector3) {
 *  // code
 * }
 * endvzip3
 *
 * Input
 * -----
 * ele1    : Pointer to each entry of first vector
 * vector1 : Vector to iterate
 * ele2    : Pointer to each entry of second vector
 * vector2 : Vector to iterate
 * ele2    : Pointer to each entry of third vector
 * vector2 : Vector to iterate
 */
#define vzip3(ele1, vector1, ele2, vector2, ele3, vector3) { \
  long ele1##_index = 0; \
  long ele1##_stride = vector1->stride[0]; \
  long vector1##_size = vector1->size[0]; \
\
  long ele2##_index = 0; \
  long ele2##_stride = vector2->stride[0]; \
  long vector2##_size = vector2->size[0]; \
\
  long ele3##_index = 0; \
  long ele3##_stride = vector3->stride[0]; \
  long vector3##_size = vector3->size[0]; \
\
  ele1 = vector1->storage->data + vector1->storageOffset; \
  ele2 = vector2->storage->data + vector2->storageOffset; \
  ele3 = vector3->storage->data + vector3->storageOffset; \
  for(; ele1##_index < vector1##_size && ele2##_index < vector2##_size; \
      ele1##_index++, ele2##_index++, ele3##_index++, \
      ele1 += ele1##_stride, ele2 += ele2##_stride, ele3 += ele3##_stride) {

#define endvzip3 }}

/**
 * Description
 * -----------
 * Zip matrix and vector row wise,
 * matrix->size[1] == vector->size[0]
 *
 * Usage
 * -----
 * mvrzip(ele1, matrix, ele2, vector) {
 *  // code
 * }
 * endmvrzip(optional code)
 *
 * Input
 * -----
 * ele1   : Pointer to each entry of matrix
 * matrix : Matrix to iterate
 * ele2   : Pointer to each entry of vector
 * vector : Vector to iterate
 */
#define mvrzip(ele1, matrix, ele2, vector) { \
  int fbreak = 0; \
  long ele1##_rindex = -1; \
  long ele1##_cindex = 0; \
  long ele1##_stride = matrix->stride[1]; \
  long matrix##_csize = matrix->size[1]; \
\
  long ele2##_index = 0; \
  long ele2##_stride = vector->stride[0]; \
  long vector##_size = vector->size[0]; \
\
  ele1 = matrix->storage->data + matrix->storageOffset - matrix->stride[0]; \
\
  while(!fbreak) { \
    ele1##_rindex++; \
    if(ele1##_rindex == matrix->size[0]) break; \
    ele1 -= ele1##_cindex * ele1##_stride; \
    ele1 += matrix->stride[0]; \
    ele1##_cindex = 0; \
\
    ele2 = vector->storage->data + vector->storageOffset; \
    ele2##_index = 0; \
\
    for(; ele2##_index < vector##_size && ele1##_cindex < matrix##_csize; \
        ele1##_cindex++, ele2##_index++, \
        ele1 += ele1##_stride, ele2 += ele2##_stride) {

#define endmvrzip(code) \
    } \
    code \
  } \
}

/**
 * Description
 * -----------
 * Zip matrix and vector column wise,
 * matrix->size[0] == vector->size[0]
 *
 * Usage
 * -----
 * mvczip(ele1, matrix, ele2, vector) {
 *  // code
 * }
 * endmvczip(optional code)
 *
 * Input
 * -----
 * ele1   : Pointer to each entry of matrix
 * matrix : Matrix to iterate
 * ele2   : Pointer to each entry of vector
 * vector : Vector to iterate
 */
#define mvczip(ele1, matrix, ele2, vector) { \
  int fbreak = 0; \
  long ele1##_rindex = 0; \
  long ele1##_cindex = -1; \
  long ele1##_stride = matrix->stride[0]; \
  long matrix##_rsize = matrix->size[0]; \
\
  long ele2##_index = 0; \
  long ele2##_stride = vector->stride[0]; \
  long vector##_size = vector->size[0]; \
\
  ele1 = matrix->storage->data + matrix->storageOffset - matrix->stride[1]; \
\
  while(!fbreak) { \
    ele1##_cindex++; \
    if(ele1##_cindex == matrix->size[1]) break; \
    ele1 -= ele1##_rindex * ele1##_stride; \
    ele1 += matrix->stride[1]; \
    ele1##_rindex = 0; \
\
    ele2 = vector->storage->data + vector->storageOffset; \
    ele2##_index = 0; \
\
    for(; ele2##_index < vector##_size && ele1##_rindex < matrix##_rsize; \
        ele1##_rindex++, ele2##_index++, \
        ele1 += ele1##_stride, ele2 += ele2##_stride) {

#define endmvczip(code) \
    } \
    code \
  } \
}

/**************** FOR *****************/

/**
 * Description
 * -----------
 * For comprehension with two matrix to iterate with side effect
 *
 * Usage
 * -----
 * mfor(ele1, matrix1, ele2, matrix2) {
 *  // code
 * }
 * endmfor(optional code)
 *
 * Input
 * -----
 * ele1    : Pointer to each entry of first matrix
 * matrix1 : Matrix to iterate
 * ele2    : Pointer to each entry of second matrix
 * matrix2 : Matrix to iterate
 */
#define mfor(ele1, matrix1, ele2, matrix2) { \
  long ele1##_index = 0; \
  long ele1##_stride = 0; \
  long matrix1##_contsz = 0; \
  long matrix1##_rowcntr = 0; \
\
  long ele2##_index = 0; \
  long ele2##_stride = 0; \
  long matrix2##_contsz = 0; \
  long matrix2##_rowcntr = 0; \
\
  int fbreak = 0; \
\
  int matrix1##_rmajr = (matrix1->stride[1] == 1 && matrix1->stride[0] == matrix1->size[1]) ? 1 : 0; \
  ele1##_stride = matrix1->stride[1]; \
\
  int matrix2##_rmajr = (matrix2->stride[1] == 1 && matrix2->stride[0] == matrix2->size[1]) ? 1 : 0; \
  ele2##_stride = matrix2->stride[1]; \
\
  if(matrix1##_rmajr) { \
    matrix1##_contsz = matrix1->size[0] * matrix1->size[1]; \
  } else { \
    matrix1##_contsz = matrix1->size[1]; \
  } \
\
  if(matrix2##_rmajr) { \
    matrix2##_contsz = matrix2->size[0] * matrix2->size[1]; \
  } else { \
    matrix2##_contsz = matrix2->size[1]; \
  } \
\
  ele1 = matrix1->storage->data + matrix1->storageOffset; \
  ele2 = matrix2->storage->data + matrix2->storageOffset; \
\
  while(!fbreak) { \
    if(ele2##_index == matrix2##_contsz) { \
      matrix2##_rowcntr += 1; \
      if(matrix2##_rmajr || (matrix2##_rowcntr == matrix2->size[0])) { \
        ele2##_index = 0; \
        ele2 = matrix2->storage->data + matrix2->storageOffset; \
        matrix2##_rowcntr = 0; \
\
        ele1##_index++; \
        ele1 += ele1##_stride; \
      } else { \
        ele2 -= ele2##_index * ele2##_stride; \
        ele2 += matrix2->stride[0]; \
        ele2##_index = 0; \
      } \
    } \
\
    if(ele1##_index == matrix1##_contsz) { \
      matrix1##_rowcntr += 1; \
      if(matrix1##_rmajr || matrix1##_rowcntr == matrix1->size[0]) break; \
      ele1 -= ele1##_index * ele1##_stride; \
      ele1 += matrix1->stride[0]; \
      ele1##_index = 0; \
    } \
\
    for(; ele2##_index < matrix2##_contsz; ele2##_index++, ele2 += ele2##_stride) {

#define endmfor(code) \
    } \
    code \
  } \
}

/**
 * Description
 * -----------
 * For comprehension with two vector to iterate with side effect
 *
 * Usage
 * -----
 * vfor(ele1, vector1, ele2, vector2) {
 *  // code
 * }
 * endvfor(optional code)
 *
 * Input
 * -----
 * ele1    : Pointer to each entry of first vector
 * vector1 : Vector to iterate
 * ele2    : Pointer to each entry of second vector
 * vector2 : Vector to iterate
 */
#define vfor(ele1, vector1, ele2, vector2) { \
  int fbreak = 0; \
  long ele1##_index = -1; \
  long ele1##_stride = vector1->stride[0]; \
  long vector1##_size = vector1->size[0]; \
\
  long ele2##_index = 0; \
  long ele2##_stride = vector2->stride[0]; \
  long vector2##_size = vector2->size[0]; \
\
  ele1 = vector1->storage->data + vector1->storageOffset - ele1##_stride; \
\
  while(!fbreak) { \
    ele1##_index++; \
    if(ele1##_index == vector1##_size) break; \
    ele1 += ele1##_stride; \
    ele2##_index = 0; \
    ele2 = vector2->storage->data + vector2->storageOffset; \
    for(; ele2##_index < vector2##_size; ele2##_index++, ele2 += ele2##_stride) {

#define endvfor(code) \
    } \
    code \
  } \
}

#endif