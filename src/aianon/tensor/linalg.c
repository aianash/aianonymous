#include <aianon/tensor/linalg.h>

#ifdef ERASED_TYPE_PRESENT

// Check if self is transpose of a contiguous matrix
static int aiatensor__(isTransposedContiguous)(AIATensor_ *this) { return 1; }

// If a matrix is a regular contiguous matrix, make sure it is transposed
// because this is what we return from Lapack calls.
static void aiatensor__(checkTransposed)(AIATensor_ *this) { }

// newContiguous followed by transpose
// Similar to (newContiguous), but checks if the transpose of the matrix
// is contiguous and also limited to 2D matrices.
static AIATensor_ *aiatensor__(newTransposedContiguous)(AIATensor_ *this) { return this; }

// Given the result tensor and src tensor, decide if the lapack call should use the
// provided result tensor or should allocate a new space to put the result in.
// The returned tensor have to be freed by the calling function.
// nrows is required, because some lapack calls, require output space smaller than
// input space, like underdetermined gels.
static AIATensor_ *aiatensor__(checkLapackClone)(AIATensor_ *result, AIATensor_ *src, int nrows) { return result; }

// Same as cloneColumnMajor, but accepts nrows argument, because some lapack calls require
// the resulting tensor to be larger than src.
static AIATensor_ *aiatensor__(cloneColumnMajorNrows)(AIATensor_ *this, AIATensor_ *src, int nrows) { return this; }

// Create a clone of src in self column major order for use with Lapack.
// If src == self, a new tensor is allocated, in any case, the return tensor should be
// freed by calling function.
static AIATensor_ *aiatensor__(cloneColumnMajor)(AIATensor_ *this, AIATensor_ *src) { return this; }

#endif
#define ERASE_FLOAT
#define ERASURE_FILE "aianon/tensor/linalg.c"
#include <aianon/core/erasure.h>