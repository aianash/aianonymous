#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aianon/tensor/tensor.h>

#include "sample_tensors.h"

AIATensor(float) *f4x4tnsr;
AIATensor(float) *f4x4tnsrT;
AIATensor(float) *f3x3tnsr;

void tensor_setup(void) {
  f4x4tnsr = mktnsr_(float, rnd4x4)();
  f4x4tnsrT = mktnsr_(float, rnd4x4T)();
  f3x3tnsr = mktnsr_(float, rnd3x3)();
}

void tensor_teardown(void) {
  aiatensor_(float, free)(f4x4tnsr);
  aiatensor_(float, free)(f4x4tnsrT);
  aiatensor_(float, free)(f3x3tnsr);
}

// Constructors
START_TEST(test_tensor_create) {
  AIATensor(float) *tmp = aiatensor_(float, empty)();

  ck_assert_msg(aiatensor_(float, nElement)(tmp) == 0,
    "Failed to calculate nElement(%d) for empty tensor", aiatensor_(float, nElement));

  ck_assert_msg(tmp->refcount == 1,
    "wrong refcount of new tensor %d != 1(true)", tmp->refcount);

  aiatensor_(float, retain)(tmp);
  ck_assert_msg(tmp->refcount == 2,
    "wrong refcount of tensor after retain %d != 2(true)", tmp->refcount);

  aiatensor_(float, free)(tmp);
  ck_assert_msg(tmp->refcount == 1,
    "wrong refcount of tmp after free %d != 1(true)", tmp->refcount);

  aiatensor_(float, free)(tmp);

  ck_assert_msg(aiatensor_(float, size)(f4x4tnsr, 0) == 4,
    "Failed to construct from data, wrong size at dim 0 = %li\n", aiatensor_(float, size)(f4x4tnsr, 0));
  ck_assert_msg(aiatensor_(float, size)(f4x4tnsr, 1) == 4,
    "Failed to construct from data, wrong size at dim 1 = %li\n", aiatensor_(float, size)(f4x4tnsr, 1));

}
END_TEST

// Storage tests
START_TEST(test_tensor_storage) {
  AIAStorage(float) *strg = aiatensor_(float, storage)(f4x4tnsr);
  ck_assert_msg(strg != NULL,
    "storage of 4x4 random tensor is null");

  ck_assert_msg(aiatensor_(float, data)(f4x4tnsr) != NULL,
    "data of 4x4 random tensor is null");

  ck_assert_msg(strg->size == 16,
    "wrong size in storage of 4x4 random tensor %d != 16(true)", strg->size);

  ck_assert_msg(strg->refcount == 1,
    "wrong refcount of storage %d != 1(true)", strg->refcount);

  aiastorage_(float, retain)(strg);
  ck_assert_msg(strg->refcount == 2,
    "wrong refcount of storage after retain %d != 2(true)", strg->refcount);

  aiastorage_(float, free)(strg);
  ck_assert_msg(strg->refcount == 1,
    "wrong refcount of storage after free %d != 1(true)", strg->refcount);
}
END_TEST

// Accessors
START_TEST(test_tensor_accessors) {
  // nDimension
  ck_assert_msg(aiatensor_(float, nDimension)(f4x4tnsr) == 2,
    "wrong number of dim = %d\n", aiatensor_(float, nDimension)(f4x4tnsr));

  // size
  ck_assert_msg(aiatensor_(float, size)(f4x4tnsr, 0) == 4,
    "wrong size at dim 0 = %li\n", aiatensor_(float, size)(f4x4tnsr, 0));
  ck_assert_msg(aiatensor_(float, size)(f4x4tnsr, 1) == 4,
    "wrong size at dim 1 = %li\n", aiatensor_(float, size)(f4x4tnsr, 1));

  // stride
  ck_assert_msg(aiatensor_(float, stride)(f4x4tnsr, 0) == 4,
    "wrong stride at dim 0 = %li\n", aiatensor_(float, stride)(f4x4tnsr, 0));
  ck_assert_msg(aiatensor_(float, stride)(f4x4tnsr, 1) == 1,
    "wrong stride at dim 1 = %li\n", aiatensor_(float, stride)(f4x4tnsr, 1));

  // nElement
  ck_assert_msg(aiatensor_(float, nElement)(f4x4tnsr) == 16,
    "wrong number of elements in 4x4 matrix, %d", aiatensor_(float, nElement)(f4x4tnsr));
}
END_TEST

// Test Methods
START_TEST(test_tensor_test_methods) {
  ck_assert_msg(aiatensor_(float, isContiguous)(f4x4tnsr),
    "Contiguous matrix identified as not contiguous");

  AIATensor(float) *tr = aiatensor_(float, empty)();
  aiatensor_(float, transpose)(tr, f4x4tnsr, 0, 1);

  // isContiguous
  ck_assert_msg(!aiatensor_(float, isContiguous)(tr),
    "Not contiguous matrix identified as contiguous");

  // isSameSize
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(tr, f4x4tnsr),
    "Same size matrix check failed between 4x4 4x4");
  ck_assert_msg(!aiatensor_(float, isSameSizeAs)(tr, f3x3tnsr),
    "Same size matrix check failed between 4x4 3x3");

  // isSameShape
  long stride4x4[2] = {4, 1};
  shape4x4.stride = stride4x4;
  ck_assert_msg(aiatensor_(float, isSameShape)(f4x4tnsr, shape4x4),
    "isSameShape failed");
  shape4x4.stride = NULL;

  aiatensor_(float, free)(tr);
}
END_TEST

// Resize
START_TEST(test_tensor_resize) {
  AIATensor(float) *tmp = aiatensor_(float, empty)();

  // resize
  aiatensor_(float, resize)(tmp, shape3x3);
  long stride3x3[2] = {3, 1};
  shape3x3.stride = stride3x3;
  ck_assert_msg(aiatensor_(float, isSameShape)(tmp, shape3x3),
    "Resize failed on empty tensor");
  shape3x3.stride = NULL;

  // resizeAs
  aiatensor_(float, resizeAs)(tmp, f4x4tnsr);
  long stride4x4[2] = {4, 1};
  shape4x4.stride = stride4x4;
  ck_assert_msg(aiatensor_(float, isSameShape)(tmp, shape4x4),
    "ResizeAs failed");
  shape4x4.stride = NULL;

  // resize1d
  long size3[1] = {3l};
  long stride3[1] = {1l};
  TensorShape shape3 = NEW_TENSOR_SHAPE(1, size3, stride3);
  aiatensor_(float, resize1d)(tmp, 3);
  ck_assert_msg(aiatensor_(float, isSameShape)(tmp, shape3),
    "Resize1d failed");

  // resize2d
  aiatensor_(float, resize2d)(tmp, 4, 4);
  shape4x4.stride = stride4x4;
  ck_assert_msg(aiatensor_(float, isSameShape)(tmp, shape4x4),
    "Resize2d failed");
  shape4x4.stride = NULL;

  aiatensor_(float, free)(tmp);
}
END_TEST

// Set
START_TEST(test_tensor_set) {
  AIATensor(float) *tmp = aiatensor_(float, empty)();

  aiatensor_(float, set)(tmp, f4x4tnsr);
  ck_assert_msg(aiatensor_(float, eq)(tmp, f4x4tnsr),
    "Failed to set tensor");

  ck_assert_msg(aiatensor_(float, isSetTo)(tmp, f4x4tnsr),
    "Failed isSetTo");

  aiatensor_(float, free)(tmp);
}
END_TEST

// Transpose
START_TEST(test_tensor_transpose) {
  AIATensor(float) *tr = aiatensor_(float, empty)();
  aiatensor_(float, transpose)(tr, f4x4tnsr, 0, 1);

  ck_assert_msg(aiatensor_(float, eq)(tr, f4x4tnsrT),
    "Failed to transpose 4x4 matrix");

  long stride4x4[2] = {1, 4};
  shape4x4.stride = stride4x4;
  ck_assert_msg(aiatensor_(float, isSameShape)(tr, shape4x4),
    "wrong shape after transpose");
  shape4x4.stride = NULL;

  aiatensor_(float, free)(tr);
}
END_TEST

// Copy
START_TEST(test_tensor_copy) {
  AIATensor(float) *tmp = aiatensor_(float, empty)();
  aiatensor_(float, resize)(tmp, shape4x4);

  // copy
  aiatensor_(float, copy)(tmp, f4x4tnsr);
  ck_assert_msg(aiatensor_(float, eq)(tmp, f4x4tnsr),
    "Failed to copy");

  // freeCopyTo
  AIATensor(float) *dest = aiatensor_(float, empty)();
  aiatensor_(float, resize)(dest, shape4x4);
  aiatensor_(float, retain)(tmp);
  aiatensor_(float, freeCopyTo)(tmp, dest);
  ck_assert_msg(aiatensor_(float, eq)(dest, f4x4tnsr) && tmp->refcount == 1,
    "Failed to free copy to");

  aiatensor_(float, free)(tmp);
}
END_TEST

// Cloning
START_TEST(test_tensor_cloning) {
  AIATensor(float) *tmp = aiatensor_(float, empty)();
  aiatensor_(float, transpose)(tmp, f4x4tnsr, 0, 1);

  AIATensor(float) *cont = aiatensor_(float, contiguous)(tmp);
  ck_assert_msg(aiatensor_(float, isContiguous)(cont),
    "aiatensor_(contiguous) couldnt create contiguous matrix");

  AIATensor(float) *cl = aiatensor_(float, clone)(f3x3tnsr);
  ck_assert_msg(aiatensor_(float, eq)(cl, f3x3tnsr),
    "Failed to clone");
}
END_TEST

//
Suite *make_tensor_suite(void) {
  Suite *s;
  TCase *tc;

  s = suite_create("Tensor");
  tc = tcase_create("Tensor");

  tcase_add_checked_fixture(tc, tensor_setup, tensor_teardown);
  tcase_add_test(tc, test_tensor_create);
  tcase_add_test(tc, test_tensor_storage);
  tcase_add_test(tc, test_tensor_accessors);
  tcase_add_test(tc, test_tensor_test_methods);
  tcase_add_test(tc, test_tensor_resize);
  tcase_add_test(tc, test_tensor_set);
  tcase_add_test(tc, test_tensor_transpose);
  tcase_add_test(tc, test_tensor_copy);
  tcase_add_test(tc, test_tensor_cloning);
  suite_add_tcase(s, tc);
  return s;
}