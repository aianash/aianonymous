#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aianon/tensor/tensor.h>

static float rnd4x4[16] =
  { 0.079918f,  0.975864f,  0.266865f,  0.912477f,
    0.849078f,  0.665310f,  0.931576f,  0.711065f,
    0.664866f,  0.303851f,  0.210187f,  0.779473f,
    0.417380f,  0.904275f,  0.164664f,  0.091194f };

static float rnd4x4T[16] =
  { 0.079918f,  0.849078f,  0.664866f,  0.417380f,
    0.975864f,  0.665310f,  0.303851f,  0.904275f,
    0.266865f,  0.931576f,  0.210187f,  0.164664f,
    0.912477f,  0.711065f,  0.779473f,  0.091194f };

static float rnd3x3[9] =
  { 0.0799180,  0.9758642,  0.2668651,
    0.8490781,  0.6653103,  0.9315766,
    0.6648669,  0.3038510,  0.2101878 };

static float rnd3x3T[9] =
  { 0.0799180,  0.8490781,  0.6648669,
    0.9758642,  0.6653103,  0.3038510,
    0.2668651,  0.9315766,  0.2101878 };

static long size4x4[2] = {4l, 4l};
static long size3x3[2] = {3l, 3l};
static long size3x4[2] = {3l, 4l};
static long size3[1] = {3l};
static long size4[1] = {4l};

static long stride3[1] = {1l};
static long stride4x4[2] = {4, 1};
static long stride3x3[2] = {3, 1};

static AIATensor(float) *f4x4tnsr;
static AIATensor(float) *f4x4tnsrT;
static AIATensor(float) *f3x3tnsr;

#define SHAPE_WITH_STRIDE(shape, strideInit) \
{ \
  long stride[shape.nDimension] = strideInit; \
  shape.stride = stride; \
}\

void tensor_setup(void) {
  f4x4tnsr = aiatensor_(float, newFromData)(arr_(float, clone)(rnd4x4, 16), 2, size4x4, NULL);
  f4x4tnsrT = aiatensor_(float, newFromData)(arr_(float, clone)(rnd4x4T, 16), 2, size4x4, NULL);
  f3x3tnsr = aiatensor_(float, newFromData)(arr_(float, clone)(rnd3x3, 9), 2, size3x3, NULL);
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

  AIATensor(float) *n = aiatensor_(float, new)(f4x4tnsr);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(n, f4x4tnsr),
    "New from another tensor failed to create from 4x4");
  ck_assert_msg(f4x4tnsr->storage->refcount == 2,
    "Refcount of storage not increased after new operation");
  aiatensor_(float, free)(n);

  AIATensor(float) *vec = aiatensor_(float, newVector)(4);
  ck_assert_msg(vec->nDimension == 1 && vec->size[0] == 4 && vec->stride[0] == 1,
    "Vector creation failed");
  aiatensor_(float, free)(vec);

  tmp = aiatensor_(float, emptyAs)(f3x3tnsr);
  ck_assert_msg(aiatensor_(float, isSameSizeAs)(tmp, f3x3tnsr),
    "EmptyAs failed to create from 3x3");
  aiatensor_(float, free)(tmp);

  tmp = aiatensor_(float, newCopy)(f3x3tnsr);
  ck_assert_msg(aiatensor_(float, eq)(tmp, f3x3tnsr),
    "Failed to set tensor");
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
  ck_assert_msg(aiatensor_(float, isSameShape)(f4x4tnsr, 2, size4x4, stride4x4),
    "isSameShape failed");

  aiatensor_(float, free)(tr);
}
END_TEST

// Resize
START_TEST(test_tensor_resize) {
  AIATensor(float) *tmp = aiatensor_(float, empty)();

  // resize
  aiatensor_(float, resize)(tmp, 2, size4x4, NULL);
  aiatensor_(float, resize)(tmp, 2, size3x3, NULL);

  ck_assert_msg(aiatensor_(float, isSameShape)(tmp, 2, size3x3, stride3x3),
    "Resize failed on empty tensor");

  // resizeAs
  aiatensor_(float, resizeAs)(tmp, f4x4tnsr);
  ck_assert_msg(aiatensor_(float, isSameShape)(tmp, 2, size4x4, stride4x4),
    "ResizeAs failed");

  // resize1d
  aiatensor_(float, resize1d)(tmp, 3);
  ck_assert_msg(aiatensor_(float, isSameShape)(tmp, 1, size3, stride3),
    "Resize1d failed");

  // resize2d
  aiatensor_(float, resize2d)(tmp, 4, 4);
  ck_assert_msg(aiatensor_(float, isSameShape)(tmp, 2, size4x4, stride4x4),
    "Resize2d failed");

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

  long trstride[2] = {1, 4};
  ck_assert_msg(aiatensor_(float, isSameShape)(tr, 2, size4x4, trstride),
    "wrong shape after transpose");

  aiatensor_(float, free)(tr);
}
END_TEST

// Copy
START_TEST(test_tensor_copy) {
  AIATensor(float) *tmp = aiatensor_(float, empty)();
  aiatensor_(float, resize)(tmp, 2, size4x4, stride4x4);

  // copy
  aiatensor_(float, copy)(tmp, f4x4tnsr);
  ck_assert_msg(aiatensor_(float, eq)(tmp, f4x4tnsr),
    "Failed to copy");

  // freeCopyTo
  AIATensor(float) *dest = aiatensor_(float, empty)();
  aiatensor_(float, resize)(dest, 2, size4x4, stride4x4);
  aiatensor_(float, retain)(tmp);
  aiatensor_(float, freeCopyTo)(tmp, dest);
  ck_assert_msg(aiatensor_(float, eq)(dest, f4x4tnsr) && tmp->refcount == 1,
    "Failed to free copy to");

  // copy float (other copies are similar)
  aiatensor_(float, resize)(tmp, 2, size3x3, NULL);
  aiatensor_(float, copyFloat)(tmp, rnd3x3);
  ck_assert_msg(aiatensor_(float, eq)(tmp, f3x3tnsr),
    "Failed to copy");

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

// Narrow
START_TEST(test_tensor_narrow) {
  AIATensor(float) *tmp = aiatensor_(float, empty)();
  aiatensor_(float, narrow)(tmp, f4x4tnsr, 0, 1, 3);
  float exp_3x4_d[12] =
    { 0.849078f,  0.665310f,  0.931576f,  0.711065f,
      0.664866f,  0.303851f,  0.210187f,  0.779473f,
      0.417380f,  0.904275f,  0.164664f,  0.091194f };
  AIATensor(float) *exp_3x4 = aiatensor_(float, newFromData)(arr_(float, clone)(exp_3x4_d, 12), 2, size3x4, NULL);

  ck_assert_msg(aiatensor_(float, eq)(exp_3x4, tmp),
    "Failed to narrow");
  aiatensor_(float, free)(exp_3x4);

  aiatensor_(float, narrow)(tmp, tmp, 1, 0, 3);
  float exp_3x3_d[12] =
    { 0.849078f,  0.665310f,  0.931576f,
      0.664866f,  0.303851f,  0.210187f,
      0.417380f,  0.904275f,  0.164664f };
  AIATensor(float) *exp_3x3 = aiatensor_(float, newFromData)(arr_(float, clone)(exp_3x3_d, 9), 2, size3x3, NULL);

  ck_assert_msg(aiatensor_(float, eq)(exp_3x3, tmp),
    "Failed to narrow second time");
  aiatensor_(float, free)(exp_3x3);

  aiatensor_(float, free)(tmp);
}
END_TEST

// Select
START_TEST(test_tensor_select) {
  AIATensor(float) *tmp = aiatensor_(float, empty)();
  aiatensor_(float, select)(tmp, f4x4tnsr, 0, 1);
  float exp_4_d[4] = { 0.849078f,  0.665310f,  0.931576f,  0.711065f };
  AIATensor(float) *exp_4 = aiatensor_(float, newFromData)(arr_(float, clone)(exp_4_d, 4), 1, size4, NULL);

  ck_assert_msg(aiatensor_(float, eq)(exp_4, tmp),
    "Failed to select");
  aiatensor_(float, free)(exp_4);
  aiatensor_(float, free)(tmp);
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
  tcase_add_test(tc, test_tensor_narrow);
  tcase_add_test(tc, test_tensor_select);
  suite_add_tcase(s, tc);
  return s;
}