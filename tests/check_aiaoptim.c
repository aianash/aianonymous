#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aiatensor/tensor.h>
#include <aiatensor/math.h>
#include <aiaoptim/optim.h>

typedef struct funca_state_ {
  AIATensor(float) *A;
  AIATensor(float) *b;
} funca_state;

static void funca(AIATensor(float) *x, float *fx, AIATensor(float) *df_dx, opfunc_ops ops, void *opstate) {
  funca_state *funcastate = (funca_state*) opstate;
  if(ops == ONLY_F || ops == F_N_GRAD) {
    *fx = aiatensor_(float, xTAx)(x, funcastate->A) / 2 + aiatensor_(float, dot)(funcastate->b, x);
  }

  if(ops == ONLY_GRAD || ops == F_N_GRAD) {
    aiatensor_(float, mv)(df_dx, funcastate->A, x);
    aiatensor_(float, csub)(df_dx, df_dx, 1, funcastate->b);
    return;
  }
}

static void rosenbrock(AIATensor(float) *x, float *f, AIATensor(float) *df_dx, opfunc_ops ops, void *state) {
  *f = 0;
  long counter = 0;
  AIA_TENSOR_APPLY2(float, x, float, df_dx,
                    float t1 = *(x_data + x_stride) - pow(*x_data, 2);
                    float t2 = 1 - *x_data;
                    (*f) += 100 * pow(t1, 2) + pow(t2, 2);
                    if(counter == 0) (*df_dx_data) = 0;
                    (*(df_dx_data + df_dx_stride)) = 200 * t1;
                    (*df_dx_data) += - 400 * *x_data * t1 - 2 * t2;
                    counter++;
                    if(counter == x->size[0] - 1) {
                      tensor_apply_finished = 1;
                      break;
                    }
                  );
}

static float fK6x6[36] =
  { 1.00000f,  0.88692f,  0.61878f,  0.47236f,  0.33959f,  0.22992f,
    0.88692f,  1.00000f,  0.88692f,  0.76337f,  0.61878f,  0.47236f,
    0.61878f,  0.88692f,  1.00000f,  0.97044f,  0.88692f,  0.76337f,
    0.47236f,  0.76337f,  0.97044f,  1.00000f,  0.97044f,  0.88692f,
    0.33959f,  0.61878f,  0.88692f,  0.97044f,  1.00000f,  0.97044f,
    0.22992f,  0.47236f,  0.76337f,  0.88692f,  0.97044f,  1.00000f };

static float fy6[6] =
  { 0.84147f,  0.42336f, -4.79462f, -1.67649f,  4.59890f,  7.91486f };

static float randx[2] =
  { 0.f,  0.f};

static float fepsi = 0.1f;

static long size6x6[2] = {6l, 6l};
static long size6[1]   = {6l};
static long size2[1]   = {2l};

AIATensor(float) *fK6x6tnsr;
AIATensor(float) *fytnsr;

void optim_setup(void) {
  fK6x6tnsr = aiatensor_(float, newFromData)(arr_(float, clone)(fK6x6, 36), 2, size6x6, NULL);
  fytnsr = aiatensor_(float, newFromData)(arr_(float, clone)(fy6, 6), 1, size6, NULL);
}

void optim_teardown(void) {
  aiatensor_(float, free)(fK6x6tnsr);
  aiatensor_(float, free)(fytnsr);
}

START_TEST(test_cg_float) {
  funca_state s = {
    .A = fK6x6tnsr,
    .b = fytnsr
  };
  float xinit[6] = {1, 1.1, 0.4, 0.6, 0.4};
  AIATensor(float) *frestnsr = aiatensor_(float, newFromData)(arr_(float, clone)(xinit, 6), 1, size6, NULL);
  optim_(float, cg)(frestnsr, funca, fK6x6tnsr, &s, &default_cg_config);

  aiatensor_(float, free)(frestnsr);
}
END_TEST

START_TEST(test_ncg_float) {
  float exp2[2] = {1.0f, 1.0f};
  AIATensor(float) *fexptnsr = aiatensor_(float, newFromData)(arr_(float, clone)(exp2, 2), 1, size2, NULL);
  funca_state s = {
    .A = fK6x6tnsr,
    .b = fytnsr
  };
  float xinit[2] = {0, 0};
  AIATensor(float) *frestnsr = aiatensor_(float, newFromData)(arr_(float, clone)(xinit, 2), 1, size2, NULL);
  optim_(float, ncg)(frestnsr, rosenbrock, NULL, NULL);
  ck_assert_msg(aiatensor_(float, epsieq)(frestnsr, fexptnsr, fepsi),
    "ncg failed. actual result = %s and expected result = %s",
    aiatensor_(float, toString)(frestnsr), aiatensor_(float, toString)(fexptnsr));

  aiatensor_(float, free)(frestnsr);
  aiatensor_(float, free)(fexptnsr);
}
END_TEST

Suite *make_cg_suite(void) {
  Suite *s;
  TCase *tc;

  s = suite_create("Optim");
  tc = tcase_create("Optim");

  tcase_add_checked_fixture(tc, optim_setup, optim_teardown);

  tcase_add_test(tc, test_cg_float);
  tcase_add_test(tc, test_ncg_float);

  suite_add_tcase(s, tc);
  return s;
}