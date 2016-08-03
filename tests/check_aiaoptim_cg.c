#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <aiatensor/tensor.h>
#include <aiatensor/math.h>
#include <aiaoptim/optim.h>

typedef struct cg_state_ {
  AIATensor(float) *A;
  AIATensor(float) *b;
} cg_state;

void funca(AIATensor(float) *x, float *fx, AIATensor(float) *df_dx, opfunc_ops ops, void *opstate) {
  cg_state *cgstate = (cg_state*) opstate;
  if(ops == ONLY_F || ops == F_N_GRAD) {
    *fx = aiatensor_(float, xTAx)(x, cgstate->A) / 2 + aiatensor_(float, dot)(cgstate->b, x);
  }

  if(ops == ONLY_GRAD || ops == F_N_GRAD) {
    aiatensor_(float, mv)(df_dx, cgstate->A, x);
    aiatensor_(float, csub)(df_dx, df_dx, 1, cgstate->b);
    return;
  }
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

static float randx[6] =
  { 0.52389f,  0.73592f,  0.16788f,  0.54593f,  0.33938f, 0.73778f };

static long size6x6[2] = {6l, 6l};
static long size6x4[2] = {6l, 4l};
static long size4x4[2] = {4l, 4l};
static long size6[1]   = {6l};

AIATensor(float) *fK6x6Ltnsr;
AIATensor(float) *fytnsr;
AIATensor(float) *fxinit;

void cg_setup(void) {
  fK6x6Ltnsr = aiatensor_(float, newFromData)(arr_(float, clone)(fK6x6, 36), 2, size6x6, NULL);
  fytnsr = aiatensor_(float, newFromData)(arr_(float, clone)(fy6, 6), 1, size6, NULL);
  fxinit = aiatensor_(float, newFromData)(arr_(float, clone)(randx, 6), 1, size6, NULL);
}

void cg_teardown(void) {

}

START_TEST(test_cg_float) {
  cg_state s = {
    .A = fK6x6Ltnsr,
    .b = fytnsr
  };
  float fx;
  AIATensor(float) *x = aiatensor_(float, emptyVector)(6);
  aiatensor_(float, copy)(x, fxinit);
  optim_(float, cg)(x, funca, fK6x6Ltnsr, &s, &default_cg_config);
  // printf("x = %s\n", aiatensor_(float, toString)(x));

  printf("x = %s\n", aiatensor_(float, toString)(x));
}
END_TEST

Suite *make_cg_suite(void) {
  Suite *s;
  TCase *tc;

  s = suite_create("GaussianProcess");
  tc = tcase_create("GaussianProcess");

  tcase_add_checked_fixture(tc, cg_setup, cg_teardown);

  tcase_add_test(tc, test_cg_float);

  suite_add_tcase(s, tc);
  return s;
}