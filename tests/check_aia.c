#include <tests/config.h>
#include <check.h>
#include <stdlib.h>
#include <stdio.h>

#include "check_aianon-math.h"

int main(void) {

  int number_failed;
  SRunner *sr;

  sr = srunner_create(make_tensor_suite());
  srunner_add_suite(sr, make_tensormath_suite());
  srunner_add_suite(sr, make_tensorlinalg_suite());
  srunner_add_suite(sr, make_kernel_suite());
  srunner_add_suite(sr, make_gp_suite());
  srunner_run_all(sr, CK_NORMAL);

  int sub_ntests = srunner_ntests_run(sr);
  printf("Ran %d tests in subordinate suite\n", sub_ntests);
  number_failed = srunner_ntests_failed(sr);
  srunner_free(sr);

  return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}