#include <stdio.h>
#include <aianon/tensor.h>

int main(int argc, char *argv[]) {
  double a = 8;
  int b = 1;
  fprintf(stdout, "Hello Universe ! I am %s v%d.%f\n", "AIAnonymous", aiatensor_(int, get)(b), aiatensor_(double, get)(a));
  return 0;
}