#include <stdio.h>
#include <aianon/tensor/tensor.h>
#include <aianon/tensor/math.h>

int main(int argc, char *argv[]) {
  double a = 8;
  int b = 1;

  AIATensor(float) *tnsr = aiatensor_(float, empty)();
  // AIATensor(double) *other = aiatensor_(double, new)();

  fprintf(stdout, "Hello Universe ! I am %s\n", "AIAnonymous");

  aiatensor_(float, free)(tnsr);
  // aiatensor_(int, free)(other);

  return 0;
}