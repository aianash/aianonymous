#include <stdio.h>
#include <aianon/tensor.h>

int main(int argc, char *argv[]) {
  double a = 8;
  int b = 1;

  AIATensor(int) *tnsr = aiatensor_(int, new)();
  AIATensor(double) *other = aiatensor_(double, new)();

  fprintf(stdout, "Hello Universe ! I am %s\n", "AIAnonymous");
  return 0;
}