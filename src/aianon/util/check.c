#include <aianon/util/check.h>

void _aia_error(const char *file, const int line, const char *fmt, ...) {
  char msg[2048];
  va_list args;

  va_start(args, fmt);
  int n = vsnprintf(msg, 2048, fmt, args);
  va_end(args);

  if(n < 2048) {
    snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
  }
  printf("Error: [AIA] %s\n", msg);
  exit(-1);
}

void _aia_assertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...) {
  char msg[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 1024, fmt, args);
  va_end(args);
  _aia_error(file, line, "Assertion `%s' failed. %s", exp, msg);
}

void _aia_argcheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...) {
  if(!condition) {
    char msg[2048];
    va_list args;

    va_start(args, fmt);
    int n = vsnprintf(msg, 2048, fmt, args);
    va_end(args);

    if(n < 2048) {
      snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
    }

    printf("Error: [AIA] Invalid argument %d: %s\n", argNumber, msg);
    exit(-1);
  }
}