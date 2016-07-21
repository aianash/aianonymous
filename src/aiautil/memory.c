#include <aiautil/memory.h>

//
void *aia_alloc(long size) {
  void *ptr;
  if(size < 0) aia_error("Invalid memory size");
  if(size == 0) return NULL;
  ptr = malloc(size);
  if(!ptr) aia_error("Not enough memory to allocate %dMB", size/1048576);
  return ptr;
}

//
void *aia_realloc(void *ptr, long size) {
  if(!ptr) return(aia_alloc(size));

  if(size == 0) {
    free(ptr);
    return NULL;
  }

  if(size < 0) aia_error("Invalid memory size");

  void *newptr = realloc(ptr, size);

  if(!newptr)
    aia_error("Not enough memory to re-allocate %dMB", size/1048576);

  return newptr;
}

//
void aia_free(void *ptr) {
  free(ptr);
}