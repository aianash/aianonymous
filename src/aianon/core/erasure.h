// #include <aianon/util.h>

#ifdef ERASED_TYPE_PRESENT

#undef T
#undef T_
#undef ERASED_TYPE_PRESENT
#line 1 ERASURE_FILE
#include ERASURE_FILE

#else

// // [TODO] move this code...
#define AIA_CONCAT_5(v,w,x,y,z) AIA_CONCAT_5_EXPAND(v,w,x,y,z)
#define AIA_CONCAT_5_EXPAND(v,w,x,y,z) v ## w ## x ## y ## z

#define AIA_CONCAT_3(x,y,z) AIA_CONCAT_3_EXPAND(x,y,z)
#define AIA_CONCAT_3_EXPAND(x,y,z) x ## y ## z

#ifndef AIA_ERASURE_HELPERS_FNS
#define AIA_ERASURE_HELPERS_FNS
#define AIA_FN_ERASE_(ns, type, name) AIA_FN_ERASE_EXPAND_(ns, type, name)
#define AIA_STRUCT_ERASE_(type, name) AIA_STRUCT_ERASE_EXPAND_(type, name)
#define AIA_FN_ERASE_EXPAND_(ns, type, name) AIA_CONCAT_5(aia,type,ns,_,name)
#define AIA_STRUCT_ERASE_EXPAND_(type, name) AIA_CONCAT_3(aia,type,name)
#endif

#if (defined(ERASE_BYTE) || defined(ERASE_ALL))
# ifndef BYTE_IS_ERASED
# define BYTE_IS_ERASED
#   define T unsigned char
#   define T_ uchar
#   define T_IS_UCHAR
#   define ERASED_TYPE_PRESENT
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#if (defined(ERASE_CHAR) || defined(ERASE_ALL))
# ifndef CHAR_IS_ERASED
# define CHAR_IS_ERASED
#   define T char
#   define T_ char
#   define T_IS_CHAR
#   define ERASED_TYPE_PRESENT
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#if (defined(ERASE_SHORT) || defined(ERASE_ALL))
# ifndef SHORT_IS_ERASED
# define SHORT_IS_ERASED
#   define T short
#   define T_ short
#   define T_IS_SHORT
#   define ERASED_TYPE_PRESENT
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#if (defined(ERASE_INT) || defined(ERASE_ALL))
# ifndef INT_IS_ERASED
# define INT_IS_ERASED
#   define T int
#   define T_ int
#   define T_IS_INT
#   define ERASED_TYPE_PRESENT
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#if (defined(ERASE_LONG) || defined(ERASE_ALL))
# ifndef LONG_IS_ERASED
# define LONG_IS_ERASED
#   define T long
#   define T_ long
#   define T_IS_LONG
#   define ERASED_TYPE_PRESENT
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#if (defined(ERASE_FLOAT) || defined(ERASE_ALL))
# ifndef FLOAT_IS_ERASED
# define FLOAT_IS_ERASED
#   define T float
#   define T_ float
#   define T_IS_FLOAT
#   define ERASED_TYPE_PRESENT
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#if (defined(ERASE_DOUBLE) || defined(ERASE_ALL))
# ifndef DOUBLE_IS_ERASED
# define DOUBLE_IS_ERASED
#   define T double
#   define T_ double
#   define T_IS_DOUBLE
#   define ERASED_TYPE_PRESENT
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#undef T
#undef T_

#undef T_IS_UCHAR
#undef T_IS_CHAR
#undef T_IS_SHORT
#undef T_IS_LONG
#undef T_IS_FLOAT
#undef T_IS_DOUBLE

#undef ERASE_BYTE
#undef ERASE_CHAR
#undef ERASE_SHORT
#undef ERASE_INT
#undef ERASE_LONG
#undef ERASE_FLOAT
#undef ERASE_DOUBLE
#undef ERASE_ALL

#undef BYTE_IS_ERASED
#undef CHAR_IS_ERASED
#undef SHORT_IS_ERASED
#undef INT_IS_ERASED
#undef LONG_IS_ERASED
#undef FLOAT_IS_ERASED
#undef DOUBLE_IS_ERASED

#undef ERASURE_FILE
#undef ERASED_TYPE_PRESENT

#endif