#ifdef ERASURE_ACTIVE

#undef T
#undef TypeName
#undef ERASURE_ACTIVE
#line 1 ERASURE_FILE
#include ERASURE_FILE

#else

#define TH_CONCAT_3(x,y,z) TH_CONCAT_3_EXPAND(x,y,z)
#define TH_CONCAT_3_EXPAND(x,y,z) x ## y ## z

#define AIAErase_(name) TH_CONCAT_3(AIA,TypeName,name)

#ifdef ERASE_BYTE
# ifndef BYTE_IS_ERASED
# define BYTE_IS_ERASED
#   define T unsigned char
#   define TypeName Byte
#   define ERASURE_ACTIVE
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#ifdef ERASE_CHAR
# ifndef CHAR_IS_ERASED
# define CHAR_IS_ERASED
#   define T char
#   define TypeName Char
#   define ERASURE_ACTIVE
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#ifdef ERASE_SHORT
# ifndef SHORT_IS_ERASED
# define SHORT_IS_ERASED
#   define T short
#   define TypeName Short
#   define ERASURE_ACTIVE
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#ifdef ERASE_INT
# ifndef INT_IS_ERASED
# define INT_IS_ERASED
#   define T int
#   define TypeName Int
#   define ERASURE_ACTIVE
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#ifdef ERASE_LONG
# ifndef LONG_IS_ERASED
# define LONG_IS_ERASED
#   define T long
#   define TypeName Long
#   define ERASURE_ACTIVE
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#ifdef ERASE_FLOAT
# ifndef FLOAT_IS_ERASED
# define FLOAT_IS_ERASED
#   define T float
#   define TypeName Float
#   define ERASURE_ACTIVE
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#ifdef ERASE_DOUBLE
# ifndef DOUBLE_IS_ERASED
# define DOUBLE_IS_ERASED
#   define T double
#   define TypeName Double
#   define ERASURE_ACTIVE
#   line 1 ERASURE_FILE
#   include ERASURE_FILE
# endif
#endif

#undef T
#undef TypeName

#undef AIAErase_

#undef ERASE_BYTE
#undef ERASE_CHAR
#undef ERASE_SHORT
#undef ERASE_INT
#undef ERASE_LONG
#undef ERASE_FLOAT
#undef ERASE_DOUBLE

#undef BYTE_IS_ERASED
#undef CHAR_IS_ERASED
#undef SHORT_IS_ERASED
#undef INT_IS_ERASED
#undef LONG_IS_ERASED
#undef FLOAT_IS_ERASED
#undef DOUBLE_IS_ERASED

#undef ERASURE_FILE
#undef ERASURE_ACTIVE

#endif