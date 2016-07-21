
# Do nothing if LAPACK was found before
if(NOT LAPACK_FOUND)

set(LAPACK_LIBRARIES)
set(LAPACK_INFO)

find_package(BLAS REQUIRED)

include(CheckFortranFunctionExists)

macro(Check_Lapack_Libraries LIBRARIES _prefix _name _flags _list _blas)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to FALSE.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.
  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})
    if(_libraries_work)
      # TODO custom find for other systems like WIN32 and APPLE
      find_library(${_prefix}_${_library}_LIBRARY
        NAMES ${_library}
        PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64
        ENV LD_LIBRARY_PATH)
      mark_as_advanced(${_prefix}_${_library}_LIBRARY)
      set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
    endif(_libraries_work)
  endforeach(_library ${_list})

  if(_libraries_work)
    # Test this combination of libraries
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}} ${_blas})
    if(CMAKE_Fortran_COMPILER_WORKS)
      check_fortran_function_exists(${_name} ${_prefix}${_combined_name}_WORKS)
    else(CMAKE_Fortran_COMPILER_WORKS)
      check_function_exists("${_name}_" ${_prefix}${_combined_name}_WORKS)
    endif(CMAKE_Fortran_COMPILER_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)
    mark_as_advanced(${_prefix}${_combined_name}_WORKS)
    set(_libraries_work ${${_prefix}${_combined_name}_WORKS})
  endif(_libraries_work)

  if(NOT _libraries_work)
    set(${LIBRARIES} FALSE)
  endif(NOT _libraries_work)
endmacro(Check_Lapack_Libraries)


if(BLAS_FOUND)

  # TODO:
  # Intel MKL
  # OpenBLAS
  # GotoBlas
  # ACML
  # Accelerate
  # vecLib

  # Generic LAPACK library
  if((NOT LAPACK_INFO) AND ((BLAS_INFO STREQUAL "generic") OR (BLAS_INFO STREQUAL "open")))
    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "lapack"
      "${BLAS_LIBRARIES}"
      )
    if(LAPACK_LIBRARIES)
      set(LAPACK_INFO "generic")
    endif(LAPACK_LIBRARIES)
  endif()

else(BLAS_FOUND)
  message(STATUS "LAPACK requires BLAS")
endif(BLAS_FOUND)

if(LAPACK_INFO)
  set(LAPACK_FOUND TRUE)
else(LAPACK_INFO)
  set(LAPACK_FOUND FALSE)
endif(LAPACK_INFO)

if(NOT LAPACK_FOUND)
  message(FATAL_ERROR "Cannot find a library with LAPACK API. Please provide library location.")
else(NOT LAPACK_FOUND)
  message(STATUS "Found a library with LAPACK API. (${LAPACK_INFO})")
endif(NOT LAPACK_FOUND)

# Do nothing if LAPACK was found before
endif(NOT LAPACK_FOUND)

# set(LAPACK_NAMES ${LAPACK_NAMES} lapack)

# # Check ATLAS paths preferentially, using this necessary hack (I love CMake).
# find_library(LAPACK_LIBRARY
#   NAMES ${LAPACK_NAMES}
#   PATHS /usr/lib64/atlas /usr/lib/atlas /usr/local/lib64/atlas /usr/local/lib/atlas
#   NO_DEFAULT_PATH)

# find_library(LAPACK_LIBRARY
#   NAMES ${LAPACK_NAMES}
#   PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
#   )

# if (LAPACK_LIBRARY)
#   set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})
#   set(LAPACK_FOUND "YES")
# else ()
#   set(LAPACK_FOUND "NO")
# endif ()


# if (LAPACK_FOUND)
#    if (NOT LAPACK_FIND_QUIETLY)
#       message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
#    endif ()
# else ()
#    if (LAPACK_FIND_REQUIRED)
#       message(FATAL_ERROR "Could not find LAPACK")
#    endif ()
# endif ()

# # Deprecated declarations.
# get_filename_component (NATIVE_LAPACK_LIB_PATH ${LAPACK_LIBRARY} PATH)

# mark_as_advanced(
#   LAPACK_LIBRARY
#   )

# endif(NOT LAPACK_FOUND)