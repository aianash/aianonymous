if(NOT BLAS_FOUND)

set(BLAS_LIBRARIES)
set(BLAS_INCLUDE_DIR)
set(BLAS_INFO)
set(BLAS_F2C)

set(WITH_BLAS "" CACHE STRING "Blas type [mkl/open/goto/acml/atlas/accelerate/veclib/generic]")

include(CheckFunctionExists)
include(CheckFortranFunctionExists)

macro(Check_Fortran_Libraries LIBRARIES _prefix _name _flags _list)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the 
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to NOTFOUND.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.

  set(__list)
  foreach(_elem ${_list})
    if(__list)
      set(__list "${__list} - ${_elem}")
    else(__list)
      set(__list "${_elem}")
    endif(__list)
  endforeach(_elem)
  message(STATUS "Checking for [${__list}]")

  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})
    if(_libraries_work)
      # TODO custom find libraries for different systems like WIN32 APPLE
      find_library(${_prefix}_${_library}_LIBRARY
        NAMES ${_library}
        PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64
        ENV LD_LIBRARY_PATH)
      mark_as_advanced(${_prefix}_${_library}_LIBRARY)
      set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
      message(STATUS " Library ${_library}: ${${_prefix}_${_library}_LIBRARY}")
    endif(_libraries_work)
  endforeach(_library)

  if(_libraries_work)
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}})
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
    set(${LIBRARIES} NOTFOUND)
  endif(NOT _libraries_work)
endmacro(Check_Fortran_Libraries)

# TODO:
# MKL
# openblas
# openblas:pthread
# libopenblas
# goto2:gfortran
# goto2:gfortran;pthread
# acml; gfortran
# Accelerate
# vecLib
# ptf77blas;atlas;gfortran

if((NOT BLAS_LIBRARIES) AND ((NOT WITH_BLAS) OR (WITH_BLAS STREQUAL "generic")))
  check_fortran_libraries(
    BLAS_LIBRARIES
    BLAS
    sgemm
    ""
    "blas")
  if(BLAS_LIBRARIES)
    set(BLAS_INFO "generic")
  endif(BLAS_LIBRARIES)
endif()

# TODO: check for f2c

if(BLAS_LIBRARIES)
  set(BLAS_FOUND TRUE)
else(BLAS_LIBRARIES)
  set(BLAS_FOUND FALSE)
endif(BLAS_LIBRARIES)

if(NOT BLAS_FOUND)
  message(FATAL_ERROR "Cannot find a library with BLAS APT. Please provide library location.")
else(NOT BLAS_FOUND)
  message(STATUS "Found a library with BLAS API (${BLAS_INFO}).")
endif(NOT BLAS_FOUND)

endif(NOT BLAS_FOUND)