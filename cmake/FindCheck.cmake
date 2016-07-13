include(FindPkgConfig)

# Take care about check.pc settings
PKG_SEARCH_MODULE(CHECK check)

# Look for CHECK include dir and libraries
if(NOT CHECK_FOUND)
    if (CHECK_INSTALL_DIR)
        message (STATUS "Using override CHECK_INSTALL_DIR to find check")
        set (CHECK_INCLUDE_DIR  "${CHECK_INSTALL_DIR}/include")
        set (CHECK_INCLUDE_DIRS "${CHECK_INCLUDE_DIR}")
        find_library(CHECK_LIBRARY NAMES check PATHS "${CHECK_INSTALL_DIR}/lib")
        find_library(COMPAT_LIBRARY NAMES compat PATHS "${CHECK_INSTALL_DIR}/lib")
        set (CHECK_LIBRARIES "${CHECK_LIBRARY}" "${COMPAT_LIBRARY}")
    else (CHECK_INSTALL_DIR)
        FIND_PATH(CHECK_INCLUDE_DIR check.h)
        find_library(CHECK_LIBRARIES NAMES check)
    endif (CHECK_INSTALL_DIR)

    if (CHECK_INCLUDE_DIR AND CHECK_LIBRARIES)
        set(CHECK_FOUND 1)
        if (NOT Check_FIND_QUIETLY)
            message (STATUS "Found CHECK: ${CHECK_LIBRARIES}")
        endif (NOT Check_FIND_QUIETLY)
    else (CHECK_INCLUDE_DIR AND CHECK_LIBRARIES)
        if (Check_FIND_REQUIRED)
            message(FATAL_ERROR "Could NOT find CHECK")
        else (Check_FIND_REQUIRED)
            if (NOT Check_FIND_QUIETLY)
                message(STATUS "Could NOT find CHECK")
            endif (NOT Check_FIND_QUIETLY)
        endif (Check_FIND_REQUIRED)
    endif (CHECK_INCLUDE_DIR AND CHECK_LIBRARIES)
endif(NOT CHECK_FOUND)

# Hide advanced variables from CMake GUIs
MARK_AS_ADVANCED(CHECK_INCLUDE_DIR CHECK_LIBRARIES)