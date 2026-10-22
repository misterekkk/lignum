cmake_minimum_required(VERSION 3.18...3.31)
project(lignum LANGUAGES CXX)

option(LIGNUM_BUILD_TESTS "Build the GoogleTest suite" OFF)
option(LIGNUM_SANITIZE "Build with sanitizers (tests, not releases)" OFF)
set(LIGNUM_SANITIZERS "address,undefined" CACHE STRING "Sanitizers for LIGNUM_SANITIZE (ASan hangs on some macOS versions: use undefined there)")
option(LIGNUM_NATIVE "Tune for the build machine (-march/-mcpu=native). Never enable for distributed wheels." OFF)
if(SKBUILD)
    set(_lignum_c_default OFF)
else()
    set(_lignum_c_default ON)
endif()
option(LIGNUM_BUILD_C_LIBRARY "Build the shared library exposing the C API (include/lignum/c_api.h)" ${_lignum_c_default})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(LIGNUM_SANITIZE AND NOT MSVC)
    add_compile_options(-fsanitize=${LIGNUM_SANITIZERS} -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=${LIGNUM_SANITIZERS})
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(lignum STATIC
    src/arrow.cpp
    src/binned.cpp
    src/builder.cpp
    src/c_api.cpp
    src/compiled.cpp
    src/io.cpp
    src/load.cpp
    src/model.cpp
    src/parsers/lightgbm.cpp
    src/parsers/xgboost.cpp
    src/validate.cpp
    extern/simdjson/simdjson.cpp
)

# Only the C API is exported from shared objects; C++ internals stay hidden. The static archive gets
# its own file name: on Windows the shared library's import library is lignum.lib.
set_target_properties(lignum PROPERTIES OUTPUT_NAME lignum_static CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

target_include_directories(lignum
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/extern/simdjson
)

# Homebrew's libomp is keg-only, so AppleClang cannot find it without a hint.
if(APPLE AND NOT OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT})
    foreach(prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if(EXISTS ${prefix}/include/omp.h)
            set(OpenMP_ROOT ${prefix})
            break()
        endif()
    endforeach()
endif()

target_link_libraries(lignum PRIVATE ${CMAKE_DL_LIBS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(lignum PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found: lignum will predict single-threaded.")
endif()

# Results must match LightGBM/XGBoost bit for bit: no fast-math, no FMA contraction.
if(MSVC)
    target_compile_options(lignum PRIVATE /fp:precise)
else()
    target_compile_options(lignum PRIVATE -ffp-contract=off -fno-math-errno -funroll-loops)
    if(LIGNUM_NATIVE)
        if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            target_compile_options(lignum PRIVATE -mcpu=native)
        else()
            target_compile_options(lignum PRIVATE -march=native)
        endif()
    endif()
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT LIGNUM_IPO_SUPPORTED OUTPUT _ipo_msg LANGUAGES CXX)
if(LIGNUM_IPO_SUPPORTED)
    set_property(TARGET lignum PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ---------------------------------------------------------------------------
# Shared library with the C API
# ---------------------------------------------------------------------------
if(LIGNUM_BUILD_C_LIBRARY)
    add_library(lignum_c SHARED src/c_api.cpp)
    set_target_properties(lignum_c PROPERTIES OUTPUT_NAME lignum CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    target_compile_definitions(lignum_c PRIVATE LIGNUM_EXPORTS PUBLIC LIGNUM_SHARED)
    target_link_libraries(lignum_c PRIVATE lignum)
    target_include_directories(lignum_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

    include(GNUInstallDirs)
    install(TARGETS lignum_c LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES include/lignum/c_api.h include/lignum/arrow.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lignum)
endif()

# ---------------------------------------------------------------------------
# Python extension (built by scikit-build-core)
# ---------------------------------------------------------------------------
if(SKBUILD)
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core python/bindings.cpp)
    target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(_core PRIVATE lignum)
    install(TARGETS _core DESTINATION lignum)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(LIGNUM_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(tests/cpp)
endif()
