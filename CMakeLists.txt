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
    src/catboost.cpp
    src/c_api.cpp
    src/compiled.cpp
    src/io.cpp
    src/load.cpp
    src/model.cpp
    src/parsers/lightgbm.cpp
    src/parsers/lightgbm_text.cpp
    src/parsers/ubjson.cpp
    src/parsers/xgboost.cpp
    src/validate.cpp
    extern/simdjson/simdjson.cpp
)

# x86-64: the CatBoost block kernel also compiled for AVX2, chosen at run time by cpuid. GCC and
# Clang switch the target inside catboost_kernel_avx2.cpp (the build stays generic); MSVC has no
# per-function target, so that one file is compiled with /arch:AVX2, and without /GL: link-time
# code generation could inline its functions into generic callers.
if(CMAKE_OSX_ARCHITECTURES)  # macOS cross builds (e.g. x86_64 on Apple silicon): the target decides
    string(COMPARE EQUAL "${CMAKE_OSX_ARCHITECTURES}" "x86_64" LIGNUM_TARGET_X86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(LIGNUM_TARGET_X86_64 ON)
endif()
if(LIGNUM_TARGET_X86_64)
    target_sources(lignum PRIVATE src/catboost_kernel_avx2.cpp)
    target_compile_definitions(lignum PRIVATE LIGNUM_CATBOOST_AVX2)
    if(MSVC)
        set_source_files_properties(src/catboost_kernel_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/GL-")
    endif()
    set(LIGNUM_CATBOOST_AVX2 ON)
endif()

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
    # SKBUILD_SABI_COMPONENT is set when scikit-build-core builds an abi3 wheel (wheel.py-api).
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
    find_package(nanobind CONFIG REQUIRED)

    # STABLE_ABI: one wheel for every Python >= 3.12 (ignored for older Pythons).
    nanobind_add_module(_core STABLE_ABI python/bindings.cpp)
    target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(_core PRIVATE lignum)
    install(TARGETS _core DESTINATION lignum)

    # macOS wheels: LightGBM and XGBoost use Homebrew's libomp, and two copies of libomp in one
    # process abort (OMP Error #15). So the extension looks for Homebrew's first and falls back to
    # a copy shipped in the wheel (lignum/.dylibs), found through the same @rpath install name.
    option(LIGNUM_BUNDLE_OPENMP "macOS: ship libomp in the wheel as a fallback for Homebrew's" OFF)
    if(APPLE AND LIGNUM_BUNDLE_OPENMP AND OpenMP_CXX_FOUND)
        set_target_properties(_core PROPERTIES
            BUILD_WITH_INSTALL_RPATH ON
            INSTALL_RPATH "/opt/homebrew/opt/libomp/lib;/usr/local/opt/libomp/lib;@loader_path/.dylibs")
        find_library(LIGNUM_LIBOMP NAMES omp PATHS ${OpenMP_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
        get_filename_component(LIGNUM_LIBOMP_REAL ${LIGNUM_LIBOMP} REALPATH)
        install(FILES ${LIGNUM_LIBOMP_REAL} DESTINATION lignum/.dylibs RENAME libomp.dylib)
        # Reference libomp as @rpath/libomp.dylib (Homebrew's own install name is an absolute path).
        add_custom_command(TARGET _core POST_BUILD
            COMMAND install_name_tool -change ${LIGNUM_LIBOMP_REAL} @rpath/libomp.dylib $<TARGET_FILE:_core>
            COMMAND install_name_tool -change ${LIGNUM_LIBOMP} @rpath/libomp.dylib $<TARGET_FILE:_core>
            COMMAND install_name_tool -change /opt/homebrew/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib $<TARGET_FILE:_core>
            COMMAND install_name_tool -change /usr/local/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib $<TARGET_FILE:_core>
            VERBATIM)
    endif()
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
