cmake_minimum_required(VERSION 3.18...3.27)
project(lignum_python LANGUAGES CXX)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core lignum_core.cpp)

target_include_directories(_core PRIVATE 
    ../include
    ../extern/simdjson
)

target_sources(_core PRIVATE
    ../src_core/model.cpp
    ../src_core/io.cpp
    ../src_core/load.cpp
    ../src_core/builder.cpp
    ../src_core/parser_lightgbm.cpp
    ../src_core/parser_xgboost.cpp
    ../extern/simdjson/simdjson.cpp
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(_core PRIVATE /O2 /DNDEBUG /fp:fast)
else()
    target_compile_options(_core PRIVATE -O3 -DNDEBUG -flto -funroll-loops -ffp-contract=fast -fno-math-errno)
    
    target_link_options(_core PRIVATE -flto)

    # target_compile_options(_core PRIVATE -march=native)
endif()

install(TARGETS _core DESTINATION lignum)