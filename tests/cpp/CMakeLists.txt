add_executable(lignum_tests
    test_layout.cpp
    test_lightgbm.cpp
    test_xgboost.cpp
    test_binary_io.cpp
    test_c_api.cpp
    test_categorical.cpp
    test_fuzz.cpp
    test_compiled.cpp
    test_arrow.cpp
    test_catboost.cpp
)

target_link_libraries(lignum_tests PRIVATE lignum GTest::gtest_main)
target_include_directories(lignum_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

include(GoogleTest)
# Discovery at test time with a generous timeout: sanitizer builds start slowly.
gtest_discover_tests(lignum_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 120)

# The AVX2 build of the CatBoost kernel must not leak AVX2 code into functions shared with the
# generic build (see src/catboost_kernel_avx2.cpp): checked on the linked C library, with MSVC on
# the AVX2 object file (dumpbin, next to the linker).
find_package(Python3 COMPONENTS Interpreter)
if(LIGNUM_CATBOOST_AVX2 AND MSVC AND Python3_Interpreter_FOUND)
    get_filename_component(_linker_dir "${CMAKE_LINKER}" DIRECTORY)
    find_program(LIGNUM_DUMPBIN dumpbin HINTS "${_linker_dir}")
    if(LIGNUM_DUMPBIN)
        add_test(NAME catboost_kernel_isa
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_kernel_isa.py --msvc ${LIGNUM_DUMPBIN}
                         "$<FILTER:$<TARGET_OBJECTS:lignum>,INCLUDE,catboost_kernel_avx2>")
    else()
        message(WARNING "dumpbin not found: the CatBoost AVX2 kernel's object is not checked.")
    endif()
elseif(LIGNUM_CATBOOST_AVX2 AND TARGET lignum_c AND Python3_Interpreter_FOUND)
    add_test(NAME catboost_kernel_isa
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_kernel_isa.py $<TARGET_FILE:lignum_c>)
endif()
