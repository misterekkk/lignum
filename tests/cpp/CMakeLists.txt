add_executable(lignum_tests
    test_layout.cpp
    test_lightgbm.cpp
    test_xgboost.cpp
    test_binary_io.cpp
    test_c_api.cpp
    test_categorical.cpp
    test_fuzz.cpp
    test_compiled.cpp
)

target_link_libraries(lignum_tests PRIVATE lignum GTest::gtest_main)
target_include_directories(lignum_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

include(GoogleTest)
# Discovery at test time with a generous timeout: sanitizer builds start slowly.
gtest_discover_tests(lignum_tests DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 120)
