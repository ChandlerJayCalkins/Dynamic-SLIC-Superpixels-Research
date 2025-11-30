# Tests CMakeLists.txt
# Unit tests and sample testing programs

cmake_minimum_required(VERSION 3.10)

# Find Google Test (optional, if not found, unit tests will be skipped)
find_package(GTest)

if(GTest_FOUND)
    message(STATUS "Google Test found - building unit tests")
    
    include_directories(${GTEST_INCLUDE_DIRS})
    
    # Preprocessing unit tests
    add_executable(test_preprocessing test_preprocessing.cpp)
    target_link_libraries(test_preprocessing 
        preprocessing 
        ${OpenCV_LIBS}
        GTest::gtest 
        GTest::gtest_main
    )
    add_test(NAME PreprocessingUnitTests COMMAND test_preprocessing)
    
    # Feature extraction unit tests
    add_executable(test_feature_extraction test_feature_extraction.cpp)
    target_link_libraries(test_feature_extraction 
        feature 
        ${OpenCV_LIBS}
        GTest::gtest 
        GTest::gtest_main
    )
    add_test(NAME FeatureExtractionUnitTests COMMAND test_feature_extraction)
else()
    message(STATUS "Google Test not found - skipping unit tests")
    message(STATUS "  Install with: brew install googletest (macOS) or apt-get install libgtest-dev (Linux)")
endif()

# MRI sample testing programs
add_executable(test_with_mri_samples test_with_mri_samples.cpp)
target_link_libraries(test_with_mri_samples 
    preprocessing 
    ${OpenCV_LIBS}
)

add_executable(test_with_mri_features test_with_mri_features.cpp)
target_link_libraries(test_with_mri_features 
    feature 
    ${OpenCV_LIBS}
)

# Print test configuration
if(GTest_FOUND)
    message(STATUS "Tests configured:")
    message(STATUS "  test_preprocessing: Preprocessing unit tests (23 tests)")
    message(STATUS "  test_feature_extraction: Feature extraction unit tests (21 tests)")
    message(STATUS "  test_with_mri_samples: MRI preprocessing tool")
    message(STATUS "  test_with_mri_features: MRI feature extraction tool")
else()
    message(STATUS "Tests configured:")
    message(STATUS "  test_with_mri_samples: MRI preprocessing tool")
    message(STATUS "  test_with_mri_features: MRI feature extraction tool")
    message(STATUS "  (unit tests skipped - GTest not found)")
endif()