# Preprocessing module
# Implements 3D histogram reconstruction and gamma transformation

# Source files
set(PREPROCESSING_SOURCES
    histogram_reconstruction.cpp
    gamma_transformation.cpp
)

# Create library
add_library(preprocessing ${PREPROCESSING_SOURCES})

# Link OpenCV
target_link_libraries(preprocessing ${OpenCV_LIBS})

# Include directories
target_include_directories(preprocessing PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
