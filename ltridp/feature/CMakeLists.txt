# Feature extraction module
# Implements Local Tri-Directional Pattern (LTriDP) texture descriptor

# Source files
set(FEATURE_SOURCES
    feature_extraction.cpp
)

# Create library
add_library(feature ${FEATURE_SOURCES})

# Link OpenCV
target_link_libraries(feature ${OpenCV_LIBS})

# Include directories
target_include_directories(feature PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
