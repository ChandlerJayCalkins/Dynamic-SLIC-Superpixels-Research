/**
 * @file preprocessor.cpp
 * @brief Preprocessor class implementation - orchestrates preprocessing pipeline
 *
 * @author Ketsia Mbaku
 * 
 * Reference:
 *         Y. Wang, Q. Qi, and X. Shen, "Image Segmentation of Brain MRI Based on
 *         LTriDP and Superpixels of Improved SLIC," Brain Sciences, vol. 10, no. 2,
 *         p. 116, 2020.
 */

#include "preprocessing.hpp"

using namespace cv;

namespace ltridp_slic_improved {

Preprocessor::Preprocessor() {
    // TODO: Initialize parameters later if needed
}

bool Preprocessor::enhance(const Mat& inputImage, Mat& outputImage, double gamma) {
    // Input validation - return false for invalid inputs
    if (inputImage.empty()) {
        return false;
    }
    
    if (inputImage.depth() != CV_8U) {
        return false;
    }
    
    if (gamma <= 0.0) {
        return false;
    }
    
    Mat reconstructed;
    apply3DHistogramReconstruction(inputImage, reconstructed);
    applyGammaTransformation(reconstructed, outputImage, gamma);
    
    return true;
}

}  // namespace ltridp_slic_improved
