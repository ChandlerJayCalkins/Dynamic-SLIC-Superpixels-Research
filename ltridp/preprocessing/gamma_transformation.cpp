/**
 * @file gamma_transformation.cpp
 * @brief Gamma correction utility for MRI preprocessing pipeline
 *
 * This file applies the gamma transformation step described in Section 3.2 of the paper
 * 
 * @author Ketsia Mbaku
 *
 * Reference:
 *         Y. Wang, Q. Qi, and X. Shen, "Image Segmentation of Brain MRI Based on
 *         LTriDP and Superpixels of Improved SLIC," Brain Sciences, vol. 10, no. 2,
 *         p. 116, 2020.
 */

#include "preprocessing.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace cv;

namespace ltridp_slic_improved {

void Preprocessor::applyGammaTransformation(const Mat& input, Mat& output, double gamma) {
    /*
     * I'(x,y) = 255 * (I(x,y)/255)^γ 
     * with γ = 0.5.
     */
    Mat lookupTable(1, 256, CV_8U);
    uchar* lutPtr = lookupTable.ptr<uchar>();
    for (int intensity = 0; intensity < 256; ++intensity) {
        const double normalized = static_cast<double>(intensity) / 255.0;
        const double corrected = std::pow(normalized, gamma);
        lutPtr[intensity] = saturate_cast<uchar>(corrected * 255.0);
    }

    LUT(input, lookupTable, output);
}

}  // namespace ltridp_slic_improved
