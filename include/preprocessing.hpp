/**
 * @file preprocessing.hpp
 * @brief brain MRI image preprocessing module interface
 * 
 * This module provides preprocessing functionality for brain MRI images,
 * including 3D histogram reconstruction and gamma transformation to enhance
 * image quality before superpixel segmentation.
 * 
 * @author Ketsia Mbaku
 *         Y. Wang, Q. Qi, and X. Shen, "Image Segmentation of Brain MRI Based on
 *         LTriDP and Superpixels of Improved SLIC," Brain Sciences, vol. 10, no. 2,
 *         p. 116, 2020.
 */

#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <opencv2/opencv.hpp>

namespace ltridp_slic_improved {

/**
 * @class Preprocessor
 * @brief Handles MRI image preprocessing and enhancement
 * 
 * Implements 3D histogram reconstruction and gamma transformation
 * to improve image quality before superpixel segmentation.
 * 
 * The preprocessing pipeline reduces intensity non-uniformity common
 * in MRI images and enhances contrast for better segmentation results.
 * This is the first step in the LTriDP improved SLIC superpixel with segmentation process.
 * 
 */
class Preprocessor {
public:
    /**
     * @brief Constructor with default parameters
     * 
     * Initializes the Preprocessor with default settings suitable
     * for general MRI image enhancement.
     * 
     * @post Preprocessor is ready to process images
     */
    Preprocessor();
    
    /**
     * @brief Enhanced preprocessing pipeline
     * 
     * Applies 3D histogram reconstruction followed by gamma correction
     * to reduce intensity non-uniformity in MRI images.
     * 
     * Processing steps:
     * 1. Apply 3D histogram reconstruction (f, g, h triple-based correction)
     * 2. Apply gamma transformation for brightness adjustment
     * 
     * @param inputImage Input MRI image (grayscale)
     * @param outputImage Enhanced output image (same format as input)
     * @param gamma Gamma correction parameter (default: 1.0, range: 0.5-2.0)
     *              - gamma < 1.0: brightens image
     *              - gamma = 1.0: no change
     *              - gamma > 1.0: darkens image
     * 
     * @return true if successful, false otherwise
     * 
     * @pre inputImage must be non-empty
     * @pre inputImage must be CV_8U type (8-bit unsigned)
     * @pre gamma must be > 0
     * 
     * @post outputImage contains enhanced image with same dimensions and type as input
     * @post inputImage is not modified
     */
    bool enhance(const cv::Mat& inputImage, 
                cv::Mat& outputImage, 
                double gamma = 1.0);
    
private:
    /**
     * @brief Apply 3D histogram reconstruction from paper Section 3.1
     * 
     * Uses three statistical measures per pixel to create a 3D histogram:
     * - f(x,y): actual gray value
     * - g(x,y): mean of 3×3 neighborhood
     * - h(x,y): median of 3×3 neighborhood
     * 
     * Corrects pixels based on their deviation from the diagonal in 3D space
     * using region-specific correction rules (8 regions).
     * 
     * @param input Input image (BGR or grayscale)
     * @param output Reconstructed output (same format as input)
     * 
     * @pre input must be non-empty CV_8U type
     * @post output has same dimensions and channels as input
     */
    void apply3DHistogramReconstruction(const cv::Mat& input, 
                                       cv::Mat& output);
};

}

#endif
