/**
 * @file preprocessing.hpp
 * @brief brain MRI image preprocessing module interface
 * 
 * This module provides preprocessing functionality for brain MRI images,
 * including 3D histogram reconstruction and gamma transformation to enhance
 * image quality before superpixel segmentation.
 * 
 * @author Ketsia Mbaku
 * 
 * Reference:
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
 * See paper Section 3.1 for details.
 */
class Preprocessor {
public:
    /**
     * @brief Constructor with default parameters
     * @post Preprocessor is ready to process images
     */
    Preprocessor();
    
    /**
     * @brief enhance applies 3D histogram reconstruction followed by gamma correction
     * to reduce intensity non-uniformity in MRI images.
     * 
     * @param inputImage Input MRI image (grayscale)
     * @param outputImage Enhanced output image
     * @param gamma Gamma correction parameter (default: 0.5 per paper Section 3.2)

     * @return true if successful, false otherwise
     * 
     * @pre inputImage must be non-empty
     * @pre inputImage must be CV_8U
     * @pre gamma > 0
     * 
     * @post outputImage contains preprocessed image with same dimensions as input
     */
    bool enhance(const cv::Mat& inputImage, 
                cv::Mat& outputImage, 
                double gamma = 0.5);
    
private:
    /**
     * @brief Apply 3D histogram reconstruction from paper Section 3.1
     * 
     * Uses three statistical measures per pixel to create a 3D histogram:
     * - f(x,y): actual gray value
     * - g(x,y): mean of 3×3 neighborhood
     * - h(x,y): median of 3×3 neighborhood
     * Corrects pixels based on their deviation from the diagonal in 3D space
     * 
     * @param input Input image (grayscale)
     * @param output Reconstructed output
     * 
     * @pre input must be non-empty CV_8U type
     * @post output has same dimensions and channels as input
     */
    void apply3DHistogramReconstruction(const cv::Mat& input, 
                                       cv::Mat& output);
    
    /**
     * applyGammaTransformation Apply gamma transformation (paper Section 3.2)
     * 
     * Performs point-wise gamma correction using:
     *   output(x,y) = 255 * ( input(x,y) / 255 )^gamma
     * where gamma controls brightness/contrast of the MRI slice.
     * 
     * Parameters:
     * input Input grayscale image
     * output Gamma-adjusted output (same size/type as input)
     * Gamma exponent (we will use 0.5 for experimentation like in paper)
     *
     * Preconditions:
     * input must be non-empty CV_8U
     * gamma > 0.0
     * 
     * Postconditions:
     * output contains gamma-corrected intensities
     */
    void applyGammaTransformation(const cv::Mat& input,
                                  cv::Mat& output,
                                  double gamma);
};

}

#endif
