/**
 * @file feature_extraction.hpp
 * @brief Local Tri-Directional Pattern (LTriDP) texture feature extraction module
 * This module provides texture feature extraction functionality for MRI images
 * using the LTriDP descriptor.
 * 
 * @author Ketsia Mbaku
 * 
 * Reference:
 *         Y. Wang, Q. Qi, and X. Shen, "Image Segmentation of Brain MRI Based on
 *         LTriDP and Superpixels of Improved SLIC," Brain Sciences, vol. 10, no. 2,
 *         p. 116, 2020.
 */

#ifndef FEATURE_EXTRACTION_HPP
#define FEATURE_EXTRACTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ltridp_slic_improved {

/**
 * @class FeatureExtractor
 * @brief Extracts Local Tri-Directional Pattern (LTriDP) texture features
 * 
 * Implements the LTriDP texture descriptor from paper Section 3.3.
 * The algorithm computes texture patterns based on:
 * - 8-directional gradients around each pixel
 * - Magnitude comparisons: M1 (center-based) vs M2 (neighbor-based)
 * - Binary encoding of magnitude relationships
 * See paper Section 3.3 for detailed algorithm description.
 */
class FeatureExtractor {
public:
    /**
     * @brief FeatureExtractor instantiates a new FeatureExtractor object
     * Pre-conditions:
     * - None
     * 
     * Post-conditions:
     * @post FeatureExtractor is ready to process images
     */
    FeatureExtractor();
    
    /**
     * @brief Extracts LTriDP texture features from an image
     * 
     * Computes the Local Tri-Directional Pattern descriptor for each pixel.
     * The output is a single-channel feature map with values [0, 255]
     * representing the LTriDP code for each pixel.
     * 
     * Parameters:
     * @param inputImage Input image (grayscale or color)
     * @param featureMap Output texture feature map (CV_8UC1, range [0, 255])
     * 
     * Return value:
     * @return true if successful, false otherwise
     * 
     * Pre-conditions:
     * @pre inputImage must be non-empty
     * @pre inputImage must be CV_8U type
     * @pre inputImage must have at least 3×3 pixels
     * 
     * Post-conditions:
     * @post featureMap contains LTriDP code for each pixel
     * @post featureMap has same dimensions as inputImage
     * @post Border pixels (1-pixel boundary) have undefined features
     */
    bool extract(const cv::Mat& inputImage, cv::Mat& featureMap);
    
private:
    unsigned char computeLTriDPCode(const float neighbors[9]) const;

    /**
     * @brief Extract 3×3 neighborhood gray values around a pixel
     * 
    * Parameters:
     * @param image Input image (CV_32F)
     * @param row center pixel row-coordinate
     * @param col center pixel column-coordinate
     * @param neighbors Output array of 9 gray values
     * 
     * Pre-conditions:
     * @pre row, col must be at least 1 pixel from image border
     * Post-conditions:
     * @post neighbors contains [g1, g2, ..., g8, gc] 8 gray neighbor values + center
     */
    void extractNeighborhood(const cv::Mat& image, int row, int col, float neighbors[9]) const;
};

} // namespace ltridp_slic_improved

#endif // FEATURE_EXTRACTION_HPP
