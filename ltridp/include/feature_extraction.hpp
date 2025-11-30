/**
 * @file feature_extraction.hpp
 * @brief Local Tri-Directional Pattern (LTriDP) texture feature extraction module
 * This module provides texture feature extraction functionality for MRI images
 * using the LTriDP descriptor, which captures directional texture patterns for
 * improved superpixel segmentation.
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
 * LTriDP extends Local Binary Pattern (LBP) by capturing neighbor-to-neighbor
 * relationships in addition to center-neighbor comparisons, providing more
 * discriminative texture features for medical image segmentation.
 * 
 * The algorithm computes texture patterns based on:
 * - 8-directional gradients around each pixel
 * - Magnitude comparisons: M1 (center-based) vs M2 (neighbor-based)
 * - Binary encoding of directional relationships
 * 
 * Neighbor indexing (1-based, clockwise from right):
 * 
 *     g6  g7  g8
 *     g5  gc  g1
 *     g4  g3  g2
 * 
 * See paper Section 3.3 for detailed algorithm description.
 */
class FeatureExtractor {
public:
    /**
     * @brief Constructor with default parameters
     * @post FeatureExtractor is ready to process images
     */
    FeatureExtractor();
    
    /**
     * @brief Extract LTriDP texture features from an image
     * 
     * Computes the Local Tri-Directional Pattern descriptor for each pixel
     * by analyzing magnitude relationships in 8 neighborhood directions.
     * The output is a single-channel feature map with values [0, 255]
     * representing the LTriDP code for each pixel.
     * 
     * @param inputImage Input image (grayscale or color)
     * @param featureMap Output texture feature map (CV_8UC1, range [0, 255])
     * 
     * @return true if successful, false otherwise
     * 
     * @pre inputImage must be non-empty
     * @pre inputImage must be CV_8U type
     * @pre inputImage must have at least 3×3 pixels
     * 
     * @post featureMap contains LTriDP code for each pixel
     * @post featureMap has same dimensions as inputImage
     * @post Border pixels (1-pixel boundary) have undefined features
     */
    bool extract(const cv::Mat& inputImage, cv::Mat& featureMap);
    
private:
    /**
     * @brief Compute LTriDP magnitude code for a single pixel
     * 
     * Implements equations (6), (7), and (8) from paper Section 3.3.
     * For each of the 8 neighbors:
     * - Computes M1: magnitude involving center pixel gc
     * - Computes M2: magnitude involving current neighbor gi
     * - Sets bit i if M1 >= M2
     * 
     * Neighbor indexing (1-based, clockwise from right):
     *     g6  g7  g8
     *     g5  gc  g1
     *     g4  g3  g2
     * 
     * For i=1: M1 = sqrt((g8-gc)² + (g2-gc)²), M2 = sqrt((g8-g1)² + (g2-g1)²)
     * For i=2-7: M1 = sqrt((g(i-1)-gc)² + (g(i+1)-gc)²), M2 = sqrt((g(i-1)-gi)² + (g(i+1)-gi)²)
     * For i=8: M1 = sqrt((g7-gc)² + (g1-gc)²), M2 = sqrt((g7-g8)² + (g1-g8)²)
     * 
     * @param neighbors Array of 9 gray values: [g1, g2, ..., g8, gc]
     * @return 8-bit LTriDP code (0-255)
     * 
     * @pre neighbors must contain exactly 9 values
     * @post Returns binary pattern encoding magnitude relationships
     */
    unsigned char computeLTriDPCode(const float neighbors[9]) const;
    
    /**
     * @brief Extract 3×3 neighborhood gray values around a pixel
     * 
     * Extracts the 8 neighbors plus center pixel in the order:
     * [g1, g2, g3, g4, g5, g6, g7, g8, gc]
     * 
     * Where neighbors are indexed clockwise starting from right (east):
     *     g6  g7  g8
     *     g5  gc  g1
     *     g4  g3  g2
     * 
     * @param image Input image (CV_32F)
     * @param x Pixel x-coordinate
     * @param y Pixel y-coordinate
     * @param neighbors Output array of 9 gray values
     * 
     * @pre x, y must be at least 1 pixel from image border
     * @pre neighbors array must have space for 9 floats
     * @post neighbors contains [g1, g2, ..., g8, gc]
     */
    void extractNeighborhood(const cv::Mat& image, int x, int y, 
                            float neighbors[9]) const;
};

} // namespace ltridp_slic_improved

#endif // FEATURE_EXTRACTION_HPP
