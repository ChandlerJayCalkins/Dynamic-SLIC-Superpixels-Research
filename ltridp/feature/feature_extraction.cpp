/**
 * @file feature_extraction.cpp
 * @brief Implementation of LTriDP texture feature extraction
 *
 * @author Ketsia Mbaku
 * 
 * Reference:
 *         Y. Wang, Q. Qi, and X. Shen, "Image Segmentation of Brain MRI Based on
 *         LTriDP and Superpixels of Improved SLIC," Brain Sciences, vol. 10, no. 2,
 *         p. 116, 2020.
 */

#include "feature_extraction.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace ltridp_slic_improved {

FeatureExtractor::FeatureExtractor() {
    // Default constructor
}

bool FeatureExtractor::extract(const cv::Mat& inputImage, cv::Mat& featureMap) {
    // Input validation
    if (inputImage.empty()) return false;
    if (inputImage.depth() != CV_8U) return false;
    if (inputImage.rows < 3 || inputImage.cols < 3) return false;
    
    // Convert to grayscale if needed
    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = inputImage.clone();
    }
    
    // Convert to float for precise computation
    cv::Mat floatImage;
    grayImage.convertTo(floatImage, CV_32F);
    
    // Initialize output feature map
    featureMap = cv::Mat::zeros(floatImage.size(), CV_8U);
    
    int rows = floatImage.rows;
    int cols = floatImage.cols;
    
    // Process each pixel (excluding 1-pixel border)
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            // Extract 3×3 neighborhood
            float neighbors[9];
            extractNeighborhood(floatImage, x, y, neighbors);
            
            // Compute LTriDP code
            unsigned char code = computeLTriDPCode(neighbors);
            
            // Store in feature map
            featureMap.at<unsigned char>(y, x) = code;
        }
    }
    
    return true;
}

void FeatureExtractor::extractNeighborhood(const cv::Mat& image, int x, int y, 
                                          float neighbors[9]) const {
    /**
     * Extract neighbors in clockwise order starting from right (East):
     * 
     *     g6  g7  g8
     *     g5  gc  g1
     *     g4  g3  g2
     * 
     * Array layout: [g1, g2, g3, g4, g5, g6, g7, g8, gc]
     * Indices:      [0,  1,  2,  3,  4,  5,  6,  7,  8]
     */
    
    neighbors[0] = image.at<float>(y,   x+1);  // g1 - Right (East)
    neighbors[1] = image.at<float>(y+1, x+1);  // g2 - Bottom-right (Southeast)
    neighbors[2] = image.at<float>(y+1, x);    // g3 - Bottom (South)
    neighbors[3] = image.at<float>(y+1, x-1);  // g4 - Bottom-left (Southwest)
    neighbors[4] = image.at<float>(y,   x-1);  // g5 - Left (West)
    neighbors[5] = image.at<float>(y-1, x-1);  // g6 - Top-left (Northwest)
    neighbors[6] = image.at<float>(y-1, x);    // g7 - Top (North)
    neighbors[7] = image.at<float>(y-1, x+1);  // g8 - Top-right (Northeast)
    neighbors[8] = image.at<float>(y,   x);    // gc - Center
}

unsigned char FeatureExtractor::computeLTriDPCode(const float neighbors[9]) const {
    /**
     * Compute LTriDP magnitude code using paper equations (6), (7), (8).
     * 
     * For each neighbor i (1-based indexing in paper, 0-based in array):
     * - Compute M1: magnitude based on center pixel gc
     * - Compute M2: magnitude based on current neighbor gi
     * - Set bit (i-1) if M1 >= M2
     * 
     * Array indices: neighbors[0..7] = g1..g8, neighbors[8] = gc
     */
    
    const float gc = neighbors[8];  // Center pixel
    unsigned char code = 0;
    
    // Process each of the 8 neighbors
    for (int i = 0; i < 8; ++i) {
        const float gi = neighbors[i];  // Current neighbor (gi+1 in 1-based paper indexing)
        
        float M1, M2;
        
        if (i == 0) {
            // i=1 in paper (g1 - first neighbor, right/East)
            // M1 = sqrt((g8 - gc)² + (g2 - gc)²)
            // M2 = sqrt((g8 - g1)² + (g2 - g1)²)
            const float g8 = neighbors[7];  // Previous neighbor (wraps around)
            const float g2 = neighbors[1];  // Next neighbor
            
            const float diff_8c = g8 - gc;
            const float diff_2c = g2 - gc;
            M1 = std::sqrt(diff_8c * diff_8c + diff_2c * diff_2c);
            
            const float diff_81 = g8 - gi;
            const float diff_21 = g2 - gi;
            M2 = std::sqrt(diff_81 * diff_81 + diff_21 * diff_21);
            
        } else if (i == 7) {
            // i=8 in paper (g8 - last neighbor, top-right/Northeast)
            // M1 = sqrt((g7 - gc)² + (g1 - gc)²)
            // M2 = sqrt((g7 - g8)² + (g1 - g8)²)
            const float g7 = neighbors[6];  // Previous neighbor
            const float g1 = neighbors[0];  // Next neighbor (wraps around)
            
            const float diff_7c = g7 - gc;
            const float diff_1c = g1 - gc;
            M1 = std::sqrt(diff_7c * diff_7c + diff_1c * diff_1c);
            
            const float diff_78 = g7 - gi;
            const float diff_18 = g1 - gi;
            M2 = std::sqrt(diff_78 * diff_78 + diff_18 * diff_18);
            
        } else {
            // i=2-7 in paper (g2-g7 - middle neighbors)
            // M1 = sqrt((g(i-1) - gc)² + (g(i+1) - gc)²)
            // M2 = sqrt((g(i-1) - gi)² + (g(i+1) - gi)²)
            const float g_prev = neighbors[i - 1];  // Previous neighbor
            const float g_next = neighbors[i + 1];  // Next neighbor
            
            const float diff_prevc = g_prev - gc;
            const float diff_nextc = g_next - gc;
            M1 = std::sqrt(diff_prevc * diff_prevc + diff_nextc * diff_nextc);
            
            const float diff_previ = g_prev - gi;
            const float diff_nexti = g_next - gi;
            M2 = std::sqrt(diff_previ * diff_previ + diff_nexti * diff_nexti);
        }
        
        // Set bit i if M1 >= M2
        if (M1 >= M2) {
            code |= (1 << i);
        }
    }
    
    return code;
}

} // namespace ltridp_slic_improved
