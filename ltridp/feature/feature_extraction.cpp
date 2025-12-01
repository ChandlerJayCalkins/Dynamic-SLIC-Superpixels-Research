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
    // TODO: Initialize attributes if needed
}

bool FeatureExtractor::extract(const cv::Mat& inputImage, cv::Mat& featureMap) {
    // Input validation
    if (inputImage.empty()) return false;
    if (inputImage.depth() != CV_8U) return false;
    if (inputImage.rows < 3 || inputImage.cols < 3) return false;
    
    // Convert to grayscale if image is in color
    cv::Mat grayImage;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = inputImage.clone();
    }
    
    cv::Mat floatImage;
    grayImage.convertTo(floatImage, CV_32F);
    featureMap = cv::Mat::zeros(floatImage.size(), CV_8U);
    int rows = floatImage.rows;
    int cols = floatImage.cols;
    
    for (int row = 1; row < rows - 1; ++row) {
        for (int col = 1; col < cols - 1; ++col) {
            // get 3x3 neighborhood
            float neighbors[9];
            extractNeighborhood(floatImage, row, col, neighbors);

            // compute and store LTriDP code
            unsigned char code = computeLTriDPCode(neighbors);
            featureMap.at<unsigned char>(row, col) = code;
        }
    }
    
    return true;
}

void FeatureExtractor::extractNeighborhood(const cv::Mat& image, int row, int col, float neighbors[9]) {
    /**
     * gc is the center pixel at (x,y)
     * gi are the neighbors indexed clockwise from right:
     *     g6  g7  g8
     *     g5  gc  g1
     *     g4  g3  g2
     */

    neighbors[0] = image.at<float>(row,   col+1);  
    neighbors[1] = image.at<float>(row+1, col+1); 
    neighbors[2] = image.at<float>(row+1, col);  
    neighbors[3] = image.at<float>(row+1, col-1);
    neighbors[4] = image.at<float>(row,   col-1); 
    neighbors[5] = image.at<float>(row-1, col-1); 
    neighbors[6] = image.at<float>(row-1, col);
    neighbors[7] = image.at<float>(row-1, col+1);
    neighbors[8] = image.at<float>(row,   col);
}

unsigned char FeatureExtractor::computeLTriDPCode(const float neighbors[9]) const {
   // TODO: Implement LTriDP code computation
   return 0;
}

} // namespace ltridp_slic_improved
