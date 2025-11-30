#include "feature_extraction.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Create small checkerboard
    cv::Mat checkerboard(10, 10, CV_8U);
    for (int r = 0; r < 10; ++r) {
        for (int c = 0; c < 10; ++c) {
            checkerboard.at<uchar>(r, c) = ((r/2 + c/2) % 2) * 255;
        }
    }
    
    std::cout << "Input checkerboard (10x10):" << std::endl;
    for (int r = 0; r < 10; ++r) {
        for (int c = 0; c < 10; ++c) {
            std::cout << (int)checkerboard.at<uchar>(r, c) << " ";
        }
        std::cout << std::endl;
    }
    
    ltridp_slic_improved::FeatureExtractor extractor;
    cv::Mat features;
    extractor.extract(checkerboard, features);
    
    std::cout << "\nLTriDP features:" << std::endl;
    for (int r = 0; r < 10; ++r) {
        for (int c = 0; c < 10; ++c) {
            std::cout << (int)features.at<uchar>(r, c) << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
