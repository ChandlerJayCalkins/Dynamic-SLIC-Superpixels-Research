/**
 * @file test_with_mri_features.cpp
 * @brief Test program for LTriDP feature extraction on real MRI samples
 *
 * This program processes MRI images from the input directory and extracts
 * LTriDP texture features, saving visualizations and feature maps.
 *
 * @author Ketsia Mbaku
 * 
 * Reference:
 *         Y. Wang, Q. Qi, and X. Shen, "Image Segmentation of Brain MRI Based on
 *         LTriDP and Superpixels of Improved SLIC," Brain Sciences, vol. 10, no. 2,
 *         p. 116, 2020.
 */

#include "feature_extraction.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;
using namespace ltridp_slic_improved;

/**
 * @brief Create a side-by-side comparison visualization
 * 
 * @param original Original grayscale MRI image
 * @param features LTriDP feature map (CV_8UC1)
 * @return Combined visualization image
 */
cv::Mat createComparison(const cv::Mat& original, const cv::Mat& features) {
    // Convert both to grayscale if needed
    cv::Mat grayOriginal;
    if (original.channels() == 3) {
        cv::cvtColor(original, grayOriginal, cv::COLOR_BGR2GRAY);
    } else {
        grayOriginal = original.clone();
    }
    
    // Apply colormap to features for better visualization
    cv::Mat colorFeatures;
    cv::applyColorMap(features, colorFeatures, cv::COLORMAP_JET);
    
    // Convert original to color for consistent display
    cv::Mat colorOriginal;
    cv::cvtColor(grayOriginal, colorOriginal, cv::COLOR_GRAY2BGR);
    
    // Combine side by side
    cv::Mat comparison;
    cv::hconcat(colorOriginal, colorFeatures, comparison);
    
    // Add labels
    cv::putText(comparison, "Original", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "LTriDP Features", cv::Point(colorOriginal.cols + 10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    
    return comparison;
}

/**
 * @brief Create a histogram of LTriDP feature codes
 * 
 * @param features LTriDP feature map
 * @return Histogram visualization (400x300)
 */
cv::Mat createHistogram(const cv::Mat& features) {
    // Compute histogram (256 bins for 0-255 codes)
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&features, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // Normalize histogram for display
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound((double)histWidth / histSize);
    
    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);
    
    // Draw histogram bars
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage, 
                 cv::Point(binWidth * (i - 1), histHeight - cvRound(hist.at<float>(i - 1))),
                 cv::Point(binWidth * i, histHeight - cvRound(hist.at<float>(i))),
                 cv::Scalar(255, 255, 255), 2);
    }
    
    // Add title
    cv::putText(histImage, "LTriDP Code Distribution", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    
    return histImage;
}

int main() {
    std::cout << "\n=== LTriDP Feature Extraction Test Program ===\n" << std::endl;
    
    // Setup paths
    fs::path inputDir = "../data/input";
    fs::path outputDir = "../data/output";
    
    // Create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }
    
    // Find all image files in input directory
    std::vector<fs::path> imageFiles;
    std::vector<std::string> extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
    
    if (fs::exists(inputDir)) {
        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    imageFiles.push_back(entry.path());
                }
            }
        }
    }
    
    std::cout << "Found " << imageFiles.size() << " images in " << inputDir << std::endl;
    
    if (imageFiles.empty()) {
        std::cout << "\nNo MRI images found!" << std::endl;
        std::cout << "Please add sample MRI images (PNG, JPG, etc.) to: " << inputDir << std::endl;
        return 1;
    }
    
    std::cout << "\nProcessing images..." << std::endl;
    
    FeatureExtractor extractor;
    int successCount = 0;
    int failCount = 0;
    
    for (const auto& imagePath : imageFiles) {
        std::string filename = imagePath.filename().string();
        std::string stem = imagePath.stem().string();
        std::string ext = imagePath.extension().string();
        
        std::cout << "Processing: " << filename << std::flush;
        
        // Load image
        cv::Mat input = cv::imread(imagePath.string(), cv::IMREAD_GRAYSCALE);
        if (input.empty()) {
            std::cout << "\n  Error: Could not load image" << std::endl;
            failCount++;
            continue;
        }
        
        // Extract features
        cv::Mat features;
        if (!extractor.extract(input, features)) {
            std::cout << "\n  Error: Feature extraction failed" << std::endl;
            failCount++;
            continue;
        }
        
        // Save feature map
        std::string featureMapPath = outputDir.string() + "/" + stem + "_features" + ext;
        cv::imwrite(featureMapPath, features);
        
        // Create and save colored feature visualization
        cv::Mat colorFeatures;
        cv::applyColorMap(features, colorFeatures, cv::COLORMAP_JET);
        std::string colorFeaturesPath = outputDir.string() + "/" + stem + "_features_color" + ext;
        cv::imwrite(colorFeaturesPath, colorFeatures);
        
        // Create and save side-by-side comparison
        cv::Mat comparison = createComparison(input, features);
        std::string comparisonPath = outputDir.string() + "/" + stem + "_comparison" + ext;
        cv::imwrite(comparisonPath, comparison);
        
        // Create and save histogram
        cv::Mat histogram = createHistogram(features);
        std::string histogramPath = outputDir.string() + "/" + stem + "_histogram" + ext;
        cv::imwrite(histogramPath, histogram);
        
        // Calculate statistics
        cv::Scalar mean, stddev;
        cv::meanStdDev(features, mean, stddev);
        double minVal, maxVal;
        cv::minMaxLoc(features, &minVal, &maxVal);
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n  Resolution: " << input.cols << "x" << input.rows;
        std::cout << "\n  Feature stats: min=" << minVal << ", max=" << maxVal 
                  << ", mean=" << mean[0] << ", stddev=" << stddev[0];
        std::cout << "\n  Saved: feature map, colored visualization, comparison, histogram";
        std::cout << std::endl;
        successCount++;
    }
    
    std::cout << "\n=== Processing Complete ===" << std::endl;
    std::cout << "  Successfully processed: " << successCount << " images" << std::endl;
    if (failCount > 0) {
        std::cout << "  Failed: " << failCount << " images" << std::endl;
    }
    std::cout << "  Output directory: " << outputDir << std::endl;
    std::cout << "\nFiles generated per image:" << std::endl;
    std::cout << "  *_features.png        - Raw LTriDP feature map (grayscale)" << std::endl;
    std::cout << "  *_features_color.png  - Colored feature visualization (jet colormap)" << std::endl;
    std::cout << "  *_comparison.png      - Side-by-side: original vs features" << std::endl;
    std::cout << "  *_histogram.png       - Distribution of LTriDP codes" << std::endl;
    std::cout << std::endl;
    
    return (successCount > 0) ? 0 : 1;
}
