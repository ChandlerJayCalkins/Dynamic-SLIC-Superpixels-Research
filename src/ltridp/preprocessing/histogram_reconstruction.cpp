/**
 * @file histogram_reconstruction.cpp
 * @brief Implementation of 3D histogram reconstruction for MRI images
 * 
 * This file implements the 3D histogram reconstruction method described in
 * "Image Segmentation of Brain MRI Based on LTriDP and Superpixels of Improved SLIC"
 * (Brain Sciences, 2020). The method uses three statistical measures per pixel
 * (gray value, mean, median) to create a 3D histogram and correct pixels based
 * on their deviation from the diagonal.
 * 
 * Reference: Section 3.1 of the paper, based on reference [19]
 * 
 * @author Ketsia Ambaku
 *         Y. Wang, Q. Qi, and X. Shen, "Image Segmentation of Brain MRI Based on
 *         LTriDP and Superpixels of Improved SLIC," Brain Sciences, vol. 10, no. 2,
 *         p. 116, 2020.
 */

#include "preprocessing.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace {

enum class RegionGroup {
    GROUP_0_1,
    GROUP_2_3,
    GROUP_4_5,
    GROUP_6_7
};

/**
 * @brief 
 * Classify (f, g, h) into the eight histogram regions described in Section 3.1
 * 
 *
 * The paper describes the regions by examining which pair among (f, g, h)
 * remains closest to the diagonal. We emulate that behavior by treating the
 * remaining value as the outlier and mapping it to the corresponding region
 * group. A small tolerance lets us collapse near-ties back to GROUP_0_1.
 */
RegionGroup classifyRegionGroup(float grayValue,
                                float localMean,
                                float localMedian,
                                float tieTolerance) {
    const float distanceFG = std::abs(grayValue - localMean);
    const float distanceFH = std::abs(grayValue - localMedian);
    const float distanceGH = std::abs(localMean - localMedian);

    if (distanceFG > distanceGH + tieTolerance &&
        distanceFH > distanceGH + tieTolerance) {
        return RegionGroup::GROUP_2_3;  // g and h stay close, f is outlier
    }

    if (distanceFG > distanceFH + tieTolerance &&
        distanceGH > distanceFH + tieTolerance) {
        return RegionGroup::GROUP_4_5;  // f and h stay close, g is outlier
    }

    if (distanceFH > distanceFG + tieTolerance &&
        distanceGH > distanceFG + tieTolerance) {
        return RegionGroup::GROUP_6_7;  // f and g stay close, h is outlier
    }

    return RegionGroup::GROUP_0_1;  // all mutually close → near-diagonal
}

}  // namespace

namespace ltridp_slic_improved {

void Preprocessor::apply3DHistogramReconstruction(const cv::Mat& input, 
                                                  cv::Mat& output) {
    /**
     * 3D Histogram Reconstruction Algorithm (from paper Section 3.1)
     * 
     * Purpose: Reconstruct gray values of medical images to reduce noise
     * and intensity non-uniformity caused by imaging equipment limitations.
     * 
     * The "3D" refers to three values per pixel:
     * - f(x,y): actual gray value
     * - g(x,y): mean of 3×3 neighborhood
     * - h(x,y): median of 3×3 neighborhood
     * 
     * For uniform images, these triples (f, g, h) should lie along the
     * diagonal of a 3D histogram. Pixels that deviate are corrected based
     * on which of 8 regions they fall into.
     * 
     * Algorithm:
     * 1. For each pixel, compute f, g (mean), h (median) from 3×3 neighborhood
     * 2. Determine which of 8 regions the triple (f, g, h) falls into
     * 3. Apply region-specific correction:
     *    - Regions 0-1: No correction
     *    - Regions 2-3: f* = (g + h)/2
     *    - Regions 4-5: g* = (f + h)/2
     *    - Regions 6-7: f* = g* = h
     * 4. Compute final value: f(x,y) = (f* + g* + h*)/3
     */
    
    // Convert to grayscale if needed
    cv::Mat grayImage;
    if (input.channels() == 3) {
        cv::cvtColor(input, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = input.clone();
    }
    
    // Convert to float for precise computation
    cv::Mat floatImage;
    grayImage.convertTo(floatImage, CV_32F);
    
    // Create output matrix
    cv::Mat reconstructed = cv::Mat::zeros(floatImage.size(), CV_32F);
    
    int rows = floatImage.rows;
    int cols = floatImage.cols;
    
    // Process each pixel
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // Get pixel value f(x,y)
            float f = floatImage.at<float>(y, x);
            
            // Compute mean g(x,y) of 3×3 neighborhood
            float sum = 0.0f;
            int count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
                        sum += floatImage.at<float>(ny, nx);
                        count++;
                    }
                }
            }
            float g = sum / count;
            
            // Compute median h(x,y) of 3×3 neighborhood
            std::vector<float> neighborhood;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
                        neighborhood.push_back(floatImage.at<float>(ny, nx));
                    }
                }
            }
            std::sort(neighborhood.begin(), neighborhood.end());
            float h = neighborhood[neighborhood.size() / 2];
            
            // Determine region and apply correction (Section 3.1, Eq. (1)-(4))
            float f_star = f;
            float g_star = g;
            float h_star = h;

            constexpr float kTieTolerance = 1.0f;  // keeps near-diagonal triples unmodified
            const RegionGroup group = classifyRegionGroup(f, g, h, kTieTolerance);

            switch (group) {
                case RegionGroup::GROUP_0_1:
                    // Keep original triple
                    break;
                case RegionGroup::GROUP_2_3:
                    f_star = (g + h) / 2.0f;
                    break;
                case RegionGroup::GROUP_4_5:
                    g_star = (f + h) / 2.0f;
                    break;
                case RegionGroup::GROUP_6_7:
                    f_star = h;
                    g_star = h;
                    break;
            }
            
            // Compute final reconstructed value (Equation 4)
            float reconstructedValue = (f_star + g_star + h_star) / 3.0f;
            reconstructed.at<float>(y, x) = reconstructedValue;
        }
    }
    
    // Convert back to appropriate format
    if (input.channels() == 3) {
        // Convert grayscale back to BGR
        reconstructed.convertTo(reconstructed, CV_8U);
        cv::cvtColor(reconstructed, output, cv::COLOR_GRAY2BGR);
    } else {
        reconstructed.convertTo(output, CV_8U);
    }
}

}
