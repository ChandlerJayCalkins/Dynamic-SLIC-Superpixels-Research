/**
 * @file histogram_reconstruction.cpp
 * @brief Implementation of 3D histogram reconstruction for MRI images
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
        return RegionGroup::GROUP_2_3;  // f is outlier
    }

    if (distanceFG > distanceFH + tieTolerance &&
        distanceGH > distanceFH + tieTolerance) {
        return RegionGroup::GROUP_4_5;  // g is outlier
    }

    if (distanceFH > distanceFG + tieTolerance &&
        distanceGH > distanceFG + tieTolerance) {
        return RegionGroup::GROUP_6_7;  // h is outlier
    }

    return RegionGroup::GROUP_0_1;  // all relatively close
}

} // namespace

namespace ltridp_slic_improved {

void Preprocessor::apply3DHistogramReconstruction(const cv::Mat& input, cv::Mat& output) {
    /**
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
    
    // Convert to float for precision
    cv::Mat floatImage;
    grayImage.convertTo(floatImage, CV_32F);
    
    cv::Mat reconstructed = cv::Mat::zeros(floatImage.size(), CV_32F);
    
    int rows = floatImage.rows;
    int cols = floatImage.cols;
    
    // Process each pixel
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
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
            
            float f_star = f;
            float g_star = g;
            float h_star = h;

            constexpr float kTieTolerance = 0.0f;  // can increase later to allow more ties
            const RegionGroup group = classifyRegionGroup(f, g, h, kTieTolerance);

            switch (group) {
                case RegionGroup::GROUP_0_1:
                    // no correction
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
            
            // Compute final reconstructed value
            float reconstructedValue = (f_star + g_star + h_star) / 3.0f;
            reconstructed.at<float>(y, x) = reconstructedValue;
        }
    }
    
    // Convert back to original format
    if (input.channels() == 3) {
        reconstructed.convertTo(reconstructed, CV_8U);
        cv::cvtColor(reconstructed, output, cv::COLOR_GRAY2BGR);
    } else {
        reconstructed.convertTo(output, CV_8U);
    }
}
}
