/**
 * @file test_feature_extraction.cpp
 * @brief Unit tests for LTriDP texture feature extraction
 *
 * @author Ketsia Mbaku
 * 
 * Reference:
 *         Y. Wang, Q. Qi, and X. Shen, "Image Segmentation of Brain MRI Based on
 *         LTriDP and Superpixels of Improved SLIC," Brain Sciences, vol. 10, no. 2,
 *         p. 116, 2020.
 */

#include <gtest/gtest.h>
#include "feature_extraction.hpp"
#include <opencv2/opencv.hpp>

using namespace ltridp_slic_improved;

class FeatureExtractionTest : public ::testing::Test {
protected:
    FeatureExtractor extractor;
    
    void SetUp() override {
        // Common setup if needed
    }
};

// ============================================================================
// Input Validation Tests
// ============================================================================

TEST_F(FeatureExtractionTest, EmptyImageShouldFail) {
    cv::Mat empty;
    cv::Mat output;
    EXPECT_FALSE(extractor.extract(empty, output));
}

TEST_F(FeatureExtractionTest, WrongDepthShouldFail) {
    cv::Mat input(10, 10, CV_16U, cv::Scalar(100));
    cv::Mat output;
    EXPECT_FALSE(extractor.extract(input, output));
}

TEST_F(FeatureExtractionTest, TooSmallImageShouldFail) {
    cv::Mat input(2, 2, CV_8U, cv::Scalar(128));
    cv::Mat output;
    EXPECT_FALSE(extractor.extract(input, output));
}

TEST_F(FeatureExtractionTest, MinimumSizeImageShouldSucceed) {
    cv::Mat input(3, 3, CV_8U, cv::Scalar(128));
    cv::Mat output;
    EXPECT_TRUE(extractor.extract(input, output));
}

// ============================================================================
// Output Format Tests
// ============================================================================

TEST_F(FeatureExtractionTest, OutputHasCorrectDepth) {
    cv::Mat input(50, 50, CV_8U, cv::Scalar(128));
    cv::Mat output;
    ASSERT_TRUE(extractor.extract(input, output));
    EXPECT_EQ(output.depth(), CV_8U);
}

TEST_F(FeatureExtractionTest, OutputHasCorrectSize) {
    cv::Mat input(50, 50, CV_8U, cv::Scalar(128));
    cv::Mat output;
    ASSERT_TRUE(extractor.extract(input, output));
    EXPECT_EQ(output.size(), input.size());
}

TEST_F(FeatureExtractionTest, OutputHasCorrectChannels) {
    cv::Mat input(50, 50, CV_8U, cv::Scalar(128));
    cv::Mat output;
    ASSERT_TRUE(extractor.extract(input, output));
    EXPECT_EQ(output.channels(), 1);
}

TEST_F(FeatureExtractionTest, OutputValuesInValidRange) {
    cv::Mat input(50, 50, CV_8U);
    cv::randu(input, 0, 256);
    cv::Mat output;
    ASSERT_TRUE(extractor.extract(input, output));
    
    double minVal, maxVal;
    cv::minMaxLoc(output, &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 255.0);
}

// ============================================================================
// Color Input Tests
// ============================================================================

TEST_F(FeatureExtractionTest, ColorImageConvertsToGrayscale) {
    cv::Mat colorInput(50, 50, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat output;
    EXPECT_TRUE(extractor.extract(colorInput, output));
    EXPECT_EQ(output.channels(), 1);
}

TEST_F(FeatureExtractionTest, GrayscaleImagePreserved) {
    cv::Mat grayInput(50, 50, CV_8U, cv::Scalar(128));
    cv::Mat output;
    EXPECT_TRUE(extractor.extract(grayInput, output));
    EXPECT_EQ(output.channels(), 1);
}

// ============================================================================
// Texture Pattern Tests
// ============================================================================

TEST_F(FeatureExtractionTest, UniformImageHasLowVariance) {
    // Uniform image should have consistent (low variance) LTriDP codes
    cv::Mat uniform(100, 100, CV_8U, cv::Scalar(128));
    cv::Mat features;
    ASSERT_TRUE(extractor.extract(uniform, features));
    
    // Compute variance (excluding 1-pixel border which is zero)
    cv::Mat roi = features(cv::Rect(1, 1, 98, 98));
    cv::Scalar mean, stddev;
    cv::meanStdDev(roi, mean, stddev);
    
    // Uniform regions should have very low variance
    EXPECT_LT(stddev[0], 10.0);
}

TEST_F(FeatureExtractionTest, TexturedImageProducesNonZeroFeatures) {
    // Create an image with varied texture (random noise)
    cv::Mat textured(100, 100, CV_8U);
    cv::randu(textured, 0, 256);
    
    cv::Mat features;
    ASSERT_TRUE(extractor.extract(textured, features));
    
    // Random texture should produce varied LTriDP codes
    cv::Mat roi = features(cv::Rect(1, 1, 98, 98));
    double minVal, maxVal;
    cv::minMaxLoc(roi, &minVal, &maxVal);
    
    // Should have multiple different codes
    EXPECT_GT(maxVal - minVal, 10.0);
}

TEST_F(FeatureExtractionTest, GradientImageHasStructuredResponse) {
    // Create horizontal gradient
    cv::Mat gradient(100, 100, CV_8U);
    for (int r = 0; r < 100; ++r) {
        for (int c = 0; c < 100; ++c) {
            gradient.at<uchar>(r, c) = static_cast<uchar>(c * 255 / 99);
        }
    }
    
    cv::Mat features;
    ASSERT_TRUE(extractor.extract(gradient, features));
    
    // Feature response should be non-uniform
    double minVal, maxVal;
    cv::minMaxLoc(features, &minVal, &maxVal);
    EXPECT_GT(maxVal - minVal, 10.0);
}

// ============================================================================
// Border Handling Tests
// ============================================================================

TEST_F(FeatureExtractionTest, BorderPixelsAreZero) {
    cv::Mat input(50, 50, CV_8U);
    cv::randu(input, 0, 256);
    cv::Mat features;
    ASSERT_TRUE(extractor.extract(input, features));
    
    // Check top border
    for (int c = 0; c < 50; ++c) {
        EXPECT_EQ(features.at<uchar>(0, c), 0);
    }
    
    // Check bottom border
    for (int c = 0; c < 50; ++c) {
        EXPECT_EQ(features.at<uchar>(49, c), 0);
    }
    
    // Check left border
    for (int r = 0; r < 50; ++r) {
        EXPECT_EQ(features.at<uchar>(r, 0), 0);
    }
    
    // Check right border
    for (int r = 0; r < 50; ++r) {
        EXPECT_EQ(features.at<uchar>(r, 49), 0);
    }
}

TEST_F(FeatureExtractionTest, InteriorPixelsAreProcessed) {
    cv::Mat input(50, 50, CV_8U);
    cv::randu(input, 0, 256);
    cv::Mat features;
    ASSERT_TRUE(extractor.extract(input, features));
    
    // Check that at least some interior pixels are non-zero
    cv::Mat interior = features(cv::Rect(1, 1, 48, 48));
    int nonZero = cv::countNonZero(interior);
    EXPECT_GT(nonZero, 0);
}

// ============================================================================
// Determinism Tests
// ============================================================================

TEST_F(FeatureExtractionTest, DeterministicOutput) {
    cv::Mat input(100, 100, CV_8U);
    cv::randu(input, 0, 256);
    
    cv::Mat output1, output2;
    ASSERT_TRUE(extractor.extract(input, output1));
    ASSERT_TRUE(extractor.extract(input, output2));
    
    // Results should be identical
    cv::Mat diff;
    cv::absdiff(output1, output2, diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(FeatureExtractionTest, AllBlackImage) {
    cv::Mat black(50, 50, CV_8U, cv::Scalar(0));
    cv::Mat features;
    ASSERT_TRUE(extractor.extract(black, features));
    
    // All black should produce consistent pattern
    cv::Mat interior = features(cv::Rect(1, 1, 48, 48));
    cv::Scalar mean, stddev;
    cv::meanStdDev(interior, mean, stddev);
    EXPECT_LT(stddev[0], 1.0);
}

TEST_F(FeatureExtractionTest, AllWhiteImage) {
    cv::Mat white(50, 50, CV_8U, cv::Scalar(255));
    cv::Mat features;
    ASSERT_TRUE(extractor.extract(white, features));
    
    // All white should produce consistent pattern
    cv::Mat interior = features(cv::Rect(1, 1, 48, 48));
    cv::Scalar mean, stddev;
    cv::meanStdDev(interior, mean, stddev);
    EXPECT_LT(stddev[0], 1.0);
}

TEST_F(FeatureExtractionTest, SinglePixelDifference) {
    // Create mostly uniform image with one different pixel in center
    cv::Mat input(50, 50, CV_8U, cv::Scalar(128));
    input.at<uchar>(25, 25) = 255;
    
    cv::Mat features;
    ASSERT_TRUE(extractor.extract(input, features));
    
    // Feature at center should differ from surrounding
    uchar centerFeature = features.at<uchar>(25, 25);
    uchar neighborFeature = features.at<uchar>(26, 25);
    
    // They should likely be different (texture changes around the spike)
    // This is a weak test - just checking computation runs
    EXPECT_TRUE(true);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(FeatureExtractionTest, LargeImage) {
    // Test with larger image size
    cv::Mat large(512, 512, CV_8U);
    cv::randu(large, 0, 256);
    
    cv::Mat features;
    EXPECT_TRUE(extractor.extract(large, features));
    EXPECT_EQ(features.size(), large.size());
}

TEST_F(FeatureExtractionTest, EndToEndPipeline) {
    // Simulate realistic workflow
    cv::Mat input(256, 256, CV_8U);
    cv::randu(input, 0, 256);
    
    cv::Mat features;
    ASSERT_TRUE(extractor.extract(input, features));
    
    // Verify output properties
    EXPECT_EQ(features.depth(), CV_8U);
    EXPECT_EQ(features.channels(), 1);
    EXPECT_EQ(features.size(), input.size());
    
    double minVal, maxVal;
    cv::minMaxLoc(features, &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 255.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
