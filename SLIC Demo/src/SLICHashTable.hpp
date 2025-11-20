#ifndef SLICHASHTABLE_HPP
#define SLICHASHTABLE_HPP

#include <opencv2/core/mat.hpp>

class SLICHashTable {
    public:
        // called for hashing segmented images, and storing them in the instance of this class
        void Hash(const cv::Mat& input_image, const cv::Mat& labels) {
            // deep copies
            input_image_cpy(input_image.clone());
            labels_cpy(labels.clone());
        }

    private:
        // copies of the Mats, owned by the class
        cv::Mat input_image_cpy;
        cv::Mat labels_cpy;

        // hash table structure
};


#endif