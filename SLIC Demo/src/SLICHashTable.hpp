#ifndef SLICHASHTABLE_HPP
#define SLICHASHTABLE_HPP

#include <opencv2/core/mat.hpp>

/*** a HashKey pointer will point to an array of size superpixel_count (+1 if labels start at 1), 
in which HashKeys will occupy the indices of the superpixels they represent.
Each pixel will be looped through and the respective HashKey structs of the superpixels they belong
to will be dynamically updated.  ***/
typedef struct {
    int l_avg;
    int a_avg;
    int b_avg;
    std::pair<int, int> x_range;
    std::pair<int, int> y_range;
    cv::Mat *original_image;
} HashKey;

class SLICHashTable {
    public:
        // called for hashing segmented images, and storing them in the instance of this class
        void Hash(const cv::Mat& input_image,
                  const cv::Mat& labels,
                  int superpixel_count)
        {
            
        }

    private:
        // hash table structure
};


#endif