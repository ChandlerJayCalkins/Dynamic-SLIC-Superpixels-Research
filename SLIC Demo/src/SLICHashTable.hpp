#ifndef SLICHASHTABLE_HPP
#define SLICHASHTABLE_HPP

#include <opencv2/core/mat.hpp>
#include <stdlib.h>

/*** a HashKey pointer will point to an array of size superpixel_count (+1 if labels start at 1), 
in which HashKeys will occupy the indices of the superpixels they represent.
Each pixel will be looped through and the respective HashKey structs of the superpixels they belong
to will be dynamically updated.  ***/
typedef struct {
    signed long l_tot;
    signed long a_tot;
    signed long b_tot;
    std::pair<int, int> x_range;
    std::pair<int, int> y_range;
    const cv::Mat *original_image;
    unsigned long pixel_count;
} HashKey;

class SLICHashTable {
    public:
        // called for hashing segmented images, and storing them in the instance of this class
        // expects a cielab image for input_image
        void Hash(const cv::Mat& input_image,
                  const cv::Mat& labels,
                  int superpixel_count,
                  unsigned long* pixel_count)
        {
            HashKey *superpixels = (HashKey*) calloc(superpixel_count, sizeof(HashKey));
            for (int row = 0; row < labels.rows; row++) {
                for (int col = 0; col < labels.cols; col++) {
                    cv::Vec3b lab_pixel = input_image.at<cv::Vec3b>(row, col);
                    int sp = labels.at<int>(row, col);
                    HashKey curr = superpixels[sp];
                    curr.l_tot += lab_pixel[0];
                    curr.a_tot += lab_pixel[1];
                    curr.b_tot += lab_pixel[2];
                    if (curr.x_range.first > col) curr.x_range.first = col;
                    if (curr.x_range.second < col) curr.x_range.second = col;
                    if (curr.y_range.first > row) curr.y_range.first = row;
                    if (curr.y_range.second < row) curr.y_range.second = row;
                    if (curr.original_image != &input_image) curr.original_image = &input_image;
                    curr.pixel_count += 1;
                    // hash superpixel if all subpixels have been found
                    if (curr.pixel_count == *pixel_count) {
                        
                    }
                }
            }
        }

    private:
        // hash table structure
};


#endif