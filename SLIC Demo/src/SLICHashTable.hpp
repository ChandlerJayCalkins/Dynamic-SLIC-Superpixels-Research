#ifndef SLICHASHTABLE_HPP
#define SLICHASHTABLE_HPP

#include <opencv2/core/mat.hpp>
#include <stdlib.h>

/*** a HashKey pointer will point to an array of size superpixel_count (+1 if labels start at 1), 
in which HashKeys will occupy the indices of the superpixels they represent.
Each pixel will be looped through and the respective HashKey structs of the superpixels they belong
to will be dynamically updated.  ***/
typedef struct {
    signed long l_tot, a_tot, b_tot;
    std::pair<int, int> x_range, y_range;
    const cv::Mat *original_image;
    unsigned long pixel_count;
} HashKey;

class SLICHashTable {
    private:
        cv::Mat hist;
        const int n = 5;
        // this implementation assumes 8-bit unsigned integer images
        const int lab_buckets = 16;
        const int lab_bucket_size = 256 / lab_buckets;
        const int max_img_w = 3840;
        const int max_img_h = 2160;
        const int x_buckets = 10;
        const int y_buckets = 10;
        const int x_bucket_size = max_img_w / x_buckets;
        const int y_bucket_size = max_img_h / y_buckets;
        int dims[5] = {lab_buckets, lab_buckets, lab_buckets, x_buckets, y_buckets};

    public:
        // hash table structure: sparse 5D histogram where bucket location is discretized based on x range, y range, and average color
        SLICHashTable() {
            hist = cv::Mat(n, dims, CV_32FC1, cv::Scalar::all(0));
        }

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
                    HashKey &curr = superpixels[sp];
                    curr.l_tot += lab_pixel[0];
                    curr.a_tot += lab_pixel[1];
                    curr.b_tot += lab_pixel[2];
                    if (curr.pixel_count == 0) {
                        curr.x_range.first = col;
                        curr.x_range.second = col;
                        curr.y_range.first = row;
                        curr.y_range.second = row;
                        curr.original_image = &input_image;
                    } else {
                        if (curr.x_range.first > col) curr.x_range.first = col;
                        if (curr.x_range.second < col) curr.x_range.second = col;
                        if (curr.y_range.first > row) curr.y_range.first = row;
                        if (curr.y_range.second < row) curr.y_range.second = row;
                    }
                    curr.pixel_count += 1;
                    // hash superpixel if all subpixels have been found
                    // sp hashes to histogram[l bucket][a bucket][b bucket][x bucket][y bucket]
                    if (curr.pixel_count == pixel_count[sp]) {

                    }
                }
            }
        }
};


#endif