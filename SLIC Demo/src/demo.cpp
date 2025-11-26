// Demo.cpp
// Demonstrates SLIC
// Author: Chandler Calkins

#include <iostream>
#include <string>
#ifdef _WIN32
    #include <direct.h>
    #define chdir _chdir
#else
    #include <unistd.h>
#endif
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc/slic.hpp>
using namespace cv;

// main - Generates superpixels for an images using SLIC and displays those superpixels on the image.
//
// Preconditions:
//
// There is a valid .jpg or .png file in the project folder and the `imread()` call that creates the `input_image` object reads from that file.
//
// Postconditions:
//
// A file called output.jpg should be in the project folder.
int main(int argc, char* argv[])
{
	// Move out of build/Debug into root of project folder
	// Use this for VSCode, comment out for Visual Studio / actual submission
	_chdir("../../");

	// Reads the input image
	const Mat input_image = imread("input.jpg");

	// Creates window to display output to
	const String window_name = "Superpixels";
	namedWindow(window_name);

	const int avg_superpixel_size = 100; // Default: 100
	const float smoothness = 100.0f; // Default: 10.0
	const int min_superpixel_size_percent = 4;
	Ptr<ximgproc::SuperpixelSLIC> slic = ximgproc::createSuperpixelSLIC(input_image, ximgproc::SLIC, avg_superpixel_size, smoothness);
	slic->iterate();
	slic->enforceLabelConnectivity(min_superpixel_size_percent);

	// Gets 2D array of the superpixel each pixel is a part of
	Mat labels;
	slic->getLabels(labels);
	int superpixel_count = slic->getNumberOfSuperpixels();

	// Counts how many pixels are in each superpixel
	unsigned long* pixel_count = (unsigned long*) calloc(superpixel_count, sizeof(unsigned long));
	for (int row = 0; row < labels.rows; row += 1)
	{
		for (int col = 0; col < labels.cols; col += 1)
		{
			pixel_count[labels.at<int>(row, col)] += 1;
		}
	}
	// Prints out the pixel count of each superpixel
	for (int i = 0; i < superpixel_count; i += 1)
	{
		std::cout << i << ": " << pixel_count[i] << std::endl;
	}

	// Gets overlay image of superpixels
	Mat superpixels;
	slic->getLabelContourMask(superpixels);

	// Creates the output image of superpixels
	Mat output(input_image);
	// Set each pixel in output to white if it's a superpixel border
	for (int row = 0; row < output.rows; row += 1)
	for (int col = 0; col < output.cols; col += 1)
	{
		if (superpixels.at<uchar>(row, col) != 0)
		{
			output.at<Vec3b>(row, col)[0] = superpixels.at<uchar>(row, col);
			output.at<Vec3b>(row, col)[1] = superpixels.at<uchar>(row, col);
			output.at<Vec3b>(row, col)[2] = superpixels.at<uchar>(row, col);
		}
	}

	// Displays output to a window
	imshow(window_name, output);
	waitKey(0);
	
	// Write output to an image file
	imwrite("output.jpg", output);

	return 0;
}
